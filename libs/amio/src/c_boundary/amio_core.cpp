// amio_core.cpp -- Production C++ implementation for the private C++ entry
// points declared in `amio_core.hpp`.
//
// Coordinates the complete C-API boundary, handle tables, Worker_Pool,
// Staging_Pool, and NetCDF/NCZarr backends.
//
// The handle-validation + exception-translation cordon in
// `amio_api.cpp` is fully functional and does NOT depend on any of
// the bodies below.
//
// `process_handle_table()` returns a function-local static so that
// the table is constructed on first use and destroyed at process
// exit, after every other static AMIO_Core object has been torn
// down.  The Meyers-singleton idiom is thread-safe in C++11+.
//
// Task 6.3 implements: open_dataset, close_dataset, flush, close.

#include "c_boundary/amio_core.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "config/config_loader.hpp"
#include "factory/backend_driver.hpp"
#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"
#include "workers/comm_split.hpp"
#include "workers/worker_pool.hpp"

// Open-time dataset configuration source.
//
// The Backend_Driver open_write/open_read methods take a
// `const conf::Config&` carrying the dataset-level keys (path / uri /
// data_model / codec / ...) that the drivers parse directly from the
// manifest file.  Those keys are NOT captured in the loader's `Config`
// struct, so the configuration must be built from the manifest file
// itself at open time (task 5, design §2).
//
// The manifest is parsed into a `conf::Config` via `from_file` and
// stored in the DatasetRecord so it outlives the driver (Req 13.1,
// 13.3).  The driver receives it by const reference (Req 13.2).
#include <conf/config.hpp>

namespace amio::detail {

HandleTable &process_handle_table() {
    static HandleTable table;
    return table;
}

// ---------------------------------------------------------------
// amio_init -- task 2: parse the manifest and build the runtime.
//
// Parses the manifest via ConfigLoader::parse and, on success,
// constructs the Staging_Pool and Worker_Pool from the parsed
// Config so that read/write paths are served from real buffers
// instead of failing with backpressure.
//
//   * Manifest missing / invalid -> the loader's error code
//     (AMIO_ERR_MANIFEST_NOT_FOUND / AMIO_ERR_MANIFEST_INVALID),
//     no core handle minted (Req 1).
//   * Pool construction failure   -> AMIO_ERR_BACKEND_FAILURE,
//     no core handle minted (Req 1.4).
//
// Validates: R1.1, R1.2, R1.3, R1.4
// ---------------------------------------------------------------
amio_status_t init(const char *manifest_path, amio_core_handle *out_core) {
    // ---- Step 1: Parse + validate the manifest (Req 1) ----
    Config cfg{};
    ValidationError verr{};
    amio_err_t parse_rc = ConfigLoader::parse(std::string(manifest_path), cfg, verr);
    if (parse_rc != AMIO_OK) {
        std::cerr << "[AMIO ERROR] in manifest '" << manifest_path << "' (field: " << verr.field_path << "): " << verr.message << std::endl;
        // MANIFEST_NOT_FOUND / MANIFEST_INVALID -- no handle minted.
        return static_cast<amio_status_t>(parse_rc);
    }

    // ---- Steps 2-4: build the runtime + mint the handle from the
    // parsed Config.  Shared with the string-based init path so the
    // pool-construction / LOGS-init / handle-mint logic lives in one
    // place (design §"Shared-body factoring").
    return init_from_config(cfg, out_core);
}

// ---------------------------------------------------------------
// init_from_config -- steps 2-4 of the original init(), factored out
// so both the path-based init() and the string-based init_from_string()
// share the identical pool-construction + LOGS-init + handle-mint path
// from an already-parsed Config (design §"Shared-body factoring").
//
// Validates: R1.1, R1.2, R1.3, R1.4
// ---------------------------------------------------------------
amio_status_t init_from_config(const Config &cfg, amio_core_handle *out_core) {
    // ---- Step 2: Construct the runtime pools (Req 1.1, 1.2) ----
    auto core = std::make_unique<AMIO_Core>();
    try {
        core->staging_pool = std::make_unique<StagingPool>(cfg.staging_pool.buffer_count, cfg.staging_pool.buffer_capacity_bytes,
                                                           static_cast<std::int64_t>(cfg.staging_timeout_ms), cfg.staging_pool.max_buffer_count,
                                                           StagingPool::GrowMode::Grow);

        if (cfg.worker_pool.threads > 0) {
            WorkerPoolConfig wp{};
            wp.thread_count = cfg.worker_pool.threads;
            core->worker_pool = std::make_unique<WorkerPool>(wp);
        } else {
            core->worker_pool = nullptr;
        }
    } catch (...) {
        // Pool construction failed -- AMIO_ERR_BACKEND_FAILURE, no
        // handle minted (Req 1.4).  `core` is destroyed here, tearing
        // down any partially-constructed pool.
        return AMIO_ERR_BACKEND_FAILURE;
    }

    core->staging_timeout_ms = static_cast<std::int64_t>(cfg.staging_timeout_ms);

    // ---- Step 2b: Initialize LOGS after communicator split (Req 6.5, 6.8) ----
    //
    // The WorkerPool has been constructed with the IOCommunicator from
    // the comm_split result.  Configure the Logger with the I/O
    // communicator handle so that MPI rank stamps reflect the I/O
    // sub-communicator rank rather than the world rank.
    //
    // memory_order_release ensures all prior writes to the logger
    // (configure_communicator, set_threshold) are visible to other
    // threads that load logs_initialized with memory_order_acquire.
#ifdef AMIO_HAS_MPI
    if (core->worker_pool) {
        core->logger.configure_communicator(core->worker_pool->io_communicator().handle());
    } else {
        core->logger.configure_communicator(g_amio_parent_comm);
    }
#else
    // Without MPI the Logger receives MPI_COMM_NULL (a no-op
    // configuration that leaves the rank as the sentinel -1).
    core->logger.configure_communicator(MPI_COMM_NULL);
#endif
    core->logger.set_threshold(logs::Severity_Level::INFO);
    core->logs_initialized.store(true, std::memory_order_release);

    // ---- Step 3: Mint the core handle (Req 1) ----
    auto token = process_handle_table().insert(HandleKind::Core, core.get());
    core->core_token = token;
    *out_core = HandleTable::to_ptr(token);
    core.release();  // ownership transferred to the handle table
    return AMIO_OK;
}

// ---------------------------------------------------------------
// init_from_string -- task 3.1: initialize the runtime from an
// in-memory manifest string.
//
// Behaves exactly like init(), except the manifest is supplied as a
// NUL-terminated string in `format` ("yaml"/"json") rather than read
// from a file path.  It parses via ConfigLoader::parse_string and, on
// success, delegates to the shared init_from_config helper so the
// pool-construction / LOGS-init / handle-mint path is identical to the
// file-based path (design §"Shared-body factoring").
//
//   * parse_string failure -> the loader's error code
//     (AMIO_ERR_MANIFEST_INVALID for bad content); because there is no
//     file, AMIO_ERR_MANIFEST_NOT_FOUND cannot occur.  No handle
//     minted (Req 8.1).
//   * Pool construction failure -> AMIO_ERR_BACKEND_FAILURE, no handle
//     minted (Req 1.4).
//
// Validates: R2.1, R2.2, R8.1
// ---------------------------------------------------------------
amio_status_t init_from_string(const char *manifest_content, const char *format, amio_core_handle *out_core) {
    // ---- Step 1: Parse + validate the in-memory manifest (Req 2.1) ----
    Config cfg{};
    ValidationError verr{};
    amio_err_t parse_rc = ConfigLoader::parse_string(std::string(manifest_content), std::string(format), cfg, verr);
    if (parse_rc != AMIO_OK) {
        std::cerr << "[AMIO ERROR] in in-memory manifest (field: " << verr.field_path << "): " << verr.message << std::endl;
        // Bad content -> MANIFEST_INVALID; no handle minted.
        return static_cast<amio_status_t>(parse_rc);
    }

    // ---- Steps 2-4: build the runtime + mint the handle, shared with
    // the path-based init() (design §"Shared-body factoring").
    return init_from_config(cfg, out_core);
}

// ---------------------------------------------------------------
// amio_finalize -- task 3: drain the runtime and invalidate the core.
//
// Teardown ordering (Req 1.5, 1.6):
//   1. Drain + join the Worker_Pool so no background prefetch/write
//      task is still running against dataset/pool state.
//   2. Flush + close every open dataset and release its handle.
//   3. Release the Staging_Pool (now that no worker thread can hold a
//      StagingBuffer reference).
//   4. Invalidate the core handle and free the core -- unconditionally,
//      even if the Staging_Pool teardown above threw.
//
// Validates: R1.5, R1.6
// ---------------------------------------------------------------
amio_status_t finalize(void *core_payload) {
    auto *core = static_cast<AMIO_Core *>(core_payload);

    // ---- Step 1: Drain + join the Worker_Pool (Req 1.5) ----
    //
    // The runtime pools are live as of task 2: background prefetch /
    // write tasks dispatched to the Worker_Pool may still hold
    // references to dataset PrefetchQueues, Backend_Drivers, and
    // Staging_Pool buffers.  Quiesce and join the Worker_Pool BEFORE
    // tearing down datasets so that no worker thread touches dataset
    // state during teardown (otherwise the dataset records below are
    // freed out from under an in-flight fetch).  Draining here also
    // establishes the Req 1.5 precondition for the Staging_Pool
    // release in step 3: once the pool is joined, no thread can hold
    // a StagingBuffer reference.
    if (core->worker_pool) {
        core->worker_pool->drain();
        core->worker_pool.reset();  // joins all worker threads
    }

    // ---- Step 2: Close all open datasets ----
    {
        std::lock_guard<std::mutex> lock(core->datasets_mu);
        for (auto &[id, record] : core->datasets) {
            if (record && record->driver) {
                try {
                    record->driver->flush();
                    record->driver->close();
                } catch (...) {
                    // Best-effort during finalize.
                }
            }
            // Release the dataset handle from the table.
            if (record && record->token != 0) {
                process_handle_table().release(record->token, HandleKind::Dataset);
            }
        }
        core->datasets.clear();
    }

    // ---- Step 3: Release the Staging_Pool (Req 1.5) ----
    //
    // The Worker_Pool has been drained/joined (step 1) and every
    // dataset is closed (step 2), so no thread can still hold a
    // StagingBuffer reference.  reset() runs the pool destructor,
    // freeing all buffers.  Wrap it so that even if teardown throws
    // the core handle is still invalidated below (Req 1.6).
    try {
        core->staging_pool.reset();
    } catch (...) {
        // Best-effort: swallow the teardown failure and fall through
        // to invalidate the core handle regardless (Req 1.6).
    }

    // ---- Step 4: Invalidate the core handle (Req 1.5, 1.6) ----
    //
    // Always reached, even when the Staging_Pool release in step 3
    // throws, so the core handle is guaranteed invalid after finalize.
    process_handle_table().release(core->core_token, HandleKind::Core);
    delete core;
    return AMIO_OK;
}

// ---------------------------------------------------------------
// amio_open_dataset -- task 5 (open the backend for reading/writing)
//
// Validates core handle → extracts backend key from config →
// calls BackendFactory::build() → opens the driver in the requested
// mode (open_write / open_read) with a conf::Config built from
// config_path → on success creates the dataset handle in the handle
// table → returns the dataset handle.
//
// On factory lookup failure → AMIO_ERR_UNKNOWN_BACKEND, no handle.
// On driver open failure → AMIO_ERR_BACKEND_FAILURE, no handle.
//
// For read-mode datasets the parsed Config is retained on the
// DatasetRecord (for prefetch depth / read timeout); the per-variable
// PrefetchQueue is created lazily on first read (design Key Design
// Decision), so no eager prefetch scheduling happens here.
// ---------------------------------------------------------------
amio_status_t open_dataset(void *core_payload, const char *config_path, std::int32_t mode, amio_dataset_handle *out_dataset) {
    // Parse the dataset configuration to extract the backend key.
    Config config{};
    ValidationError verr{};
    amio_err_t parse_rc = ConfigLoader::parse(std::string(config_path), config, verr);
    if (parse_rc != AMIO_OK) {
        // If the file doesn't exist, return MANIFEST_NOT_FOUND;
        // otherwise return the parse error.
        return static_cast<amio_status_t>(parse_rc);
    }

    // Build a conf::Config from the manifest file for the driver's
    // open_write / open_read (Req 13.1, 13.2).  The dataset-level
    // configuration (path / uri / data_model / codec / ...) is parsed
    // by each driver directly from this Config.
    conf::Config manifest_cfg = conf::Config::from_file(std::string(config_path));

    // Delegate the factory-build + communicator-set + open + record
    // construction to the shared config-based helper so the path-based
    // and string-based (task 3.1) open variants share one body.
    return open_dataset_from_config(core_payload, config, std::move(manifest_cfg), mode, out_dataset);
}

// ---------------------------------------------------------------
// open_dataset_from_config -- shared open path.
//
// Runs the factory-build + communicator-set + open_read/open_write +
// DatasetRecord construction path that open_dataset used inline.  Both
// the path-based open_dataset and the string-based
// open_dataset_from_string (task 3.1) reach this helper after producing
// a parsed `config` (backend key + read-path knobs) and a
// `manifest_cfg` (the driver's dataset-level configuration).  The
// `manifest_cfg` is taken by value and moved into the DatasetRecord so
// it outlives the driver (Req 13.1, 13.3).
//
// On factory lookup failure -> AMIO_ERR_UNKNOWN_BACKEND, no handle.
// On driver open failure     -> AMIO_ERR_BACKEND_FAILURE, no handle.
// ---------------------------------------------------------------
amio_status_t open_dataset_from_config(void *core_payload, const Config &config, conf::Config manifest_cfg, std::int32_t mode,
                                       amio_dataset_handle *out_dataset) {
    auto *core = static_cast<AMIO_Core *>(core_payload);

    // Extract backend key from configuration.
    const std::string &backend_key = config.backend;

    // Look up the driver in the factory.
    amio_err_t factory_err = AMIO_OK;
    auto driver = BackendFactory::instance().build(backend_key, factory_err);
    if (!driver) {
        // Factory lookup failed — AMIO_ERR_UNKNOWN_BACKEND, no handle.
        return static_cast<amio_status_t>(factory_err);
    }

#ifdef AMIO_HAS_MPI
    try {
        if (core != nullptr) {
            if (core->worker_pool) {
                driver->set_communicator(core->worker_pool->io_communicator().handle());
            } else {
                driver->set_communicator(g_amio_parent_comm);
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "[AMIO ERROR] open_dataset failed while setting communicator: " << e.what() << std::endl;
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        std::cerr << "[AMIO ERROR] open_dataset failed while setting communicator: unknown exception" << std::endl;
        return AMIO_ERR_BACKEND_FAILURE;
    }
#endif

    // Attempt to open the driver in the requested mode (Req 2.1).
    // On any open failure the driver throws; the catch below translates
    // to AMIO_ERR_BACKEND_FAILURE and returns no dataset handle
    // (Req 2.2).
    try {
        if (mode == AMIO_MODE_WRITE) {
            driver->open_write(manifest_cfg);
        } else /* AMIO_MODE_READ */ {
            driver->open_read(manifest_cfg);
        }
    } catch (const std::exception &e) {
        std::cerr << "[AMIO ERROR] open_dataset failed: " << e.what() << std::endl;
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        std::cerr << "[AMIO ERROR] open_dataset failed: unknown exception" << std::endl;
        return AMIO_ERR_BACKEND_FAILURE;
    }

    // Create the dataset record.  manifest_config is declared before
    // driver in DatasetRecord so that in C++ member destruction order
    // (reverse of declaration) the driver is destroyed first, ensuring
    // the driver's destructor can still safely access config values
    // during teardown (Req 13.4).
    auto record = std::make_unique<DatasetRecord>();
    record->manifest_config = std::move(manifest_cfg);
    record->driver = std::move(driver);
    record->mode = mode;
    record->dataset_id = core->next_dataset_id.fetch_add(1);
    record->core = core;  // back-pointer for write path access to staging/worker

    // Retain the parsed Config so the read path can source the prefetch
    // depth / read timeout when the per-variable PrefetchQueue is
    // created lazily on first read (Req 2.3, 2.4).  The per-variable
    // PrefetchQueue is intentionally NOT created here: with one queue
    // per variable created lazily on first amio_read (design Key Design
    // Decision), there is no variable to prefetch at open time, so the
    // eager schedule_initial() / placeholder total_timesteps block is
    // removed.
    if (mode == AMIO_MODE_READ) {
        record->dataset_config = config;
        record->prefetch_depth = config.prefetch.depth;
        record->read_timeout_s = static_cast<std::int64_t>(config.prefetch.read_timeout_s);
    }

    // Insert into the handle table.
    auto token = process_handle_table().insert(HandleKind::Dataset, record.get());
    record->token = token;

    // Register in the core's dataset map.
    {
        std::lock_guard<std::mutex> lock(core->datasets_mu);
        core->datasets[record->dataset_id] = std::move(record);
    }

    *out_dataset = HandleTable::to_ptr(token);
    return AMIO_OK;
}

// ---------------------------------------------------------------
// open_dataset_from_string -- task 3.1: open a dataset from an
// in-memory config string.
//
// Behaves exactly like open_dataset(), except the dataset config is
// supplied as a NUL-terminated string in `format` ("yaml"/"json")
// rather than read from a file path.  It parses the backend key via
// ConfigLoader::parse_string, builds the driver conf::Config via
// conf::Config::from_string, then delegates to the shared
// open_dataset_from_config helper so the factory-build +
// communicator-set + open + DatasetRecord path is identical to the
// file-based open_dataset (design §"Shared-body factoring").
//
//   * parse_string failure -> the loader's error code
//     (AMIO_ERR_MANIFEST_INVALID for bad content); because there is no
//     file, AMIO_ERR_MANIFEST_NOT_FOUND cannot occur.  No handle
//     minted (Req 8.1).
//   * factory lookup failure -> AMIO_ERR_UNKNOWN_BACKEND, no handle.
//   * driver open failure     -> AMIO_ERR_BACKEND_FAILURE, no handle.
//
// Validates: R2.1, R2.2, R8.1
// ---------------------------------------------------------------
amio_status_t open_dataset_from_string(void *core_payload, const char *config_content, const char *format, std::int32_t mode,
                                       amio_dataset_handle *out_dataset) {
    // Parse the in-memory dataset configuration to extract the backend
    // key.  Bad content maps to the loader's error code
    // (AMIO_ERR_MANIFEST_INVALID); MANIFEST_NOT_FOUND cannot occur here
    // because there is no file.
    Config config{};
    ValidationError verr{};
    amio_err_t parse_rc = ConfigLoader::parse_string(std::string(config_content), std::string(format), config, verr);
    if (parse_rc != AMIO_OK) {
        return static_cast<amio_status_t>(parse_rc);
    }

    // Build a conf::Config from the in-memory config for the driver's
    // open_write / open_read (Req 13.1, 13.2), mirroring open_dataset's
    // conf::Config::from_file.
    conf::Config manifest_cfg = conf::Config::from_string(std::string(config_content));

    // Delegate the factory-build + communicator-set + open + record
    // construction to the shared config-based helper so the path-based
    // and string-based open variants share one body.
    return open_dataset_from_config(core_payload, config, std::move(manifest_cfg), mode, out_dataset);
}

// ---------------------------------------------------------------
// amio_close_dataset -- task 6.3 / task 11
//
// Outstanding-view guard → cancel read prefetches → flush pending
// writes → surface failures → call driver->close() → release handle
// from table.
//
// Step 0 (Req 8.1): if any Memory_View for this dataset is still
// outstanding, close fails with AMIO_ERR_VIEWS_OUTSTANDING and NO
// teardown is performed -- the backend driver, its handle, and every
// staging buffer behind the outstanding views are left intact so the
// host can still release them.  The guard runs BEFORE any flush /
// cancel / close so close is a no-op when views are live.
//
// Step 1 (Req 8.3): for a read-mode dataset with no outstanding views,
// cancel every per-variable PrefetchQueue.  cancel_pending() marks the
// queue cancelled, drops in-flight pending fetches, and releases any
// completed-but-unread staging buffers back to the pool, so closing a
// reader does not leak the look-ahead window.
//
// Steps 2-4 (Req 8.2): the existing write-path flush / failure
// surfacing and the driver flush+close run unchanged once views are
// drained, releasing the Backend_Driver resources.
//
// Validates: R8.1, R8.2, R8.3
// ---------------------------------------------------------------
amio_status_t close_dataset(void *dataset_payload) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    // Step 0: Outstanding-view guard (Req 8.1).
    //
    // Reject the close before ANY teardown when one or more
    // Memory_Views for this dataset are still outstanding.  Returning
    // here leaves the driver open, the handle valid, and the staged
    // buffers retained, so the host can release the views and retry.
    if (record->outstanding_views.load() > 0) {
        return AMIO_ERR_VIEWS_OUTSTANDING;
    }

    amio_status_t result = AMIO_OK;

    // Step 1: Cancel pending prefetches for read-mode datasets (Req 8.3).
    //
    // With no views outstanding, every completed-but-unread buffer in a
    // PrefetchQueue is safe to reclaim.  Iterate the per-variable read
    // state under variables_mu and cancel each queue: cancel_pending()
    // marks the queue cancelled, clears the pending set, and releases
    // completed buffers back to the Staging_Pool.  Write-mode datasets
    // hold no PrefetchQueues, so this loop is a no-op for them.
    {
        std::lock_guard<std::mutex> lock(record->variables_mu);
        for (auto &[name, vs] : record->variables) {
            (void)name;
            if (vs && vs->queue) {
                vs->queue->cancel_pending();
            }
        }
    }

    // Step 2: Drain pending writes (block until all complete or fail).
    if (record->pending_writes.load() > 0) {
        // Block until all pending writes for this dataset complete
        // (timeout at 30s as a safety net; close should not hang forever).
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (record->pending_writes.load() > 0) {
            if (std::chrono::steady_clock::now() >= deadline) {
                break;  // best-effort: proceed to flush/close even if writes linger
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Step 3: Surface any recorded failures.
    if (record->has_failure.load()) {
        result = static_cast<amio_status_t>(record->first_failure_code);
    }

    // Step 4: Flush and close the driver.
    try {
        if (record->driver) {
            record->driver->flush();
        }
    } catch (...) {
        if (result == AMIO_OK) {
            result = AMIO_ERR_BACKEND_FAILURE;
        }
    }

    try {
        if (record->driver) {
            record->driver->close();
        }
    } catch (...) {
        if (result == AMIO_OK) {
            result = AMIO_ERR_BACKEND_FAILURE;
        }
    }

    // Step 5: Release the handle from the table.
    process_handle_table().release(record->token, HandleKind::Dataset);

    // Step 6: Remove from the core's dataset map.
    // Note: We don't have a back-pointer to the core here, so the
    // record will be cleaned up when the core is finalized or when
    // the dataset record is destroyed.  The handle is already
    // invalidated in the table, so subsequent lookups will fail.

    return result;
}

// ---------------------------------------------------------------
// amio_flush -- task 6.3
//
// Block until all pending writes for dataset complete or fail.
// Return aggregate status.
// ---------------------------------------------------------------
amio_status_t flush(void *dataset_payload, std::int64_t timeout_ms) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    // If no pending writes, return immediately.
    if (record->pending_writes.load() == 0) {
        // Check for previously recorded failures.
        if (record->has_failure.load()) {
            return static_cast<amio_status_t>(record->first_failure_code);
        }
        return AMIO_OK;
    }

    // Block until pending writes drain or timeout.
    // In the full implementation (task 9.x), this would wait on a
    // condition variable signaled by the worker pool.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 86400000);

    while (record->pending_writes.load() > 0) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return AMIO_ERR_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Surface any failures.
    if (record->has_failure.load()) {
        return static_cast<amio_status_t>(record->first_failure_code);
    }

    return AMIO_OK;
}

// ---------------------------------------------------------------
// amio_close -- delegates to close_dataset (task 6.3)
// ---------------------------------------------------------------
amio_status_t close(void *dataset_payload) {
    return close_dataset(dataset_payload);
}

// ---------------------------------------------------------------
// Remaining stubs (tasks 9.x)
// ---------------------------------------------------------------

// ---------------------------------------------------------------
// Helper: compute element size in bytes for a given dtype.
// Returns 0 for unsupported/invalid dtype values.
//
// Delegates to the shared element_size() helper in backend_driver.hpp
// so the dtype-size mapping lives in one place (Req 4.3); retained as a
// thin alias to keep this translation unit's call sites readable.
// ---------------------------------------------------------------
static std::size_t dtype_element_size(amio_dtype_t dtype) noexcept {
    return element_size(dtype);
}

// ---------------------------------------------------------------
// Helper: validate shape descriptor.
// Returns true if the shape is valid (rank in [1,7], all extents
// strictly positive).
// ---------------------------------------------------------------
static bool validate_shape(const amio_shape_t *shape) noexcept {
    if (shape->rank < 1 || shape->rank > AMIO_MAX_RANK) {
        return false;
    }
    for (int32_t i = 0; i < shape->rank; ++i) {
        if (shape->extents[i] <= 0) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------
// Helper: compute total element count from shape extents.
// Precondition: shape has been validated (rank >= 1, all extents > 0).
// ---------------------------------------------------------------
static std::size_t compute_element_count(const amio_shape_t *shape) noexcept {
    std::size_t count = 1;
    for (int32_t i = 0; i < shape->rank; ++i) {
        count *= static_cast<std::size_t>(shape->extents[i]);
    }
    return count;
}

// ---------------------------------------------------------------
// amio_write -- task 9.1: synchronous snapshot write path
//
// Validates inputs → computes payload size → acquires staging
// buffer → deep copies host data → enqueues WriteTask (or marks
// complete if worker pool is null) → returns io_handle.
//
// After this function returns, no Worker_Pool thread retains a
// reference to the host pointer.  The host may immediately reuse
// or free its buffer.
//
// Validates: R2.1, R2.2, R2.3, R2.4, R2.5, R2.7, R2.8, R2.10, R6.1
// ---------------------------------------------------------------
amio_status_t write(void *dataset_payload, const char *var_name, const void *host_data, amio_dtype_t dtype, const amio_shape_t *shape,
                    amio_io_handle *out_io) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    // ---- Step 1: Validate inputs (R2.10) ----

    // Null pointer check (host_data, var_name, shape already checked
    // in amio_api.cpp, but we double-check here for safety).
    if (host_data == nullptr || var_name == nullptr || shape == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // Unsupported dtype check.
    const std::size_t elem_size = dtype_element_size(dtype);
    if (elem_size == 0) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // Invalid shape check (rank 0, zero/negative extents).
    if (!validate_shape(shape)) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // ---- Step 2: Compute payload size ----
    const std::size_t element_count = compute_element_count(shape);
    const std::size_t payload_bytes = element_count * elem_size;

    // Sanity: payload must be at least 1 byte.
    if (payload_bytes == 0) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // ---- Step 3: Access AMIO_Core via back-pointer ----
    AMIO_Core *core = record->core;

    // ---- Step 4: Acquire staging buffer (R2.2, R2.6) ----
    StagingBuffer *staging_buf = nullptr;
    // Fallback buffer for stub mode (no staging pool).
    std::unique_ptr<std::byte[]> fallback_storage;
    StagingBuffer fallback_buf{};

    if (core != nullptr && core->staging_pool != nullptr) {
        staging_buf = core->staging_pool.get()->acquire(payload_bytes);
        if (staging_buf == nullptr) {
            // Timeout: no buffer available within staging timeout.
            return AMIO_ERR_STAGING_BACKPRESSURE;
        }
    } else {
        // No staging pool configured (stub/test mode).
        // Allocate a temporary buffer for the snapshot so that the
        // host pointer is still captured synchronously (R2.2, R2.3).
        // In production this path should not be hit; the staging
        // pool is always present after amio_init.
        fallback_storage = std::make_unique<std::byte[]>(payload_bytes);
        fallback_buf.data = fallback_storage.get();
        fallback_buf.capacity_bytes = payload_bytes;
        staging_buf = &fallback_buf;
    }

    // ---- Step 5: Deep copy from host pointer into staging buffer (R2.2, R2.3) ----
    // After this memcpy, the host pointer is no longer referenced.
    // The WriteTask will carry only the staging buffer pointer.
    staging_buf->used_bytes = payload_bytes;
    std::memcpy(staging_buf->data, host_data, payload_bytes);

    // ---- Step 6: Create IoRecord for tracking this write ----
    auto *io_rec = new IoRecord{};
    io_rec->dataset_id = record->dataset_id;
    io_rec->core = core;
    io_rec->staging_buf = staging_buf;

    // Insert into handle table as an Io handle.
    auto io_token = process_handle_table().insert(HandleKind::Io, io_rec);
    io_rec->token = io_token;

    // ---- Step 7: Enqueue WriteTask onto Worker_Pool (R6.1) ----
    // Determine if this buffer is pool-managed or fallback-allocated.
    const bool is_pool_buffer = (core != nullptr && core->staging_pool != nullptr);

    if (core != nullptr && core->worker_pool != nullptr && is_pool_buffer) {
        // Compute a variable_id from the var_name for ordering.
        // Use a simple hash for the variable key.
        std::uint64_t var_hash = 0;
        for (const char *p = var_name; *p != '\0'; ++p) {
            var_hash = var_hash * 31 + static_cast<std::uint64_t>(*p);
        }

        DatasetVariableKey dv_key{};
        dv_key.dataset_id = record->dataset_id;
        dv_key.variable_id = var_hash;

        // Build the variable metadata the backend driver needs to
        // encode this field (name, dtype, shape).  The GRIB2 driver,
        // for example, uses the shape to derive Ni/Nj and gates the
        // contiguity fast/slow path on the strides.
        VarMeta meta{};
        meta.dataset_id = record->dataset_id;
        meta.variable_id = var_hash;
        meta.name = var_name;
        meta.dtype = dtype;
        meta.shape = *shape;
        meta.timestep = -1;

        // Capture staging buffer pointer (NOT host pointer) in the
        // callback.  This is the critical safety property: after
        // this function returns, no worker thread holds a reference
        // to host_data.
        StagingBuffer *buf_for_worker = staging_buf;
        std::uint64_t io_handle_id = io_token;

        std::uint64_t seq = core->worker_pool.get()->submit_write(dv_key, io_handle_id, [buf_for_worker, record, io_rec, core, meta]() {
            // Worker thread callback: serialize buffer via backend.
            // The driver operates on the staging buffer, never
            // the host pointer.
            try {
                if (record->driver) {
                    record->driver->write(*buf_for_worker, meta);
                }
                io_rec->completed.store(true);
            } catch (...) {
                io_rec->failed.store(true);
                io_rec->failure_code = AMIO_ERR_BACKEND_FAILURE;
                record->has_failure.store(true);
                if (record->first_failure_code == AMIO_OK) {
                    record->first_failure_code = AMIO_ERR_BACKEND_FAILURE;
                }
            }

            // Release staging buffer back to pool after write
            // completes (R3.10).
            if (buf_for_worker != nullptr && core->staging_pool != nullptr) {
                core->staging_pool.get()->release(buf_for_worker);
            }

            // Decrement pending write count.
            record->pending_writes.fetch_sub(1);
        });

        io_rec->dv_seq = seq;

        // Track pending write.
        record->pending_writes.fetch_add(1);
    } else {
        // No worker pool or no pool buffer (stub mode): mark write
        // as immediately complete.  Release staging buffer if we
        // acquired one from the pool.
        io_rec->completed.store(true);

        if (is_pool_buffer && staging_buf != nullptr && core != nullptr && core->staging_pool != nullptr) {
            core->staging_pool.get()->release(staging_buf);
        }
        // If fallback_storage was used, it will be freed when this
        // function returns (unique_ptr goes out of scope).  The
        // io_rec does not retain a reference to it since the write
        // is already marked complete.
    }

    // ---- Step 8: Return io_handle to caller (R2.7) ----
    *out_io = HandleTable::to_ptr(io_token);
    return AMIO_OK;
}

// ---------------------------------------------------------------
// validate_bbox -- task 9: selective-read bounds validation (Req 12).
//
// Validates a caller-supplied Bounding_Box against the variable's
// shape (sourced from Dataset_Metadata via describe_variable) before
// it is handed to the PrefetchQueue / Backend_Driver, so out-of-range
// or malformed requests fail clearly instead of corrupting memory.
//
//   * null bbox                     -> AMIO_OK (full read)            (Req 12 full-read)
//   * rank != variable rank         -> AMIO_ERR_INVALID_INPUT         (Req 12.2)
//   * stride < 1 in any dimension   -> AMIO_ERR_INVALID_INPUT         (Req 12.4)
//   * negative offset               -> AMIO_ERR_INVALID_INPUT         (Req 12.3)
//   * extent < 1                    -> AMIO_ERR_INVALID_INPUT         (Req 12.3)
//   * offset + (extent-1)*stride
//       selects an index >= the
//       variable extent in any dim  -> AMIO_ERR_INVALID_INPUT         (Req 12.3)
//
// A validated box is passed through to the driver so only intersecting
// byte ranges are requested from storage (Req 12.1).
//
// Validates: R12.1, R12.2, R12.3, R12.4
// ---------------------------------------------------------------
static amio_status_t validate_bbox(const amio_bbox_t *b, const amio_shape_t &shape) noexcept {
    if (b == nullptr) {
        // No Bounding_Box -- full read (Req 12, full-read case).
        return AMIO_OK;
    }
    if (b->rank != shape.rank) {
        // Rank mismatch (Req 12.2).
        return AMIO_ERR_INVALID_INPUT;
    }
    for (int d = 0; d < b->rank; ++d) {
        if (b->strides[d] < 1) {
            // Stride less than one (Req 12.4).
            return AMIO_ERR_INVALID_INPUT;
        }
        if (b->offsets[d] < 0) {
            // Negative offset (Req 12.3).
            return AMIO_ERR_INVALID_INPUT;
        }
        std::int64_t last = b->offsets[d] + (b->extents[d] - 1) * b->strides[d];
        if (b->extents[d] < 1 || last >= shape.extents[d]) {
            // Empty extent or selection outside the variable extents
            // in this dimension (Req 12.3).
            return AMIO_ERR_INVALID_INPUT;
        }
    }
    return AMIO_OK;
}

// ---------------------------------------------------------------
// resolve_variable -- task 8: lazily create per-variable read state.
//
// A read-mode dataset opens once but may be read for many variables;
// each variable gets its own PrefetchQueue so the completed / pending
// / failed look-ahead windows never alias on the same timestep keys
// (design.md Key Design Decision: per-variable prefetch).
//
// On the first amio_read for `var_name` this:
//   1. probes the backend via describe_variable for the variable's
//      element type, shape, and total timestep count (Dataset_Metadata,
//      Req 4.5);
//   2. if the variable is found, constructs its dedicated PrefetchQueue
//      with the depth / read timeout sourced from the retained dataset
//      Config (Req 2.3) and the shared Staging_Pool / Worker_Pool, bound
//      to the variable's metadata and total timestep count (Req 2.4);
//   3. kicks off the initial min(depth, total_timesteps) look-ahead
//      fetches via schedule_initial() (Req 5.1).
//
// Subsequent reads for the same variable reuse the cached state.
//
// Thread-safe: serialized on record->variables_mu so concurrent
// amio_read calls create each VariableReadState exactly once (Req 3.1).
//
// Returns a pointer to the cached VariableReadState, or nullptr when
// the driver reports the variable is absent / undescribable
// (describe_variable returns found == false); the caller maps that to
// AMIO_ERR_BACKEND_FAILURE (Req 4.5).
//
// Validates: R2.3, R2.4, R3.1, R4.5, R5.1
// ---------------------------------------------------------------
static VariableReadState *resolve_variable(DatasetRecord *record, const std::string &var_name) {
    std::lock_guard<std::mutex> lock(record->variables_mu);

    // std::cerr << "[AMIO STUBS DEBUG] resolve_variable: var_name = '" << var_name << "'" << std::endl;

    // Reuse existing state if the variable has already been resolved.
    auto it = record->variables.find(var_name);
    if (it != record->variables.end()) {
        // std::cerr << "[AMIO STUBS DEBUG] resolve_variable: found cached state for '" << var_name << "'" << std::endl;
        return it->second.get();
    }

    // First use of this variable: probe the backend for its metadata.
    if (!record->driver) {
        // std::cerr << "[AMIO STUBS DEBUG] resolve_variable error: record->driver is NULL" << std::endl;
        return nullptr;
    }
    VariableInfo info = record->driver->describe_variable(var_name);
    // std::cerr << "[AMIO STUBS DEBUG] resolve_variable: record->driver->describe_variable returned found = " << (info.found ? "true" : "false")
    //          << std::endl;
    if (!info.found) {
        // Variable absent or driver cannot introspect it (Req 4.5).
        return nullptr;
    }

    // Construct the variable's read state and its dedicated queue.
    auto state = std::make_unique<VariableReadState>();
    state->name = var_name;
    state->info = info;

    AMIO_Core *core = record->core;
    StagingPool *pool = (core != nullptr) ? core->staging_pool.get() : nullptr;
    WorkerPool *workers = (core != nullptr) ? core->worker_pool.get() : nullptr;

    // Depth / read timeout come from the retained dataset Config
    // (Req 2.3); the queue is bound to this variable's metadata and
    // total timestep count (Req 2.4).
    state->queue = std::make_unique<PrefetchQueue>(record->dataset_config.prefetch.depth,
                                                   static_cast<std::int64_t>(record->dataset_config.prefetch.read_timeout_s), pool, workers,
                                                   record->driver.get(), record->dataset_id, var_name, info, info.total_timesteps);

    // The initial look-ahead window is NOT scheduled here: the first
    // get_buffer call self-kicks-off, so the window starts at the timestep
    // the host actually requests and uses the caller's latched selection
    // (bbox) rather than eagerly staging full records from timestep 0.
    // (schedule_initial remains public for tests and explicit warm-up.)

    VariableReadState *raw = state.get();
    record->variables.emplace(var_name, std::move(state));
    return raw;
}

// ---------------------------------------------------------------
// get_var_attribute_text -- read a CF text attribute for a variable (or a
// global attribute when var_name is empty/NULL). Two-call sizing: pass
// out_buf == NULL to obtain the length via out_len, then call again with
// a buffer. Returns AMIO_ERR_BACKEND_FAILURE when the attribute is absent
// or the backend cannot provide it (callers treat that as "no attribute").
amio_status_t get_var_attribute_text(void *dataset_payload, const char *var_name, const char *attr_name, char *out_buf, std::size_t buf_cap,
                                     std::size_t *out_len) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    if (attr_name == nullptr || attr_name[0] == '\0' || out_len == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    // A buffer with no capacity cannot be NUL-terminated; size-only queries pass out_buf == NULL.
    if (out_buf != nullptr && buf_cap == 0) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (record->mode != AMIO_MODE_READ) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (!record->driver) {
        return AMIO_ERR_BACKEND_FAILURE;
    }

    std::optional<std::string> value =
        record->driver->get_text_attribute(var_name != nullptr ? std::string(var_name) : std::string(), std::string(attr_name));
    if (!value) {
        return AMIO_ERR_BACKEND_FAILURE;
    }

    *out_len = value->size();
    if (out_buf != nullptr) {
        // memcpy, not strncpy: the value may contain interior NULs, which strncpy would truncate at.
        const std::size_t n = std::min(value->size(), buf_cap - 1);
        std::memcpy(out_buf, value->data(), n);
        out_buf[n] = '\0';
    }
    return AMIO_OK;
}

// ---------------------------------------------------------------
// get_var_attribute_double -- read a numeric attribute (CF packing
// attributes such as "scale_factor"/"add_offset", or "_FillValue") for a
// variable, or a global attribute when var_name is empty/NULL. The stored
// value is converted to double whatever its on-disk type. Returns
// AMIO_ERR_BACKEND_FAILURE when the attribute is absent or non-numeric
// (callers treat that as "no attribute").
amio_status_t get_var_attribute_double(void *dataset_payload, const char *var_name, const char *attr_name, double *out_value) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    if (attr_name == nullptr || attr_name[0] == '\0' || out_value == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (record->mode != AMIO_MODE_READ) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (!record->driver) {
        return AMIO_ERR_BACKEND_FAILURE;
    }

    std::optional<double> value =
        record->driver->get_numeric_attribute(var_name != nullptr ? std::string(var_name) : std::string(), std::string(attr_name));
    if (!value) {
        return AMIO_ERR_BACKEND_FAILURE;
    }

    *out_value = *value;
    return AMIO_OK;
}

// amio_read -- task 8: read coordinator + lazy per-variable resolve
//
// Validation order (design §5):
//   1. null / empty var_name        -> AMIO_ERR_INVALID_INPUT (Req 3.4, 6.5)
//   2. write-mode dataset           -> AMIO_ERR_INVALID_INPUT (Req 6.6)
//   3. timestep < 0                 -> AMIO_ERR_INVALID_INPUT (Req 6.4)
//   4. unresolved variable          -> AMIO_ERR_BACKEND_FAILURE (Req 4.5)
//   5. timestep >= total_timesteps  -> AMIO_ERR_INVALID_INPUT (Req 6.4)
//
// On the first read of a variable, resolve_variable creates its
// PrefetchQueue and launches the initial look-ahead (Req 2.3, 2.4,
// 3.1, 5.1).  get_buffer returns the staged buffer for timestep T
// without calling-thread I/O when the fetch has completed, or blocks
// until it completes / fails / times out (Req 5.2, 5.3, 6.1, 6.2).
// On success a ViewRecord is minted, the outstanding-view count is
// incremented (Req 7), and T + depth is scheduled to replenish the
// look-ahead window (Req 5.3).
//
// Bounding-box validation (task 9) is inserted between variable
// resolution and get_buffer: the bbox is checked against the
// variable's shape (Req 12) and the validated box is then passed
// through to the queue / driver.
//
// Validates: R2.3, R2.4, R3.1, R3.2, R4.5, R5.1, R5.2, R5.3, R6.4, R6.5, R6.6, R12.1, R12.2, R12.3, R12.4
// ---------------------------------------------------------------
amio_status_t describe(void *dataset_payload, const char *var_name, amio_shape_t *out_shape, std::int64_t *out_total_timesteps) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    if (var_name == nullptr || var_name[0] == '\0' || out_shape == nullptr || out_total_timesteps == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (record->mode != AMIO_MODE_READ) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // resolve_variable caches the driver's describe_variable result in the
    // per-variable read state.  Since the look-ahead window now self-kicks-
    // off on the first get_buffer (not at resolve time), calling describe
    // resolves metadata WITHOUT staging any payload.
    VariableReadState *vs = resolve_variable(record, var_name);
    if (vs == nullptr) {
        return AMIO_ERR_BACKEND_FAILURE;
    }

    *out_shape = vs->info.shape;
    *out_total_timesteps = vs->info.total_timesteps;
    return AMIO_OK;
}

amio_status_t read(void *dataset_payload, const char *var_name, std::int64_t timestep, const amio_bbox_t *bbox, amio_view_handle *out_view) {
    auto *record = static_cast<DatasetRecord *>(dataset_payload);

    // ---- Step 1: Validate inputs (fixed order, design §5) ----

    // Null / empty variable name (Req 3.4, 6.5).
    if (var_name == nullptr || var_name[0] == '\0') {
        return AMIO_ERR_INVALID_INPUT;
    }

    // Read on a write-mode dataset (Req 6.6).
    if (record->mode != AMIO_MODE_READ) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // Negative timestep (Req 6.4).
    if (timestep < 0) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // ---- Step 2: Resolve per-variable read state (lazy) ----
    // Creates the VariableReadState + PrefetchQueue on first use and
    // launches the initial look-ahead (Req 2.3, 2.4, 3.1, 4.5, 5.1).
    VariableReadState *vs = resolve_variable(record, var_name);
    if (vs == nullptr) {
        // Variable not found / driver cannot describe it (Req 4.5).
        return AMIO_ERR_BACKEND_FAILURE;
    }

    // Timestep at or beyond the variable's total timestep count
    // (Req 6.4) -- validated against Dataset_Metadata, not a
    // placeholder.
    if (timestep >= vs->info.total_timesteps) {
        return AMIO_ERR_INVALID_INPUT;
    }

    // ---- Step 2b: Validate the Bounding_Box against the variable shape ----
    // Rank mismatch, negative offset, empty extent, stride < 1, or a
    // selection that runs past the variable extents in any dimension
    // is rejected before any fetch is scheduled (Req 12.2, 12.3, 12.4).
    // A null bbox is a full read and passes through.  The validated box
    // is then handed to the queue / driver so only intersecting byte
    // ranges are requested from storage (Req 12.1).
    amio_status_t bbox_rc = validate_bbox(bbox, vs->info.shape);
    if (bbox_rc != AMIO_OK) {
        return bbox_rc;
    }

    // ---- Step 3: Get buffer from the variable's PrefetchQueue ----
    // The validated bounding box is passed straight through to the
    // queue / driver (Req 12.1).  A completed fetch returns immediately
    // with no calling-thread I/O; otherwise this blocks until the fetch
    // completes / fails / times out (Req 5.2, 5.3, 6.1, 6.2).
    StagingBuffer *buf = nullptr;
    amio_status_t get_rc = vs->queue->get_buffer(timestep, bbox, &buf);
    if (get_rc != AMIO_OK) {
        // Timeout or backend failure -- surface the error (Req 6.1, 6.2).
        return get_rc;
    }
    if (buf == nullptr) {
        // Should not happen if get_buffer returned AMIO_OK, but guard.
        return AMIO_ERR_BACKEND_FAILURE;
    }

    // ---- Step 4: Create ViewRecord and track the outstanding view ----
    // The staging buffer carries ref_count=1 from acquire; that
    // reference is held by the view until amio_release_view (R5.6,
    // R5.9).
    AMIO_Core *core = record->core;

    auto *view_rec = new ViewRecord{};
    view_rec->staging_buf = buf;
    view_rec->core = core;
    view_rec->dataset_id = record->dataset_id;
    view_rec->timestep = timestep;
    view_rec->dtype = vs->info.dtype;
    if (bbox != nullptr) {
        view_rec->shape.rank = bbox->rank;
        for (int d = 0; d < bbox->rank && d < AMIO_MAX_RANK; ++d) {
            view_rec->shape.extents[d] = bbox->extents[d];
            view_rec->shape.strides[d] = bbox->strides[d];
        }
    } else {
        view_rec->shape = vs->info.shape;
    }

    // Insert into handle table as a View handle.
    auto view_token = process_handle_table().insert(HandleKind::View, view_rec);
    view_rec->token = view_token;

    // Track outstanding views for close-time validation (R5.10).
    record->outstanding_views.fetch_add(1);

    // ---- Step 5: Replenish look-ahead -- schedule T + depth (Req 5.3) ----
    vs->queue->schedule_next(timestep);

    // ---- Step 6: Return view handle to caller ----
    *out_view = HandleTable::to_ptr(view_token);
    return AMIO_OK;
}

amio_status_t wait(void *io_payload, std::int64_t timeout_ms) {
    auto *io_rec = static_cast<IoRecord *>(io_payload);

    // Already completed?
    if (io_rec->completed.load()) {
        if (io_rec->failed.load()) {
            return static_cast<amio_status_t>(io_rec->failure_code);
        }
        return AMIO_OK;
    }

    // Block until completed or timeout.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms > 0 ? timeout_ms : 86400000);

    while (!io_rec->completed.load()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return AMIO_ERR_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (io_rec->failed.load()) {
        return static_cast<amio_status_t>(io_rec->failure_code);
    }
    return AMIO_OK;
}

// ---------------------------------------------------------------
// amio_release_view -- task 9.2/9.3: drop reference to staging buffer
//
// Decrements the ref_count on the staging buffer.  When the last
// reference is dropped, the buffer returns to the Staging_Pool
// free list.  Decrements the outstanding view count on the dataset.
//
// Validates: R5.6, R5.9
// ---------------------------------------------------------------
amio_status_t release_view(void *view_payload) {
    auto *view_rec = static_cast<ViewRecord *>(view_payload);

    if (view_rec == nullptr) {
        return AMIO_ERR_INVALID_HANDLE;
    }

    // Release the staging buffer back to the pool.
    AMIO_Core *core = view_rec->core;
    if (view_rec->staging_buf != nullptr && core != nullptr && core->staging_pool != nullptr) {
        core->staging_pool.get()->release(view_rec->staging_buf);
    }

    // Decrement outstanding view count on the dataset.
    // Find the dataset record to decrement its view count.
    if (core != nullptr) {
        std::lock_guard<std::mutex> lock(core->datasets_mu);
        auto it = core->datasets.find(view_rec->dataset_id);
        if (it != core->datasets.end() && it->second) {
            it->second->outstanding_views.fetch_sub(1);
        }
    }

    // Release the view handle from the table.
    process_handle_table().release(view_rec->token, HandleKind::View);

    // Free the view record.
    delete view_rec;
    return AMIO_OK;
}

amio_status_t view_data(void *view_payload, const void **out_data, std::size_t *out_size) {
    auto *view_rec = static_cast<ViewRecord *>(view_payload);
    if (view_rec == nullptr || view_rec->staging_buf == nullptr) {
        return AMIO_ERR_INVALID_HANDLE;
    }
    *out_data = view_rec->staging_buf->data;
    *out_size = view_rec->staging_buf->used_bytes;
    return AMIO_OK;
}

amio_status_t view_shape(void *view_payload, amio_shape_t *out_shape) {
    auto *view_rec = static_cast<ViewRecord *>(view_payload);
    if (view_rec == nullptr) {
        return AMIO_ERR_INVALID_HANDLE;
    }
    *out_shape = view_rec->shape;
    return AMIO_OK;
}

amio_status_t view_dtype(void *view_payload, amio_dtype_t *out_dtype) {
    auto *view_rec = static_cast<ViewRecord *>(view_payload);
    if (view_rec == nullptr) {
        return AMIO_ERR_INVALID_HANDLE;
    }
    *out_dtype = view_rec->dtype;
    return AMIO_OK;
}

}  // namespace amio::detail
