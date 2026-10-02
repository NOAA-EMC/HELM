// amio_core.hpp -- Private C++ entry points called by the C-Boundary.
//
// This header is PRIVATE to the AMIO_Core build and is never installed.
// It bridges the `extern "C"` AMIO_C_API entry points (in
// `src/c_boundary/amio_api.cpp`) and the C++ implementations of
// `amio_init`, the dataset / I/O paths, and `amio_finalize` that
// land in tasks 4.x, 6.x, and 9.x.
//
// Each function below is the post-handle-validation, post-exception-
// translation entry into the C++ private API.  The C-Boundary calls
// them under a try/catch cordon (see amio_api.cpp) so they are free
// to throw `conf::Conf_Error`, `std::exception`, or any other type
// without the host application ever observing a C++ exception.
//
// Validates: R10.5, R10.6, R10.7, R12.2

#ifndef AMIO_SRC_C_BOUNDARY_AMIO_CORE_HPP
#define AMIO_SRC_C_BOUNDARY_AMIO_CORE_HPP

#include <atomic>
#include <conf/config.hpp>
#include <cstdint>
#include <logs/logger.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "amio/amio_errors.h"
#include "amio/amio_types.h"
#include "c_boundary/handle_table.hpp"
#include "config/config_loader.hpp"
#include "factory/backend_driver.hpp"
#include "prefetch/prefetch_queue.hpp"

namespace amio::detail {

// Forward declarations.
class WorkerPool;
class PrefetchQueue;
class StagingPool;
struct StagingBuffer;
struct AMIO_Core;

// ---------------------------------------------------------------
// IoRecord -- internal state for a pending asynchronous I/O op.
//
// Created by write() and tracked in the handle table as an Io
// handle.  The worker pool signals completion or failure.
// ---------------------------------------------------------------
struct IoRecord {
    std::uint64_t dataset_id = 0;
    std::uint64_t dv_seq = 0;  // per-(dataset,variable) sequence
    HandleTable::Token token = 0;
    StagingBuffer *staging_buf = nullptr;  // non-owning; pool owns the buffer
    AMIO_Core *core = nullptr;             // back-pointer for pool release

    // Completion state.
    std::atomic<bool> completed{false};
    std::atomic<bool> failed{false};
    amio_err_t failure_code = AMIO_OK;
};

// ---------------------------------------------------------------
// ViewRecord -- internal state for an outstanding read view.
//
// Created by amio_read when a prefetched buffer is returned to the
// host.  The view holds a reference to the staging buffer (via
// ref_count) and is released by amio_release_view.
// ---------------------------------------------------------------
struct ViewRecord {
    StagingBuffer *staging_buf = nullptr;  // non-owning; pool owns the buffer
    HandleTable::Token token = 0;
    AMIO_Core *core = nullptr;  // back-pointer for pool release
    std::uint64_t dataset_id = 0;
    std::int64_t timestep = -1;
    amio_shape_t shape = {};
    amio_dtype_t dtype = AMIO_DTYPE_F32;  // element type of staging_buf
};

// ---------------------------------------------------------------
// VariableReadState -- per-variable read look-ahead state.
//
// A read-mode dataset opens once but may be read for many
// variables.  Each variable gets its own PrefetchQueue so that the
// completed/pending/failed look-ahead windows for different
// variables never alias on the same timestep keys (see design.md
// Key Design Decision: per-variable prefetch).
//
// Created lazily on the first amio_read for a given variable name
// (task 8): the driver's describe_variable result is cached in
// `info`, and `queue` is constructed bound to that variable's name,
// dtype, shape, and total timestep count.
//
// Fields:
//   name   - the variable name this state tracks
//   info   - cached Dataset_Metadata (dtype, shape, total_timesteps)
//            from Backend_Driver::describe_variable
//   queue  - the variable's look-ahead PrefetchQueue (owned)
struct VariableReadState {
    std::string name;
    VariableInfo info;
    std::unique_ptr<PrefetchQueue> queue;
};

// ---------------------------------------------------------------
// DatasetRecord -- internal state for an open dataset.
//
// Holds the Backend_Driver instance, the dataset's mode (read or
// write), and the handle table token so that close can release it.
// Also tracks pending write count for flush semantics.
//
// Ownership & destruction order: `manifest_config` is declared
// before `driver` so that in C++ member destruction (reverse of
// declaration order) the driver is destroyed first.  This ensures
// the driver's destructor may still safely access configuration
// values during teardown (Req 13.4).
// ---------------------------------------------------------------
struct DatasetRecord {
    // Parsed manifest as a conf::Config -- owns the configuration
    // document for the driver's lifetime.  Declared before `driver`
    // so it outlives the driver (destroyed after driver, per C++
    // reverse-declaration-order destruction).  Passed by const
    // reference to open_write / open_read (Req 13.1, 13.2).
    std::optional<conf::Config> manifest_config;

    std::unique_ptr<Backend_Driver> driver;
    std::int32_t mode = AMIO_MODE_WRITE;
    HandleTable::Token token = 0;
    std::uint64_t dataset_id = 0;

    // Back-pointer to the owning AMIO_Core (non-owning).
    AMIO_Core *core = nullptr;

    // Pending write tracking for flush/close.
    mutable std::mutex pending_mu;
    std::atomic<std::uint64_t> pending_writes{0};
    std::atomic<bool> has_failure{false};
    amio_err_t first_failure_code = AMIO_OK;

    // Next variable ID counter for ordering.
    std::atomic<std::uint64_t> next_variable_id{1};

    // ---- Read path state (task 9.2) ----

    // Retained dataset configuration (parsed at open_dataset).
    // Read-mode datasets use this to source the prefetch depth and
    // read timeout when the per-variable PrefetchQueue is created
    // lazily on first read (Req 2.3, 2.4); see task 6/8.
    Config dataset_config;

    // ---- Per-variable read state (task 6) ----

    // Guards `variables` against concurrent lazy creation from
    // multiple amio_read calls (Req 3.1).
    mutable std::mutex variables_mu;

    // Per-variable read look-ahead state, keyed by variable name.
    // Populated lazily on the first amio_read for each variable
    // (task 8): each entry owns the variable's resolved metadata
    // and its dedicated PrefetchQueue so look-ahead windows for
    // different variables stay independent (Req 2.3, 2.4, 3.1).
    std::unordered_map<std::string, std::unique_ptr<VariableReadState>> variables;

    // Prefetch queue for read-mode datasets.  Created during
    // open_dataset when mode == AMIO_MODE_READ.  Null for write
    // datasets.
    std::unique_ptr<PrefetchQueue> prefetch_queue;

    // Total timesteps in the dataset (from config, for read mode).
    std::int64_t total_timesteps = 0;

    // Read timeout in seconds (from config).
    std::int64_t read_timeout_s = 60;

    // Prefetch depth N (from config).
    std::size_t prefetch_depth = 4;

    // Outstanding view count for close-time validation (R5.10).
    std::atomic<std::uint64_t> outstanding_views{0};
};

// ---------------------------------------------------------------
// AMIO_Core -- internal context holding all subsystem references.
//
// Created by amio_init, destroyed by amio_finalize.  Holds the
// dataset records, worker pool reference, and staging pool reference.
// ---------------------------------------------------------------
struct AMIO_Core {
    HandleTable::Token core_token = 0;

    // Active dataset records keyed by dataset_id.
    mutable std::mutex datasets_mu;
    std::unordered_map<std::uint64_t, std::unique_ptr<DatasetRecord>> datasets;
    std::atomic<std::uint64_t> next_dataset_id{1};

    // Staging pool (owned by AMIO_Core, may be null in stub mode).
    std::unique_ptr<StagingPool> staging_pool;

    // Staging timeout from config (milliseconds).
    std::int64_t staging_timeout_ms = 5000;

    // Worker pool (owned by AMIO_Core, may be null in stub mode).
    std::unique_ptr<WorkerPool> worker_pool;

    // ---- LOGS integration (Req 6.8) ----

    // Owned Logger instance.  Lifetime is tied to the core lifecycle:
    // created during amio_init, destroyed during amio_finalize.
    logs::Logger logger;

    // Pre-initialization fallback flag.  Before amio_init completes
    // communicator setup the Exception_Bridge falls back to direct
    // stderr output rather than invoking LOGS (Req 6.7).
    std::atomic<bool> logs_initialized{false};
};

// process_handle_table() -- accessor for the singleton HandleTable
// owned by the AMIO_Core build.  All AMIO_C_API entry points share
// a single table because tokens are minted as raw `void*` values
// and the C ABI offers no place to thread per-context state.
HandleTable &process_handle_table();

// ---------------------------------------------------------------
// Private C++ entry points (one per AMIO_C_API function).
//
// `core_payload`, `dataset_payload`, `io_payload`, `view_payload`
// are the validated payload pointers extracted from the handle
// table by the C-Boundary.  They are guaranteed non-null on entry
// (the C-Boundary returns AMIO_ERR_INVALID_HANDLE before calling
// these functions otherwise).
//
// Each function returns an AMIO_ERR_* code.  Throwing is also
// permitted -- the C-Boundary catches `conf::Conf_Error`,
// `std::exception`, and `...` and translates them to the
// appropriate AMIO_ERR_* code (R12.2).
// ---------------------------------------------------------------

amio_status_t init(const char *manifest_path, amio_core_handle *out_core);

// ---------------------------------------------------------------
// init_from_config -- shared core-construction path for init().
//
// Runs the pool-construction + LOGS-init + handle-mint sequence
// (steps 2-4 of the current init()) from an already-parsed Config.
// Both the path-based init() and the forthcoming string-based
// init_from_string() converge on this helper so the runtime-build
// logic lives in exactly one place.
//
//   * Pool construction failure -> AMIO_ERR_BACKEND_FAILURE, no
//     handle minted (Req 1.4).
//   * On success mints the core handle and writes *out_core.
// ---------------------------------------------------------------
amio_status_t init_from_config(const Config &cfg, amio_core_handle *out_core);

// ---------------------------------------------------------------
// init_from_string -- in-memory manifest variant of init().
//
// Parses `manifest_content` in the given `format` ("yaml"/"json")
// via ConfigLoader::parse_string, then delegates to init_from_config
// so the pool-construction + LOGS-init + handle-mint path is shared
// with the file-based init() (design §"Shared-body factoring").
//
//   * parse_string failure -> AMIO_ERR_MANIFEST_INVALID (never
//     AMIO_ERR_MANIFEST_NOT_FOUND -- there is no file), no handle
//     minted.
//   * Otherwise behaves exactly like init().
// ---------------------------------------------------------------
amio_status_t init_from_string(const char *manifest_content, const char *format, amio_core_handle *out_core);

amio_status_t finalize(void *core_payload);

amio_status_t open_dataset(void *core_payload, const char *config_path, std::int32_t mode, amio_dataset_handle *out_dataset);

// Shared open path: builds the backend driver from the parsed backend
// key, sets the communicator, opens the driver in the requested mode
// with `manifest_cfg`, and constructs/registers the DatasetRecord.
// Both the path-based `open_dataset` and the string-based
// `open_dataset_from_string` (task 3.1) converge here after producing
// their `config` (backend key) and `manifest_cfg` (driver config).
amio_status_t open_dataset_from_config(void *core_payload, const Config &config, conf::Config manifest_cfg, std::int32_t mode,
                                       amio_dataset_handle *out_dataset);

// ---------------------------------------------------------------
// open_dataset_from_string -- in-memory config variant of
// open_dataset().
//
// Parses the backend key from `config_content` (in the given
// `format`) via ConfigLoader::parse_string, builds the driver
// conf::Config via conf::Config::from_string, then delegates to
// open_dataset_from_config so the factory-build + communicator-set +
// open + DatasetRecord path is shared with the file-based
// open_dataset (design §"Shared-body factoring").
//
//   * parse_string failure -> AMIO_ERR_MANIFEST_INVALID (never
//     AMIO_ERR_MANIFEST_NOT_FOUND -- there is no file), no handle
//     minted.
//   * Otherwise behaves exactly like open_dataset().
// ---------------------------------------------------------------
amio_status_t open_dataset_from_string(void *core_payload, const char *config_content, const char *format, std::int32_t mode,
                                       amio_dataset_handle *out_dataset);

amio_status_t close_dataset(void *dataset_payload);

// describe -- query a variable's per-timestep shape and record count WITHOUT
// staging any payload.  Returns AMIO_ERR_INVALID_INPUT for a null/empty name
// or null outputs, AMIO_ERR_INVALID_INPUT on a write-mode dataset, and
// AMIO_ERR_BACKEND_FAILURE when the driver cannot describe the variable.
// `out_shape` receives the variable's Dataset_Metadata shape (the same shape
// amio_read validates a bbox against); `out_total_timesteps` the number of
// temporal records.  Lets a caller size a bounding box (and its record loop)
// without the binary-search-over-amio_read probe it would otherwise need.
amio_status_t describe(void *dataset_payload, const char *var_name, amio_shape_t *out_shape, std::int64_t *out_total_timesteps);

amio_status_t write(void *dataset_payload, const char *var_name, const void *host_data, amio_dtype_t dtype, const amio_shape_t *shape,
                    amio_io_handle *out_io);

amio_status_t read(void *dataset_payload, const char *var_name, std::int64_t timestep, const amio_bbox_t *bbox, amio_view_handle *out_view);

amio_status_t get_var_attribute_text(void *dataset_payload, const char *var_name, const char *attr_name, char *out_buf, std::size_t buf_cap,
                                     std::size_t *out_len);

amio_status_t get_var_attribute_double(void *dataset_payload, const char *var_name, const char *attr_name, double *out_value);

amio_status_t flush(void *dataset_payload, std::int64_t timeout_ms);

amio_status_t close(void *dataset_payload);

amio_status_t wait(void *io_payload, std::int64_t timeout_ms);

amio_status_t release_view(void *view_payload);

amio_status_t view_data(void *view_payload, const void **out_data, std::size_t *out_size);

amio_status_t view_shape(void *view_payload, amio_shape_t *out_shape);

amio_status_t view_dtype(void *view_payload, amio_dtype_t *out_dtype);

}  // namespace amio::detail

#endif  // AMIO_SRC_C_BOUNDARY_AMIO_CORE_HPP
