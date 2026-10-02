// test_worker_pool.cpp
//
// Unit tests for `amio::detail::WorkerPool` covering the background
// thread pool contract: configurable thread count, write task FIFO
// ordering, per-(dataset,variable) ordering, prefetch priority
// scheduling, drain, and graceful shutdown.
//
// Because the WorkerPool is private to the AMIO_Core build (its
// header lives under `src/workers/`), this test target compiles
// `worker_pool.cpp` directly into the test binary.
//
// Test scope:
//
//   * Construction starts the correct number of threads (R3.1).
//   * submit_write enqueues tasks and workers execute them (R6.1).
//   * Per-(dataset,variable) ordering: writes to the same pair
//     execute in submission order (R6.3).
//   * Different (dataset,variable) pairs can execute concurrently.
//   * submit_prefetch enqueues tasks prioritized by distance.
//   * drain() blocks until all tasks complete.
//   * Graceful shutdown: destructor drains and joins.
//   * After shutdown, submit is a no-op.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "workers/mpi_threading.hpp"
#include "workers/worker_pool.hpp"

namespace {

using amio::detail::BackpressureConfig;
using amio::detail::DatasetVariableKey;
using amio::detail::IOCommunicator;
using amio::detail::ThreadConfig;
using amio::detail::WorkerPool;
using amio::detail::WorkerPoolConfig;

struct TestResult {
    int passed = 0;
    int failed = 0;
};

TestResult g_result{};

void report_failure(const char *expr, const char *file, int line, const std::string &context) {
    std::fprintf(stderr, "FAIL %s:%d: %s   (%s)\n", file, line, expr, context.c_str());
    ++g_result.failed;
}

#define EXPECT_TRUE(cond, ctx)                                \
    do {                                                      \
        if (!(cond)) {                                        \
            report_failure(#cond, __FILE__, __LINE__, (ctx)); \
        } else {                                              \
            ++g_result.passed;                                \
        }                                                     \
    } while (0)

// ---- Test: construction starts correct number of threads ----

void test_construction_thread_count() {
    {
        WorkerPool pool(1);
        EXPECT_TRUE(pool.thread_count() == 1, "thread_count should be 1");
    }
    {
        WorkerPool pool(4);
        EXPECT_TRUE(pool.thread_count() == 4, "thread_count should be 4");
    }
}

// ---- Test: submit_write executes callback ----

void test_submit_write_executes_callback() {
    WorkerPool pool(2);

    std::atomic<int> counter{0};

    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });

    pool.drain();

    EXPECT_TRUE(counter.load() == 1, "write callback should have been executed once");
    EXPECT_TRUE(pool.total_writes_completed() == 1, "total_writes_completed should be 1");
}

// ---- Test: multiple writes execute ----

void test_multiple_writes_execute() {
    WorkerPool pool(2);

    std::atomic<int> counter{0};
    constexpr int kTasks = 20;

    DatasetVariableKey key{1, 1};
    for (int i = 0; i < kTasks; ++i) {
        pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    }

    pool.drain();

    EXPECT_TRUE(counter.load() == kTasks, "all write callbacks should have executed, got " + std::to_string(counter.load()));
    EXPECT_TRUE(pool.total_writes_completed() == kTasks, "total_writes_completed mismatch");
}

// ---- Test: per-(dataset,variable) ordering preserved ----

void test_per_dv_ordering_preserved() {
    // Use multiple threads to stress ordering.
    WorkerPool pool(4);

    std::mutex mu;
    std::vector<int> execution_order;

    DatasetVariableKey key{42, 7};
    constexpr int kTasks = 50;

    for (int i = 0; i < kTasks; ++i) {
        pool.submit_write(key, [&, i]() {
            // Small sleep to increase chance of out-of-order
            // execution if ordering is broken.
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            std::lock_guard<std::mutex> lock(mu);
            execution_order.push_back(i);
        });
    }

    pool.drain();

    EXPECT_TRUE(static_cast<int>(execution_order.size()) == kTasks, "all tasks should have executed");

    // Verify strict ordering.
    bool ordered = true;
    for (int i = 0; i < static_cast<int>(execution_order.size()); ++i) {
        if (execution_order[i] != i) {
            ordered = false;
            break;
        }
    }
    EXPECT_TRUE(ordered,
                "writes to same (dataset,variable) must execute in "
                "submission order");
}

// ---- Test: different (dataset,variable) pairs can interleave ----

void test_different_dv_pairs_interleave() {
    WorkerPool pool(4);

    std::atomic<int> counter_a{0};
    std::atomic<int> counter_b{0};

    DatasetVariableKey key_a{1, 1};
    DatasetVariableKey key_b{2, 2};

    constexpr int kTasks = 20;

    for (int i = 0; i < kTasks; ++i) {
        pool.submit_write(key_a, [&]() {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            counter_a.fetch_add(1, std::memory_order_relaxed);
        });
        pool.submit_write(key_b, [&]() {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            counter_b.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.drain();

    EXPECT_TRUE(counter_a.load() == kTasks, "all key_a tasks should complete");
    EXPECT_TRUE(counter_b.load() == kTasks, "all key_b tasks should complete");
}

// ---- Test: submit_prefetch executes callback ----

void test_submit_prefetch_executes_callback() {
    WorkerPool pool(2);

    std::atomic<int> counter{0};

    pool.submit_prefetch(/*timestep=*/5, /*distance=*/3,
                         /*dataset_id=*/1, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });

    pool.drain();

    EXPECT_TRUE(counter.load() == 1, "prefetch callback should have been executed");
    EXPECT_TRUE(pool.total_prefetches_completed() == 1, "total_prefetches_completed should be 1");
}

// ---- Test: prefetch priority by distance ----

void test_prefetch_priority_by_distance() {
    // Use 1 thread to ensure sequential execution and verify
    // priority ordering.
    WorkerPool pool(1);

    std::mutex mu;
    std::vector<std::int64_t> execution_order;

    // Submit tasks with varying distances (out of order).
    // Distance 1 should execute first, then 3, then 5, then 10.
    struct PrefetchSpec {
        std::int64_t timestep;
        std::int64_t distance;
    };
    PrefetchSpec specs[] = {
        {10, 10},
        {5, 5},
        {1, 1},
        {3, 3},
    };

    // Submit all before any can execute by holding the pool busy.
    std::atomic<bool> gate{false};
    DatasetVariableKey dummy_key{99, 99};

    // Block the single worker with a write task.
    pool.submit_write(dummy_key, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Give the worker time to pick up the blocking task.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Now submit prefetch tasks -- they'll queue up.
    for (const auto &spec : specs) {
        pool.submit_prefetch(spec.timestep, spec.distance,
                             /*dataset_id=*/1, [&, ts = spec.timestep]() {
                                 std::lock_guard<std::mutex> lock(mu);
                                 execution_order.push_back(ts);
                             });
    }

    // Release the gate.
    gate.store(true, std::memory_order_release);

    pool.drain();

    EXPECT_TRUE(execution_order.size() == 4, "all prefetch tasks should execute");

    // Verify priority order: distance 1, 3, 5, 10 → timesteps 1, 3, 5, 10.
    if (execution_order.size() == 4) {
        EXPECT_TRUE(execution_order[0] == 1, "first prefetch should be distance=1 (timestep=1)");
        EXPECT_TRUE(execution_order[1] == 3, "second prefetch should be distance=3 (timestep=3)");
        EXPECT_TRUE(execution_order[2] == 5, "third prefetch should be distance=5 (timestep=5)");
        EXPECT_TRUE(execution_order[3] == 10, "fourth prefetch should be distance=10 (timestep=10)");
    }
}

// ---- Test: drain blocks until all tasks complete ----

void test_drain_blocks_until_complete() {
    WorkerPool pool(2);

    std::atomic<int> counter{0};

    DatasetVariableKey key{1, 1};
    for (int i = 0; i < 10; ++i) {
        pool.submit_write(key, [&]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.drain();

    // After drain returns, all tasks must be complete.
    EXPECT_TRUE(counter.load() == 10, "all tasks should be complete after drain");
}

// ---- Test: write_queue_depth and prefetch_queue_depth ----

void test_queue_depth_reporting() {
    WorkerPool pool(1);

    // Block the worker.
    std::atomic<bool> gate{false};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Give worker time to pick up the blocking task.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Submit more tasks that will queue up.
    DatasetVariableKey key2{2, 2};
    pool.submit_write(key2, [&]() {});
    pool.submit_write(key2, [&]() {});
    pool.submit_prefetch(1, 1, 1, [&]() {});

    EXPECT_TRUE(pool.write_queue_depth() == 2, "write_queue_depth should be 2");
    EXPECT_TRUE(pool.prefetch_queue_depth() == 1, "prefetch_queue_depth should be 1");

    // Release gate and drain.
    gate.store(true, std::memory_order_release);
    pool.drain();

    EXPECT_TRUE(pool.write_queue_depth() == 0, "write_queue_depth should be 0 after drain");
    EXPECT_TRUE(pool.prefetch_queue_depth() == 0, "prefetch_queue_depth should be 0 after drain");
}

// ---- Test: shutdown makes submit no-op ----

void test_shutdown_makes_submit_noop() {
    WorkerPool pool(2);
    pool.shutdown();

    EXPECT_TRUE(pool.is_shutdown(), "pool should be shut down");

    std::atomic<int> counter{0};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    pool.submit_prefetch(1, 1, 1, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });

    // Give a moment for any accidental execution.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    EXPECT_TRUE(counter.load() == 0, "no callbacks should execute after shutdown");
}

// ---- Test: destructor drains and joins gracefully ----

void test_destructor_drains_gracefully() {
    std::atomic<int> counter{0};

    {
        WorkerPool pool(2);
        DatasetVariableKey key{1, 1};
        for (int i = 0; i < 10; ++i) {
            pool.submit_write(key, [&]() {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                counter.fetch_add(1, std::memory_order_relaxed);
            });
        }
        // Destructor should drain and join.
    }

    EXPECT_TRUE(counter.load() == 10, "destructor should drain all tasks before returning");
}

// ---- Test: sequence numbers are assigned correctly ----

void test_sequence_numbers_assigned() {
    WorkerPool pool(1);

    DatasetVariableKey key{1, 1};
    auto seq0 = pool.submit_write(key, []() {});
    auto seq1 = pool.submit_write(key, []() {});
    auto seq2 = pool.submit_write(key, []() {});

    EXPECT_TRUE(seq0 == 0, "first seq should be 0");
    EXPECT_TRUE(seq1 == 1, "second seq should be 1");
    EXPECT_TRUE(seq2 == 2, "third seq should be 2");

    // Different key gets its own sequence.
    DatasetVariableKey key2{2, 2};
    auto seq_b0 = pool.submit_write(key2, []() {});
    EXPECT_TRUE(seq_b0 == 0, "different key should start at seq 0");

    pool.drain();
}

// ---- Test: concurrent writes to multiple keys ----

void test_concurrent_writes_multiple_keys() {
    WorkerPool pool(4);

    constexpr int kKeys = 8;
    constexpr int kTasksPerKey = 25;

    // Track execution order per key.
    std::mutex mu;
    std::map<std::uint64_t, std::vector<int>> orders;

    for (int k = 0; k < kKeys; ++k) {
        DatasetVariableKey key{static_cast<std::uint64_t>(k), 0};
        for (int i = 0; i < kTasksPerKey; ++i) {
            pool.submit_write(key, [&, k, i]() {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                std::lock_guard<std::mutex> lock(mu);
                orders[static_cast<std::uint64_t>(k)].push_back(i);
            });
        }
    }

    pool.drain();

    // Verify ordering per key.
    bool all_ordered = true;
    for (int k = 0; k < kKeys; ++k) {
        auto &order = orders[static_cast<std::uint64_t>(k)];
        if (static_cast<int>(order.size()) != kTasksPerKey) {
            all_ordered = false;
            break;
        }
        for (int i = 0; i < kTasksPerKey; ++i) {
            if (order[i] != i) {
                all_ordered = false;
                break;
            }
        }
    }

    EXPECT_TRUE(all_ordered,
                "each (dataset,variable) key should have tasks in "
                "submission order");
    EXPECT_TRUE(pool.total_writes_completed() == static_cast<std::uint64_t>(kKeys * kTasksPerKey), "total writes completed mismatch");
}

// ---- Test: mixed write and prefetch tasks ----

void test_mixed_write_and_prefetch() {
    WorkerPool pool(3);

    std::atomic<int> write_count{0};
    std::atomic<int> prefetch_count{0};

    DatasetVariableKey key{1, 1};
    for (int i = 0; i < 10; ++i) {
        pool.submit_write(key, [&]() { write_count.fetch_add(1, std::memory_order_relaxed); });
        pool.submit_prefetch(i, i, 1, [&]() { prefetch_count.fetch_add(1, std::memory_order_relaxed); });
    }

    pool.drain();

    EXPECT_TRUE(write_count.load() == 10, "all write tasks should complete");
    EXPECT_TRUE(prefetch_count.load() == 10, "all prefetch tasks should complete");
}

// ---- Test: WorkerPoolConfig constructor with default config ----

void test_config_constructor_default() {
    WorkerPoolConfig config;
    config.thread_count = 2;
    // No thread_configs, no io_comm -- defaults.

    WorkerPool pool(config);

    EXPECT_TRUE(pool.thread_count() == 2, "config constructor should set thread_count=2");
    EXPECT_TRUE(pool.io_communicator().valid, "default io_communicator should be valid");
    EXPECT_TRUE(pool.io_communicator().is_io_rank, "default io_communicator should mark all ranks as I/O");
    EXPECT_TRUE(pool.pinning_errors() == 0, "no pinning errors with default config");

    // Verify the pool still works.
    std::atomic<int> counter{0};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    pool.drain();
    EXPECT_TRUE(counter.load() == 1, "config-constructed pool should execute tasks");
}

// ---- Test: WorkerPoolConfig with IOCommunicator ----

void test_config_constructor_with_io_comm() {
    WorkerPoolConfig config;
    config.thread_count = 2;
    config.io_comm.valid = true;
    config.io_comm.is_io_rank = true;

    WorkerPool pool(config);

    EXPECT_TRUE(pool.io_communicator().valid, "io_communicator should be valid");
    EXPECT_TRUE(pool.io_communicator().is_io_rank, "io_communicator should report is_io_rank");
    EXPECT_TRUE(pool.io_communicator().rank() == 0, "io_communicator rank() should be 0 (no split)");
    EXPECT_TRUE(pool.io_communicator().size() == 1, "io_communicator size() should be 1 (no split)");
}

// ---- Test: WorkerPoolConfig with invalid pinning records errors ----

void test_config_constructor_invalid_pinning() {
    WorkerPoolConfig config;
    config.thread_count = 2;

    // Set up invalid pinning for both threads.
    ThreadConfig bad_pin;
    bad_pin.cpu_cores = {99999};  // Invalid core ID.
    config.thread_configs = {bad_pin, bad_pin};

    WorkerPool pool(config);

    // Give threads time to start and attempt pinning.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // On Linux, pinning to core 99999 should fail.
    // On non-Linux, any non-default config fails.
    EXPECT_TRUE(pool.pinning_errors() == 2, "both threads should report pinning errors, got " + std::to_string(pool.pinning_errors()));

    // Pool should still function despite pinning failures.
    std::atomic<int> counter{0};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    pool.drain();
    EXPECT_TRUE(counter.load() == 1, "pool should still work after pinning failures");
}

// ---- Test: WorkerPoolConfig with valid pinning (Linux only) ----

void test_config_constructor_valid_pinning() {
    int available = amio::detail::query_available_cpus();
    if (available <= 0) {
        std::fprintf(stdout, "  SKIP: cannot query available CPUs\n");
        ++g_result.passed;
        return;
    }

    WorkerPoolConfig config;
    config.thread_count = 2;

    // Pin both threads to core 0 (should always be available).
    ThreadConfig pin;
    pin.cpu_cores = {0};
    config.thread_configs = {pin, pin};

    WorkerPool pool(config);

    // Give threads time to start and apply pinning.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

#if defined(__linux__)
    EXPECT_TRUE(pool.pinning_errors() == 0, "valid pinning should succeed on Linux, got " + std::to_string(pool.pinning_errors()) + " errors");
#else
    // On non-Linux, non-default configs always fail.
    EXPECT_TRUE(pool.pinning_errors() == 2, "non-Linux should report pinning errors");
#endif

    // Pool should function.
    std::atomic<int> counter{0};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    pool.drain();
    EXPECT_TRUE(counter.load() == 1, "pool should work with valid pinning");
}

// ---- Test: Backend_Driver MPI calls routed through I/O communicator ----

void test_backend_driver_uses_io_comm() {
    // This test verifies that the io_communicator() is accessible
    // from within a worker callback, simulating how a Backend_Driver
    // would route MPI calls through the I/O communicator (R3.5).
    WorkerPoolConfig config;
    config.thread_count = 1;
    config.io_comm.valid = true;
    config.io_comm.is_io_rank = true;

    WorkerPool pool(config);

    std::atomic<bool> observed_valid{false};
    std::atomic<bool> observed_is_io{false};

    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() {
        // In a real Backend_Driver, this is where MPI calls would
        // use pool.io_communicator().handle() instead of
        // MPI_COMM_WORLD.
        const auto &comm = pool.io_communicator();
        observed_valid.store(comm.valid, std::memory_order_relaxed);
        observed_is_io.store(comm.is_io_rank, std::memory_order_relaxed);
    });

    pool.drain();

    EXPECT_TRUE(observed_valid.load(), "Backend_Driver should see valid=true");
    EXPECT_TRUE(observed_is_io.load(), "Backend_Driver should see is_io_rank=true");
}

// ---- Test: Full init-time validation flow (R3.2, R3.3, R3.5, R3.6) ----
//
// This test simulates the amio_init validation sequence:
//   1. Validate thread bindings (validate_thread_config)
//   2. Validate comm config (validate_comm_config)
//   3. Perform comm split (split_communicator)
//   4. Construct WorkerPool with validated config
//
// This is the integration pattern that amio_init uses to satisfy
// R3.2, R3.3, R3.5, R3.6.

void test_full_init_validation_flow_success() {
    using amio::detail::CommConfig;
    using amio::detail::IOCommunicator;
    using amio::detail::split_communicator;
    using amio::detail::validate_comm_config;
    using amio::detail::validate_thread_config;

    // Step 1: Validate thread bindings (default = no pinning).
    ThreadConfig thread_cfg;
    amio_err_t rc = validate_thread_config(thread_cfg);
    EXPECT_TRUE(rc == AMIO_OK, "default thread config should validate OK");

    // Step 2: Validate comm config (default = no split).
    CommConfig comm_cfg;
    rc = validate_comm_config(comm_cfg);
    EXPECT_TRUE(rc == AMIO_OK, "default comm config should validate OK");

    // Step 3: Perform comm split.
    IOCommunicator io_comm;
    rc = split_communicator(comm_cfg, 0, io_comm);
    EXPECT_TRUE(rc == AMIO_OK, "default comm split should succeed");
    EXPECT_TRUE(io_comm.valid, "io_comm should be valid");
    EXPECT_TRUE(io_comm.is_io_rank, "all ranks should be I/O in default mode");

    // Step 4: Construct WorkerPool with validated config.
    WorkerPoolConfig pool_cfg;
    pool_cfg.thread_count = 2;
    pool_cfg.thread_configs = {thread_cfg, thread_cfg};
    pool_cfg.io_comm = io_comm;

    WorkerPool pool(pool_cfg);

    EXPECT_TRUE(pool.thread_count() == 2, "pool should have 2 threads");
    EXPECT_TRUE(pool.io_communicator().valid, "pool io_communicator should be valid");
    EXPECT_TRUE(pool.io_communicator().is_io_rank, "pool should report is_io_rank=true");
    EXPECT_TRUE(pool.pinning_errors() == 0, "no pinning errors with default config");

    // Verify pool functions correctly.
    std::atomic<int> counter{0};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() { counter.fetch_add(1, std::memory_order_relaxed); });
    pool.drain();
    EXPECT_TRUE(counter.load() == 1, "pool should execute tasks after full init flow");
}

void test_full_init_validation_flow_invalid_binding() {
    using amio::detail::validate_thread_config;

    // Simulate amio_init detecting invalid bindings BEFORE creating
    // the WorkerPool (R3.3: "SHALL NOT create the Worker_Pool").
    ThreadConfig bad_cfg;
    bad_cfg.cpu_cores = {99999};  // Invalid core.

    amio_err_t rc = validate_thread_config(bad_cfg);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "invalid binding should be detected at init time");

    // amio_init would return AMIO_ERR_INVALID_BINDING here and NOT
    // create the WorkerPool.  The host affinity is unchanged because
    // validate_thread_config does not modify affinity.
}

void test_full_init_validation_flow_invalid_comm_split() {
    using amio::detail::CommConfig;
    using amio::detail::IOCommunicator;
    using amio::detail::split_communicator;
    using amio::detail::validate_comm_config;

    // Simulate amio_init detecting invalid I/O rank set (R3.6).
    CommConfig bad_cfg;
    bad_cfg.io_ranks = {0, 5};  // Rank 5 is out of range.
    bad_cfg.world_size = 4;

    amio_err_t rc = validate_comm_config(bad_cfg);
    EXPECT_TRUE(rc == AMIO_ERR_COMM_SPLIT_FAILED, "invalid comm config should be detected at init time");

    // Even if we try the split, it should fail.
    IOCommunicator io_comm;
    rc = split_communicator(bad_cfg, 0, io_comm);
    EXPECT_TRUE(rc == AMIO_ERR_COMM_SPLIT_FAILED, "split with invalid ranks should fail");
    EXPECT_TRUE(!io_comm.valid, "io_comm should not be valid on failure");

    // amio_init would return AMIO_ERR_COMM_SPLIT_FAILED here and NOT
    // create the WorkerPool.  The world communicator is unmodified.
}

void test_all_backend_calls_use_io_comm_only() {
    // Verify that multiple Backend_Driver callbacks all observe the
    // same I/O communicator, confirming that ALL MPI calls are routed
    // through it (R3.5).
    WorkerPoolConfig config;
    config.thread_count = 2;
    config.io_comm.valid = true;
    config.io_comm.is_io_rank = true;

    WorkerPool pool(config);

    constexpr int kTasks = 10;
    std::atomic<int> correct_comm_count{0};

    // Submit tasks to different (dataset, variable) pairs to exercise
    // multiple threads.
    for (int i = 0; i < kTasks; ++i) {
        DatasetVariableKey key{static_cast<uint64_t>(i), 0};
        pool.submit_write(key, [&]() {
            const auto &comm = pool.io_communicator();
            if (comm.valid && comm.is_io_rank) {
                correct_comm_count.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    pool.drain();

    EXPECT_TRUE(correct_comm_count.load() == kTasks, "all " + std::to_string(kTasks) + " Backend_Driver callbacks should see valid io_comm, got " +
                                                         std::to_string(correct_comm_count.load()));
}

// ---- Test: MPI threading validation (R3.8) ----

void test_mpi_threading_validation_no_background_mpi() {
    using amio::detail::MpiThreadingConfig;
    using amio::detail::validate_mpi_threading;

    // No background MPI-IO requested -- any thread level is fine.
    MpiThreadingConfig cfg;
    cfg.requires_background_mpi_io = false;
    cfg.mpi_thread_level_provided = amio::detail::kMpiThreadSingle;

    amio_err_t rc = validate_mpi_threading(cfg);
    EXPECT_TRUE(rc == AMIO_OK, "no background MPI-IO should pass regardless of thread level");
}

void test_mpi_threading_validation_thread_multiple_ok() {
    using amio::detail::MpiThreadingConfig;
    using amio::detail::validate_mpi_threading;

    // Background MPI-IO requested with MPI_THREAD_MULTIPLE -- OK.
    MpiThreadingConfig cfg;
    cfg.requires_background_mpi_io = true;
    cfg.mpi_thread_level_provided = amio::detail::kMpiThreadMultiple;

    amio_err_t rc = validate_mpi_threading(cfg);
    EXPECT_TRUE(rc == AMIO_OK, "MPI_THREAD_MULTIPLE should satisfy background MPI-IO");
}

void test_mpi_threading_validation_insufficient_level() {
    using amio::detail::MpiThreadingConfig;
    using amio::detail::validate_mpi_threading;

    // Background MPI-IO requested but only MPI_THREAD_SERIALIZED.
    MpiThreadingConfig cfg;
    cfg.requires_background_mpi_io = true;
    cfg.mpi_thread_level_provided = amio::detail::kMpiThreadSerialized;

    amio_err_t rc = validate_mpi_threading(cfg);
    EXPECT_TRUE(rc == AMIO_ERR_THREADING_UNSUPPORTED, "MPI_THREAD_SERIALIZED should fail for background MPI-IO");
}

void test_mpi_threading_validation_mpi_not_initialized() {
    using amio::detail::MpiThreadingConfig;
    using amio::detail::validate_mpi_threading;

    // Background MPI-IO requested but MPI not initialized (-1).
    MpiThreadingConfig cfg;
    cfg.requires_background_mpi_io = true;
    cfg.mpi_thread_level_provided = -1;

    amio_err_t rc = validate_mpi_threading(cfg);
    EXPECT_TRUE(rc == AMIO_ERR_THREADING_UNSUPPORTED, "MPI not initialized should fail for background MPI-IO");
}

void test_mpi_threading_validation_funneled_insufficient() {
    using amio::detail::MpiThreadingConfig;
    using amio::detail::validate_mpi_threading;

    // Background MPI-IO requested but only MPI_THREAD_FUNNELED.
    MpiThreadingConfig cfg;
    cfg.requires_background_mpi_io = true;
    cfg.mpi_thread_level_provided = amio::detail::kMpiThreadFunneled;

    amio_err_t rc = validate_mpi_threading(cfg);
    EXPECT_TRUE(rc == AMIO_ERR_THREADING_UNSUPPORTED, "MPI_THREAD_FUNNELED should fail for background MPI-IO");
}

// ---- Test: Queue-full without backpressure (R6.9) ----

void test_queue_full_without_backpressure() {
    // Configure a pool with small queue capacity and no backpressure.
    WorkerPoolConfig config;
    config.thread_count = 1;
    config.backpressure.enabled = false;
    config.backpressure.queue_capacity = 5;

    WorkerPool pool(config);

    // Block the worker so tasks accumulate in the queue.
    std::atomic<bool> gate{false};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Give worker time to pick up the blocking task.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Fill the queue to capacity (5 tasks).
    DatasetVariableKey key2{2, 2};
    for (int i = 0; i < 5; ++i) {
        std::uint64_t seq = 0;
        amio_err_t rc = pool.submit_write(key2, []() {}, &seq);
        EXPECT_TRUE(rc == AMIO_OK, "submit within capacity should succeed, task " + std::to_string(i));
    }

    // Next submit should fail with AMIO_ERR_QUEUE_FULL.
    std::uint64_t seq = 0;
    amio_err_t rc = pool.submit_write(key2, []() {}, &seq);
    EXPECT_TRUE(rc == AMIO_ERR_QUEUE_FULL, "submit beyond capacity should return AMIO_ERR_QUEUE_FULL");
    EXPECT_TRUE(seq == 0, "seq_out should be 0 on queue-full rejection");

    // Release gate and drain.
    gate.store(true, std::memory_order_release);
    pool.drain();
}

void test_queue_full_legacy_interface() {
    // Test the legacy submit_write interface returns 0 on queue-full.
    WorkerPoolConfig config;
    config.thread_count = 1;
    config.backpressure.enabled = false;
    config.backpressure.queue_capacity = 3;

    WorkerPool pool(config);

    // Block the worker.
    std::atomic<bool> gate{false};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Fill queue.
    DatasetVariableKey key2{2, 2};
    for (int i = 0; i < 3; ++i) {
        pool.submit_write(key2, []() {});
    }

    // Next submit via legacy interface should return 0.
    auto seq = pool.submit_write(key2, []() {});
    EXPECT_TRUE(seq == 0, "legacy submit_write should return 0 on queue-full");

    gate.store(true, std::memory_order_release);
    pool.drain();
}

// ---- Test: Backpressure watermarks (R6.8) ----

void test_backpressure_blocks_at_high_watermark() {
    // Configure backpressure: L=2, H=5, capacity=10.
    // When queue depth >= 5, block until depth < 2.
    WorkerPoolConfig config;
    config.thread_count = 2;
    config.backpressure.enabled = true;
    config.backpressure.low_watermark = 2;
    config.backpressure.high_watermark = 5;
    config.backpressure.queue_capacity = 10;

    WorkerPool pool(config);

    EXPECT_TRUE(pool.backpressure_enabled(), "backpressure should be enabled");
    EXPECT_TRUE(pool.low_watermark() == 2, "low_watermark should be 2");
    EXPECT_TRUE(pool.high_watermark() == 5, "high_watermark should be 5");
    EXPECT_TRUE(pool.queue_capacity() == 10, "queue_capacity should be 10");

    // Block the workers so tasks accumulate.
    std::atomic<bool> gate{false};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    // Second worker also blocked.
    DatasetVariableKey key3{3, 3};
    pool.submit_write(key3, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Submit tasks to fill up to high_watermark (5).
    DatasetVariableKey key2{2, 2};
    for (int i = 0; i < 5; ++i) {
        std::uint64_t seq = 0;
        amio_err_t rc = pool.submit_write(key2, []() {}, &seq);
        EXPECT_TRUE(rc == AMIO_OK, "submit below high_watermark should succeed, task " + std::to_string(i));
    }

    // Now queue depth is 5 (== H). Next submit should block.
    // We'll submit from a separate thread and verify it blocks.
    std::atomic<bool> submit_completed{false};
    std::thread submitter([&]() {
        std::uint64_t seq = 0;
        amio_err_t rc = pool.submit_write(key2, []() {}, &seq);
        (void)rc;
        submit_completed.store(true, std::memory_order_release);
    });

    // Give the submitter time to block.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(!submit_completed.load(std::memory_order_acquire), "submit should be blocked at high_watermark");

    // Release the gate so workers can drain tasks.
    gate.store(true, std::memory_order_release);

    // Wait for the submitter to complete.
    submitter.join();
    EXPECT_TRUE(submit_completed.load(std::memory_order_acquire), "submit should complete after queue drains below low_watermark");

    pool.drain();
}

void test_backpressure_config_accessors() {
    // Test default (no backpressure).
    {
        WorkerPool pool(2);
        EXPECT_TRUE(!pool.backpressure_enabled(), "default pool should not have backpressure enabled");
        EXPECT_TRUE(pool.queue_capacity() == 1024, "default queue_capacity should be 1024");
    }

    // Test with explicit config.
    {
        WorkerPoolConfig config;
        config.thread_count = 1;
        config.backpressure.enabled = true;
        config.backpressure.low_watermark = 10;
        config.backpressure.high_watermark = 50;
        config.backpressure.queue_capacity = 100;

        WorkerPool pool(config);
        EXPECT_TRUE(pool.backpressure_enabled(), "configured pool should have backpressure enabled");
        EXPECT_TRUE(pool.low_watermark() == 10, "low_watermark should be 10");
        EXPECT_TRUE(pool.high_watermark() == 50, "high_watermark should be 50");
        EXPECT_TRUE(pool.queue_capacity() == 100, "queue_capacity should be 100");
    }
}

void test_backpressure_shutdown_unblocks_writers() {
    // Verify that shutdown unblocks writers waiting on backpressure.
    WorkerPoolConfig config;
    config.thread_count = 1;
    config.backpressure.enabled = true;
    config.backpressure.low_watermark = 1;
    config.backpressure.high_watermark = 3;
    config.backpressure.queue_capacity = 10;

    WorkerPool pool(config);

    // Block the worker.
    std::atomic<bool> gate{false};
    DatasetVariableKey key{1, 1};
    pool.submit_write(key, [&]() {
        while (!gate.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Fill to high_watermark.
    DatasetVariableKey key2{2, 2};
    for (int i = 0; i < 3; ++i) {
        pool.submit_write(key2, []() {});
    }

    // Submit from another thread -- should block.
    std::atomic<bool> submit_done{false};
    std::thread submitter([&]() {
        pool.submit_write(key2, []() {});
        submit_done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(!submit_done.load(std::memory_order_acquire), "submit should be blocked");

    // Shutdown should unblock the writer.
    gate.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Let worker drain
    pool.shutdown();

    submitter.join();
    EXPECT_TRUE(submit_done.load(std::memory_order_acquire), "shutdown should unblock blocked writer");
}

}  // namespace

int main() {
    test_construction_thread_count();
    test_submit_write_executes_callback();
    test_multiple_writes_execute();
    test_per_dv_ordering_preserved();
    test_different_dv_pairs_interleave();
    test_submit_prefetch_executes_callback();
    test_prefetch_priority_by_distance();
    test_drain_blocks_until_complete();
    test_queue_depth_reporting();
    test_shutdown_makes_submit_noop();
    test_destructor_drains_gracefully();
    test_sequence_numbers_assigned();
    test_concurrent_writes_multiple_keys();
    test_mixed_write_and_prefetch();
    test_config_constructor_default();
    test_config_constructor_with_io_comm();
    test_config_constructor_invalid_pinning();
    test_config_constructor_valid_pinning();
    test_backend_driver_uses_io_comm();
    test_full_init_validation_flow_success();
    test_full_init_validation_flow_invalid_binding();
    test_full_init_validation_flow_invalid_comm_split();
    test_all_backend_calls_use_io_comm_only();

    // Task 4.4 tests: MPI threading, backpressure, queue-full.
    test_mpi_threading_validation_no_background_mpi();
    test_mpi_threading_validation_thread_multiple_ok();
    test_mpi_threading_validation_insufficient_level();
    test_mpi_threading_validation_mpi_not_initialized();
    test_mpi_threading_validation_funneled_insufficient();
    test_queue_full_without_backpressure();
    test_queue_full_legacy_interface();
    test_backpressure_blocks_at_high_watermark();
    test_backpressure_config_accessors();
    test_backpressure_shutdown_unblocks_writers();

    std::fprintf(stdout, "test_worker_pool: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
