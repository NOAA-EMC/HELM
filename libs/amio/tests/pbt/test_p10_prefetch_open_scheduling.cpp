// test_p10_prefetch_open_scheduling.cpp -- Property test P10: Prefetch open scheduling.
//
// For any (look-ahead depth N, dataset timestep count M): opening read
// dataset schedules exactly min(N,M) background fetch tasks before
// returning.
//
// Min 100 iterations.
//
// **Validates: Requirements R5.2**

#include <algorithm>
#include <cstdint>
#include <memory>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Property Test P10: Prefetch open scheduling
//
// For any generated:
//   - look-ahead depth N in [1, 64]
//   - dataset timestep count M in [1, 128]
//
// Construct a PrefetchQueue with depth N and total_timesteps M,
// call schedule_initial(), and verify that exactly min(N, M)
// background fetch tasks are scheduled (visible as pending or
// completed entries in the queue).
//
// Uses the real PrefetchQueue implementation with a
// MockBackendDriver that records read calls. The driver is "real"
// in the sense that it implements the Backend_Driver interface and
// performs actual memory operations -- it just doesn't require
// filesystem access for this scheduling-focused test.
// ===================================================================

TEST_CASE("P10: Prefetch open scheduling - schedules exactly min(N,M) fetches", "[pbt][p10][prefetch][scheduling]") {
    auto result = rc::check("opening read dataset schedules exactly min(N,M) background fetch tasks", []() {
        // Generate look-ahead depth N in [1, 64].
        auto depth = *rc::gen::inRange<std::size_t>(1, 65);

        // Generate dataset timestep count M in [1, 128].
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(1, 129);

        // Create a staging pool with enough buffers to hold all
        // prefetch requests without backpressure.
        std::size_t buffer_count = std::min(static_cast<std::size_t>(total_timesteps),
                                            depth) + 4;  // extra headroom
        std::size_t buffer_capacity = 4096;
        std::int64_t timeout_ms = 5000;

        StagingPool pool(buffer_count, buffer_capacity, timeout_ms);

        // Create a MockBackendDriver that records read calls.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate stored payloads so reads succeed.
        // The mock driver returns stored_payloads_[timestep % size].
        // We need at least one payload for reads to succeed.
        {
            std::vector<std::byte> dummy(64, std::byte{0xAB});
            // Write a dummy payload so reads have something to return.
            StagingBuffer dummy_buf;
            dummy_buf.data = dummy.data();
            dummy_buf.capacity_bytes = dummy.size();
            dummy_buf.used_bytes = dummy.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            driver->write(dummy_buf, meta);
        }

        // Create PrefetchQueue WITHOUT a worker pool (synchronous
        // fallback mode). This means schedule_initial() will
        // perform fetches synchronously and they will appear as
        // completed entries.
        //
        // The VariableInfo sizes the staging acquisition (F32, 16
        // elements = 64 bytes) -- well within the 4096-byte buffers.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth,
                         60,  // read_timeout_s
                         &pool,
                         nullptr,  // no worker pool -> synchronous fetch
                         driver.get(),
                         1,  // dataset_id
                         "test_var", info, total_timesteps);

        // Schedule initial fetches.
        pq.schedule_initial();

        // Expected number of scheduled fetches.
        std::size_t expected = std::min(static_cast<std::int64_t>(depth), total_timesteps);

        // In synchronous mode, all fetches complete immediately.
        // Verify completed + pending + failed == expected.
        std::size_t total_tracked = pq.completed_count() + pq.pending_count() + pq.failed_count();

        RC_ASSERT(total_tracked == expected);

        // Since we're in synchronous mode with a working driver,
        // all should be completed (no failures expected).
        RC_ASSERT(pq.completed_count() == expected);
        RC_ASSERT(pq.pending_count() == 0);

        // Verify the driver received exactly min(N,M) read calls.
        auto read_calls = driver->get_calls(CallRecord::Method::Read);
        RC_ASSERT(read_calls.size() == expected);

        // Verify the read calls cover timesteps [0, expected).
        for (std::size_t i = 0; i < expected; ++i) {
            RC_ASSERT(read_calls[i].timestep == static_cast<std::int64_t>(i));
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P10b: Prefetch open scheduling with Worker_Pool
//
// Same property but using a real Worker_Pool for asynchronous
// dispatch. After draining the pool, verify the same invariant.
// ===================================================================

TEST_CASE("P10: Prefetch open scheduling - async with Worker_Pool", "[pbt][p10][prefetch][scheduling][async]") {
    auto result = rc::check("async scheduling dispatches exactly min(N,M) prefetch tasks to Worker_Pool", []() {
        // Generate look-ahead depth N in [1, 32].
        auto depth = *rc::gen::inRange<std::size_t>(1, 33);

        // Generate dataset timestep count M in [1, 64].
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(1, 65);

        // Create staging pool.
        std::size_t buffer_count = std::min(static_cast<std::size_t>(total_timesteps), depth) + 4;
        std::size_t buffer_capacity = 4096;

        StagingPool pool(buffer_count, buffer_capacity, 5000);

        // Create MockBackendDriver.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate a dummy payload.
        {
            std::vector<std::byte> dummy(64, std::byte{0xCD});
            StagingBuffer dummy_buf;
            dummy_buf.data = dummy.data();
            dummy_buf.capacity_bytes = dummy.size();
            dummy_buf.used_bytes = dummy.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            driver->write(dummy_buf, meta);
        }

        // Create Worker_Pool with 2 threads.
        WorkerPoolConfig wp_config;
        wp_config.thread_count = 2;
        wp_config.backpressure.queue_capacity = 1024;
        wp_config.backpressure.enabled = false;

        auto workers = std::make_unique<WorkerPool>(wp_config);

        // Create PrefetchQueue with the worker pool.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, workers.get(), driver.get(), 1, "test_var", info, total_timesteps);

        // Schedule initial fetches.
        pq.schedule_initial();

        // Drain the worker pool to ensure all tasks complete.
        workers->drain();

        // Expected number of scheduled fetches.
        std::size_t expected = std::min(static_cast<std::int64_t>(depth), total_timesteps);

        // After draining, all fetches should be completed.
        std::size_t total_tracked = pq.completed_count() + pq.pending_count() + pq.failed_count();

        RC_ASSERT(total_tracked == expected);

        // Verify the driver received exactly min(N,M) read calls.
        // Subtract 1 for the initial dummy write call.
        auto read_calls = driver->get_calls(CallRecord::Method::Read);
        RC_ASSERT(read_calls.size() == expected);
    });

    REQUIRE(result);
}
