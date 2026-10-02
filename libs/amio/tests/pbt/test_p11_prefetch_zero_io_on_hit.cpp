// test_p11_prefetch_zero_io_on_hit.cpp -- Property test P11: Prefetch
// zero-I/O on hit.
//
// For any read dataset and timestep T with completed buffer:
// amio_read(T) returns Memory_View without invoking
// Backend_Driver::read or storage I/O on calling thread.
//
// Min 100 iterations with real backend driver; verify via
// timing/ordering that no Backend_Driver::read is invoked on
// calling thread.
//
// **Validates: Requirements R5.3**

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Property Test P11: Prefetch zero-I/O on hit
//
// For any generated:
//   - look-ahead depth N in [2, 32]
//   - dataset timestep count M in [2, 64]
//   - timestep T in [0, min(N,M)-1] (guaranteed to be prefetched)
//
// After schedule_initial() completes (synchronous mode), the buffer
// for timestep T is already in the completed map. Calling
// get_buffer(T) should:
//   1. Return immediately (no blocking)
//   2. NOT invoke any additional Backend_Driver::read call
//   3. Return AMIO_OK with a valid buffer pointer
//
// We verify this by recording the driver's read call count before
// and after get_buffer(T) -- the count must not increase.
// We also verify the calling thread ID is NOT seen in any new
// driver read calls.
// ===================================================================

TEST_CASE("P11: Prefetch zero-I/O on hit - no driver read on completed buffer", "[pbt][p11][prefetch][zero_io]") {
    auto result = rc::check("amio_read(T) with completed buffer does not invoke Backend_Driver::read", []() {
        // Generate look-ahead depth N in [2, 32].
        auto depth = *rc::gen::inRange<std::size_t>(2, 33);

        // Generate dataset timestep count M in [2, 64].
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(2, 65);

        // Timestep T must be within the initially prefetched range.
        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);
        auto timestep_t = *rc::gen::inRange<std::int64_t>(0, max_prefetched);

        // Create staging pool with enough buffers.
        std::size_t buffer_count = static_cast<std::size_t>(max_prefetched) + 4;
        std::size_t buffer_capacity = 4096;

        StagingPool pool(buffer_count, buffer_capacity, 5000);

        // Create MockBackendDriver.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate a dummy payload for reads.
        {
            std::vector<std::byte> dummy(128, std::byte{0x42});
            StagingBuffer dummy_buf;
            dummy_buf.data = dummy.data();
            dummy_buf.capacity_bytes = dummy.size();
            dummy_buf.used_bytes = dummy.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            driver->write(dummy_buf, meta);
        }

        // Create PrefetchQueue in synchronous mode.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool,
                         nullptr,  // synchronous mode
                         driver.get(), 1, "test_var", info, total_timesteps);

        // Schedule initial fetches -- all complete synchronously.
        pq.schedule_initial();

        // Verify the buffer for timestep T is completed.
        RC_ASSERT(pq.completed_count() == static_cast<std::size_t>(max_prefetched));

        // Record the read call count BEFORE get_buffer.
        std::size_t reads_before = driver->call_count(CallRecord::Method::Read);

        // Record the calling thread ID.
        auto calling_thread = std::this_thread::get_id();

        // Call get_buffer(T) -- should return immediately from cache.
        auto start_time = std::chrono::steady_clock::now();
        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(timestep_t, nullptr, &buf);
        auto elapsed = std::chrono::steady_clock::now() - start_time;

        // Verify: AMIO_OK returned.
        RC_ASSERT(status == AMIO_OK);

        // Verify: buffer is non-null.
        RC_ASSERT(buf != nullptr);

        // Verify: no additional Backend_Driver::read was invoked.
        std::size_t reads_after = driver->call_count(CallRecord::Method::Read);
        RC_ASSERT(reads_after == reads_before);

        // Verify: the operation completed quickly (< 10ms).
        // A cache hit should be essentially instantaneous.
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        RC_ASSERT(elapsed_ms < 10);

        // Verify: no new read calls were made on the calling thread.
        auto all_reads = driver->get_calls(CallRecord::Method::Read);
        for (std::size_t i = reads_before; i < all_reads.size(); ++i) {
            RC_ASSERT(all_reads[i].thread_id != calling_thread);
        }

        // Release the buffer back to the pool.
        pool.release(buf);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P11b: Multiple sequential reads from cache
//
// After prefetching, reading multiple timesteps in sequence should
// all be cache hits with zero additional I/O.
// ===================================================================

TEST_CASE("P11: Prefetch zero-I/O on hit - sequential reads all from cache", "[pbt][p11][prefetch][zero_io][sequential]") {
    auto result = rc::check("sequential reads of prefetched timesteps produce zero additional I/O", []() {
        // Generate look-ahead depth N in [4, 16].
        auto depth = *rc::gen::inRange<std::size_t>(4, 17);

        // Generate dataset timestep count M >= depth.
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(static_cast<std::int64_t>(depth), static_cast<std::int64_t>(depth) + 32);

        // Create staging pool.
        std::size_t buffer_count = depth + 8;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create driver.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate dummy payload.
        {
            std::vector<std::byte> dummy(128, std::byte{0x55});
            StagingBuffer dummy_buf;
            dummy_buf.data = dummy.data();
            dummy_buf.capacity_bytes = dummy.size();
            dummy_buf.used_bytes = dummy.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            driver->write(dummy_buf, meta);
        }

        // Create PrefetchQueue in synchronous mode.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, nullptr, driver.get(), 1, "test_var", info, total_timesteps);

        // Schedule initial fetches.
        pq.schedule_initial();

        // Record reads after initial scheduling.
        std::size_t reads_after_init = driver->call_count(CallRecord::Method::Read);

        // Read all initially prefetched timesteps.
        std::int64_t prefetched_count = std::min(static_cast<std::int64_t>(depth), total_timesteps);

        for (std::int64_t t = 0; t < prefetched_count; ++t) {
            StagingBuffer *buf = nullptr;
            amio_status_t status = pq.get_buffer(t, nullptr, &buf);
            RC_ASSERT(status == AMIO_OK);
            RC_ASSERT(buf != nullptr);
            pool.release(buf);
        }

        // Verify: no additional reads were performed during get_buffer
        // calls (all were cache hits from the initial prefetch).
        std::size_t reads_after_gets = driver->call_count(CallRecord::Method::Read);
        RC_ASSERT(reads_after_gets == reads_after_init);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P11c (read-pipeline Property 3): No calling-thread
// I/O on hit -- async dispatch variant.
//
// Design "Correctness Properties", Property 3:
//   If a timestep's fetch has completed, amio_read returns its buffer
//   without invoking Backend_Driver::read on the calling thread.
//
// This strengthens the synchronous P11 above: fetches are dispatched
// through a real Worker_Pool, so every Backend_Driver::read executes
// on a worker thread.  After draining, get_buffer(T) for a completed
// timestep must (a) return immediately, (b) invoke no additional
// driver read, and (c) confirm that NO driver read ever ran on the
// calling thread.
//
// **Validates: Requirements 5.2**
// ===================================================================

TEST_CASE("P11: read-pipeline Property 3 - completed hit performs no calling-thread driver read (async)",
          "[pbt][p11][prefetch][zero_io][p3][async]") {
    auto result = rc::check("completed-fetch hit returns without any Backend_Driver::read on the calling thread", []() {
        auto depth = *rc::gen::inRange<std::size_t>(2, 17);
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(2, 33);
        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);
        auto timestep_t = *rc::gen::inRange<std::int64_t>(0, max_prefetched);

        std::size_t buffer_count = static_cast<std::size_t>(max_prefetched) + 4;
        StagingPool pool(buffer_count, 4096, 5000);

        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);
        {
            std::vector<std::byte> dummy(64, std::byte{0x24});
            StagingBuffer dummy_buf;
            dummy_buf.data = dummy.data();
            dummy_buf.capacity_bytes = dummy.size();
            dummy_buf.used_bytes = dummy.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            driver->write(dummy_buf, meta);
        }

        WorkerPoolConfig wp_config;
        wp_config.thread_count = 2;
        wp_config.backpressure.queue_capacity = 1024;
        wp_config.backpressure.enabled = false;
        auto workers = std::make_unique<WorkerPool>(wp_config);

        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, workers.get(), driver.get(), 1, "test_var", info, total_timesteps);

        pq.schedule_initial();
        workers->drain();  // all fetches now complete on worker threads

        auto calling_thread = std::this_thread::get_id();
        std::size_t reads_before = driver->call_count(CallRecord::Method::Read);

        auto start = std::chrono::steady_clock::now();
        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(timestep_t, nullptr, &buf);
        auto elapsed = std::chrono::steady_clock::now() - start;

        RC_ASSERT(status == AMIO_OK);
        RC_ASSERT(buf != nullptr);

        // (a) No additional driver read on hit.
        RC_ASSERT(driver->call_count(CallRecord::Method::Read) == reads_before);

        // (b) Returned promptly (cache hit, no blocking I/O).
        RC_ASSERT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() < 10);

        // (c) No Backend_Driver::read ever executed on the calling thread.
        for (const auto &rec : driver->get_calls(CallRecord::Method::Read)) {
            RC_ASSERT(rec.thread_id != calling_thread);
        }

        pool.release(buf);
    });

    REQUIRE(result);
}
