// test_p12_prefetch_lookahead_replenishment.cpp -- Property test P12:
// Prefetch look-ahead replenishment.
//
// For any timestep T returned by amio_read: if T+N within bounds →
// schedule fetch for T+N; if T+N outside bounds → no additional
// fetch scheduled.
//
// Min 100 iterations.
//
// **Validates: Requirements R5.4**

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
// Property Test P12: Prefetch look-ahead replenishment
//
// For any generated:
//   - look-ahead depth N in [1, 32]
//   - dataset timestep count M in [1, 128]
//   - timestep T in [0, min(N,M)-1] (initially prefetched)
//
// After reading timestep T via get_buffer(T), call schedule_next(T).
// Then verify:
//   - If T+N < M: a fetch for T+N is scheduled (pending or completed)
//   - If T+N >= M: no additional fetch is scheduled
// ===================================================================

TEST_CASE("P12: Prefetch look-ahead replenishment - T+N within bounds", "[pbt][p12][prefetch][replenishment]") {
    auto result = rc::check("after reading T, if T+N < M then fetch for T+N is scheduled", []() {
        // Generate look-ahead depth N in [1, 32].
        auto depth = *rc::gen::inRange<std::size_t>(1, 33);

        // Generate dataset timestep count M such that there exist
        // timesteps beyond the initial prefetch window.
        // M must be > depth so that T+N can be within bounds.
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(static_cast<std::int64_t>(depth) + 1, static_cast<std::int64_t>(depth) + 65);

        // Timestep T in [0, min(N,M)-1] -- initially prefetched.
        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);
        auto timestep_t = *rc::gen::inRange<std::int64_t>(0, max_prefetched);

        // Ensure T+N is within bounds.
        std::int64_t next_t = timestep_t + static_cast<std::int64_t>(depth);
        RC_PRE(next_t < total_timesteps);

        // Create staging pool with enough buffers.
        std::size_t buffer_count = static_cast<std::size_t>(total_timesteps) + 4;
        if (buffer_count > 4096) buffer_count = 4096;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create driver.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate dummy payload.
        {
            std::vector<std::byte> dummy(128, std::byte{0x77});
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

        // Schedule initial fetches.
        pq.schedule_initial();

        // Read timestep T.
        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(timestep_t, nullptr, &buf);
        RC_ASSERT(status == AMIO_OK);
        RC_ASSERT(buf != nullptr);
        pool.release(buf);

        // Record read count before schedule_next.
        std::size_t reads_before = driver->call_count(CallRecord::Method::Read);

        // Call schedule_next(T) to trigger replenishment.
        pq.schedule_next(timestep_t);

        // In synchronous mode, the fetch for T+N completes immediately.
        // Verify a new read was performed.
        std::size_t reads_after = driver->call_count(CallRecord::Method::Read);
        RC_ASSERT(reads_after == reads_before + 1);

        // Verify the new read was for timestep T+N.
        auto all_reads = driver->get_calls(CallRecord::Method::Read);
        RC_ASSERT(all_reads.back().timestep == next_t);

        // Verify the buffer for T+N is now completed.
        StagingBuffer *next_buf = nullptr;
        status = pq.get_buffer(next_t, nullptr, &next_buf);
        RC_ASSERT(status == AMIO_OK);
        RC_ASSERT(next_buf != nullptr);
        pool.release(next_buf);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P12b: No replenishment when T+N is out of bounds
//
// If T+N >= M (total timesteps), schedule_next(T) should NOT
// schedule any additional fetch.
// ===================================================================

TEST_CASE("P12: Prefetch look-ahead replenishment - T+N out of bounds", "[pbt][p12][prefetch][replenishment][out_of_bounds]") {
    auto result = rc::check("after reading T, if T+N >= M then no additional fetch is scheduled", []() {
        // Generate look-ahead depth N in [1, 32].
        auto depth = *rc::gen::inRange<std::size_t>(1, 33);

        // Generate dataset timestep count M in [1, depth].
        // This ensures that for any T in [0, M-1], T+N >= M.
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(1, static_cast<std::int64_t>(depth) + 1);

        // Timestep T: choose the last prefetched timestep.
        // For T+N to be out of bounds, we need T+N >= total_timesteps.
        // Since total_timesteps <= depth, any T >= 0 satisfies
        // T + depth >= depth >= total_timesteps.
        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);
        auto timestep_t = *rc::gen::inRange<std::int64_t>(0, max_prefetched);

        // Verify precondition: T+N >= M.
        std::int64_t next_t = timestep_t + static_cast<std::int64_t>(depth);
        RC_PRE(next_t >= total_timesteps);

        // Create staging pool.
        std::size_t buffer_count = static_cast<std::size_t>(max_prefetched) + 4;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create driver.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate dummy payload.
        {
            std::vector<std::byte> dummy(128, std::byte{0x88});
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

        // Read timestep T.
        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(timestep_t, nullptr, &buf);
        RC_ASSERT(status == AMIO_OK);
        RC_ASSERT(buf != nullptr);
        pool.release(buf);

        // Record read count before schedule_next.
        std::size_t reads_before = driver->call_count(CallRecord::Method::Read);

        // Call schedule_next(T).
        pq.schedule_next(timestep_t);

        // Verify: NO additional read was performed.
        std::size_t reads_after = driver->call_count(CallRecord::Method::Read);
        RC_ASSERT(reads_after == reads_before);

        // Verify: pending count did not increase.
        RC_ASSERT(pq.pending_count() == 0);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P12c: Replenishment chain maintains look-ahead
//
// Reading timesteps sequentially and calling schedule_next after
// each read maintains the look-ahead window sliding forward.
// ===================================================================

TEST_CASE("P12: Prefetch look-ahead replenishment - sliding window", "[pbt][p12][prefetch][replenishment][sliding]") {
    auto result = rc::check("sequential reads with schedule_next maintain sliding look-ahead window", []() {
        // Generate look-ahead depth N in [2, 8].
        auto depth = *rc::gen::inRange<std::size_t>(2, 9);

        // Generate dataset timestep count M > 2*N to allow sliding.
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(static_cast<std::int64_t>(2 * depth), static_cast<std::int64_t>(2 * depth) + 32);

        // Create staging pool with generous buffer count.
        std::size_t buffer_count = static_cast<std::size_t>(total_timesteps) + 4;
        if (buffer_count > 4096) buffer_count = 4096;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create driver.
        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // Pre-populate dummy payload.
        {
            std::vector<std::byte> dummy(128, std::byte{0x99});
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

        // Read timesteps sequentially, calling schedule_next after each.
        // Read up to depth timesteps to test the sliding window.
        std::int64_t reads_to_do = std::min(static_cast<std::int64_t>(depth), total_timesteps);

        for (std::int64_t t = 0; t < reads_to_do; ++t) {
            StagingBuffer *buf = nullptr;
            amio_status_t status = pq.get_buffer(t, nullptr, &buf);
            RC_ASSERT(status == AMIO_OK);
            RC_ASSERT(buf != nullptr);
            pool.release(buf);

            // Trigger replenishment.
            pq.schedule_next(t);

            // If T+N is within bounds, verify it was fetched.
            std::int64_t next_t = t + static_cast<std::int64_t>(depth);
            if (next_t < total_timesteps) {
                // The fetch for next_t should now be completed
                // (synchronous mode).
                StagingBuffer *next_buf = nullptr;
                status = pq.get_buffer(next_t, nullptr, &next_buf);
                RC_ASSERT(status == AMIO_OK);
                RC_ASSERT(next_buf != nullptr);

                // Put it back as completed for future reads.
                // Actually, get_buffer removes it from completed_,
                // so we need to release it.
                pool.release(next_buf);
            }
        }
    });

    REQUIRE(result);
}
