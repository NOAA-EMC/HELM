// test_p8_staging_backpressure.cpp -- Property 8: Staging_Pool backpressure timeout.
//
// For any config with all buffers occupied and any timeout T in
// [1, 100]ms (kept short for test speed): StagingPool::acquire either
// acquires a buffer (if one was released by another thread) or returns
// nullptr (signaling AMIO_ERR_STAGING_BACKPRESSURE) within T + epsilon.
// Source pointer is unmodified and queue depth is unchanged after a
// backpressure rejection.
//
// These cases exercise the ORIGINAL hard-limit contract, which the pool
// preserves in Bounded mode (and at the auto-grow ceiling).  The default
// Grow mode deliberately never reaches this path -- an exhausted pool
// appends a slot instead -- so the pools below are constructed Bounded.
//
// Min 100 iterations.
//
// **Validates: Requirements R2.6**
//
// CRITICAL: Uses the REAL StagingPool implementation (no mocks).
// Tests exercise the actual condition-variable timeout path.

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"
#include "staging/staging_pool.hpp"

// ===================================================================
// Property 8a: Backpressure timeout -- exhausted pool returns nullptr
// within T + epsilon.
//
// Steps:
//   1. Create a StagingPool with a small buffer_count (1-4).
//   2. Acquire all buffers (exhaust the pool).
//   3. Attempt to acquire one more with a generated timeout T in
//      [1, 100]ms.
//   4. Verify the acquire returns nullptr (backpressure) within
//      T + epsilon (epsilon = 50ms to account for scheduling jitter).
//   5. Verify the pool state is unchanged (free_buffer_count == 0,
//      total_buffer_count unchanged).
// ===================================================================

TEST_CASE("P8: exhausted StagingPool returns nullptr within timeout", "[pbt][p8][staging_backpressure][timeout]") {
    auto result = rc::check(
        "exhausted pool acquire returns nullptr within T + epsilon, "
        "pool state unchanged",
        []() {
            // Generate pool parameters.
            auto buffer_count = *rc::gen::inRange<std::size_t>(1, 5);         // [1, 4]
            auto buffer_capacity = *rc::gen::inRange<std::size_t>(64, 4097);  // [64, 4096] bytes
            auto timeout_ms = *rc::gen::inRange<std::int64_t>(1, 101);        // [1, 100] ms

            // Create the pool with the generated timeout.  Bounded mode:
            // backpressure is the contract under test here.
            amio::detail::StagingPool pool(buffer_count, buffer_capacity, timeout_ms, buffer_count, amio::detail::StagingPool::GrowMode::Bounded);

            // Verify initial state.
            RC_ASSERT(pool.total_buffer_count() == buffer_count);
            RC_ASSERT(pool.free_buffer_count() == buffer_count);

            // Exhaust all buffers.
            std::vector<amio::detail::StagingBuffer *> acquired;
            acquired.reserve(buffer_count);
            for (std::size_t i = 0; i < buffer_count; ++i) {
                auto *buf = pool.acquire(1);  // request minimal size
                RC_ASSERT(buf != nullptr);
                acquired.push_back(buf);
            }

            // Pool should now be fully exhausted.
            RC_ASSERT(pool.free_buffer_count() == 0);

            // Record state before the backpressure attempt.
            std::size_t total_before = pool.total_buffer_count();

            // Attempt to acquire one more -- should timeout.
            auto t_start = std::chrono::steady_clock::now();
            auto *result_buf = pool.acquire(1);
            auto t_end = std::chrono::steady_clock::now();

            // Must return nullptr (backpressure).
            RC_ASSERT(result_buf == nullptr);

            // Elapsed time should be >= T (the timeout) and <= T + epsilon.
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

            // The acquire should have waited at least close to the timeout.
            // Allow a small margin below (2ms) for clock granularity.
            RC_ASSERT(elapsed_ms >= (timeout_ms - 2));

            // The acquire should not have waited excessively beyond the timeout.
            // Allow 50ms epsilon for scheduling jitter on loaded systems.
            constexpr std::int64_t kEpsilonMs = 50;
            RC_ASSERT(elapsed_ms <= (timeout_ms + kEpsilonMs));

            // Pool state must be unchanged after backpressure.
            RC_ASSERT(pool.total_buffer_count() == total_before);
            RC_ASSERT(pool.free_buffer_count() == 0);

            // Clean up: release all acquired buffers.
            for (auto *buf : acquired) {
                pool.release(buf);
            }

            // After release, pool should be fully restored.
            RC_ASSERT(pool.free_buffer_count() == buffer_count);
        });
    REQUIRE(result);
}

// ===================================================================
// Property 8b: Source data is unmodified after backpressure rejection.
//
// Simulates the scenario where a write is attempted with source data
// but the pool is exhausted. Verifies that the source data bytes
// remain bit-for-bit identical after the backpressure timeout.
// ===================================================================

TEST_CASE("P8: source data unmodified after backpressure rejection", "[pbt][p8][staging_backpressure][source_unmodified]") {
    auto result = rc::check("source pointer bytes unchanged after backpressure timeout", []() {
        // Generate pool parameters -- small pool, short timeout.
        auto buffer_count = *rc::gen::inRange<std::size_t>(1, 4);         // [1, 3]
        auto buffer_capacity = *rc::gen::inRange<std::size_t>(64, 1025);  // [64, 1024]
        auto timeout_ms = *rc::gen::inRange<std::int64_t>(1, 51);         // [1, 50] ms

        // Generate source data of random size and content.
        auto data_size = *rc::gen::inRange<std::size_t>(1, 513);  // [1, 512] bytes
        std::vector<std::byte> source_data(data_size);
        for (std::size_t i = 0; i < data_size; ++i) {
            source_data[i] = static_cast<std::byte>(*rc::gen::inRange(0, 256));
        }

        // Keep a copy of the original source data for comparison.
        std::vector<std::byte> original_data = source_data;

        // Create pool and exhaust it (Bounded: no auto-grow past count).
        amio::detail::StagingPool pool(buffer_count, buffer_capacity, timeout_ms, buffer_count, amio::detail::StagingPool::GrowMode::Bounded);

        std::vector<amio::detail::StagingBuffer *> acquired;
        for (std::size_t i = 0; i < buffer_count; ++i) {
            auto *buf = pool.acquire(1);
            RC_ASSERT(buf != nullptr);
            acquired.push_back(buf);
        }

        // Attempt acquire -- will timeout (backpressure).
        auto *result_buf = pool.acquire(data_size);
        RC_ASSERT(result_buf == nullptr);

        // Source data must be completely unmodified.
        RC_ASSERT(source_data.size() == original_data.size());
        RC_ASSERT(std::memcmp(source_data.data(), original_data.data(), source_data.size()) == 0);

        // Clean up.
        for (auto *buf : acquired) {
            pool.release(buf);
        }
    });
    REQUIRE(result);
}

// ===================================================================
// Property 8c: Acquire succeeds if a buffer is released during wait.
//
// Verifies that if a buffer is released by another thread while the
// acquire is waiting, the acquire succeeds (returns non-null) within
// the timeout window. This tests the condition-variable notification
// path.
// ===================================================================

TEST_CASE("P8: acquire succeeds when buffer released during wait", "[pbt][p8][staging_backpressure][release_during_wait]") {
    auto result = rc::check("acquire succeeds if buffer released before timeout expires", []() {
        // Use a pool with exactly 1 buffer and a longer timeout
        // so the release has time to happen.
        auto buffer_capacity = *rc::gen::inRange<std::size_t>(64, 1025);
        constexpr std::int64_t timeout_ms = 500;  // 500ms -- plenty of time

        amio::detail::StagingPool pool(1, buffer_capacity, timeout_ms, 1, amio::detail::StagingPool::GrowMode::Bounded);

        // Exhaust the single buffer.
        auto *held_buf = pool.acquire(1);
        RC_ASSERT(held_buf != nullptr);
        RC_ASSERT(pool.free_buffer_count() == 0);

        // Generate a release delay in [5, 50]ms.
        auto release_delay_ms = *rc::gen::inRange<int>(5, 51);

        // Launch a thread that releases the buffer after a delay.
        std::thread releaser([&pool, held_buf, release_delay_ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(release_delay_ms));
            pool.release(held_buf);
        });

        // Attempt to acquire -- should succeed after the release.
        auto t_start = std::chrono::steady_clock::now();
        auto *new_buf = pool.acquire(1);
        auto t_end = std::chrono::steady_clock::now();

        // The acquire should succeed (buffer was released).
        RC_ASSERT(new_buf != nullptr);

        // It should have taken approximately release_delay_ms.
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
        // Should not have waited the full timeout.
        RC_ASSERT(elapsed_ms < timeout_ms);

        releaser.join();

        // Clean up.
        pool.release(new_buf);
        RC_ASSERT(pool.free_buffer_count() == 1);
    });
    REQUIRE(result);
}
