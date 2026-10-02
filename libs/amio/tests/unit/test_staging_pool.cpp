// test_staging_pool.cpp
//
// Unit tests for `amio::detail::StagingPool` covering the buffer pool
// contract: lazy slot storage, best-fit acquisition, grow-to-required,
// auto-grow slot count, condition-variable backpressure timeout
// (Bounded mode / at the hard ceiling), release, and ref-counting.
//
// Because the StagingPool is private to the AMIO_Core build (its
// header lives under `src/staging/`), this test target compiles
// `staging_pool.cpp` directly into the test binary.
//
// Test scope:
//
//   * Construction allocates the correct number of buffers with
//     the specified capacity (R1.3, R1.5).
//   * acquire returns a buffer with capacity >= requested bytes
//     and sets ref_count=1, seq incremented (R2.2, R2.9).
//   * Best-fit-or-larger selection: smallest sufficient buffer is
//     chosen from the free list.
//   * release decrements ref_count; when it reaches 0 the buffer
//     returns to the free list (R3.10, R5.6, R5.9).
//   * add_ref increments ref_count; release only returns buffer
//     when ref_count drops to 0.
//   * In the default Grow mode an exhausted pool appends a new slot
//     (up to max_buffer_count) instead of failing, so a host model
//     cannot crash on an under-provisioned buffer_count.
//   * In Bounded mode (and at the Grow ceiling), acquire blocks and
//     returns nullptr after the configured timeout (R2.6 backpressure).
//   * When a buffer is released while another thread is waiting,
//     the waiting thread wakes and acquires it (Bounded pool).
//   * Generation counter (seq) increments on each acquire.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "staging/staging_pool.hpp"

namespace {

using amio::detail::StagingBuffer;
using amio::detail::StagingPool;

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

// ---- Test: construction allocates correct buffers ----

void test_construction_allocates_buffers() {
    constexpr std::size_t count = 8;
    constexpr std::size_t capacity = 1024;
    constexpr std::int64_t timeout = 100;

    StagingPool pool(count, capacity, timeout);

    EXPECT_TRUE(pool.total_buffer_count() == count, "total_buffer_count mismatch");
    EXPECT_TRUE(pool.free_buffer_count() == count, "all buffers should be free after construction");
    EXPECT_TRUE(pool.buffer_capacity() == capacity, "buffer_capacity mismatch");
    EXPECT_TRUE(pool.timeout_ms() == timeout, "timeout_ms mismatch");
}

// ---- Test: acquire returns valid buffer ----

void test_acquire_returns_valid_buffer() {
    StagingPool pool(4, 2048, 100);

    StagingBuffer *buf = pool.acquire(512);
    EXPECT_TRUE(buf != nullptr, "acquire returned nullptr for available pool");
    EXPECT_TRUE(buf->capacity_bytes >= 512, "acquired buffer capacity < requested");
    EXPECT_TRUE(buf->capacity_bytes == 2048, "acquired buffer should have pool's capacity");
    EXPECT_TRUE(buf->ref_count == 1, "acquired buffer ref_count should be 1");
    EXPECT_TRUE(buf->seq == 1, "first acquire should set seq to 1");
    EXPECT_TRUE(buf->data != nullptr, "acquired buffer data pointer is null");
    EXPECT_TRUE(pool.free_buffer_count() == 3, "free count should decrease by 1 after acquire");

    pool.release(buf);
}

// ---- Test: acquire all buffers exhausts pool (Bounded mode) ----

void test_acquire_all_buffers_exhausts_pool() {
    constexpr std::size_t count = 3;
    StagingPool pool(count, 512, 50, count, StagingPool::GrowMode::Bounded);

    StagingBuffer *bufs[3];
    for (std::size_t i = 0; i < count; ++i) {
        bufs[i] = pool.acquire(256);
        EXPECT_TRUE(bufs[i] != nullptr, "acquire failed before pool exhaustion");
    }

    EXPECT_TRUE(pool.free_buffer_count() == 0, "free count should be 0 after acquiring all");

    // Release all.
    for (std::size_t i = 0; i < count; ++i) {
        pool.release(bufs[i]);
    }

    EXPECT_TRUE(pool.free_buffer_count() == count, "free count should be restored after releasing all");
}

// ---- Test: backpressure timeout returns nullptr (Bounded mode) ----

void test_backpressure_timeout_returns_nullptr() {
    // Bounded pool with 1 buffer and a short timeout: the historical
    // hard-limit contract, preserved for callers that opt out of auto-grow.
    StagingPool pool(1, 1024, 10, 1, StagingPool::GrowMode::Bounded);  // 10ms timeout

    StagingBuffer *buf = pool.acquire(512);
    EXPECT_TRUE(buf != nullptr, "first acquire should succeed");

    auto start = std::chrono::steady_clock::now();
    StagingBuffer *timeout_buf = pool.acquire(512);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(timeout_buf == nullptr, "acquire should return nullptr on timeout");

    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    EXPECT_TRUE(elapsed_ms >= 9,  // allow 1ms jitter
                "timeout should wait at least ~10ms, got " + std::to_string(elapsed_ms) + "ms");

    pool.release(buf);
}

// ---- Test: release wakes waiting thread ----

void test_release_wakes_waiting_thread() {
    // Bounded: with auto-grow the waiter would simply get a new slot and
    // this test would not exercise the condition-variable wake path.
    StagingPool pool(1, 1024, 5000, 1, StagingPool::GrowMode::Bounded);  // 5s timeout (long)

    StagingBuffer *buf = pool.acquire(512);
    EXPECT_TRUE(buf != nullptr, "first acquire should succeed");

    std::atomic<bool> acquired{false};
    std::atomic<bool> started{false};

    // Spawn a thread that will block on acquire.
    std::thread waiter([&]() {
        started.store(true, std::memory_order_release);
        StagingBuffer *b = pool.acquire(256);
        if (b != nullptr) {
            acquired.store(true, std::memory_order_release);
            pool.release(b);
        }
    });

    // Wait for the waiter thread to start.
    while (!started.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Give it a moment to actually block on the CV.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // Release the buffer -- this should wake the waiter.
    pool.release(buf);

    waiter.join();

    EXPECT_TRUE(acquired.load(std::memory_order_acquire), "waiter should have acquired buffer after release");
}

// ---- Test: ref_count and add_ref ----

void test_ref_count_and_add_ref() {
    StagingPool pool(2, 512, 100);

    StagingBuffer *buf = pool.acquire(256);
    EXPECT_TRUE(buf != nullptr, "acquire should succeed");
    EXPECT_TRUE(buf->ref_count == 1, "initial ref_count should be 1");

    // Add a reference (simulating multi-view sharing).
    pool.add_ref(buf);
    EXPECT_TRUE(buf->ref_count == 2, "ref_count should be 2 after add_ref");

    // First release: ref_count drops to 1, buffer stays in-use.
    pool.release(buf);
    EXPECT_TRUE(buf->ref_count == 1, "ref_count should be 1 after first release");
    EXPECT_TRUE(pool.free_buffer_count() == 1, "buffer should NOT be on free list with ref_count=1");

    // Second release: ref_count drops to 0, buffer returns to pool.
    pool.release(buf);
    EXPECT_TRUE(pool.free_buffer_count() == 2, "buffer should be on free list after final release");
}

// ---- Test: generation counter increments on each acquire ----

void test_generation_counter_increments() {
    StagingPool pool(1, 1024, 100);

    StagingBuffer *buf = pool.acquire(512);
    EXPECT_TRUE(buf != nullptr, "first acquire");
    EXPECT_TRUE(buf->seq == 1, "seq should be 1 on first acquire");

    pool.release(buf);

    buf = pool.acquire(512);
    EXPECT_TRUE(buf != nullptr, "second acquire");
    EXPECT_TRUE(buf->seq == 2, "seq should be 2 on second acquire");

    pool.release(buf);

    buf = pool.acquire(512);
    EXPECT_TRUE(buf != nullptr, "third acquire");
    EXPECT_TRUE(buf->seq == 3, "seq should be 3 on third acquire");

    pool.release(buf);
}

// ---- Test: buffer data is writable ----

void test_buffer_data_is_writable() {
    StagingPool pool(2, 256, 100);

    StagingBuffer *buf = pool.acquire(128);
    EXPECT_TRUE(buf != nullptr, "acquire should succeed");

    // Write to the buffer (simulating a snapshot copy).
    std::memset(buf->data, 0xAB, 128);
    buf->used_bytes = 128;

    // Verify the write.
    EXPECT_TRUE(static_cast<unsigned char>(buf->data[0]) == 0xAB, "buffer data should be writable and readable");
    EXPECT_TRUE(static_cast<unsigned char>(buf->data[127]) == 0xAB, "buffer data end should be writable");
    EXPECT_TRUE(buf->used_bytes == 128, "used_bytes should reflect the payload size");

    pool.release(buf);
}

// ---- Test: acquire with required_bytes > nominal capacity grows --------
//
// Grow-to-required: a pool provisioned with a small nominal capacity still
// serves an oversized slab by expanding a free buffer's storage to the
// requested size, rather than returning AMIO_ERR_STAGING_BACKPRESSURE.  This
// is what lets CECE lower the read-pool default capacity without risking a
// failure on an unexpectedly large record.

void test_acquire_oversized_request_grows() {
    // All buffers nominally 512 bytes; requesting 1024 should grow one and
    // succeed (not time out).
    StagingPool pool(2, 512, 10);  // 10ms timeout

    StagingBuffer *buf = pool.acquire(1024);
    EXPECT_TRUE(buf != nullptr, "acquire for oversized request should grow and succeed");
    EXPECT_TRUE(buf->capacity_bytes >= 1024, "grown buffer capacity < requested");
    EXPECT_TRUE(buf->data != nullptr, "grown buffer data pointer is null");
    EXPECT_TRUE(pool.committed_bytes() >= 1024, "committed_bytes should reflect the grown buffer");

    pool.release(buf);
    // The grown capacity persists after release (no shrink).
    EXPECT_TRUE(pool.committed_bytes() >= 1024, "grown buffer should retain capacity after release");
}

// ---- Test: oversized request still backpressures at the hard ceiling ----

void test_acquire_oversized_backpressures_when_exhausted() {
    // Bounded pool: no free buffer and no growth allowed, so even a
    // growable request must time out.
    StagingPool pool(1, 512, 10, 1, StagingPool::GrowMode::Bounded);  // one buffer, 10ms timeout

    StagingBuffer *held = pool.acquire(256);
    EXPECT_TRUE(held != nullptr, "first acquire should succeed");

    StagingBuffer *buf = pool.acquire(4096);
    EXPECT_TRUE(buf == nullptr, "oversized acquire with no free buffer should return nullptr");

    pool.release(held);
}

// ---- Test: exhausted pool auto-grows a new slot instead of failing ----
//
// This is the "it should always work" property: buffer_count is a
// provisioning hint, not a limit a host model can crash on.  When every
// slot is busy and the ceiling has not been reached, acquire appends a
// fresh slot (committed lazily at the requested size) and succeeds
// immediately rather than waiting out the staging timeout.

void test_exhausted_pool_auto_grows_slot() {
    constexpr std::size_t count = 2;
    constexpr std::size_t ceiling = 6;
    StagingPool pool(count, 512, 10, ceiling);  // 10ms timeout: a failure would be fast

    EXPECT_TRUE(pool.total_buffer_count() == count, "initial slots == buffer_count");
    EXPECT_TRUE(pool.max_buffer_count() == ceiling, "max_buffer_count reported");
    EXPECT_TRUE(pool.initial_buffer_count() == count, "initial_buffer_count reported");

    StagingBuffer *b1 = pool.acquire(256);
    StagingBuffer *b2 = pool.acquire(256);
    EXPECT_TRUE(b1 != nullptr && b2 != nullptr, "within-count acquires should succeed");
    EXPECT_TRUE(pool.free_buffer_count() == 0, "pool exhausted after count acquires");

    // The contract under test: an exhausted pool must NOT fail.
    auto start = std::chrono::steady_clock::now();
    StagingBuffer *b3 = pool.acquire(256);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(b3 != nullptr, "exhausted pool should auto-grow a slot, not return nullptr");
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    EXPECT_TRUE(elapsed_ms < 10, "auto-grow must not wait out the staging timeout, got " + std::to_string(elapsed_ms) + "ms");

    EXPECT_TRUE(pool.total_buffer_count() == count + 1, "a slot was appended");
    EXPECT_TRUE(pool.allocated_buffer_count() == count + 1, "the new slot committed its storage");

    // Grown slots are independent buffers with distinct storage.
    EXPECT_TRUE(b3->data != nullptr, "grown slot has data");
    EXPECT_TRUE(b3->data != b1->data && b3->data != b2->data, "grown slot must not alias an in-use buffer");
    EXPECT_TRUE(b3->capacity_bytes >= 256, "grown slot sized to the request");

    // Held pointers stay valid after growth (vectors are reserved to the
    // ceiling), so release by pointer arithmetic still works.
    pool.release(b1);
    pool.release(b2);
    pool.release(b3);
    EXPECT_TRUE(pool.free_buffer_count() == count + 1, "all slots returned after growth");
}

// ---- Test: auto-grow honours the hard ceiling ----
//
// At max_buffer_count the pool stops growing and the original
// backpressure contract returns: the request blocks and then yields
// nullptr, which now signals genuine saturation (stuck IO / leaked
// views) rather than a sizing mistake.

void test_auto_grow_stops_at_ceiling() {
    constexpr std::size_t count = 1;
    constexpr std::size_t ceiling = 2;
    StagingPool pool(count, 512, 10, ceiling);  // 10ms timeout

    StagingBuffer *b1 = pool.acquire(256);
    EXPECT_TRUE(b1 != nullptr, "first acquire should succeed");

    StagingBuffer *b2 = pool.acquire(256);
    EXPECT_TRUE(b2 != nullptr, "second acquire should auto-grow");
    EXPECT_TRUE(pool.total_buffer_count() == ceiling, "grown to the ceiling");

    // Ceiling reached: backpressure is preserved.
    StagingBuffer *b3 = pool.acquire(256);
    EXPECT_TRUE(b3 == nullptr, "acquire at the ceiling should return nullptr (backpressure)");
    EXPECT_TRUE(pool.total_buffer_count() == ceiling, "no slots appended past the ceiling");

    pool.release(b1);
    pool.release(b2);

    // Once a slot frees up, service resumes without further growth.
    StagingBuffer *b4 = pool.acquire(256);
    EXPECT_TRUE(b4 != nullptr, "acquire should succeed again after a release");
    EXPECT_TRUE(pool.total_buffer_count() == ceiling, "total count unchanged after release + acquire");
    pool.release(b4);
}

// ---- Test: auto-grown slot keeps earlier buffers' pointers valid ----
//
// Guards the reserve()-to-ceiling invariant: release() recovers a slot's
// index from its pointer, so a reallocation during growth would corrupt
// the free list.  Interleave acquires, a growth event, and releases.

void test_growth_preserves_outstanding_pointers() {
    StagingPool pool(1, 256, 50, 8);

    StagingBuffer *held[6];
    held[0] = pool.acquire(64);
    EXPECT_TRUE(held[0] != nullptr, "acquire 0");

    for (int i = 1; i < 6; ++i) {
        held[i] = pool.acquire(64);
        EXPECT_TRUE(held[i] != nullptr, "acquire should succeed via auto-grow");
    }

    EXPECT_TRUE(pool.total_buffer_count() == 6, "five slots auto-grown");
    EXPECT_TRUE(pool.free_buffer_count() == 0, "all slots in use");

    // Release in a scrambled order; each must find its own slot.
    pool.release(held[3]);
    pool.release(held[0]);
    pool.release(held[5]);
    EXPECT_TRUE(pool.free_buffer_count() == 3, "three slots returned");

    // Re-acquire must reuse the freed slots, not grow further.
    StagingBuffer *again = pool.acquire(64);
    EXPECT_TRUE(again != nullptr, "re-acquire after release");
    EXPECT_TRUE(pool.total_buffer_count() == 6, "reuse must not grow the pool");
    pool.release(again);

    pool.release(held[1]);
    pool.release(held[2]);
    pool.release(held[4]);
    EXPECT_TRUE(pool.free_buffer_count() == 6, "all slots free at the end");
}

// ---- Test: lazy allocation commits nothing until first acquire -----------

void test_lazy_allocation_commits_on_demand() {
    // A pool with a large count and capacity must not reserve count*capacity
    // bytes up front; that eager reservation is the OOM this change removes.
    constexpr std::size_t count = 8;
    constexpr std::size_t capacity = 1024 * 1024;  // 1 MiB nominal
    StagingPool pool(count, capacity, 100);

    EXPECT_TRUE(pool.committed_bytes() == 0, "construction must not commit any byte storage");
    EXPECT_TRUE(pool.allocated_buffer_count() == 0, "construction must allocate no buffer storage");
    EXPECT_TRUE(pool.free_buffer_count() == count, "all slots should be free after construction");

    StagingBuffer *b1 = pool.acquire(512);
    EXPECT_TRUE(b1 != nullptr, "first acquire should succeed");
    EXPECT_TRUE(pool.allocated_buffer_count() == 1, "exactly one buffer committed after one acquire");
    EXPECT_TRUE(pool.committed_bytes() == capacity, "a within-nominal acquire commits the nominal capacity");

    // A second small acquire commits a second buffer.
    StagingBuffer *b2 = pool.acquire(512);
    EXPECT_TRUE(b2 != nullptr, "second acquire should succeed");
    EXPECT_TRUE(pool.allocated_buffer_count() == 2, "two buffers committed after two acquires");

    pool.release(b1);
    pool.release(b2);
    // Released buffers keep their committed storage (they stay in the pool).
    EXPECT_TRUE(pool.allocated_buffer_count() == 2, "released buffers retain committed storage");
    EXPECT_TRUE(pool.free_buffer_count() == count, "all slots free again after release");
}

// ---- Test: concurrent acquire and release ----

void test_concurrent_acquire_release() {
    constexpr std::size_t count = 4;
    StagingPool pool(count, 1024, 2000);

    constexpr int kThreads = 4;
    constexpr int kIterations = 200;

    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (int tid = 0; tid < kThreads; ++tid) {
        workers.emplace_back([&]() {
            for (int i = 0; i < kIterations; ++i) {
                StagingBuffer *buf = pool.acquire(256);
                if (buf == nullptr) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (buf->ref_count != 1) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
                // Simulate brief work.
                buf->used_bytes = 256;
                std::this_thread::yield();
                pool.release(buf);
            }
        });
    }

    for (auto &w : workers) {
        w.join();
    }

    EXPECT_TRUE(failures.load(std::memory_order_relaxed) == 0, "concurrent acquire/release had " + std::to_string(failures.load()) + " failures");
    EXPECT_TRUE(pool.free_buffer_count() == count, "all buffers should be free after concurrent test");
}

}  // namespace

int main() {
    test_construction_allocates_buffers();
    test_acquire_returns_valid_buffer();
    test_acquire_all_buffers_exhausts_pool();
    test_backpressure_timeout_returns_nullptr();
    test_release_wakes_waiting_thread();
    test_ref_count_and_add_ref();
    test_generation_counter_increments();
    test_buffer_data_is_writable();
    test_acquire_oversized_request_grows();
    test_acquire_oversized_backpressures_when_exhausted();
    test_lazy_allocation_commits_on_demand();
    test_exhausted_pool_auto_grows_slot();
    test_auto_grow_stops_at_ceiling();
    test_growth_preserves_outstanding_pointers();
    test_concurrent_acquire_release();

    std::fprintf(stdout, "test_staging_pool: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
