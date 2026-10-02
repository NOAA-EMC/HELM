// test_p9_worker_pool_admission.cpp -- Property test P9: Worker_Pool admission control invariant.
//
// For any config with watermarks (L,H) where 0<=L<H<=capacity:
//   queue>=H blocks until <L; without backpressure: exceeding
//   capacity → AMIO_ERR_QUEUE_FULL without enqueue.
//
// Min 100 iterations.
// Uses the REAL WorkerPool (not mocked).
//
// **Validates: Requirements R6.8, R6.9**

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"
#include "workers/worker_pool.hpp"

namespace {

// ===================================================================
// Helper: Generate valid backpressure watermarks.
//
// Generates (capacity, H, L) such that:
//   capacity in [3, 32]  (small for fast testing)
//   H in [2, capacity]
//   L in [0, H-1]
//
// These constraints satisfy the invariant 0 <= L < H <= capacity.
// ===================================================================

struct WatermarkConfig {
    std::size_t capacity;
    std::size_t high_watermark;
    std::size_t low_watermark;
};

rc::Gen<WatermarkConfig> genWatermarkConfig() {
    return rc::gen::exec([]() {
        WatermarkConfig cfg;
        // Keep capacity small for fast test iterations.
        cfg.capacity = *rc::gen::inRange<std::size_t>(3, 33);
        // H must be at least 2 (so L can be at least 0 and L < H).
        // H must be <= capacity.
        std::size_t h_min = 2;
        std::size_t h_max = cfg.capacity;
        cfg.high_watermark = *rc::gen::inRange<std::size_t>(h_min, h_max + 1);
        // L must be in [0, H-1].
        cfg.low_watermark = *rc::gen::inRange<std::size_t>(0, cfg.high_watermark);
        return cfg;
    });
}

// Helper: Create a WorkerPoolConfig with backpressure enabled.
amio::detail::WorkerPoolConfig make_backpressure_config(const WatermarkConfig &wm) {
    amio::detail::WorkerPoolConfig config;
    config.thread_count = 1;  // Single worker thread for deterministic testing.
    config.backpressure.enabled = true;
    config.backpressure.queue_capacity = wm.capacity;
    config.backpressure.high_watermark = wm.high_watermark;
    config.backpressure.low_watermark = wm.low_watermark;
    return config;
}

// Helper: Create a WorkerPoolConfig WITHOUT backpressure (R6.9 case).
amio::detail::WorkerPoolConfig make_no_backpressure_config(std::size_t capacity) {
    amio::detail::WorkerPoolConfig config;
    config.thread_count = 1;
    config.backpressure.enabled = false;
    config.backpressure.queue_capacity = capacity;
    config.backpressure.high_watermark = 0;
    config.backpressure.low_watermark = 0;
    return config;
}

// A DatasetVariableKey for testing.
amio::detail::DatasetVariableKey test_dv_key(std::uint64_t ds = 1, std::uint64_t var = 1) {
    amio::detail::DatasetVariableKey key;
    key.dataset_id = ds;
    key.variable_id = var;
    return key;
}

}  // anonymous namespace

// ===================================================================
// Property Test P9a: Backpressure blocking behavior (R6.8)
//
// For any config with watermarks (L, H) where 0 <= L < H <= capacity:
//   1. Create a WorkerPool with backpressure enabled.
//   2. Block the worker thread so tasks accumulate in the queue.
//   3. Submit tasks until queue depth reaches H.
//   4. Verify that the next submission blocks (does not return
//      immediately).
//   5. Unblock the worker thread to drain tasks until queue < L.
//   6. Verify that the blocked submission completes (unblocks).
// ===================================================================

TEST_CASE("P9a: Worker_Pool backpressure blocks at H, resumes at L", "[pbt][p9][worker_pool][admission][backpressure]") {
    auto result = rc::check("queue>=H blocks until depth<L (R6.8)", []() {
        auto wm = *genWatermarkConfig();

        RC_TAG(wm.capacity);
        RC_CLASSIFY(wm.high_watermark == wm.capacity, "H == capacity");
        RC_CLASSIFY(wm.low_watermark == 0, "L == 0");

        // --- Set up a WorkerPool with backpressure ---
        auto config = make_backpressure_config(wm);

        // Use a gate to block the worker thread from processing tasks.
        // This lets us fill the queue to the high watermark.
        std::mutex gate_mu;
        std::condition_variable gate_cv;
        bool gate_open = false;

        // Track how many tasks have been executed.
        std::atomic<std::size_t> tasks_executed{0};

        amio::detail::WorkerPool pool(config);

        // Verify configuration was applied.
        RC_ASSERT(pool.backpressure_enabled());
        RC_ASSERT(pool.high_watermark() == wm.high_watermark);
        RC_ASSERT(pool.low_watermark() == wm.low_watermark);
        RC_ASSERT(pool.queue_capacity() == wm.capacity);

        // --- Fill the queue to H by submitting tasks that block ---
        // Each task waits on the gate before completing.
        auto blocking_task = [&]() {
            std::unique_lock<std::mutex> lock(gate_mu);
            gate_cv.wait(lock, [&]() { return gate_open; });
            tasks_executed.fetch_add(1, std::memory_order_release);
        };

        // Submit H tasks.  The first task will be dequeued by the
        // worker immediately but will block on the gate.  The
        // remaining H-1 tasks stay in the queue.  So after
        // submitting H tasks, the queue depth is H-1 (one is
        // in-flight).  We need to submit enough to get the queue
        // depth to >= H.
        //
        // Since one task is immediately dequeued and blocks, we
        // need to submit H+1 tasks to get queue depth == H.
        // But we need to be careful: the worker dequeues one task
        // immediately, so queue depth after N submissions is N-1
        // (if the worker is blocked on the first task).
        //
        // Strategy: submit H tasks.  The worker picks up the first
        // one and blocks.  Queue depth = H - 1.  If H - 1 >= H,
        // that's impossible.  So we need to submit H + 1 tasks
        // to get queue depth = H.
        //
        // Actually, let's think more carefully:
        // - Worker thread starts and waits for work.
        // - We submit task 1: worker wakes, dequeues it, starts
        //   executing (blocks on gate).  Queue depth = 0.
        // - We submit task 2: queue depth = 1.
        // - ...
        // - We submit task H+1: queue depth = H.
        //
        // At this point, queue depth == H == high_watermark, so
        // the next submit should block.
        //
        // However, there's a race: the worker might not have
        // dequeued the first task yet.  To handle this, we use
        // a separate "first task started" signal.

        std::mutex started_mu;
        std::condition_variable started_cv;
        bool first_task_started = false;

        // Submit the first task with a "started" signal.
        auto first_task = [&]() {
            {
                std::lock_guard<std::mutex> lock(started_mu);
                first_task_started = true;
            }
            started_cv.notify_all();
            // Now block on the gate.
            std::unique_lock<std::mutex> lock(gate_mu);
            gate_cv.wait(lock, [&]() { return gate_open; });
            tasks_executed.fetch_add(1, std::memory_order_release);
        };

        std::uint64_t seq = 0;
        amio_err_t rc = pool.submit_write(test_dv_key(), first_task, &seq);
        RC_ASSERT(rc == AMIO_OK);

        // Wait for the worker to pick up the first task.
        {
            std::unique_lock<std::mutex> lock(started_mu);
            started_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return first_task_started; });
        }
        RC_ASSERT(first_task_started);

        // Now the worker is blocked on the gate.  Submit H more
        // tasks to fill the queue to depth H.
        for (std::size_t i = 0; i < wm.high_watermark; ++i) {
            rc = pool.submit_write(test_dv_key(2, i + 1), blocking_task, &seq);
            RC_ASSERT(rc == AMIO_OK);
        }

        // Queue depth should now be == H (the first task is
        // in-flight, not in the queue).
        RC_ASSERT(pool.write_queue_depth() >= wm.high_watermark);

        // --- Verify that the next submission BLOCKS ---
        // We submit from a separate thread and check that it
        // doesn't return within a short timeout.
        std::atomic<bool> submit_completed{false};
        std::thread submitter([&]() {
            std::uint64_t s = 0;
            pool.submit_write(test_dv_key(3, 1), []() {}, &s);
            submit_completed.store(true, std::memory_order_release);
        });

        // Give the submitter a chance to block.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // The submitter should still be blocked (queue >= H).
        RC_ASSERT(!submit_completed.load(std::memory_order_acquire));

        // --- Open the gate to drain tasks ---
        {
            std::lock_guard<std::mutex> lock(gate_mu);
            gate_open = true;
        }
        gate_cv.notify_all();

        // Wait for the submitter to complete (it should unblock
        // once queue depth drops below L).
        submitter.join();
        RC_ASSERT(submit_completed.load(std::memory_order_acquire));

        // --- Shutdown and drain ---
        pool.shutdown();
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P9b: No-backpressure queue-full rejection (R6.9)
//
// For any capacity in [2, 32]:
//   1. Create a WorkerPool WITHOUT backpressure.
//   2. Block the worker thread so tasks accumulate.
//   3. Fill the queue to capacity.
//   4. Verify that the next submission returns AMIO_ERR_QUEUE_FULL
//      immediately without enqueuing.
// ===================================================================

TEST_CASE("P9b: Worker_Pool no-backpressure returns AMIO_ERR_QUEUE_FULL", "[pbt][p9][worker_pool][admission][queue_full]") {
    auto result = rc::check("exceeding capacity without backpressure → AMIO_ERR_QUEUE_FULL (R6.9)", []() {
        // Generate a small capacity for fast testing.
        std::size_t capacity = *rc::gen::inRange<std::size_t>(2, 33);

        RC_TAG(capacity);

        // --- Set up a WorkerPool without backpressure ---
        auto config = make_no_backpressure_config(capacity);

        // Gate to block the worker thread.
        std::mutex gate_mu;
        std::condition_variable gate_cv;
        bool gate_open = false;

        std::atomic<std::size_t> tasks_executed{0};

        amio::detail::WorkerPool pool(config);

        // Verify no backpressure.
        RC_ASSERT(!pool.backpressure_enabled());
        RC_ASSERT(pool.queue_capacity() == capacity);

        // --- Block the worker with the first task ---
        std::mutex started_mu;
        std::condition_variable started_cv;
        bool first_task_started = false;

        auto first_task = [&]() {
            {
                std::lock_guard<std::mutex> lock(started_mu);
                first_task_started = true;
            }
            started_cv.notify_all();
            std::unique_lock<std::mutex> lock(gate_mu);
            gate_cv.wait(lock, [&]() { return gate_open; });
            tasks_executed.fetch_add(1, std::memory_order_release);
        };

        auto blocking_task = [&]() {
            std::unique_lock<std::mutex> lock(gate_mu);
            gate_cv.wait(lock, [&]() { return gate_open; });
            tasks_executed.fetch_add(1, std::memory_order_release);
        };

        std::uint64_t seq = 0;
        amio_err_t rc = pool.submit_write(test_dv_key(), first_task, &seq);
        RC_ASSERT(rc == AMIO_OK);

        // Wait for the worker to pick up the first task.
        {
            std::unique_lock<std::mutex> lock(started_mu);
            started_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return first_task_started; });
        }
        RC_ASSERT(first_task_started);

        // --- Fill the queue to capacity ---
        // The first task is in-flight (not in queue), so we can
        // submit `capacity` more tasks to fill the queue.
        for (std::size_t i = 0; i < capacity; ++i) {
            rc = pool.submit_write(test_dv_key(2, i + 1), blocking_task, &seq);
            RC_ASSERT(rc == AMIO_OK);
        }

        // Queue depth should now be == capacity.
        RC_ASSERT(pool.write_queue_depth() >= capacity);

        // --- Verify that the next submission returns QUEUE_FULL ---
        std::uint64_t rejected_seq = 999;
        amio_err_t reject_rc = pool.submit_write(test_dv_key(3, 1), []() {}, &rejected_seq);

        RC_ASSERT(reject_rc == AMIO_ERR_QUEUE_FULL);
        RC_ASSERT(rejected_seq == 0);  // No sequence assigned.

        // Queue depth should NOT have increased.
        RC_ASSERT(pool.write_queue_depth() <= capacity);

        // --- Cleanup: open gate and shutdown ---
        {
            std::lock_guard<std::mutex> lock(gate_mu);
            gate_open = true;
        }
        gate_cv.notify_all();

        pool.shutdown();
    });

    REQUIRE(result);
}
