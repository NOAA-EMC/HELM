// test_p14_prefetch_failure_surfaces.cpp -- Property test P14:
// Prefetch failure surfaces on read.
//
// For any timestep T whose background fetch failed: Prefetch_Queue
// retains failure record; next amio_read(T) returns non-zero
// AMIO_ERR_* and no Memory_View.
//
// Min 100 iterations with injected failures.
//
// **Validates: Requirements R5.8**

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

#include "factory/backend_driver.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// FailingDriver -- a Backend_Driver that fails on specific timesteps.
//
// This driver throws an exception when read() is called for any
// timestep in the `fail_timesteps` set. For other timesteps, it
// performs a normal (minimal) read operation.
//
// This is NOT a mock -- it implements real Backend_Driver behavior
// with controlled failure injection for testing the failure
// propagation path.
// ===================================================================

namespace {

class FailingDriver : public Backend_Driver {
   public:
    FailingDriver() = default;
    ~FailingDriver() override = default;

    void open_write(const conf::Config & /*config*/) override {}
    void open_read(const conf::Config & /*config*/) override {}
    void flush() override {}
    void close() override {}

    void write(const StagingBuffer & /*src*/, const VarMeta & /*meta*/) override {}

    void read(StagingBuffer &dst, const VarMeta & /*meta*/, std::int64_t timestep, const std::optional<BoundingBox> & /*bbox*/) override {
        std::lock_guard<std::mutex> lock(mu_);

        read_count_++;

        // Check if this timestep should fail.
        if (fail_timesteps_.count(timestep) > 0) {
            throw std::runtime_error("Injected read failure for timestep " + std::to_string(timestep));
        }

        // Normal read: write some data to the buffer.
        std::size_t write_size = std::min(std::size_t{64}, dst.capacity_bytes);
        std::memset(dst.data, static_cast<int>(timestep & 0xFF), write_size);
        dst.used_bytes = write_size;
    }

    // Configure which timesteps should fail.
    void set_fail_timesteps(const std::set<std::int64_t> &timesteps) {
        std::lock_guard<std::mutex> lock(mu_);
        fail_timesteps_ = timesteps;
    }

    void add_fail_timestep(std::int64_t timestep) {
        std::lock_guard<std::mutex> lock(mu_);
        fail_timesteps_.insert(timestep);
    }

    std::size_t read_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return read_count_;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        fail_timesteps_.clear();
        read_count_ = 0;
    }

   private:
    mutable std::mutex mu_;
    std::set<std::int64_t> fail_timesteps_;
    std::size_t read_count_ = 0;
};

}  // anonymous namespace

// ===================================================================
// Property Test P14: Prefetch failure surfaces on read
//
// For any generated:
//   - look-ahead depth N in [2, 16]
//   - dataset timestep count M in [4, 64]
//   - failing timestep T in [0, min(N,M)-1]
//
// When the background fetch for timestep T fails, the PrefetchQueue
// retains the failure record. The next call to get_buffer(T) must:
//   1. Return a non-zero AMIO_ERR_* status code
//   2. NOT return a Memory_View (out_buf == nullptr)
//   3. The error code must be AMIO_ERR_BACKEND_FAILURE
// ===================================================================

TEST_CASE("P14: Prefetch failure surfaces - failed fetch returns error on read", "[pbt][p14][prefetch][failure]") {
    auto result = rc::check("failed background fetch surfaces as AMIO_ERR_* on next amio_read(T)", []() {
        // Generate look-ahead depth N in [2, 16].
        auto depth = *rc::gen::inRange<std::size_t>(2, 17);

        // Generate dataset timestep count M in [4, 64].
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(4, 65);

        // Choose a failing timestep within the initial prefetch window.
        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);
        auto fail_timestep = *rc::gen::inRange<std::int64_t>(0, max_prefetched);

        // Create staging pool.
        std::size_t buffer_count = static_cast<std::size_t>(max_prefetched) + 4;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create failing driver.
        auto driver = std::make_shared<FailingDriver>();
        driver->add_fail_timestep(fail_timestep);

        // Create PrefetchQueue in synchronous mode.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool,
                         nullptr,  // synchronous mode
                         driver.get(), 1, "test_var", info, total_timesteps);

        // Schedule initial fetches.
        // The fetch for fail_timestep will throw and be recorded
        // as a failure.
        pq.schedule_initial();

        // Verify: the failed timestep is in the failed set.
        RC_ASSERT(pq.failed_count() >= 1);

        // Verify: successful fetches completed for other timesteps.
        // Total = completed + failed should equal min(N, M).
        std::size_t expected_total = static_cast<std::size_t>(max_prefetched);
        RC_ASSERT(pq.completed_count() + pq.failed_count() == expected_total);

        // Now read the failed timestep.
        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(fail_timestep, nullptr, &buf);

        // Verify: non-zero error code returned.
        RC_ASSERT(status != AMIO_OK);

        // Verify: the error is AMIO_ERR_BACKEND_FAILURE.
        RC_ASSERT(status == AMIO_ERR_BACKEND_FAILURE);

        // Verify: no Memory_View returned (buf is nullptr).
        RC_ASSERT(buf == nullptr);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P14b: Multiple failures are retained independently
//
// When multiple timesteps fail, each failure is retained and
// surfaced independently on the corresponding read.
// ===================================================================

TEST_CASE("P14: Prefetch failure surfaces - multiple failures retained", "[pbt][p14][prefetch][failure][multiple]") {
    auto result = rc::check("multiple failed fetches are retained and surfaced independently", []() {
        // Generate look-ahead depth N in [4, 16].
        auto depth = *rc::gen::inRange<std::size_t>(4, 17);

        // Generate dataset timestep count M >= depth.
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(static_cast<std::int64_t>(depth), static_cast<std::int64_t>(depth) + 32);

        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);

        // Generate 2-3 failing timesteps within the prefetch window.
        auto num_failures = *rc::gen::inRange<std::size_t>(2, 4);
        num_failures = std::min(num_failures, static_cast<std::size_t>(max_prefetched));

        // Pick `num_failures` DISTINCT timesteps in [0, max_prefetched).
        //
        // We must not call the generator inside an unbounded
        // "while (set.size() < num_failures)" loop: during shrinking
        // RapidCheck can repeatedly yield the same minimal value (e.g. 0),
        // so the set never reaches the target size, the loop spins
        // forever, and the per-iteration generator bookkeeping grows
        // without bound until std::bad_alloc.  Instead, draw a single
        // starting index and select distinct timesteps deterministically.
        auto start = *rc::gen::inRange<std::int64_t>(0, max_prefetched);
        std::set<std::int64_t> fail_set;
        for (std::size_t i = 0; i < num_failures; ++i) {
            fail_set.insert((start + static_cast<std::int64_t>(i)) % max_prefetched);
        }
        // The modulo selection above yields exactly num_failures distinct
        // values because num_failures <= max_prefetched.
        num_failures = fail_set.size();
        // Create staging pool.
        std::size_t buffer_count = static_cast<std::size_t>(max_prefetched) + 4;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create failing driver.
        auto driver = std::make_shared<FailingDriver>();
        driver->set_fail_timesteps(fail_set);

        // Create PrefetchQueue in synchronous mode.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, nullptr, driver.get(), 1, "test_var", info, total_timesteps);

        // Schedule initial fetches.
        pq.schedule_initial();

        // Verify: failed count matches the number of injected failures.
        RC_ASSERT(pq.failed_count() == num_failures);

        // Verify: each failed timestep returns an error.
        for (std::int64_t t : fail_set) {
            StagingBuffer *buf = nullptr;
            amio_status_t status = pq.get_buffer(t, nullptr, &buf);
            RC_ASSERT(status != AMIO_OK);
            RC_ASSERT(status == AMIO_ERR_BACKEND_FAILURE);
            RC_ASSERT(buf == nullptr);
        }

        // Verify: non-failed timesteps return successfully.
        for (std::int64_t t = 0; t < max_prefetched; ++t) {
            if (fail_set.count(t) == 0) {
                StagingBuffer *buf = nullptr;
                amio_status_t status = pq.get_buffer(t, nullptr, &buf);
                RC_ASSERT(status == AMIO_OK);
                RC_ASSERT(buf != nullptr);
                pool.release(buf);
            }
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P14c: Failure with Worker_Pool (async mode)
//
// Same property but with asynchronous dispatch through Worker_Pool.
// After draining, the failure should still be surfaced correctly.
// ===================================================================

TEST_CASE("P14: Prefetch failure surfaces - async with Worker_Pool", "[pbt][p14][prefetch][failure][async]") {
    auto result = rc::check("async failed fetch surfaces as AMIO_ERR_* after Worker_Pool drain", []() {
        // Generate look-ahead depth N in [2, 8].
        auto depth = *rc::gen::inRange<std::size_t>(2, 9);

        // Generate dataset timestep count M in [4, 32].
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(4, 33);

        std::int64_t max_prefetched = std::min(static_cast<std::int64_t>(depth), total_timesteps);

        // Choose a failing timestep.
        auto fail_timestep = *rc::gen::inRange<std::int64_t>(0, max_prefetched);

        // Create staging pool.
        std::size_t buffer_count = static_cast<std::size_t>(max_prefetched) + 4;
        StagingPool pool(buffer_count, 4096, 5000);

        // Create failing driver.
        auto driver = std::make_shared<FailingDriver>();
        driver->add_fail_timestep(fail_timestep);

        // Create Worker_Pool.
        WorkerPoolConfig wp_config;
        wp_config.thread_count = 2;
        wp_config.backpressure.queue_capacity = 1024;
        wp_config.backpressure.enabled = false;

        auto workers = std::make_unique<WorkerPool>(wp_config);

        // Create PrefetchQueue with worker pool.
        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, workers.get(), driver.get(), 1, "test_var", info, total_timesteps);

        // Schedule initial fetches (dispatched to worker pool).
        pq.schedule_initial();

        // Drain the worker pool.
        workers->drain();

        // Verify: the failed timestep returns an error.
        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(fail_timestep, nullptr, &buf);
        RC_ASSERT(status != AMIO_OK);
        RC_ASSERT(status == AMIO_ERR_BACKEND_FAILURE);
        RC_ASSERT(buf == nullptr);

        // Verify: non-failed timesteps succeed.
        for (std::int64_t t = 0; t < max_prefetched; ++t) {
            if (t != fail_timestep) {
                StagingBuffer *nbuf = nullptr;
                amio_status_t nstatus = pq.get_buffer(t, nullptr, &nbuf);
                RC_ASSERT(nstatus == AMIO_OK);
                RC_ASSERT(nbuf != nullptr);
                pool.release(nbuf);
            }
        }
    });

    REQUIRE(result);
}
