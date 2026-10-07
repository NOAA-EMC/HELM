// test_p15_driver_thread_invariant.cpp -- Property test P15: Driver-thread
// invariant.
//
// For any amio_write, amio_read, or amio_flush: no Backend_Driver virtual
// method executes on calling thread; all execute exclusively on Worker_Pool
// threads.
//
// Min 100 iterations with real backend driver; verify thread IDs via
// instrumentation hooks.
//
// **Validates: Requirements R3.4, R6.1**

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// InstrumentedDriver -- records the thread ID of every Backend_Driver
// virtual method invocation.  Used to verify that no driver method
// executes on the calling (test main) thread.
// ===================================================================

namespace {

class InstrumentedDriver : public amio::detail::Backend_Driver {
   public:
    InstrumentedDriver() = default;
    ~InstrumentedDriver() override = default;

    void open_write(const conf::Config & /*config*/) override {
        record_thread();
    }

    void open_read(const conf::Config & /*config*/) override {
        record_thread();
    }

    void write(const amio::detail::StagingBuffer & /*src*/, const amio::detail::VarMeta & /*meta*/) override {
        record_thread();
    }

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta & /*meta*/, std::int64_t /*timestep*/,
              const std::optional<amio::detail::BoundingBox> & /*bbox*/) override {
        record_thread();
        // Fill with dummy data so the read "succeeds".
        if (dst.capacity_bytes > 0) {
            std::memset(dst.data, 0xAB, std::min(dst.capacity_bytes, std::size_t{64}));
            dst.used_bytes = std::min(dst.capacity_bytes, std::size_t{64});
        }
    }

    void flush() override {
        record_thread();
    }

    void close() override {
        record_thread();
    }

    // Return all thread IDs that invoked any driver method.
    std::set<std::thread::id> get_driver_threads() const {
        std::lock_guard<std::mutex> lock(mu_);
        return driver_threads_;
    }

    // Return the total number of driver method invocations.
    std::size_t invocation_count() const {
        return invocation_count_.load();
    }

   private:
    void record_thread() {
        std::lock_guard<std::mutex> lock(mu_);
        driver_threads_.insert(std::this_thread::get_id());
        invocation_count_.fetch_add(1);
    }

    mutable std::mutex mu_;
    std::set<std::thread::id> driver_threads_;
    std::atomic<std::size_t> invocation_count_{0};
};

}  // anonymous namespace

// ===================================================================
// Property Test P15a: Write path -- driver methods never execute on
// the calling thread.
//
// For any generated:
//   - thread_count in [1, 4]
//   - N writes (2..10) to the same (dataset, variable) pair
//
// Submit all N writes to the WorkerPool (each callback invokes the
// InstrumentedDriver::write), drain, and verify that the calling
// thread's ID is NOT in the set of driver thread IDs.
// ===================================================================

TEST_CASE("P15: Driver-thread invariant - write path", "[pbt][p15][driver_thread][write]") {
    auto result = rc::check("no Backend_Driver::write executes on calling thread", []() {
        // Capture the calling (test main) thread ID.
        const auto calling_thread = std::this_thread::get_id();

        // Generate thread count [1, 4].
        auto thread_count = *rc::gen::inRange<std::size_t>(1, 5);

        // Generate number of writes [2, 10].
        auto num_writes = *rc::gen::inRange<std::size_t>(2, 11);

        // Create the instrumented driver.
        auto driver = std::make_shared<InstrumentedDriver>();

        // Create WorkerPool.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        // Generate a (dataset, variable) key.
        DatasetVariableKey dv_key{1, 1};

        // Submit N writes.  Each callback invokes driver->write().
        std::vector<std::byte> dummy_data(64, std::byte{0x42});
        StagingBuffer staging_buf;
        staging_buf.data = dummy_data.data();
        staging_buf.capacity_bytes = dummy_data.size();
        staging_buf.used_bytes = dummy_data.size();

        for (std::size_t i = 0; i < num_writes; ++i) {
            pool.submit_write(dv_key, [driver, &staging_buf]() {
                VarMeta meta;
                meta.dataset_id = 1;
                meta.variable_id = 1;
                meta.name = "test_var";
                meta.dtype = AMIO_DTYPE_F32;
                driver->write(staging_buf, meta);
            });
        }

        // Drain all tasks.
        pool.drain();

        // Verify: the instrumented driver was actually called.
        RC_ASSERT(driver->invocation_count() == num_writes);

        // Verify: the calling thread's ID is NOT in the set of
        // threads that invoked driver methods.
        auto driver_threads = driver->get_driver_threads();
        RC_ASSERT(driver_threads.find(calling_thread) == driver_threads.end());

        // Verify: all driver invocations happened on worker threads
        // (i.e., threads different from the calling thread).
        for (const auto &tid : driver_threads) {
            RC_ASSERT(tid != calling_thread);
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P15b: Read path -- driver methods never execute on
// the calling thread.
//
// Submit prefetch tasks that invoke InstrumentedDriver::read on
// worker threads, then verify the calling thread never invoked
// any driver method.
// ===================================================================

TEST_CASE("P15: Driver-thread invariant - read/prefetch path", "[pbt][p15][driver_thread][read]") {
    auto result = rc::check("no Backend_Driver::read executes on calling thread", []() {
        // Capture the calling (test main) thread ID.
        const auto calling_thread = std::this_thread::get_id();

        // Generate thread count [1, 4].
        auto thread_count = *rc::gen::inRange<std::size_t>(1, 5);

        // Generate number of prefetch tasks [2, 8].
        auto num_reads = *rc::gen::inRange<std::size_t>(2, 9);

        // Create the instrumented driver.
        auto driver = std::make_shared<InstrumentedDriver>();

        // Create WorkerPool.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        // Submit prefetch tasks that invoke driver->read().
        for (std::size_t i = 0; i < num_reads; ++i) {
            std::int64_t timestep = static_cast<std::int64_t>(i);
            pool.submit_prefetch(timestep, timestep, /*dataset_id=*/1, [driver]() {
                // Simulate a read by calling driver->read().
                std::vector<std::byte> buf(64, std::byte{0});
                StagingBuffer staging_buf;
                staging_buf.data = buf.data();
                staging_buf.capacity_bytes = buf.size();
                staging_buf.used_bytes = 0;

                VarMeta meta;
                meta.dataset_id = 1;
                meta.variable_id = 1;
                meta.name = "test_var";
                meta.dtype = AMIO_DTYPE_F32;

                driver->read(staging_buf, meta, 0, std::nullopt);
            });
        }

        // Drain all tasks.
        pool.drain();

        // Verify: the instrumented driver was actually called.
        RC_ASSERT(driver->invocation_count() == num_reads);

        // Verify: the calling thread's ID is NOT in the set of
        // threads that invoked driver methods.
        auto driver_threads = driver->get_driver_threads();
        RC_ASSERT(driver_threads.find(calling_thread) == driver_threads.end());
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P15c: Flush path -- driver flush method never
// executes on the calling thread.
//
// Submit write tasks that invoke driver->flush() as part of their
// callback, then verify the calling thread never invoked flush.
// ===================================================================

TEST_CASE("P15: Driver-thread invariant - flush via worker", "[pbt][p15][driver_thread][flush]") {
    auto result = rc::check("no Backend_Driver::flush executes on calling thread", []() {
        // Capture the calling (test main) thread ID.
        const auto calling_thread = std::this_thread::get_id();

        // Generate thread count [1, 4].
        auto thread_count = *rc::gen::inRange<std::size_t>(1, 5);

        // Generate number of flush-triggering tasks [1, 5].
        auto num_flushes = *rc::gen::inRange<std::size_t>(1, 6);

        // Create the instrumented driver.
        auto driver = std::make_shared<InstrumentedDriver>();

        // Create WorkerPool.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        DatasetVariableKey dv_key{1, 1};

        // Submit tasks that call driver->flush() on worker threads.
        for (std::size_t i = 0; i < num_flushes; ++i) {
            pool.submit_write(dv_key, [driver]() { driver->flush(); });
        }

        // Drain all tasks.
        pool.drain();

        // Verify: the instrumented driver was actually called.
        RC_ASSERT(driver->invocation_count() == num_flushes);

        // Verify: the calling thread's ID is NOT in the set of
        // threads that invoked driver methods.
        auto driver_threads = driver->get_driver_threads();
        RC_ASSERT(driver_threads.find(calling_thread) == driver_threads.end());
    });

    REQUIRE(result);
}
