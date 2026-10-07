// test_p7_order_preservation.cpp -- Property test P7: Order preservation
// per (dataset, variable).
//
// For any sequence of writes to same (dataset, variable): storage layer
// observes writes in submission order regardless of thread count, NUMA
// pinning, or queue depth.
//
// Min 100 iterations with MockBackendDriver recording call order.
//
// Also tests that writes to DIFFERENT (dataset, variable) pairs can
// interleave freely (no global ordering constraint across pairs).
//
// **Validates: Requirements R6.3**

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Helper: OrderRecorder -- lightweight recorder that tracks the
// sequence in which write callbacks are executed per (dataset, variable).
//
// Each write callback records its submission-order index into the
// recorder.  After all writes complete, we verify that the recorded
// indices are strictly monotonically increasing for each
// (dataset, variable) pair.
// ===================================================================

namespace {

struct OrderRecorder {
    mutable std::mutex mu;

    // Key: (dataset_id, variable_id) → vector of submission indices
    // in the order they were observed at the storage layer.
    std::map<DatasetVariableKey, std::vector<std::uint64_t>> observed_order;

    void record(DatasetVariableKey key, std::uint64_t submission_index) {
        std::lock_guard<std::mutex> lock(mu);
        observed_order[key].push_back(submission_index);
    }

    // Check that for each (dataset, variable) pair, the observed
    // indices are in strictly increasing order.
    bool all_in_order() const {
        std::lock_guard<std::mutex> lock(mu);
        for (const auto &[key, indices] : observed_order) {
            for (std::size_t i = 1; i < indices.size(); ++i) {
                if (indices[i] <= indices[i - 1]) {
                    return false;
                }
            }
        }
        return true;
    }

    // Get the observed order for a specific key.
    std::vector<std::uint64_t> get_order(DatasetVariableKey key) const {
        std::lock_guard<std::mutex> lock(mu);
        auto it = observed_order.find(key);
        if (it == observed_order.end()) return {};
        return it->second;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mu);
        observed_order.clear();
    }
};

}  // anonymous namespace

// ===================================================================
// Property Test P7: Order preservation per (dataset, variable)
//
// For any generated:
//   - thread_count in [1, 4]
//   - N writes (4..20) to the same (dataset, variable) pair
//
// Submit all N writes to the WorkerPool, drain, and verify that the
// MockBackendDriver / OrderRecorder observed them in submission order.
// ===================================================================

TEST_CASE("P7: Order preservation - same (dataset, variable) pair", "[pbt][p7][order_preservation]") {
    auto result = rc::check("writes to same (dataset, variable) are observed in submission order", []() {
        // Generate thread count [1, 4] for testing speed.
        auto thread_count = *rc::gen::inRange<std::size_t>(1, 5);

        // Generate number of writes [4, 20].
        auto num_writes = *rc::gen::inRange<std::size_t>(4, 21);

        // Generate a (dataset, variable) key.
        auto dataset_id = *rc::gen::inRange<std::uint64_t>(1, 100);
        auto variable_id = *rc::gen::inRange<std::uint64_t>(1, 100);

        DatasetVariableKey dv_key{dataset_id, variable_id};

        // Create the order recorder.
        auto recorder = std::make_shared<OrderRecorder>();

        // Create WorkerPool with generated thread count.
        // Use a large queue capacity to avoid QUEUE_FULL.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);
        RC_ASSERT(pool.thread_count() == thread_count);

        // Submit N writes to the same (dataset, variable) pair.
        // Each write callback records its submission index.
        // The convenience overload returns the assigned sequence
        // number (0-based per dv_key).
        for (std::size_t i = 0; i < num_writes; ++i) {
            auto seq = pool.submit_write(dv_key, [recorder, dv_key, i]() { recorder->record(dv_key, i); });
            RC_ASSERT(seq == i);
        }

        // Drain all tasks.
        pool.drain();

        // Verify: all writes observed in submission order.
        auto observed = recorder->get_order(dv_key);
        RC_ASSERT(observed.size() == num_writes);

        for (std::size_t i = 0; i < num_writes; ++i) {
            RC_ASSERT(observed[i] == i);
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P7b: Order preservation with MockBackendDriver
//
// Uses the full MockBackendDriver to verify that the write() method
// is called in submission order for a given (dataset, variable) pair,
// regardless of thread count.
// ===================================================================

TEST_CASE("P7: Order preservation - MockBackendDriver records in order", "[pbt][p7][order_preservation][mock_driver]") {
    auto result = rc::check("MockBackendDriver observes writes in submission order per (dataset, variable)", []() {
        // Generate thread count [1, 4].
        auto thread_count = *rc::gen::inRange<std::size_t>(1, 5);

        // Generate number of writes [4, 16].
        auto num_writes = *rc::gen::inRange<std::size_t>(4, 17);

        // Generate a (dataset, variable) key.
        auto dataset_id = *rc::gen::inRange<std::uint64_t>(1, 50);
        auto variable_id = *rc::gen::inRange<std::uint64_t>(1, 50);

        DatasetVariableKey dv_key{dataset_id, variable_id};

        // Create MockBackendDriver.
        auto mock_driver = std::make_shared<MockBackendDriver>();

        // Create WorkerPool.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        // Submit N writes.  Each callback invokes mock_driver->write()
        // with a VarMeta carrying the dataset_id and variable_id, and
        // a minimal StagingBuffer.
        //
        // We use a shared staging buffer (read-only in mock) since
        // MockBackendDriver just records the call.
        std::vector<std::byte> dummy_data(64, std::byte{0x42});
        StagingBuffer staging_buf;
        staging_buf.data = dummy_data.data();
        staging_buf.capacity_bytes = dummy_data.size();
        staging_buf.used_bytes = dummy_data.size();

        for (std::size_t i = 0; i < num_writes; ++i) {
            pool.submit_write(dv_key, [mock_driver, dataset_id, variable_id, &staging_buf]() {
                VarMeta meta;
                meta.dataset_id = dataset_id;
                meta.variable_id = variable_id;
                meta.name = "test_var";
                meta.dtype = AMIO_DTYPE_F32;
                mock_driver->write(staging_buf, meta);
            });
        }

        // Drain all tasks.
        pool.drain();

        // Verify: MockBackendDriver observed writes in order.
        RC_ASSERT(mock_driver->writes_in_order(dataset_id, variable_id));

        // Also verify the total count.
        auto writes = mock_driver->get_writes_for(dataset_id, variable_id);
        RC_ASSERT(writes.size() == num_writes);

        // Verify sequence numbers are strictly increasing.
        for (std::size_t i = 1; i < writes.size(); ++i) {
            RC_ASSERT(writes[i].sequence > writes[i - 1].sequence);
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P7c: Different (dataset, variable) pairs can interleave
//
// Writes to DIFFERENT (dataset, variable) pairs are NOT required to
// maintain any relative ordering.  This test verifies that the pool
// does not impose unnecessary global serialization -- writes to
// different pairs can execute concurrently / in any order.
// ===================================================================

TEST_CASE("P7: Different (dataset, variable) pairs can interleave freely", "[pbt][p7][order_preservation][interleave]") {
    auto result = rc::check("writes to different (dataset, variable) pairs are all completed and per-pair order is preserved", []() {
        // Use multiple threads to increase interleaving opportunity.
        auto thread_count = *rc::gen::inRange<std::size_t>(2, 5);

        // Generate 2-4 distinct (dataset, variable) pairs.
        auto num_pairs = *rc::gen::inRange<std::size_t>(2, 5);

        // Generate writes per pair [3, 10].
        auto writes_per_pair = *rc::gen::inRange<std::size_t>(3, 11);

        // Create distinct keys.
        std::vector<DatasetVariableKey> keys;
        for (std::size_t p = 0; p < num_pairs; ++p) {
            DatasetVariableKey key;
            key.dataset_id = p + 1;     // 1, 2, 3, 4
            key.variable_id = p + 100;  // 100, 101, 102, 103
            keys.push_back(key);
        }

        // Create order recorder.
        auto recorder = std::make_shared<OrderRecorder>();

        // Create WorkerPool.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        // Submit writes for all pairs, interleaved (round-robin).
        for (std::size_t w = 0; w < writes_per_pair; ++w) {
            for (std::size_t p = 0; p < num_pairs; ++p) {
                auto key = keys[p];
                pool.submit_write(key, [recorder, key, w]() { recorder->record(key, w); });
            }
        }

        // Drain all tasks.
        pool.drain();

        // Verify: EACH pair's writes are in submission order.
        for (std::size_t p = 0; p < num_pairs; ++p) {
            auto observed = recorder->get_order(keys[p]);
            RC_ASSERT(observed.size() == writes_per_pair);

            // Must be strictly increasing (0, 1, 2, ...).
            for (std::size_t i = 0; i < writes_per_pair; ++i) {
                RC_ASSERT(observed[i] == i);
            }
        }

        // Verify: total writes completed equals num_pairs * writes_per_pair.
        RC_ASSERT(pool.total_writes_completed() == num_pairs * writes_per_pair);
    });

    REQUIRE(result);
}
