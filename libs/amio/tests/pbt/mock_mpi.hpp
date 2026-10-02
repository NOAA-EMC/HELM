// mock_mpi.hpp -- MockMpi for testing MPI-related paths in PBT.
//
// Provides a lightweight MPI mock that:
//   * Simulates single-rank MPI environment (rank 0, size 1)
//   * Records communicator split calls
//   * Allows injecting split failures
//   * Tracks MPI_THREAD_MULTIPLE initialization state
//   * Records collective operations for verification
//
// This mock is used for testing AMIO_Core's MPI-related logic
// (communicator splits, thread-level checks, I/O rank routing)
// without requiring a real multi-rank MPI environment.
//
// Validates: R11.2 (testing infrastructure)

#ifndef AMIO_TESTS_PBT_MOCK_MPI_HPP
#define AMIO_TESTS_PBT_MOCK_MPI_HPP

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace amio::pbt {

// ===================================================================
// MpiCallRecord -- records a single MPI-related operation.
// ===================================================================

struct MpiCallRecord {
    enum class Operation { Init, Finalize, CommSplit, CommFree, Barrier, Allreduce, Bcast, Send, Recv, QueryThreadLevel };

    Operation operation;
    int color = 0;  // for CommSplit
    int key = 0;    // for CommSplit
    int source_rank = 0;
    int dest_rank = 0;
    std::string tag;  // descriptive tag
};

// ===================================================================
// MockMpi -- simulates MPI environment for testing.
//
// Thread-safe: all public methods are safe to call concurrently.
// ===================================================================

class MockMpi {
   public:
    MockMpi() = default;
    ~MockMpi() = default;

    // Non-copyable, non-movable (singleton-like usage in tests).
    MockMpi(const MockMpi &) = delete;
    MockMpi &operator=(const MockMpi &) = delete;
    MockMpi(MockMpi &&) = delete;
    MockMpi &operator=(MockMpi &&) = delete;

    // ----- Configuration -----

    // Set the simulated world size and rank.
    void set_world(int rank, int size) {
        std::lock_guard<std::mutex> lock(mu_);
        world_rank_ = rank;
        world_size_ = size;
    }

    // Set the MPI thread level (e.g., MPI_THREAD_MULTIPLE = 3).
    void set_thread_level(int level) {
        std::lock_guard<std::mutex> lock(mu_);
        thread_level_ = level;
    }

    // Inject a communicator split failure.
    void inject_split_failure(const std::string &message = "CommSplit failed") {
        std::lock_guard<std::mutex> lock(mu_);
        split_failure_message_ = message;
        split_should_fail_ = true;
    }

    // Clear injected split failure.
    void clear_split_failure() {
        std::lock_guard<std::mutex> lock(mu_);
        split_should_fail_ = false;
        split_failure_message_.clear();
    }

    // ----- Simulated MPI operations -----

    // Query world rank.
    int rank() const {
        std::lock_guard<std::mutex> lock(mu_);
        return world_rank_;
    }

    // Query world size.
    int size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return world_size_;
    }

    // Query thread level provided at init.
    int query_thread_level() const {
        std::lock_guard<std::mutex> lock(mu_);
        record_call_internal(MpiCallRecord::Operation::QueryThreadLevel);
        return thread_level_;
    }

    // Simulate MPI_Comm_split.
    // Returns 0 on success, non-zero on failure.
    int comm_split(int color, int key) {
        std::lock_guard<std::mutex> lock(mu_);

        MpiCallRecord rec;
        rec.operation = MpiCallRecord::Operation::CommSplit;
        rec.color = color;
        rec.key = key;
        calls_.push_back(rec);

        if (split_should_fail_) {
            return -1;  // Simulate failure
        }

        // Record the new communicator
        split_count_++;
        return 0;
    }

    // Simulate MPI_Comm_free.
    void comm_free() {
        std::lock_guard<std::mutex> lock(mu_);
        record_call_internal(MpiCallRecord::Operation::CommFree);
        if (split_count_ > 0) split_count_--;
    }

    // Simulate MPI_Barrier.
    void barrier() {
        std::lock_guard<std::mutex> lock(mu_);
        record_call_internal(MpiCallRecord::Operation::Barrier);
    }

    // ----- Observation interface -----

    // Get all recorded calls.
    std::vector<MpiCallRecord> get_calls() const {
        std::lock_guard<std::mutex> lock(mu_);
        return calls_;
    }

    // Get calls filtered by operation.
    std::vector<MpiCallRecord> get_calls(MpiCallRecord::Operation op) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<MpiCallRecord> filtered;
        for (const auto &c : calls_) {
            if (c.operation == op) {
                filtered.push_back(c);
            }
        }
        return filtered;
    }

    // Get total call count.
    std::size_t call_count() const {
        std::lock_guard<std::mutex> lock(mu_);
        return calls_.size();
    }

    // Get number of active (un-freed) communicator splits.
    int active_splits() const {
        std::lock_guard<std::mutex> lock(mu_);
        return split_count_;
    }

    // Check if split was attempted.
    bool split_was_attempted() const {
        std::lock_guard<std::mutex> lock(mu_);
        for (const auto &c : calls_) {
            if (c.operation == MpiCallRecord::Operation::CommSplit) {
                return true;
            }
        }
        return false;
    }

    // Reset all state.
    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        calls_.clear();
        split_count_ = 0;
        split_should_fail_ = false;
        split_failure_message_.clear();
        world_rank_ = 0;
        world_size_ = 1;
        thread_level_ = 3;  // MPI_THREAD_MULTIPLE
    }

   private:
    void record_call_internal(MpiCallRecord::Operation op) const {
        MpiCallRecord rec;
        rec.operation = op;
        calls_.push_back(rec);
    }

    mutable std::mutex mu_;
    mutable std::vector<MpiCallRecord> calls_;
    int world_rank_ = 0;
    int world_size_ = 1;
    int thread_level_ = 3;  // MPI_THREAD_MULTIPLE
    int split_count_ = 0;
    bool split_should_fail_ = false;
    std::string split_failure_message_;
};

}  // namespace amio::pbt

#endif  // AMIO_TESTS_PBT_MOCK_MPI_HPP
