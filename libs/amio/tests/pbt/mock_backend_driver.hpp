// mock_backend_driver.hpp -- MockBackendDriver for PBT tests.
//
// A test double that implements the Backend_Driver interface and:
//   * Records all write/read/flush/close calls with timestamps
//     and thread IDs
//   * Allows injecting failures (throw on next write, etc.)
//   * Tracks call ordering for order-preservation tests
//
// This mock is used for testing AMIO_Core infrastructure (staging,
// worker pool, prefetch, ordering) without requiring real file I/O.
// The actual round-trip property tests (P1) use real backend drivers.
//
// Validates: R11.2 (testing infrastructure)

#ifndef AMIO_TESTS_PBT_MOCK_BACKEND_DRIVER_HPP
#define AMIO_TESTS_PBT_MOCK_BACKEND_DRIVER_HPP

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"

namespace amio::pbt {

// ===================================================================
// CallRecord -- records a single Backend_Driver method invocation.
// ===================================================================

struct CallRecord {
    enum class Method { OpenWrite, OpenRead, Write, Read, Flush, Close };

    Method method;
    std::thread::id thread_id;
    std::chrono::steady_clock::time_point timestamp;
    std::uint64_t sequence;  // global call order

    // For write calls: dataset_id, variable_id, payload size
    std::uint64_t dataset_id = 0;
    std::uint64_t variable_id = 0;
    std::size_t payload_size = 0;

    // For read calls: timestep
    std::int64_t timestep = -1;

    // VarMeta captured on write/read (added for Property 6: VarMeta
    // completeness).  Records the exact name/dtype/shape the driver
    // received so a property can assert they match the variable's
    // describe_variable() result and the caller-supplied name.
    std::string name;
    amio_dtype_t dtype = AMIO_DTYPE_F32;
    amio_shape_t shape = {};

    // Bounding box captured on read (has_bbox=false for a full read).
    bool has_bbox = false;
    amio::detail::BoundingBox bbox = {};
};

// ===================================================================
// MockBackendDriver -- records calls and optionally injects failures.
//
// Thread-safe: all public methods are safe to call concurrently.
// ===================================================================

class MockBackendDriver : public amio::detail::Backend_Driver {
   public:
    MockBackendDriver() = default;
    ~MockBackendDriver() override = default;

    // ----- Backend_Driver interface implementation -----

    void open_write(const conf::Config & /*config*/) override {
        std::lock_guard<std::mutex> lock(mu_);
        record_call(CallRecord::Method::OpenWrite);
        check_and_throw(CallRecord::Method::OpenWrite);
    }

    void open_read(const conf::Config & /*config*/) override {
        std::lock_guard<std::mutex> lock(mu_);
        record_call(CallRecord::Method::OpenRead);
        check_and_throw(CallRecord::Method::OpenRead);
    }

    void write(const amio::detail::StagingBuffer &src, const amio::detail::VarMeta &meta) override {
        std::lock_guard<std::mutex> lock(mu_);

        CallRecord rec;
        rec.method = CallRecord::Method::Write;
        rec.thread_id = std::this_thread::get_id();
        rec.timestamp = std::chrono::steady_clock::now();
        rec.sequence = next_seq_++;
        rec.dataset_id = meta.dataset_id;
        rec.variable_id = meta.variable_id;
        rec.payload_size = src.used_bytes;
        rec.name = meta.name;
        rec.dtype = meta.dtype;
        rec.shape = meta.shape;
        calls_.push_back(rec);

        // Store the written payload for round-trip verification
        if (store_payloads_) {
            std::vector<std::byte> payload(src.used_bytes);
            std::memcpy(payload.data(), src.data, src.used_bytes);
            stored_payloads_.push_back(std::move(payload));
        }

        check_and_throw(CallRecord::Method::Write);
    }

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta &meta, std::int64_t timestep,
              const std::optional<amio::detail::BoundingBox> &bbox) override {
        std::lock_guard<std::mutex> lock(mu_);

        CallRecord rec;
        rec.method = CallRecord::Method::Read;
        rec.thread_id = std::this_thread::get_id();
        rec.timestamp = std::chrono::steady_clock::now();
        rec.sequence = next_seq_++;
        rec.dataset_id = meta.dataset_id;
        rec.variable_id = meta.variable_id;
        rec.timestep = timestep;
        rec.name = meta.name;
        rec.dtype = meta.dtype;
        rec.shape = meta.shape;
        if (bbox.has_value()) {
            rec.has_bbox = true;
            rec.bbox = bbox.value();
        }
        calls_.push_back(rec);

        check_and_throw(CallRecord::Method::Read);

        // If we have stored payloads, return the appropriate one
        if (!stored_payloads_.empty()) {
            std::size_t idx = static_cast<std::size_t>(timestep) % stored_payloads_.size();
            const auto &payload = stored_payloads_[idx];
            std::size_t copy_size = std::min(payload.size(), dst.capacity_bytes);
            std::memcpy(dst.data, payload.data(), copy_size);
            dst.used_bytes = copy_size;
        }
    }

    void flush() override {
        std::lock_guard<std::mutex> lock(mu_);
        record_call(CallRecord::Method::Flush);
        check_and_throw(CallRecord::Method::Flush);
    }

    void close() override {
        std::lock_guard<std::mutex> lock(mu_);
        record_call(CallRecord::Method::Close);
        check_and_throw(CallRecord::Method::Close);
    }

    // ----- Test control interface -----

    // Inject a failure: the next call to `method` will throw.
    void inject_failure(CallRecord::Method method, const std::string &message = "Injected failure") {
        std::lock_guard<std::mutex> lock(mu_);
        pending_failures_.push_back({method, message});
    }

    // Inject a failure that triggers after N calls to `method`.
    void inject_failure_after(CallRecord::Method method, std::size_t after_n_calls, const std::string &message = "Injected failure") {
        std::lock_guard<std::mutex> lock(mu_);
        deferred_failures_.push_back({method, after_n_calls, message, 0});
    }

    // Enable/disable payload storage for round-trip tests.
    void set_store_payloads(bool enable) {
        std::lock_guard<std::mutex> lock(mu_);
        store_payloads_ = enable;
    }

    // ----- Observation interface -----

    // Get all recorded calls (thread-safe copy).
    std::vector<CallRecord> get_calls() const {
        std::lock_guard<std::mutex> lock(mu_);
        return calls_;
    }

    // Get calls filtered by method.
    std::vector<CallRecord> get_calls(CallRecord::Method method) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<CallRecord> filtered;
        for (const auto &c : calls_) {
            if (c.method == method) {
                filtered.push_back(c);
            }
        }
        return filtered;
    }

    // Get write calls for a specific (dataset, variable) pair, in order.
    std::vector<CallRecord> get_writes_for(std::uint64_t dataset_id, std::uint64_t variable_id) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<CallRecord> filtered;
        for (const auto &c : calls_) {
            if (c.method == CallRecord::Method::Write && c.dataset_id == dataset_id && c.variable_id == variable_id) {
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

    // Get call count for a specific method.
    std::size_t call_count(CallRecord::Method method) const {
        std::lock_guard<std::mutex> lock(mu_);
        std::size_t count = 0;
        for (const auto &c : calls_) {
            if (c.method == method) ++count;
        }
        return count;
    }

    // Get stored payloads (for round-trip verification).
    std::vector<std::vector<std::byte>> get_stored_payloads() const {
        std::lock_guard<std::mutex> lock(mu_);
        return stored_payloads_;
    }

    // Reset all recorded state.
    void reset() {
        std::lock_guard<std::mutex> lock(mu_);
        calls_.clear();
        stored_payloads_.clear();
        pending_failures_.clear();
        deferred_failures_.clear();
        next_seq_ = 0;
    }

    // Check if all write calls for a (dataset, variable) pair arrived
    // in sequence order (useful for order-preservation tests).
    bool writes_in_order(std::uint64_t dataset_id, std::uint64_t variable_id) const {
        auto writes = get_writes_for(dataset_id, variable_id);
        for (std::size_t i = 1; i < writes.size(); ++i) {
            if (writes[i].sequence <= writes[i - 1].sequence) {
                return false;
            }
        }
        return true;
    }

   private:
    struct PendingFailure {
        CallRecord::Method method;
        std::string message;
    };

    struct DeferredFailure {
        CallRecord::Method method;
        std::size_t trigger_after;
        std::string message;
        std::size_t current_count;
    };

    void record_call(CallRecord::Method method) {
        CallRecord rec;
        rec.method = method;
        rec.thread_id = std::this_thread::get_id();
        rec.timestamp = std::chrono::steady_clock::now();
        rec.sequence = next_seq_++;
        calls_.push_back(rec);
    }

    void check_and_throw(CallRecord::Method method) {
        // Check immediate pending failures
        for (auto it = pending_failures_.begin(); it != pending_failures_.end(); ++it) {
            if (it->method == method) {
                std::string msg = it->message;
                pending_failures_.erase(it);
                throw std::runtime_error(msg);
            }
        }

        // Check deferred failures
        for (auto &df : deferred_failures_) {
            if (df.method == method) {
                df.current_count++;
                if (df.current_count >= df.trigger_after) {
                    std::string msg = df.message;
                    // Remove this deferred failure (one-shot)
                    df.trigger_after = std::size_t(-1);  // disable
                    throw std::runtime_error(msg);
                }
            }
        }
    }

    mutable std::mutex mu_;
    std::vector<CallRecord> calls_;
    std::vector<std::vector<std::byte>> stored_payloads_;
    std::vector<PendingFailure> pending_failures_;
    std::vector<DeferredFailure> deferred_failures_;
    std::uint64_t next_seq_ = 0;
    bool store_payloads_ = false;
};

}  // namespace amio::pbt

#endif  // AMIO_TESTS_PBT_MOCK_BACKEND_DRIVER_HPP
