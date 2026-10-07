// test_p16_exception_bridge.cpp -- Property test P16: Exception bridge
// invariant.
//
// For any exception on Worker_Pool thread (std::exception,
// unknown): C boundary catches, records AMIO_ERR_* against handle, ensures
// parallel stack-trace emission completes before surfacing, surfaces on next
// flush/close/wait without remapping, preserves trace for AMIO_Core lifetime.
//
// Min 100 iterations with injected exceptions.
//
// **Validates: Requirements R12.1, R12.2, R12.4, R12.9, R12.10**

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"
#include "workers/exception_bridge.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Exception types for injection testing.
// ===================================================================

namespace {

// A custom exception type simulating a domain-specific exception.
// Uses std::runtime_error as the base for testing exception bridge
// handling of derived exception types.
class DomainSpecificException : public std::runtime_error {
   public:
    explicit DomainSpecificException(const std::string &msg) : std::runtime_error(msg) {}
};

// A standard library exception.
class StandardException : public std::logic_error {
   public:
    explicit StandardException(const std::string &msg) : std::logic_error(msg) {}
};

// Exception kind enum for generation.
enum class ExceptionKind {
    StdRuntimeError,
    StdLogicError,
    DomainSpecific,
    UnknownThrow  // throw an int (non-std::exception)
};

// Throw the specified kind of exception.
[[noreturn]] void throw_exception(ExceptionKind kind, const std::string &msg) {
    switch (kind) {
        case ExceptionKind::StdRuntimeError:
            throw std::runtime_error(msg);
        case ExceptionKind::StdLogicError:
            throw StandardException(msg);
        case ExceptionKind::DomainSpecific:
            throw DomainSpecificException(msg);
        case ExceptionKind::UnknownThrow:
            throw 42;  // non-std::exception type
    }
    // Unreachable, but silence compiler warnings.
    throw std::runtime_error("unreachable");
}

}  // anonymous namespace

// ===================================================================
// Property Test P16a: Exception on worker thread is recorded against
// the originating handle.
//
// For any generated:
//   - exception kind (std::runtime_error, std::logic_error, unknown)
//   - handle_id (arbitrary uint64)
//   - thread_count [1, 4]
//
// Submit a write task that throws the generated exception.  After
// drain, verify that the OutcomeRegistry has a failure recorded
// against the handle_id with a non-zero AMIO_ERR_* code.
// ===================================================================

TEST_CASE("P16: Exception bridge - exception recorded against handle", "[pbt][p16][exception_bridge][record]") {
    auto result = rc::check("exceptions on worker threads are recorded against originating handle", []() {
        // Generate exception kind.
        auto kind = *rc::gen::elementOf(std::vector<ExceptionKind>{ExceptionKind::StdRuntimeError, ExceptionKind::StdLogicError,
                                                                   ExceptionKind::DomainSpecific, ExceptionKind::UnknownThrow});

        // Generate handle_id.
        auto handle_id = *rc::gen::inRange<std::uint64_t>(1, 10000);

        // Generate thread count [1, 4].
        auto thread_count = *rc::gen::inRange<std::size_t>(1, 5);

        // Create WorkerPool with exception cordon.
        WorkerPoolConfig config;
        config.thread_count = thread_count;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        DatasetVariableKey dv_key{1, 1};

        // Submit a write task that throws.
        std::string msg = "test_exception_" + std::to_string(handle_id);
        pool.submit_write(dv_key, handle_id, [kind, msg]() { throw_exception(kind, msg); });

        // Drain all tasks (exception should be caught by cordon).
        pool.drain();

        // Verify: the outcome registry has a failure for this handle.
        const auto &registry = pool.outcome_registry();
        RC_ASSERT(registry.has_failure(handle_id));

        // Verify: the recorded error code is non-zero (AMIO_ERR_*).
        auto first_failure = registry.get_first_failure(handle_id);
        RC_ASSERT(first_failure.is_failure());
        RC_ASSERT(first_failure.error_code != AMIO_OK);

        // Verify: the error code is a valid AMIO_ERR_* value.
        // For std::exception derivatives, expect AMIO_ERR_BACKEND_FAILURE.
        // For unknown throws, also expect AMIO_ERR_BACKEND_FAILURE.
        RC_ASSERT(first_failure.error_code == AMIO_ERR_BACKEND_FAILURE);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P16b: Exception message is preserved in the outcome.
//
// For any std::exception-derived throw: the outcome message contains
// the what() string from the exception.
// ===================================================================

TEST_CASE("P16: Exception bridge - message preserved", "[pbt][p16][exception_bridge][message]") {
    auto result = rc::check("exception what() message is preserved in outcome", []() {
        // Generate a unique message.
        auto suffix = *rc::gen::inRange<int>(1, 100000);
        std::string msg = "error_msg_" + std::to_string(suffix);

        // Generate handle_id.
        auto handle_id = *rc::gen::inRange<std::uint64_t>(1, 10000);

        // Use std::runtime_error so we can verify the message.
        WorkerPoolConfig config;
        config.thread_count = 1;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        DatasetVariableKey dv_key{1, 1};

        pool.submit_write(dv_key, handle_id, [msg]() { throw std::runtime_error(msg); });

        pool.drain();

        // Verify: the outcome message contains the exception text.
        const auto &registry = pool.outcome_registry();
        auto first_failure = registry.get_first_failure(handle_id);
        RC_ASSERT(first_failure.is_failure());

        // The message should contain our injected text.
        RC_ASSERT(first_failure.message.find(msg) != std::string::npos);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P16c: Multiple exceptions accumulate against the
// same handle.
//
// For any N exceptions (2..5) thrown against the same handle_id:
// all are recorded and retrievable.
// ===================================================================

TEST_CASE("P16: Exception bridge - multiple failures accumulate", "[pbt][p16][exception_bridge][accumulate]") {
    auto result = rc::check("multiple exceptions against same handle are all recorded", []() {
        // Generate number of failures [2, 5].
        auto num_failures = *rc::gen::inRange<std::size_t>(2, 6);

        // Generate handle_id.
        auto handle_id = *rc::gen::inRange<std::uint64_t>(1, 10000);

        WorkerPoolConfig config;
        config.thread_count = 1;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        DatasetVariableKey dv_key{1, 1};

        // Submit N tasks that all throw.
        for (std::size_t i = 0; i < num_failures; ++i) {
            std::string msg = "failure_" + std::to_string(i);
            pool.submit_write(dv_key, handle_id, [msg]() { throw std::runtime_error(msg); });
        }

        pool.drain();

        // Verify: all failures are recorded.
        const auto &registry = pool.outcome_registry();
        RC_ASSERT(registry.has_failure(handle_id));

        auto outcomes = registry.get_outcomes(handle_id);
        RC_ASSERT(outcomes.size() == num_failures);

        // Each outcome should be a failure.
        for (const auto &outcome : outcomes) {
            RC_ASSERT(outcome.is_failure());
            RC_ASSERT(outcome.error_code == AMIO_ERR_BACKEND_FAILURE);
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P16d: Outcomes are preserved for the pool's lifetime.
//
// After recording failures, the outcomes remain accessible until
// the pool is destroyed (simulating AMIO_Core lifetime preservation).
// ===================================================================

TEST_CASE("P16: Exception bridge - outcomes preserved for lifetime", "[pbt][p16][exception_bridge][lifetime]") {
    auto result = rc::check("outcomes are preserved and accessible after drain", []() {
        // Generate handle_id.
        auto handle_id = *rc::gen::inRange<std::uint64_t>(1, 10000);

        WorkerPoolConfig config;
        config.thread_count = 2;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        DatasetVariableKey dv_key{1, 1};

        // Submit a failing task.
        pool.submit_write(dv_key, handle_id, []() { throw std::runtime_error("persistent_failure"); });

        pool.drain();

        // Verify: outcome is accessible.
        RC_ASSERT(pool.outcome_registry().has_failure(handle_id));

        // Submit more successful tasks (should not clear the failure).
        for (int i = 0; i < 5; ++i) {
            pool.submit_write(dv_key, []() { /* no-op success */ });
        }
        pool.drain();

        // Verify: the original failure is STILL recorded (R12.10).
        RC_ASSERT(pool.outcome_registry().has_failure(handle_id));

        auto first = pool.outcome_registry().get_first_failure(handle_id);
        RC_ASSERT(first.is_failure());
        RC_ASSERT(first.message.find("persistent_failure") != std::string::npos);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P16e: Stack trace is recorded (non-empty for
// std::exception derivatives).
//
// For any std::exception throw: the outcome's stack_trace field is
// populated (non-empty string).
// ===================================================================

TEST_CASE("P16: Exception bridge - stack trace recorded", "[pbt][p16][exception_bridge][stacktrace]") {
    auto result = rc::check("stack trace is recorded for exceptions", []() {
        auto handle_id = *rc::gen::inRange<std::uint64_t>(1, 10000);

        WorkerPoolConfig config;
        config.thread_count = 1;
        config.backpressure.queue_capacity = 1024;
        config.backpressure.enabled = false;

        WorkerPool pool(config);

        DatasetVariableKey dv_key{1, 1};

        pool.submit_write(dv_key, handle_id, []() { throw std::runtime_error("trace_test"); });

        pool.drain();

        // Verify: outcome exists and has a stack trace.
        const auto &registry = pool.outcome_registry();
        auto first = registry.get_first_failure(handle_id);
        RC_ASSERT(first.is_failure());

        // The stack trace should be non-empty (even if it's just
        // a local log without MPI collective).
        // Note: In non-MPI builds, the trace may be minimal but
        // should still be populated.
        RC_ASSERT(!first.stack_trace.empty());
    });

    REQUIRE(result);
}
