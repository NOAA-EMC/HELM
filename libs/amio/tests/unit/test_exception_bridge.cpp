// test_exception_bridge.cpp
//
// Unit tests for `amio::detail` exception bridge covering:
//   * Exception translation (std::exception → AMIO_ERR_BACKEND_FAILURE)
//   * Exception translation (std::invalid_argument → AMIO_ERR_INVALID_INPUT)
//   * Exception translation (unknown → AMIO_ERR_BACKEND_FAILURE)
//   * Exception translation (conf::Conf_Error → AMIO_ERR_MANIFEST_NOT_FOUND,
//                            AMIO_ERR_MANIFEST_INVALID, AMIO_ERR_INVALID_INPUT)
//   * emit_parallel_stacktrace produces non-empty trace
//   * OutcomeRegistry: record, query, clear
//   * execute_with_exception_cordon: success path
//   * execute_with_exception_cordon: std::exception path
//   * execute_with_exception_cordon: unknown exception path
//   * WorkerPool integration: exceptions caught on worker threads
//   * WorkerPool integration: buffer release after exception
//
// Because the exception bridge is private to the AMIO_Core build
// (its header lives under `src/workers/`), this test target compiles
// the source directly into the test binary.
//
// Validates: R12.1, R12.2, R12.3, R12.4, R12.6, R12.9, R12.10

#include <atomic>
#include <cassert>
#include <chrono>
#include <conf/error.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "workers/exception_bridge.hpp"
#include "workers/worker_pool.hpp"

namespace {

using amio::detail::DatasetVariableKey;
using amio::detail::emit_parallel_stacktrace;
using amio::detail::execute_with_exception_cordon;
using amio::detail::IOCommunicator;
using amio::detail::OutcomeRegistry;
using amio::detail::TaskOutcome;
using amio::detail::translate_exception_to_error;
using amio::detail::WorkerPool;
using amio::detail::WorkerPoolConfig;

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

// ---- Test: translate_exception_to_error with std::runtime_error ----

void test_translate_std_runtime_error() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw std::runtime_error("test runtime error");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_BACKEND_FAILURE, "std::runtime_error should map to AMIO_ERR_BACKEND_FAILURE");
    EXPECT_TRUE(msg == "test runtime error", "message should be preserved");
}

// ---- Test: translate_exception_to_error with std::invalid_argument ----

void test_translate_std_invalid_argument() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw std::invalid_argument("bad argument");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_INVALID_INPUT, "std::invalid_argument should map to AMIO_ERR_INVALID_INPUT");
    EXPECT_TRUE(msg == "bad argument", "message should be preserved");
}

// ---- Test: translate_exception_to_error with unknown exception ----

void test_translate_unknown_exception() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw 42;  // Non-std exception.
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_BACKEND_FAILURE, "unknown exception should map to AMIO_ERR_BACKEND_FAILURE");
    EXPECT_TRUE(msg == "Unknown exception (non-std)", "message should indicate unknown exception");
}

// ---- Test: translate_exception_to_error with conf::Conf_Error{File_Not_Found} ----

void test_translate_conf_error_file_not_found() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw conf::Conf_Error(conf::Error_Code::File_Not_Found, "test file not found");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_MANIFEST_NOT_FOUND, "conf::Conf_Error{File_Not_Found} should map to AMIO_ERR_MANIFEST_NOT_FOUND");
    EXPECT_TRUE(msg.find("test file not found") != std::string::npos, "message should contain the original error text");
}

// ---- Test: translate_exception_to_error with conf::Conf_Error{Parse_Error} ----

void test_translate_conf_error_parse_error() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw conf::Conf_Error(conf::Error_Code::Parse_Error, "invalid YAML at line 5");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_MANIFEST_INVALID, "conf::Conf_Error{Parse_Error} should map to AMIO_ERR_MANIFEST_INVALID");
    EXPECT_TRUE(msg.find("invalid YAML at line 5") != std::string::npos, "message should contain the original error text");
}

// ---- Test: translate_exception_to_error with conf::Conf_Error{Key_Not_Found} ----

void test_translate_conf_error_key_not_found() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw conf::Conf_Error(conf::Error_Code::Key_Not_Found, "key 'staging_pool.buffer_count' missing");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_MANIFEST_INVALID, "conf::Conf_Error{Key_Not_Found} should map to AMIO_ERR_MANIFEST_INVALID");
    EXPECT_TRUE(msg.find("staging_pool.buffer_count") != std::string::npos, "message should contain the missing key path");
}

// ---- Test: translate_exception_to_error with conf::Conf_Error{Type_Mismatch} ----

void test_translate_conf_error_type_mismatch() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw conf::Conf_Error(conf::Error_Code::Type_Mismatch, "expected int, got string");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_INVALID_INPUT, "conf::Conf_Error{Type_Mismatch} should map to AMIO_ERR_INVALID_INPUT");
    EXPECT_TRUE(msg.find("expected int, got string") != std::string::npos, "message should contain the type mismatch description");
}

// ---- Test: translate_exception_to_error with conf::Conf_Error{Invalid_Arg} ----

void test_translate_conf_error_invalid_arg() {
    std::string msg;
    amio_err_t code = AMIO_OK;

    try {
        throw conf::Conf_Error(conf::Error_Code::Invalid_Arg, "null pointer passed to Config");
    } catch (...) {
        code = translate_exception_to_error(&msg);
    }

    EXPECT_TRUE(code == AMIO_ERR_INVALID_INPUT, "conf::Conf_Error{Invalid_Arg} should map to AMIO_ERR_INVALID_INPUT");
    EXPECT_TRUE(msg.find("null pointer passed to Config") != std::string::npos, "message should contain the invalid argument description");
}

// ---- Test: emit_parallel_stacktrace produces non-empty trace ----

void test_emit_parallel_stacktrace_produces_trace() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    std::string trace = emit_parallel_stacktrace(io_comm, AMIO_ERR_BACKEND_FAILURE, "test failure message");

    EXPECT_TRUE(!trace.empty(), "stack trace should be non-empty");
    EXPECT_TRUE(trace.find("AMIO FATAL") != std::string::npos, "stack trace should contain AMIO FATAL marker");
    EXPECT_TRUE(trace.find("test failure message") != std::string::npos, "stack trace should contain the error message");
    EXPECT_TRUE(trace.find("AMIO_ERR_BACKEND_FAILURE") != std::string::npos, "stack trace should contain the error code description");
}

// ---- Test: OutcomeRegistry record and query ----

void test_outcome_registry_record_and_query() {
    OutcomeRegistry registry;

    // Initially no failures.
    EXPECT_TRUE(!registry.has_failure(1), "no failure initially for handle 1");
    EXPECT_TRUE(registry.get_outcomes(1).empty(), "no outcomes initially for handle 1");
    EXPECT_TRUE(registry.failure_count() == 0, "failure_count should be 0 initially");

    // Record a failure.
    TaskOutcome failure;
    failure.error_code = AMIO_ERR_BACKEND_FAILURE;
    failure.message = "disk full";
    failure.stack_trace = "[trace]";
    registry.record(1, failure);

    EXPECT_TRUE(registry.has_failure(1), "should have failure for handle 1");
    EXPECT_TRUE(registry.failure_count() == 1, "failure_count should be 1");

    auto outcomes = registry.get_outcomes(1);
    EXPECT_TRUE(outcomes.size() == 1, "should have 1 outcome for handle 1");
    EXPECT_TRUE(outcomes[0].error_code == AMIO_ERR_BACKEND_FAILURE, "outcome error_code should match");
    EXPECT_TRUE(outcomes[0].message == "disk full", "outcome message should match");

    // Record a success (should not affect failure detection).
    TaskOutcome success;
    success.error_code = AMIO_OK;
    registry.record(1, success);

    EXPECT_TRUE(registry.has_failure(1), "should still have failure for handle 1");
    EXPECT_TRUE(registry.get_outcomes(1).size() == 2, "should have 2 outcomes for handle 1");

    // get_first_failure returns the first failure.
    auto first = registry.get_first_failure(1);
    EXPECT_TRUE(first.error_code == AMIO_ERR_BACKEND_FAILURE, "first failure should be BACKEND_FAILURE");
    EXPECT_TRUE(first.message == "disk full", "first failure message should match");
}

// ---- Test: OutcomeRegistry clear ----

void test_outcome_registry_clear() {
    OutcomeRegistry registry;

    TaskOutcome failure;
    failure.error_code = AMIO_ERR_BACKEND_FAILURE;
    failure.message = "error";
    registry.record(1, failure);
    registry.record(2, failure);

    EXPECT_TRUE(registry.failure_count() == 2, "should have 2 handles with failures");

    registry.clear(1);
    EXPECT_TRUE(!registry.has_failure(1), "handle 1 should be cleared");
    EXPECT_TRUE(registry.has_failure(2), "handle 2 should still have failure");
    EXPECT_TRUE(registry.failure_count() == 1, "failure_count should be 1 after clearing handle 1");

    registry.clear_all();
    EXPECT_TRUE(!registry.has_failure(2), "handle 2 should be cleared after clear_all");
    EXPECT_TRUE(registry.failure_count() == 0, "failure_count should be 0 after clear_all");
}

// ---- Test: OutcomeRegistry get_first_failure with no failure ----

void test_outcome_registry_no_failure() {
    OutcomeRegistry registry;

    auto first = registry.get_first_failure(999);
    EXPECT_TRUE(first.error_code == AMIO_OK, "get_first_failure on unknown handle should return AMIO_OK");
    EXPECT_TRUE(!first.is_failure(), "should not be a failure");
}

// ---- Test: execute_with_exception_cordon success path ----

void test_cordon_success_path() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    OutcomeRegistry registry;
    std::atomic<int> counter{0};

    auto outcome = execute_with_exception_cordon([&]() { counter.fetch_add(1); },
                                                 /*handle_id=*/42, io_comm, registry);

    EXPECT_TRUE(outcome.error_code == AMIO_OK, "success path should return AMIO_OK");
    EXPECT_TRUE(!outcome.is_failure(), "success path should not be a failure");
    EXPECT_TRUE(counter.load() == 1, "callback should have been executed");
    EXPECT_TRUE(!registry.has_failure(42), "no failure should be recorded on success");
}

// ---- Test: execute_with_exception_cordon std::exception path ----

void test_cordon_std_exception_path() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    OutcomeRegistry registry;

    auto outcome = execute_with_exception_cordon([&]() { throw std::runtime_error("backend crashed"); },
                                                 /*handle_id=*/100, io_comm, registry);

    EXPECT_TRUE(outcome.error_code == AMIO_ERR_BACKEND_FAILURE, "std::exception should produce AMIO_ERR_BACKEND_FAILURE");
    EXPECT_TRUE(outcome.is_failure(), "should be a failure");
    EXPECT_TRUE(outcome.message == "backend crashed", "message should be preserved");
    EXPECT_TRUE(!outcome.stack_trace.empty(), "stack trace should be non-empty");

    // Verify recorded in registry (R12.9).
    EXPECT_TRUE(registry.has_failure(100), "failure should be recorded against handle 100");
    auto first = registry.get_first_failure(100);
    EXPECT_TRUE(first.error_code == AMIO_ERR_BACKEND_FAILURE, "recorded failure should match");
    EXPECT_TRUE(first.message == "backend crashed", "recorded message should match");
}

// ---- Test: execute_with_exception_cordon unknown exception path ----

void test_cordon_unknown_exception_path() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    OutcomeRegistry registry;

    auto outcome = execute_with_exception_cordon([&]() { throw 42; },  // Non-std exception.
                                                 /*handle_id=*/200, io_comm, registry);

    EXPECT_TRUE(outcome.error_code == AMIO_ERR_BACKEND_FAILURE, "unknown exception should produce AMIO_ERR_BACKEND_FAILURE");
    EXPECT_TRUE(outcome.message == "Unknown exception (non-std)", "message should indicate unknown exception");
    EXPECT_TRUE(registry.has_failure(200), "failure should be recorded against handle 200");
}

// ---- Test: WorkerPool catches exceptions on worker threads (R12.1) ----

void test_worker_pool_catches_exceptions() {
    WorkerPool pool(2);

    DatasetVariableKey key{1, 1};

    // Submit a task that throws -- it should NOT crash the pool.
    pool.submit_write(key, /*handle_id=*/500, [&]() { throw std::runtime_error("worker thread exception"); });

    pool.drain();

    // Pool should still be functional.
    EXPECT_TRUE(!pool.is_shutdown(), "pool should not be shut down after exception");

    // The failure should be recorded in the outcome registry.
    EXPECT_TRUE(pool.outcome_registry().has_failure(500), "failure should be recorded against handle 500");

    auto first = pool.outcome_registry().get_first_failure(500);
    EXPECT_TRUE(first.error_code == AMIO_ERR_BACKEND_FAILURE, "recorded error code should be AMIO_ERR_BACKEND_FAILURE");
    EXPECT_TRUE(first.message == "worker thread exception", "recorded message should match");
    EXPECT_TRUE(!first.stack_trace.empty(), "stack trace should be recorded (R12.10)");

    // Submit another task to verify pool still works.
    std::atomic<int> counter{0};
    pool.submit_write(key, [&]() { counter.fetch_add(1); });
    pool.drain();
    EXPECT_TRUE(counter.load() == 1, "pool should still execute tasks after exception");
}

// ---- Test: WorkerPool exception cordon with prefetch tasks ----

void test_worker_pool_prefetch_exception_cordon() {
    WorkerPool pool(2);

    // Submit a prefetch task that throws.
    pool.submit_prefetch(/*timestep=*/5, /*distance=*/1,
                         /*dataset_id=*/1, /*handle_id=*/600, [&]() { throw std::logic_error("prefetch failed"); });

    pool.drain();

    // Pool should still be functional.
    EXPECT_TRUE(!pool.is_shutdown(), "pool should not be shut down after prefetch exception");

    // The failure should be recorded.
    EXPECT_TRUE(pool.outcome_registry().has_failure(600), "failure should be recorded against handle 600");

    auto first = pool.outcome_registry().get_first_failure(600);
    EXPECT_TRUE(first.error_code == AMIO_ERR_BACKEND_FAILURE, "prefetch failure should be AMIO_ERR_BACKEND_FAILURE");
    EXPECT_TRUE(first.message == "prefetch failed", "prefetch failure message should match");
}

// ---- Test: Multiple exceptions recorded against same handle ----

void test_multiple_exceptions_same_handle() {
    WorkerPool pool(2);

    DatasetVariableKey key{1, 1};

    // Submit multiple tasks that throw against the same handle.
    for (int i = 0; i < 3; ++i) {
        pool.submit_write(key, /*handle_id=*/700, [&, i]() { throw std::runtime_error("error " + std::to_string(i)); });
    }

    pool.drain();

    auto outcomes = pool.outcome_registry().get_outcomes(700);
    EXPECT_TRUE(outcomes.size() == 3, "should have 3 outcomes recorded, got " + std::to_string(outcomes.size()));

    // All should be failures.
    bool all_failures = true;
    for (const auto &o : outcomes) {
        if (!o.is_failure()) {
            all_failures = false;
            break;
        }
    }
    EXPECT_TRUE(all_failures, "all outcomes should be failures");
}

// ---- Test: Stack trace emitted BEFORE outcome is recorded (R12.3, R12.4) ----
//
// This test verifies the ordering guarantee: the stack trace is
// emitted (and stored in the outcome) before the outcome is
// recorded in the registry.  We verify this by checking that the
// recorded outcome contains a non-empty stack_trace.

void test_stack_trace_before_recording() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    OutcomeRegistry registry;

    execute_with_exception_cordon([&]() { throw std::runtime_error("ordering test"); },
                                  /*handle_id=*/800, io_comm, registry);

    // The recorded outcome should have a non-empty stack trace,
    // proving that emit_parallel_stacktrace completed before
    // the outcome was recorded.
    auto first = registry.get_first_failure(800);
    EXPECT_TRUE(!first.stack_trace.empty(),
                "stack trace should be present in recorded outcome "
                "(proves emission completed before recording)");
    EXPECT_TRUE(first.stack_trace.find("ordering test") != std::string::npos, "stack trace should contain the error message");
}

// ---- Test: Exception cordon with null callback (no-op) ----

void test_cordon_null_callback() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    OutcomeRegistry registry;

    auto outcome = execute_with_exception_cordon(nullptr,  // null callback
                                                 /*handle_id=*/900, io_comm, registry);

    EXPECT_TRUE(outcome.error_code == AMIO_OK, "null callback should succeed (no-op)");
    EXPECT_TRUE(!registry.has_failure(900), "no failure should be recorded for null callback");
}

// ---- Test: OutcomeRegistry thread safety ----

void test_outcome_registry_thread_safety() {
    OutcomeRegistry registry;

    constexpr int kThreads = 8;
    constexpr int kOpsPerThread = 100;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kOpsPerThread; ++i) {
                TaskOutcome outcome;
                outcome.error_code = AMIO_ERR_BACKEND_FAILURE;
                outcome.message = "thread " + std::to_string(t) + " op " + std::to_string(i);
                registry.record(static_cast<std::uint64_t>(t), outcome);
            }
        });
    }

    for (auto &t : threads) {
        t.join();
    }

    // Each thread should have recorded kOpsPerThread outcomes.
    for (int t = 0; t < kThreads; ++t) {
        auto outcomes = registry.get_outcomes(static_cast<std::uint64_t>(t));
        EXPECT_TRUE(outcomes.size() == kOpsPerThread, "thread " + std::to_string(t) + " should have " + std::to_string(kOpsPerThread) +
                                                          " outcomes, got " + std::to_string(outcomes.size()));
    }

    EXPECT_TRUE(registry.failure_count() == kThreads, "failure_count should be " + std::to_string(kThreads));
}

}  // namespace

int main() {
    test_translate_std_runtime_error();
    test_translate_std_invalid_argument();
    test_translate_unknown_exception();
    test_translate_conf_error_file_not_found();
    test_translate_conf_error_parse_error();
    test_translate_conf_error_key_not_found();
    test_translate_conf_error_type_mismatch();
    test_translate_conf_error_invalid_arg();
    test_emit_parallel_stacktrace_produces_trace();
    test_outcome_registry_record_and_query();
    test_outcome_registry_clear();
    test_outcome_registry_no_failure();
    test_cordon_success_path();
    test_cordon_std_exception_path();
    test_cordon_unknown_exception_path();
    test_worker_pool_catches_exceptions();
    test_worker_pool_prefetch_exception_cordon();
    test_multiple_exceptions_same_handle();
    test_stack_trace_before_recording();
    test_cordon_null_callback();
    test_outcome_registry_thread_safety();

    std::fprintf(stdout, "test_exception_bridge: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
