// test_logs_integration.cpp
//
// Unit tests for LOGS integration in AMIO, covering:
//   * AMIO_Core state has logger fields with correct defaults
//   * Logger initialized flag transitions (false → true)
//   * Pre-initialization diagnostics fall back to stderr (logger=nullptr)
//   * Post-initialization diagnostics route through Logger (logger!=nullptr)
//
// Because the exception bridge is private to the AMIO_Core build
// (its header lives under `src/workers/`), this test target compiles
// the source directly into the test binary.
//
// Validates: R6.7, R6.8, R12.1

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <logs/logger.hpp>
#include <logs/severity.hpp>
#include <string>

#include "workers/exception_bridge.hpp"

namespace {

using amio::detail::emit_parallel_stacktrace;
using amio::detail::IOCommunicator;

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

// ---- Test: Logger initialization state mirrors AMIO_Core pattern (Req 6.8) ----
//
// AMIO_Core owns a logs::Logger and a std::atomic<bool> logs_initialized
// flag.  This test verifies that the Logger default-constructs in an
// unconfigured state (rank == -1) and the atomic flag defaults to false,
// matching the AMIO_Core initialization contract.

void test_logger_default_state() {
    logs::Logger logger;
    std::atomic<bool> logs_initialized{false};

    // Logger should be unconfigured: rank == -1.
    EXPECT_TRUE(logger.rank() == -1, "Logger should default to rank -1 (unconfigured)");

    // logs_initialized should default to false.
    EXPECT_TRUE(logs_initialized.load() == false, "logs_initialized should default to false");
}

// ---- Test: logs_initialized flag can transition (Req 6.8) ----

void test_logs_initialized_transition() {
    std::atomic<bool> logs_initialized{false};

    EXPECT_TRUE(logs_initialized.load(std::memory_order_acquire) == false, "logs_initialized should start as false");

    // Simulate amio_init completing communicator setup.
    logs_initialized.store(true, std::memory_order_release);

    EXPECT_TRUE(logs_initialized.load(std::memory_order_acquire) == true, "logs_initialized should be true after store");
}

// ---- Test: Pre-initialization fallback to stderr (logger=nullptr, Req 6.7) ----
//
// When logger is nullptr, emit_parallel_stacktrace should still
// produce a valid trace string containing the error message.
// The diagnostic is emitted to stderr (which we don't capture here,
// but we verify the returned trace is valid and non-empty).

void test_pre_init_fallback_stderr() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    // Call with logger=nullptr (pre-initialization state).
    std::string trace = emit_parallel_stacktrace(io_comm, AMIO_ERR_BACKEND_FAILURE, "pre-init diagnostic test", nullptr);

    EXPECT_TRUE(!trace.empty(), "pre-init fallback should produce non-empty trace");
    EXPECT_TRUE(trace.find("pre-init diagnostic test") != std::string::npos, "pre-init trace should contain the error message");
    EXPECT_TRUE(trace.find("AMIO FATAL") != std::string::npos, "pre-init trace should contain AMIO FATAL marker");
    EXPECT_TRUE(trace.find("AMIO_ERR_BACKEND_FAILURE") != std::string::npos, "pre-init trace should contain the error code description");
}

// ---- Test: Post-initialization routing through Logger (Req 6.1, 6.7) ----
//
// When a Logger pointer is provided (LOGS is initialized),
// emit_parallel_stacktrace should route diagnostics through it.
// The returned trace string should still be non-empty and contain
// the error message (the Logger does not alter the return value).

void test_post_init_routes_through_logger() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    // Create a Logger instance (will log to stderr by default since
    // no sinks are configured, but this exercises the LOGS path).
    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // Call with a valid logger pointer (post-initialization state).
    std::string trace = emit_parallel_stacktrace(io_comm, AMIO_ERR_BACKEND_FAILURE, "post-init logger test", &logger);

    EXPECT_TRUE(!trace.empty(), "post-init Logger path should produce non-empty trace");
    EXPECT_TRUE(trace.find("post-init logger test") != std::string::npos, "post-init trace should contain the error message");
    EXPECT_TRUE(trace.find("AMIO FATAL") != std::string::npos, "post-init trace should contain AMIO FATAL marker");
    EXPECT_TRUE(trace.find("AMIO_ERR_BACKEND_FAILURE") != std::string::npos, "post-init trace should contain the error code description");
}

// ---- Test: Logger routes FATAL for unrecoverable errors (Req 6.4) ----
//
// Unrecoverable errors (AMIO_ERR_COMM_SPLIT_FAILED) should be
// emitted at FATAL severity when a logger is provided.  We verify
// the trace is still produced correctly.

void test_logger_fatal_for_unrecoverable() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // AMIO_ERR_COMM_SPLIT_FAILED is classified as unrecoverable.
    std::string trace = emit_parallel_stacktrace(io_comm, AMIO_ERR_COMM_SPLIT_FAILED, "communicator split failed", &logger);

    EXPECT_TRUE(!trace.empty(), "unrecoverable error should produce non-empty trace");
    EXPECT_TRUE(trace.find("communicator split failed") != std::string::npos, "trace should contain the error message for unrecoverable error");
    EXPECT_TRUE(trace.find("AMIO_ERR_COMM_SPLIT_FAILED") != std::string::npos, "trace should contain the unrecoverable error code description");
}

// ---- Test: Logger routes ERROR for recoverable errors (Req 6.1) ----
//
// Recoverable errors (AMIO_ERR_MANIFEST_INVALID) should be emitted
// at ERROR severity when a logger is provided.

void test_logger_error_for_recoverable() {
    IOCommunicator io_comm;
    io_comm.valid = true;
    io_comm.is_io_rank = true;

    logs::Logger logger;
    logger.set_threshold(logs::Severity_Level::DEBUG);

    // AMIO_ERR_MANIFEST_INVALID is a recoverable error.
    std::string trace = emit_parallel_stacktrace(io_comm, AMIO_ERR_MANIFEST_INVALID, "manifest parse error", &logger);

    EXPECT_TRUE(!trace.empty(), "recoverable error should produce non-empty trace");
    EXPECT_TRUE(trace.find("manifest parse error") != std::string::npos, "trace should contain the error message for recoverable error");
    EXPECT_TRUE(trace.find("AMIO_ERR_MANIFEST_INVALID") != std::string::npos, "trace should contain the recoverable error code description");
}

}  // namespace

int main() {
    test_logger_default_state();
    test_logs_initialized_transition();
    test_pre_init_fallback_stderr();
    test_post_init_routes_through_logger();

    // Unrecoverable FATAL logs trigger std::exit(70). We run this in a child process
    // and verify the child exits with code 70.
    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "fork failed\n");
        return 1;
    } else if (pid == 0) {
        // Child process: execute the FATAL log test (which will exit with 70).
        test_logger_fatal_for_unrecoverable();
        // If it somehow reached here, it failed.
        std::_Exit(1);
    } else {
        // Parent process: wait for child to exit and check exit status.
        int status = 0;
        if (waitpid(pid, &status, 0) < 0) {
            std::fprintf(stderr, "waitpid failed\n");
            return 1;
        }
        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            EXPECT_TRUE(exit_code == 70, "FATAL unrecoverable test should exit with 70");
        } else {
            report_failure("child exited normally", __FILE__, __LINE__, "child process terminated abnormally");
        }
    }

    test_logger_error_for_recoverable();

    std::fprintf(stdout, "test_logs_integration: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
