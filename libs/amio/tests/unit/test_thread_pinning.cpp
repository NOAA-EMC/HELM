// test_thread_pinning.cpp
//
// Unit tests for `amio::detail::apply_thread_pinning` covering the
// CPU/NUMA pinning abstraction layer.
//
// Because thread_pinning is private to the AMIO_Core build (its
// header lives under `src/workers/`), this test target compiles
// `thread_pinning.cpp` directly into the test binary.
//
// Test scope:
//
//   * Default config (no pinning) returns AMIO_OK without changing
//     affinity (R3.2 no-op case).
//   * Empty cpu_cores with numa_domain=-1 is treated as default.
//   * Invalid CPU core IDs (negative, beyond available CPUs) return
//     AMIO_ERR_INVALID_BINDING (R3.3).
//   * Invalid NUMA domain returns AMIO_ERR_INVALID_BINDING (R3.3).
//   * Valid CPU core IDs succeed on Linux (R3.2).
//   * query_available_cpus returns a positive value on supported
//     platforms.

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>

#include "workers/thread_pinning.hpp"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace {

using amio::detail::apply_thread_pinning;
using amio::detail::query_available_cpus;
using amio::detail::ThreadConfig;
using amio::detail::validate_thread_config;

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

// ---- Test: default config is no-op ----

void test_default_config_returns_ok() {
    ThreadConfig config;  // empty cores, numa_domain = -1
    EXPECT_TRUE(config.is_default(), "default config should report is_default");

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_OK, "default config should return AMIO_OK, got " + std::to_string(rc));
}

// ---- Test: empty cores with explicit numa_domain=-1 is default ----

void test_explicit_default_config() {
    ThreadConfig config;
    config.cpu_cores = {};
    config.numa_domain = -1;

    EXPECT_TRUE(config.is_default(), "explicit default should be is_default");

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_OK, "explicit default config should return AMIO_OK");
}

// ---- Test: invalid CPU core (negative) returns INVALID_BINDING ----

void test_negative_cpu_core_returns_error() {
    ThreadConfig config;
    config.cpu_cores = {-1};

    EXPECT_TRUE(!config.is_default(), "non-default config");

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "negative core should return AMIO_ERR_INVALID_BINDING, got " + std::to_string(rc));
}

// ---- Test: invalid CPU core (way beyond available) returns INVALID_BINDING ----

void test_oversized_cpu_core_returns_error() {
    ThreadConfig config;
    // Use a core ID that's almost certainly beyond any real system.
    config.cpu_cores = {99999};

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "oversized core ID should return AMIO_ERR_INVALID_BINDING, got " + std::to_string(rc));
}

// ---- Test: invalid NUMA domain returns INVALID_BINDING ----

void test_invalid_numa_domain_returns_error() {
    ThreadConfig config;
    config.numa_domain = 9999;  // Almost certainly doesn't exist.

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "invalid NUMA domain should return AMIO_ERR_INVALID_BINDING, got " + std::to_string(rc));
}

// ---- Test: valid CPU core succeeds (Linux only) ----

void test_valid_cpu_core_succeeds() {
    int available = query_available_cpus();
    if (available <= 0) {
        std::fprintf(stdout, "  SKIP: cannot query available CPUs\n");
        ++g_result.passed;  // Count as pass (platform limitation).
        return;
    }

    // Pin to core 0 -- should always be available.
    ThreadConfig config;
    config.cpu_cores = {0};

    amio_err_t rc = apply_thread_pinning(config);

#if defined(__linux__)
    EXPECT_TRUE(rc == AMIO_OK, "pinning to core 0 should succeed on Linux, got " + std::to_string(rc));
#else
    // On non-Linux platforms, non-default configs return INVALID_BINDING.
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "non-Linux should return INVALID_BINDING for non-default config");
#endif
}

// ---- Test: pinning from a worker thread ----

void test_pinning_from_worker_thread() {
    int available = query_available_cpus();
    if (available <= 0) {
        std::fprintf(stdout, "  SKIP: cannot query available CPUs\n");
        ++g_result.passed;
        return;
    }

    amio_err_t thread_result = AMIO_ERR_INVALID_BINDING;

    std::thread worker([&]() {
        ThreadConfig config;
        config.cpu_cores = {0};
        thread_result = apply_thread_pinning(config);
    });
    worker.join();

#if defined(__linux__)
    EXPECT_TRUE(thread_result == AMIO_OK, "worker thread pinning to core 0 should succeed on Linux");
#else
    EXPECT_TRUE(thread_result == AMIO_ERR_INVALID_BINDING, "non-Linux worker thread should return INVALID_BINDING");
#endif
}

// ---- Test: query_available_cpus returns positive on supported platforms ----

void test_query_available_cpus() {
    int cpus = query_available_cpus();
#if defined(__linux__) || defined(__APPLE__)
    EXPECT_TRUE(cpus > 0, "query_available_cpus should return > 0 on Linux/macOS, got " + std::to_string(cpus));
#else
    // On unsupported platforms, 0 is acceptable.
    EXPECT_TRUE(cpus >= 0, "query_available_cpus should return >= 0");
#endif
}

// ---- Test: multiple valid cores ----

void test_multiple_valid_cores() {
    int available = query_available_cpus();
    if (available < 2) {
        std::fprintf(stdout, "  SKIP: need at least 2 CPUs for multi-core test\n");
        ++g_result.passed;
        return;
    }

#if defined(__linux__)
    // Reset affinity to all available CPUs before testing multi-core
    // pinning.  Previous tests may have narrowed the affinity mask.
    {
        cpu_set_t all_cpus;
        CPU_ZERO(&all_cpus);
        for (int i = 0; i < available; ++i) {
            CPU_SET(i, &all_cpus);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(all_cpus), &all_cpus);
    }
#endif

    ThreadConfig config;
    config.cpu_cores = {0, 1};

    amio_err_t rc = apply_thread_pinning(config);

#if defined(__linux__)
    EXPECT_TRUE(rc == AMIO_OK, "pinning to cores {0,1} should succeed on Linux, got " + std::to_string(rc));
#else
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "non-Linux should return INVALID_BINDING");
#endif
}

// ---- Test: mix of valid and invalid cores returns error ----

void test_mixed_valid_invalid_cores() {
    ThreadConfig config;
    config.cpu_cores = {0, 99999};  // 0 is valid, 99999 is not.

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "mix of valid/invalid cores should return INVALID_BINDING, got " + std::to_string(rc));
}

// ---- Test: validate_thread_config default is OK ----

void test_validate_default_config() {
    ThreadConfig config;
    amio_err_t rc = validate_thread_config(config);
    EXPECT_TRUE(rc == AMIO_OK, "validate default config should return AMIO_OK, got " + std::to_string(rc));
}

// ---- Test: validate_thread_config with invalid core ----

void test_validate_invalid_core() {
    ThreadConfig config;
    config.cpu_cores = {99999};

    amio_err_t rc = validate_thread_config(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "validate invalid core should return INVALID_BINDING, got " + std::to_string(rc));
}

// ---- Test: validate_thread_config with valid core ----

void test_validate_valid_core() {
    int available = query_available_cpus();
    if (available <= 0) {
        std::fprintf(stdout, "  SKIP: cannot query available CPUs\n");
        ++g_result.passed;
        return;
    }

    ThreadConfig config;
    config.cpu_cores = {0};

    amio_err_t rc = validate_thread_config(config);

#if defined(__linux__)
    EXPECT_TRUE(rc == AMIO_OK, "validate core 0 should succeed on Linux, got " + std::to_string(rc));
#else
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "non-Linux should return INVALID_BINDING for non-default config");
#endif
}

// ---- Test: validate_thread_config with invalid NUMA domain ----

void test_validate_invalid_numa() {
    ThreadConfig config;
    config.numa_domain = 9999;

    amio_err_t rc = validate_thread_config(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "validate invalid NUMA domain should return INVALID_BINDING, got " + std::to_string(rc));
}

}  // namespace

int main() {
    test_default_config_returns_ok();
    test_explicit_default_config();
    test_negative_cpu_core_returns_error();
    test_oversized_cpu_core_returns_error();
    test_invalid_numa_domain_returns_error();
    test_valid_cpu_core_succeeds();
    test_pinning_from_worker_thread();
    test_query_available_cpus();
    test_multiple_valid_cores();
    test_mixed_valid_invalid_cores();
    test_validate_default_config();
    test_validate_invalid_core();
    test_validate_valid_core();
    test_validate_invalid_numa();

    std::fprintf(stdout, "test_thread_pinning: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
