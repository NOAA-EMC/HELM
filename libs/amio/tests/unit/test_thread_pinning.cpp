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

#include "affinity_fixture.hpp"
#include "workers/thread_pinning.hpp"

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

#if defined(__linux__)
void report_affinity_failure(const std::string &context) {
    report_failure("affinity fixture", __FILE__, __LINE__, context);
}
#endif

// Every test gets the actual inherited mask and restores it before the next
// test.  The guard reports query/restore/readback failures as test failures.
void run_test(void (*test)()) {
#if defined(__linux__)
    amio_test::AffinityGuard affinity(report_affinity_failure);
    if (!affinity.valid()) {
        return;
    }
#endif
    test();
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
#if defined(__linux__)
    const auto original = amio_test::current_cpu_ids();
#endif
    ThreadConfig config;  // empty cores, numa_domain = -1
    EXPECT_TRUE(config.is_default(), "default config should report is_default");

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_OK, "default config should return AMIO_OK, got " + std::to_string(rc));
#if defined(__linux__)
    EXPECT_TRUE(!original.empty() && amio_test::current_cpu_ids() == original, "default config must leave affinity unchanged");
#endif
}

// ---- Test: empty cores with explicit numa_domain=-1 is default ----

void test_explicit_default_config() {
#if defined(__linux__)
    const auto original = amio_test::current_cpu_ids();
#endif
    ThreadConfig config;
    config.cpu_cores = {};
    config.numa_domain = -1;

    EXPECT_TRUE(config.is_default(), "explicit default should be is_default");

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_OK, "explicit default config should return AMIO_OK");
#if defined(__linux__)
    EXPECT_TRUE(!original.empty() && amio_test::current_cpu_ids() == original, "explicit default config must leave affinity unchanged");
#endif
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
    ThreadConfig config;
#if defined(__linux__)
    const auto allowed = amio_test::current_cpu_ids();
    EXPECT_TRUE(!allowed.empty(), "must query the actual allowed CPU IDs");
    if (allowed.empty()) {
        return;
    }
    config.cpu_cores = {allowed.back()};
#else
    config.cpu_cores = {0};
#endif

    amio_err_t rc = apply_thread_pinning(config);

#if defined(__linux__)
    EXPECT_TRUE(rc == AMIO_OK, "pinning to an allowed CPU should succeed on Linux, got " + std::to_string(rc));
    EXPECT_TRUE(amio_test::current_cpu_ids() == config.cpu_cores, "kernel affinity must equal the requested singleton CPU mask");
#else
    // On non-Linux platforms, non-default configs return INVALID_BINDING.
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "non-Linux should return INVALID_BINDING for non-default config");
#endif
}

// ---- Test: pinning from a worker thread ----

void test_pinning_from_worker_thread() {
    ThreadConfig config;
#if defined(__linux__)
    const auto original = amio_test::current_cpu_ids();
    EXPECT_TRUE(!original.empty(), "must query the calling thread's allowed CPU IDs");
    if (original.empty()) {
        return;
    }
    config.cpu_cores = {original.back()};
    std::vector<int> worker_affinity;
#else
    config.cpu_cores = {0};
#endif

    amio_err_t thread_result = AMIO_ERR_INVALID_BINDING;

    std::thread worker([&]() {
        thread_result = apply_thread_pinning(config);
#if defined(__linux__)
        worker_affinity = amio_test::current_cpu_ids();
#endif
    });
    worker.join();

#if defined(__linux__)
    EXPECT_TRUE(thread_result == AMIO_OK, "worker thread pinning to an allowed CPU should succeed on Linux");
    EXPECT_TRUE(worker_affinity == config.cpu_cores, "worker kernel affinity must equal the requested CPU mask");
    EXPECT_TRUE(amio_test::current_cpu_ids() == original, "worker pinning must not change calling thread affinity");
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

// ---- Test: allowed CPU subset (two distinct IDs when allocation permits) ----

void test_multiple_valid_cores() {
    ThreadConfig config;
#if defined(__linux__)
    const auto allowed = amio_test::current_cpu_ids();
    EXPECT_TRUE(!allowed.empty(), "must query the actual allowed CPU IDs");
    if (allowed.empty()) {
        return;
    }
    config.cpu_cores = {allowed.front()};
    if (allowed.size() > 1) {
        config.cpu_cores.push_back(allowed.back());
    } else {
        std::fprintf(stdout, "  INFO: one allowed CPU; exercising singleton subset without widening affinity\n");
    }
#else
    config.cpu_cores = {0, 1};
#endif

    amio_err_t rc = apply_thread_pinning(config);

#if defined(__linux__)
    EXPECT_TRUE(rc == AMIO_OK, "pinning to allowed CPU subset should succeed on Linux, got " + std::to_string(rc));
    EXPECT_TRUE(amio_test::current_cpu_ids() == config.cpu_cores, "kernel affinity must equal the requested CPU subset");
#else
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "non-Linux should return INVALID_BINDING");
#endif
}

// ---- Test: mix of valid and invalid cores returns error ----

void test_mixed_valid_invalid_cores() {
    ThreadConfig config;
#if defined(__linux__)
    const auto original = amio_test::current_cpu_ids();
    EXPECT_TRUE(!original.empty(), "must query a genuinely valid CPU ID for the mixed test");
    if (original.empty()) {
        return;
    }
    config.cpu_cores = {original.back(), 99999};
#else
    config.cpu_cores = {0, 99999};
#endif

    amio_err_t rc = apply_thread_pinning(config);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_BINDING, "mix of valid/invalid cores should return INVALID_BINDING, got " + std::to_string(rc));
#if defined(__linux__)
    EXPECT_TRUE(amio_test::current_cpu_ids() == original, "mixed invalid binding must leave affinity unchanged");
#endif
}

// ---- Test: reject a representable CPU outside the current mask ----

void test_cpu_outside_current_affinity() {
#if defined(__linux__)
    const auto allowed = amio_test::current_cpu_ids();
    EXPECT_TRUE(!allowed.empty(), "must query allowed CPU IDs before narrowing");
    if (allowed.empty()) {
        return;
    }
    ThreadConfig pin;
    pin.cpu_cores = {allowed.back()};
    EXPECT_TRUE(apply_thread_pinning(pin) == AMIO_OK, "narrowing to an allowed CPU must succeed");
    EXPECT_TRUE(amio_test::current_cpu_ids() == pin.cpu_cores, "narrowed mask must match the requested singleton");

    ThreadConfig excluded;
    // Prefer a CPU known to be allowed before narrowing.  A singleton
    // allocation still tests an in-range CPU outside its current mask.
    excluded.cpu_cores = {allowed.size() > 1 ? allowed.front() : (allowed.back() == 0 ? 1 : 0)};
    EXPECT_TRUE(validate_thread_config(excluded) == AMIO_ERR_INVALID_BINDING, "validation must reject a CPU outside the current mask");
    EXPECT_TRUE(apply_thread_pinning(excluded) == AMIO_ERR_INVALID_BINDING, "pinning must reject a CPU outside the current mask");
    EXPECT_TRUE(amio_test::current_cpu_ids() == pin.cpu_cores, "rejected binding must not widen the current mask");
#else
    ThreadConfig config;
    config.cpu_cores = {0};
    EXPECT_TRUE(apply_thread_pinning(config) == AMIO_ERR_INVALID_BINDING, "non-Linux must reject non-default pinning");
#endif
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
    ThreadConfig config;
#if defined(__linux__)
    const auto original = amio_test::current_cpu_ids();
    EXPECT_TRUE(!original.empty(), "must query allowed CPU IDs for validation");
    if (original.empty()) {
        return;
    }
    config.cpu_cores = {original.back()};
#else
    config.cpu_cores = {0};
#endif

    amio_err_t rc = validate_thread_config(config);

#if defined(__linux__)
    EXPECT_TRUE(rc == AMIO_OK, "validate an allowed CPU should succeed on Linux, got " + std::to_string(rc));
    EXPECT_TRUE(amio_test::current_cpu_ids() == original, "validation must not change the calling thread's affinity");
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
    run_test(test_default_config_returns_ok);
    run_test(test_explicit_default_config);
    run_test(test_negative_cpu_core_returns_error);
    run_test(test_oversized_cpu_core_returns_error);
    run_test(test_invalid_numa_domain_returns_error);
    run_test(test_valid_cpu_core_succeeds);
    run_test(test_pinning_from_worker_thread);
    run_test(test_query_available_cpus);
    run_test(test_multiple_valid_cores);
    run_test(test_mixed_valid_invalid_cores);
    run_test(test_cpu_outside_current_affinity);
    run_test(test_validate_default_config);
    run_test(test_validate_invalid_core);
    run_test(test_validate_valid_core);
    run_test(test_validate_invalid_numa);

    std::fprintf(stdout, "test_thread_pinning: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
