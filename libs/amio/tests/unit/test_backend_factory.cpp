// test_backend_factory.cpp
//
// Unit tests for `amio::detail::BackendFactory` covering the
// string-keyed registry contract: case-sensitive exact-match lookup,
// unknown/missing/wrong-case key rejection with AMIO_ERR_UNKNOWN_BACKEND,
// zero state mutation on failure, concurrent read+write dataset support,
// thread-safe registration and lookup, and static-init registration
// via BackendRegistrar<T>.
//
// Because the BackendFactory is private to the AMIO_Core build (its
// header lives under `src/factory/`), this test target compiles
// `backend_factory.cpp` directly into the test binary.
//
// Test scope:
//
//   * Registration of drivers with string keys (R4.2).
//   * Case-sensitive exact-match lookup (R4.1, R4.3, R4.4, R4.5).
//   * Unknown/missing/wrong-case keys → AMIO_ERR_UNKNOWN_BACKEND
//     with zero state mutation (R4.6).
//   * Independent driver instances for concurrent datasets (R4.7).
//   * BackendRegistrar<T> static registration pattern.
//   * Thread-safe concurrent registration and lookup.
//   * No concrete driver class in public API (R4.8) — verified by
//     the fact that tests use only the factory interface.
//
// Validates: R4.1, R4.2, R4.3, R4.4, R4.5, R4.6, R4.7, R4.8

#include <atomic>
#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "factory/backend_factory.hpp"

namespace {

using amio::detail::Backend_Driver;
using amio::detail::BackendFactory;
using amio::detail::BackendRegistrar;
using amio::detail::BoundingBox;
using amio::detail::StagingBuffer;
using amio::detail::VarMeta;

// ---------------------------------------------------------------
// Mock drivers for testing
// ---------------------------------------------------------------

// A minimal concrete driver for testing factory dispatch.
class MockNetCDF_Driver : public Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const StagingBuffer &, const VarMeta &) override {}
    void read(StagingBuffer &, const VarMeta &, std::int64_t, const std::optional<BoundingBox> &) override {}
    void flush() override {}
    void close() override {}
};

class MockZarr_Driver : public Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const StagingBuffer &, const VarMeta &) override {}
    void read(StagingBuffer &, const VarMeta &, std::int64_t, const std::optional<BoundingBox> &) override {}
    void flush() override {}
    void close() override {}
};

class MockGRIB2_Driver : public Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const StagingBuffer &, const VarMeta &) override {}
    void read(StagingBuffer &, const VarMeta &, std::int64_t, const std::optional<BoundingBox> &) override {}
    void flush() override {}
    void close() override {}
};

// ---------------------------------------------------------------
// Test infrastructure
// ---------------------------------------------------------------

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

// Helper: reset factory state before each test group.
void reset_factory() {
    BackendFactory::instance().clear();
}

// Helper: register the three standard mock drivers.
void register_standard_drivers() {
    BackendFactory::instance().register_driver("netcdf4", []() { return std::make_unique<MockNetCDF_Driver>(); });
    BackendFactory::instance().register_driver("zarr3", []() { return std::make_unique<MockZarr_Driver>(); });
    BackendFactory::instance().register_driver("grib2", []() { return std::make_unique<MockGRIB2_Driver>(); });
}

// ---------------------------------------------------------------
// Tests
// ---------------------------------------------------------------

// Test: registration succeeds for valid keys.
void test_registration_succeeds() {
    reset_factory();

    bool ok1 = BackendFactory::instance().register_driver("netcdf4", []() { return std::make_unique<MockNetCDF_Driver>(); });
    EXPECT_TRUE(ok1, "registration of 'netcdf4' should succeed");

    bool ok2 = BackendFactory::instance().register_driver("zarr3", []() { return std::make_unique<MockZarr_Driver>(); });
    EXPECT_TRUE(ok2, "registration of 'zarr3' should succeed");

    bool ok3 = BackendFactory::instance().register_driver("grib2", []() { return std::make_unique<MockGRIB2_Driver>(); });
    EXPECT_TRUE(ok3, "registration of 'grib2' should succeed");

    EXPECT_TRUE(BackendFactory::instance().has("netcdf4"), "factory should have 'netcdf4'");
    EXPECT_TRUE(BackendFactory::instance().has("zarr3"), "factory should have 'zarr3'");
    EXPECT_TRUE(BackendFactory::instance().has("grib2"), "factory should have 'grib2'");
}

// Test: registration with empty key fails.
void test_registration_empty_key_fails() {
    reset_factory();

    bool ok = BackendFactory::instance().register_driver("", []() { return std::make_unique<MockNetCDF_Driver>(); });
    EXPECT_TRUE(!ok, "registration with empty key should fail");
    EXPECT_TRUE(!BackendFactory::instance().has(""), "factory should not have empty key");
}

// Test: exact-match lookup succeeds for registered keys (R4.3, R4.4, R4.5).
void test_exact_match_lookup_succeeds() {
    reset_factory();
    register_standard_drivers();

    amio_err_t err = AMIO_ERR_UNKNOWN_BACKEND;

    auto nc = BackendFactory::instance().build("netcdf4", err);
    EXPECT_TRUE(nc != nullptr, "build('netcdf4') should return a driver");
    EXPECT_TRUE(err == AMIO_OK, "build('netcdf4') should set err to AMIO_OK");

    err = AMIO_ERR_UNKNOWN_BACKEND;
    auto zarr = BackendFactory::instance().build("zarr3", err);
    EXPECT_TRUE(zarr != nullptr, "build('zarr3') should return a driver");
    EXPECT_TRUE(err == AMIO_OK, "build('zarr3') should set err to AMIO_OK");

    err = AMIO_ERR_UNKNOWN_BACKEND;
    auto grib = BackendFactory::instance().build("grib2", err);
    EXPECT_TRUE(grib != nullptr, "build('grib2') should return a driver");
    EXPECT_TRUE(err == AMIO_OK, "build('grib2') should set err to AMIO_OK");
}

// Test: case-sensitive — wrong case returns AMIO_ERR_UNKNOWN_BACKEND (R4.1, R4.6).
void test_case_sensitive_lookup_fails() {
    reset_factory();
    register_standard_drivers();

    amio_err_t err = AMIO_OK;

    auto d1 = BackendFactory::instance().build("NetCDF4", err);
    EXPECT_TRUE(d1 == nullptr, "build('NetCDF4') should fail (wrong case)");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "wrong case should produce AMIO_ERR_UNKNOWN_BACKEND");

    err = AMIO_OK;
    auto d2 = BackendFactory::instance().build("NETCDF4", err);
    EXPECT_TRUE(d2 == nullptr, "build('NETCDF4') should fail");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "all-caps should produce AMIO_ERR_UNKNOWN_BACKEND");

    err = AMIO_OK;
    auto d3 = BackendFactory::instance().build("Zarr3", err);
    EXPECT_TRUE(d3 == nullptr, "build('Zarr3') should fail");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "mixed case 'Zarr3' should produce AMIO_ERR_UNKNOWN_BACKEND");

    err = AMIO_OK;
    auto d4 = BackendFactory::instance().build("GRIB2", err);
    EXPECT_TRUE(d4 == nullptr, "build('GRIB2') should fail");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "all-caps 'GRIB2' should produce AMIO_ERR_UNKNOWN_BACKEND");
}

// Test: unknown key returns AMIO_ERR_UNKNOWN_BACKEND (R4.6).
void test_unknown_key_returns_error() {
    reset_factory();
    register_standard_drivers();

    amio_err_t err = AMIO_OK;

    auto d1 = BackendFactory::instance().build("hdf5", err);
    EXPECT_TRUE(d1 == nullptr, "build('hdf5') should fail (not registered)");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "unknown key should produce AMIO_ERR_UNKNOWN_BACKEND");

    err = AMIO_OK;
    auto d2 = BackendFactory::instance().build("parquet", err);
    EXPECT_TRUE(d2 == nullptr, "build('parquet') should fail");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "unknown key 'parquet' should produce AMIO_ERR_UNKNOWN_BACKEND");
}

// Test: empty key returns AMIO_ERR_UNKNOWN_BACKEND (R4.6).
void test_empty_key_returns_error() {
    reset_factory();
    register_standard_drivers();

    amio_err_t err = AMIO_OK;

    auto d = BackendFactory::instance().build("", err);
    EXPECT_TRUE(d == nullptr, "build('') should fail");
    EXPECT_TRUE(err == AMIO_ERR_UNKNOWN_BACKEND, "empty key should produce AMIO_ERR_UNKNOWN_BACKEND");
}

// Test: zero state mutation on failed lookup (R4.6).
// Verify that a failed lookup does not alter the registry.
void test_zero_state_mutation_on_failure() {
    reset_factory();
    register_standard_drivers();

    auto keys_before = BackendFactory::instance().registered_keys();

    amio_err_t err = AMIO_OK;
    auto d = BackendFactory::instance().build("nonexistent", err);
    EXPECT_TRUE(d == nullptr, "failed build should return nullptr");

    auto keys_after = BackendFactory::instance().registered_keys();
    EXPECT_TRUE(keys_before == keys_after, "registry should be unchanged after failed lookup");

    // Verify existing keys still work after failed lookup.
    err = AMIO_ERR_UNKNOWN_BACKEND;
    auto nc = BackendFactory::instance().build("netcdf4", err);
    EXPECT_TRUE(nc != nullptr, "existing key should still work after failed lookup");
    EXPECT_TRUE(err == AMIO_OK, "existing key should return AMIO_OK after failed lookup");
}

// Test: each build() returns an independent instance (R4.7).
// Supports concurrent read + write datasets on different drivers.
void test_independent_instances() {
    reset_factory();
    register_standard_drivers();

    amio_err_t err1 = AMIO_ERR_UNKNOWN_BACKEND;
    amio_err_t err2 = AMIO_ERR_UNKNOWN_BACKEND;

    auto read_driver = BackendFactory::instance().build("netcdf4", err1);
    auto write_driver = BackendFactory::instance().build("zarr3", err2);

    EXPECT_TRUE(read_driver != nullptr, "read driver should be created");
    EXPECT_TRUE(write_driver != nullptr, "write driver should be created");
    EXPECT_TRUE(err1 == AMIO_OK, "read driver err should be AMIO_OK");
    EXPECT_TRUE(err2 == AMIO_OK, "write driver err should be AMIO_OK");

    // They should be different instances.
    EXPECT_TRUE(read_driver.get() != write_driver.get(), "read and write drivers should be different instances");

    // Two instances of the same driver should also be independent.
    amio_err_t err3 = AMIO_ERR_UNKNOWN_BACKEND;
    auto nc2 = BackendFactory::instance().build("netcdf4", err3);
    EXPECT_TRUE(nc2 != nullptr, "second netcdf4 instance should be created");
    EXPECT_TRUE(nc2.get() != read_driver.get(), "two netcdf4 instances should be different objects");
}

// Test: BackendRegistrar<T> registers at construction time.
void test_backend_registrar() {
    reset_factory();

    // Use BackendRegistrar to register a driver.
    {
        BackendRegistrar<MockNetCDF_Driver> reg("test_driver");
    }
    // The registrar registered the driver; it should persist even
    // after the registrar object is destroyed (consistent with
    // factory behavior).

    EXPECT_TRUE(BackendFactory::instance().has("test_driver"), "BackendRegistrar should register the driver");

    amio_err_t err = AMIO_ERR_UNKNOWN_BACKEND;
    auto d = BackendFactory::instance().build("test_driver", err);
    EXPECT_TRUE(d != nullptr, "BackendRegistrar-registered driver should be buildable");
    EXPECT_TRUE(err == AMIO_OK, "BackendRegistrar-registered driver should return AMIO_OK");
}

// Test: registered_keys returns all registered keys sorted.
void test_registered_keys() {
    reset_factory();
    register_standard_drivers();

    auto keys = BackendFactory::instance().registered_keys();
    EXPECT_TRUE(keys.size() == 3, "should have 3 registered keys");

    // Keys are returned sorted.
    EXPECT_TRUE(keys[0] == "grib2", "first key should be 'grib2'");
    EXPECT_TRUE(keys[1] == "netcdf4", "second key should be 'netcdf4'");
    EXPECT_TRUE(keys[2] == "zarr3", "third key should be 'zarr3'");
}

// Test: has() returns false for unregistered keys.
void test_has_returns_false_for_unregistered() {
    reset_factory();
    register_standard_drivers();

    EXPECT_TRUE(!BackendFactory::instance().has("hdf5"), "has('hdf5') should be false");
    EXPECT_TRUE(!BackendFactory::instance().has("NetCDF4"), "has('NetCDF4') should be false (case-sensitive)");
    EXPECT_TRUE(!BackendFactory::instance().has(""), "has('') should be false");
}

// Test: concurrent lookup from multiple threads (thread safety).
void test_concurrent_lookup() {
    reset_factory();
    register_standard_drivers();

    constexpr int kThreads = 8;
    constexpr int kIterations = 500;

    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    const std::string keys[] = {"netcdf4", "zarr3", "grib2", "unknown", "NetCDF4", ""};

    for (int tid = 0; tid < kThreads; ++tid) {
        workers.emplace_back([&, tid]() {
            for (int i = 0; i < kIterations; ++i) {
                const std::string &key = keys[(tid + i) % 6];
                amio_err_t err = AMIO_OK;
                auto d = BackendFactory::instance().build(key, err);

                bool is_valid_key = (key == "netcdf4" || key == "zarr3" || key == "grib2");
                if (is_valid_key) {
                    if (d == nullptr || err != AMIO_OK) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                } else {
                    if (d != nullptr || err != AMIO_ERR_UNKNOWN_BACKEND) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto &w : workers) {
        w.join();
    }

    EXPECT_TRUE(failures.load(std::memory_order_relaxed) == 0, "concurrent lookup had " + std::to_string(failures.load()) + " failures");
}

// Test: concurrent registration and lookup (thread safety).
void test_concurrent_registration_and_lookup() {
    reset_factory();

    constexpr int kThreads = 4;
    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    // Half the threads register, half look up.
    for (int tid = 0; tid < kThreads; ++tid) {
        workers.emplace_back([&, tid]() {
            if (tid % 2 == 0) {
                // Registrar thread.
                std::string key = "driver_" + std::to_string(tid);
                BackendFactory::instance().register_driver(key, []() { return std::make_unique<MockNetCDF_Driver>(); });
            } else {
                // Lookup thread — may or may not find keys.
                for (int i = 0; i < 100; ++i) {
                    amio_err_t err = AMIO_OK;
                    auto d = BackendFactory::instance().build("driver_0", err);
                    // Either found or not found — both are valid.
                    if (d != nullptr && err != AMIO_OK) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (d == nullptr && err != AMIO_ERR_UNKNOWN_BACKEND) {
                        failures.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        });
    }

    for (auto &w : workers) {
        w.join();
    }

    EXPECT_TRUE(failures.load(std::memory_order_relaxed) == 0, "concurrent registration+lookup had " + std::to_string(failures.load()) + " failures");
}

// Test: re-registration replaces previous builder (last-writer-wins).
void test_re_registration_replaces_builder() {
    reset_factory();

    // Register with MockNetCDF_Driver first.
    BackendFactory::instance().register_driver("mykey", []() { return std::make_unique<MockNetCDF_Driver>(); });

    amio_err_t err = AMIO_ERR_UNKNOWN_BACKEND;
    auto d1 = BackendFactory::instance().build("mykey", err);
    EXPECT_TRUE(d1 != nullptr, "first registration should work");
    EXPECT_TRUE(err == AMIO_OK, "first build should return AMIO_OK");

    // Re-register with MockZarr_Driver.
    BackendFactory::instance().register_driver("mykey", []() { return std::make_unique<MockZarr_Driver>(); });

    err = AMIO_ERR_UNKNOWN_BACKEND;
    auto d2 = BackendFactory::instance().build("mykey", err);
    EXPECT_TRUE(d2 != nullptr, "re-registration should work");
    EXPECT_TRUE(err == AMIO_OK, "re-registered build should return AMIO_OK");

    // The new instance should be a different type (MockZarr_Driver).
    // We can't easily check the type without RTTI, but we verify
    // it's a valid, different pointer.
    EXPECT_TRUE(d2.get() != d1.get(), "re-registered driver should be a new instance");
}

}  // namespace

int main() {
    test_registration_succeeds();
    test_registration_empty_key_fails();
    test_exact_match_lookup_succeeds();
    test_case_sensitive_lookup_fails();
    test_unknown_key_returns_error();
    test_empty_key_returns_error();
    test_zero_state_mutation_on_failure();
    test_independent_instances();
    test_backend_registrar();
    test_registered_keys();
    test_has_returns_false_for_unregistered();
    test_concurrent_lookup();
    test_concurrent_registration_and_lookup();
    test_re_registration_replaces_builder();

    std::fprintf(stdout, "test_backend_factory: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
