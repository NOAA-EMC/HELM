// test_zarr_driver.cpp
//
// Unit tests for `amio::detail::Zarr_Driver` covering configuration
// validation, sharding validation, codec validation, missing field
// detection, cloud URI routing, and the non-TensorStore fallback path.
//
// These tests exercise the driver's validation logic which works
// regardless of whether TensorStore is available at compile time.
// The actual TensorStore I/O paths require TensorStore to be linked.
//
// Test scope:
//
//   * Missing required config fields → single error naming each (R8.10)
//   * Chunk dims must divide shard dims (positive integers) (R8.3)
//   * Codec must be one of {blosc, zstandard} (R8.4)
//   * Cloud URI detection (s3://, gs://, https://) (R8.2)
//   * Network/auth error categorization (R8.9)
//   * Registration with factory key "zarr3"
//   * Non-TensorStore build: open_write/open_read throw
//
// Validates: R8.1, R8.2, R8.3, R8.4, R8.5, R8.9, R8.10

// Standard headers.
#include <cassert>
#include <conf/config.hpp>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/zarr/zarr_driver.hpp"
#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"

namespace {

using amio::detail::Backend_Driver;
using amio::detail::BackendFactory;
using amio::detail::BoundingBox;
using amio::detail::StagingBuffer;
using amio::detail::VarMeta;
using amio::detail::Zarr_Driver;

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

#define EXPECT_THROWS(expr, ctx)                                              \
    do {                                                                      \
        bool threw = false;                                                   \
        try {                                                                 \
            expr;                                                             \
        } catch (...) {                                                       \
            threw = true;                                                     \
        }                                                                     \
        if (!threw) {                                                         \
            report_failure(#expr " should throw", __FILE__, __LINE__, (ctx)); \
        } else {                                                              \
            ++g_result.passed;                                                \
        }                                                                     \
    } while (0)

#define EXPECT_THROWS_WITH(expr, substring, ctx)                                         \
    do {                                                                                 \
        bool threw = false;                                                              \
        std::string msg;                                                                 \
        try {                                                                            \
            expr;                                                                        \
        } catch (const std::exception &e) {                                              \
            threw = true;                                                                \
            msg = e.what();                                                              \
        } catch (...) {                                                                  \
            threw = true;                                                                \
            msg = "(unknown exception)";                                                 \
        }                                                                                \
        if (!threw) {                                                                    \
            report_failure(#expr " should throw", __FILE__, __LINE__, (ctx));            \
        } else if (msg.find(substring) == std::string::npos) {                           \
            std::string detail = "expected '" + std::string(substring) + "' in: " + msg; \
            report_failure(#expr " wrong message", __FILE__, __LINE__, detail);          \
        } else {                                                                         \
            ++g_result.passed;                                                           \
        }                                                                                \
    } while (0)

// ---------------------------------------------------------------
// Helper: build a conf::Config from YAML text for testing.
// ---------------------------------------------------------------

// A simple key-value configuration helper that builds a conf::Config
// from YAML strings for Zarr_Driver testing.
conf::Config make_config_from_yaml(const std::string &yaml) {
    return conf::Config::from_string(yaml);
}

// Helper: create a valid Zarr configuration.
conf::Config make_valid_config() {
    return conf::Config::from_string(
        "uri: /tmp/test_zarr_output\n"
        "chunk_shape: [64, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n"
        "dtype: float32\n");
}

// ---------------------------------------------------------------
// Tests: Factory registration
// ---------------------------------------------------------------

void test_zarr_driver_registered_with_factory() {
    // The static BackendRegistrar<Zarr_Driver>("zarr3") should have
    // registered the driver at static init time.
    EXPECT_TRUE(BackendFactory::instance().has("zarr3"), "Zarr_Driver should be registered under key 'zarr3'");

    amio_err_t err = AMIO_ERR_UNKNOWN_BACKEND;
    auto driver = BackendFactory::instance().build("zarr3", err);
    EXPECT_TRUE(driver != nullptr, "build('zarr3') should return a driver instance");
    EXPECT_TRUE(err == AMIO_OK, "build('zarr3') should set err to AMIO_OK");
}

// ---------------------------------------------------------------
// Tests: Missing required config fields (R8.10)
// ---------------------------------------------------------------

void test_missing_all_fields() {
    Zarr_Driver driver;
    auto cfg = conf::Config::from_string("dummy: true\n");  // Empty config — all fields missing.

    EXPECT_THROWS_WITH(driver.open_write(cfg), "missing required configuration fields", "empty config should throw naming missing fields");
}

void test_missing_uri_field() {
    // Provide everything except uri.
    auto cfg = conf::Config::from_string(
        "chunk_shape: [64, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "uri", "missing uri should be named in error");
}

void test_missing_chunk_shape_field() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "chunk_shape", "missing chunk_shape should be named in error");
}

void test_missing_multiple_fields() {
    // Only provide uri — missing chunk_shape, shard_shape, array_shape, codec.
    auto cfg = conf::Config::from_string("uri: /tmp/test\n");

    Zarr_Driver driver;
    try {
        driver.open_write(cfg);
        report_failure("should throw", __FILE__, __LINE__, "missing multiple fields should throw");
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        // Should name all missing fields in a single error.
        EXPECT_TRUE(msg.find("chunk_shape") != std::string::npos, "error should name 'chunk_shape'");
        EXPECT_TRUE(msg.find("shard_shape") != std::string::npos, "error should name 'shard_shape'");
        EXPECT_TRUE(msg.find("array_shape") != std::string::npos, "error should name 'array_shape'");
        EXPECT_TRUE(msg.find("codec") != std::string::npos, "error should name 'codec'");
    }
}

// ---------------------------------------------------------------
// Tests: Sharding validation (R8.3)
// ---------------------------------------------------------------

void test_chunk_dims_must_divide_shard_dims() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "chunk_shape: [100, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "must evenly divide", "chunk not dividing shard should throw");
}

void test_chunk_dims_must_be_positive() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "chunk_shape: [0, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "must be a positive integer", "zero chunk dim should throw");
}

void test_shard_dims_must_be_positive() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "chunk_shape: [64, 64]\n"
        "shard_shape: [-1, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "must be a positive integer", "negative shard dim should throw");
}

void test_chunk_shard_dimension_mismatch() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "chunk_shape: [64, 64, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "same number of dimensions", "mismatched chunk/shard dims should throw");
}

void test_valid_sharding_accepted() {
    // 64 divides 256, 128 divides 512 — should pass validation.
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "chunk_shape: [64, 128]\n"
        "shard_shape: [256, 512]\n"
        "array_shape: [1024, 1024]\n"
        "codec: blosc\n");

    Zarr_Driver driver;
    // In non-TensorStore builds, this will throw about TensorStore
    // not being available (after passing validation).
#ifndef AMIO_HAS_TENSORSTORE
    EXPECT_THROWS_WITH(driver.open_write(cfg), "TensorStore is not available", "valid config should pass validation but fail on TensorStore absence");
#endif
}

// ---------------------------------------------------------------
// Tests: Codec validation (R8.4)
// ---------------------------------------------------------------

void test_invalid_codec_rejected() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test\n"
        "chunk_shape: [64, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: lz4\n");

    Zarr_Driver driver;
    EXPECT_THROWS_WITH(driver.open_write(cfg), "must be one of {blosc, zstandard}", "invalid codec should throw");
}

void test_blosc_codec_accepted() {
    auto cfg = make_valid_config();

    Zarr_Driver driver;
#ifndef AMIO_HAS_TENSORSTORE
    // Should pass codec validation, fail on TensorStore absence.
    EXPECT_THROWS_WITH(driver.open_write(cfg), "TensorStore is not available", "blosc codec should pass validation");
#endif
}

void test_zstandard_codec_accepted() {
    auto cfg = conf::Config::from_string(
        "uri: /tmp/test_zarr_output\n"
        "chunk_shape: [64, 64]\n"
        "shard_shape: [256, 256]\n"
        "array_shape: [1024, 1024]\n"
        "codec: zstandard\n"
        "dtype: float32\n");

    Zarr_Driver driver;
#ifndef AMIO_HAS_TENSORSTORE
    // Should pass codec validation, fail on TensorStore absence.
    EXPECT_THROWS_WITH(driver.open_write(cfg), "TensorStore is not available", "zstandard codec should pass validation");
#endif
}

// ---------------------------------------------------------------
// Tests: Cloud URI detection (R8.2)
// ---------------------------------------------------------------

void test_cloud_uri_detection() {
    EXPECT_TRUE(Zarr_Driver::is_cloud_uri("s3://bucket/path"), "s3:// should be detected as cloud URI");
    EXPECT_TRUE(Zarr_Driver::is_cloud_uri("gs://bucket/path"), "gs:// should be detected as cloud URI");
    EXPECT_TRUE(Zarr_Driver::is_cloud_uri("https://example.com/path"), "https:// should be detected as cloud URI");
    EXPECT_TRUE(!Zarr_Driver::is_cloud_uri("/local/path"), "local path should not be cloud URI");
    EXPECT_TRUE(!Zarr_Driver::is_cloud_uri("file:///local/path"), "file:// should not be cloud URI");
    EXPECT_TRUE(!Zarr_Driver::is_cloud_uri("http://example.com"), "http:// (not https) should not be cloud URI");
}

// ---------------------------------------------------------------
// Tests: Error categorization (R8.9)
// ---------------------------------------------------------------

void test_error_categorization() {
    EXPECT_TRUE(Zarr_Driver::categorize_error("authentication failed") == "authentication/authorization error", "auth error should be categorized");
    EXPECT_TRUE(Zarr_Driver::categorize_error("403 Forbidden") == "authentication/authorization error", "403 should be categorized as auth error");
    EXPECT_TRUE(Zarr_Driver::categorize_error("network timeout") == "network error", "network timeout should be categorized");
    EXPECT_TRUE(Zarr_Driver::categorize_error("connection refused") == "network error", "connection refused should be categorized as network error");
    EXPECT_TRUE(Zarr_Driver::categorize_error("unsupported scheme") == "unsupported URI scheme", "unsupported scheme should be categorized");
    EXPECT_TRUE(Zarr_Driver::categorize_error("some other error") == "I/O error", "unknown error should be categorized as I/O error");
}

// ---------------------------------------------------------------
// Tests: Non-TensorStore fallback behavior
// ---------------------------------------------------------------

#ifndef AMIO_HAS_TENSORSTORE
void test_open_write_throws_without_tensorstore() {
    auto cfg = make_valid_config();
    Zarr_Driver driver;

    EXPECT_THROWS_WITH(driver.open_write(cfg), "TensorStore is not available", "open_write should throw without TensorStore");
}

void test_open_read_throws_without_tensorstore() {
    auto cfg = make_valid_config();
    Zarr_Driver driver;

    EXPECT_THROWS_WITH(driver.open_read(cfg), "TensorStore is not available", "open_read should throw without TensorStore");
}
#endif

// ---------------------------------------------------------------
// Tests: Driver state management
// ---------------------------------------------------------------

void test_flush_noop_when_not_open() {
    Zarr_Driver driver;
    // flush() should be a no-op when not open (no throw).
    driver.flush();
    ++g_result.passed;
}

void test_close_noop_when_not_open() {
    Zarr_Driver driver;
    // close() should be a no-op when not open (no throw).
    driver.close();
    ++g_result.passed;
}

void test_write_throws_when_not_open() {
    Zarr_Driver driver;
    StagingBuffer buf{};
    VarMeta meta{};

    EXPECT_THROWS_WITH(driver.write(buf, meta), "not open for writing", "write on unopened driver should throw");
}

void test_read_throws_when_not_open() {
    Zarr_Driver driver;
    StagingBuffer buf{};
    VarMeta meta{};

    EXPECT_THROWS_WITH(driver.read(buf, meta, 0, std::nullopt), "not open for reading", "read on unopened driver should throw");
}

// ---------------------------------------------------------------
// Tests: dtype conversion helpers
// ---------------------------------------------------------------

void test_dtype_to_string() {
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_F32) == "float32", "F32 -> float32");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_F64) == "float64", "F64 -> float64");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_I8) == "int8", "I8 -> int8");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_I16) == "int16", "I16 -> int16");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_I32) == "int32", "I32 -> int32");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_I64) == "int64", "I64 -> int64");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_U8) == "uint8", "U8 -> uint8");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_U16) == "uint16", "U16 -> uint16");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_U32) == "uint32", "U32 -> uint32");
    EXPECT_TRUE(Zarr_Driver::dtype_to_string(AMIO_DTYPE_U64) == "uint64", "U64 -> uint64");
}

void test_dtype_size() {
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_F32) == 4, "F32 = 4 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_F64) == 8, "F64 = 8 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_I8) == 1, "I8 = 1 byte");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_I16) == 2, "I16 = 2 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_I32) == 4, "I32 = 4 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_I64) == 8, "I64 = 8 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_U8) == 1, "U8 = 1 byte");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_U16) == 2, "U16 = 2 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_U32) == 4, "U32 = 4 bytes");
    EXPECT_TRUE(Zarr_Driver::dtype_size(AMIO_DTYPE_U64) == 8, "U64 = 8 bytes");
}

}  // namespace

int main() {
    // Factory registration.
    test_zarr_driver_registered_with_factory();

    // Missing fields (R8.10).
    test_missing_all_fields();
    test_missing_uri_field();
    test_missing_chunk_shape_field();
    test_missing_multiple_fields();

    // Sharding validation (R8.3).
    test_chunk_dims_must_divide_shard_dims();
    test_chunk_dims_must_be_positive();
    test_shard_dims_must_be_positive();
    test_chunk_shard_dimension_mismatch();
    test_valid_sharding_accepted();

    // Codec validation (R8.4).
    test_invalid_codec_rejected();
    test_blosc_codec_accepted();
    test_zstandard_codec_accepted();

    // Cloud URI detection (R8.2).
    test_cloud_uri_detection();

    // Error categorization (R8.9).
    test_error_categorization();

    // Non-TensorStore fallback.
#ifndef AMIO_HAS_TENSORSTORE
    test_open_write_throws_without_tensorstore();
    test_open_read_throws_without_tensorstore();
#endif

    // Driver state management.
    test_flush_noop_when_not_open();
    test_close_noop_when_not_open();
    test_write_throws_when_not_open();
    test_read_throws_when_not_open();

    // Dtype helpers.
    test_dtype_to_string();
    test_dtype_size();

    std::fprintf(stdout, "test_zarr_driver: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
