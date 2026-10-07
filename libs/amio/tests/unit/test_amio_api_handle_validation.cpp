// test_amio_api_handle_validation.cpp
//
// Unit tests for the C-Boundary handle-validation cordon implemented
// in `src/c_boundary/amio_api.cpp`.  These tests exercise the FFI
// surface as a host application would: link against `libamio.so`,
// include only the public header, never call into private C++.
//
// The bodies behind every entry point are still stubs (task 3.2
// installs the cordon; tasks 4.x / 6.x / 9.x install the bodies),
// so a successful path through the cordon ends in
// AMIO_ERR_BACKEND_FAILURE.  What this test file pins is the *cordon
// itself*: handle / pointer / argument validation, NULL-handle
// rejection, NULL-out-pointer rejection, and the contract that
// caller-supplied output arguments are zeroed before any failure.
//
// Validates: R10.5, R10.6, R10.7, R12.2 (subset reachable through
// the public C surface in task 3.2).

#include <cstdint>
#include <cstdio>
#include <string>

#include "amio/amio.h"

namespace {

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

// -------------------------------------------------------------------
// amio_init: NULL inputs must return AMIO_ERR_INVALID_INPUT and
// must not mutate caller-supplied output.
// -------------------------------------------------------------------
void test_init_rejects_null_arguments() {
    amio_core_handle out = reinterpret_cast<amio_core_handle>(0xDEADBEEF);

    int rc = amio_init(/*manifest_path=*/nullptr, &out);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_INPUT, "amio_init(NULL path) did not return AMIO_ERR_INVALID_INPUT");
    EXPECT_TRUE(out == nullptr, "amio_init failed but did not zero out_core");

    rc = amio_init("ignored.yml", /*out_core=*/nullptr);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_INPUT, "amio_init(NULL out_core) did not return AMIO_ERR_INVALID_INPUT");
}

// -------------------------------------------------------------------
// amio_finalize: NULL handle must surface AMIO_ERR_NULL_HANDLE,
// garbage handle must surface AMIO_ERR_INVALID_HANDLE, and neither
// case may dereference the pointer.
// -------------------------------------------------------------------
void test_finalize_rejects_null_and_garbage_handles() {
    int rc = amio_finalize(/*core=*/nullptr);
    EXPECT_TRUE(rc == AMIO_ERR_NULL_HANDLE, "amio_finalize(NULL) did not return AMIO_ERR_NULL_HANDLE");

    // A clearly-fabricated, non-null but never-issued handle.  The
    // cordon must reject this without dereferencing.
    auto bogus = reinterpret_cast<amio_core_handle>(static_cast<std::uintptr_t>(0x0123456789ABCDEFull));
    rc = amio_finalize(bogus);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "amio_finalize(garbage) did not return AMIO_ERR_INVALID_HANDLE");
}

// -------------------------------------------------------------------
// amio_write: argument-validation precedes handle validation; NULL
// out_io is rejected even when the dataset handle is otherwise OK.
// -------------------------------------------------------------------
void test_write_argument_validation() {
    amio_shape_t shape{};
    shape.rank = 1;
    shape.extents[0] = 4;

    int payload = 0;

    // NULL out_io short-circuits with AMIO_ERR_INVALID_INPUT,
    // regardless of dataset handle validity.
    int rc = amio_write(/*dataset=*/nullptr, "var", &payload, AMIO_DTYPE_F32, &shape,
                        /*out_io=*/nullptr);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_INPUT, "amio_write(NULL out_io) did not return AMIO_ERR_INVALID_INPUT");

    amio_io_handle io = reinterpret_cast<amio_io_handle>(0xCAFEFACE);

    // NULL dataset surfaces AMIO_ERR_NULL_HANDLE *after* out_io
    // is zeroed.
    rc = amio_write(/*dataset=*/nullptr, "var", &payload, AMIO_DTYPE_F32, &shape, &io);
    EXPECT_TRUE(rc == AMIO_ERR_NULL_HANDLE, "amio_write(NULL dataset) did not return AMIO_ERR_NULL_HANDLE");
    EXPECT_TRUE(io == nullptr, "amio_write failed but did not zero out_io");

    // NULL var_name with a fabricated dataset handle still fails
    // pre-lookup with AMIO_ERR_INVALID_INPUT.
    io = reinterpret_cast<amio_io_handle>(0xCAFEFACE);
    auto bogus_dataset = reinterpret_cast<amio_dataset_handle>(static_cast<std::uintptr_t>(0xFEEDFACE12345678ull));
    rc = amio_write(bogus_dataset,
                    /*var_name=*/nullptr, &payload, AMIO_DTYPE_F32, &shape, &io);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_INPUT, "amio_write(NULL var_name) did not return AMIO_ERR_INVALID_INPUT");
    EXPECT_TRUE(io == nullptr, "amio_write failed pre-lookup but did not zero out_io");

    // Garbage dataset handle with otherwise-valid arguments must
    // surface AMIO_ERR_INVALID_HANDLE.
    io = reinterpret_cast<amio_io_handle>(0xCAFEFACE);
    rc = amio_write(bogus_dataset, "var", &payload, AMIO_DTYPE_F32, &shape, &io);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "amio_write(garbage dataset) did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(io == nullptr, "amio_write(garbage dataset) leaked a value into out_io");
}

// -------------------------------------------------------------------
// amio_read: same NULL-handle / NULL-out behavior as write.
// -------------------------------------------------------------------
void test_read_argument_validation() {
    amio_view_handle view = reinterpret_cast<amio_view_handle>(0xC0FFEE);

    int rc = amio_read(/*dataset=*/nullptr, "var",
                       /*timestep=*/0,
                       /*bbox=*/nullptr,
                       /*out_view=*/nullptr);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_INPUT, "amio_read(NULL out_view) did not return AMIO_ERR_INVALID_INPUT");

    view = reinterpret_cast<amio_view_handle>(0xC0FFEE);
    rc = amio_read(/*dataset=*/nullptr, "var",
                   /*timestep=*/0,
                   /*bbox=*/nullptr, &view);
    EXPECT_TRUE(rc == AMIO_ERR_NULL_HANDLE, "amio_read(NULL dataset) did not return AMIO_ERR_NULL_HANDLE");
    EXPECT_TRUE(view == nullptr, "amio_read failed but did not zero out_view");
}

// -------------------------------------------------------------------
// amio_flush / amio_close / amio_wait / amio_release_view all share
// the same kind_dispatch scaffolding; smoke-test that NULL handles
// surface AMIO_ERR_NULL_HANDLE rather than crashing.
// -------------------------------------------------------------------
void test_flush_close_wait_release_null_handle() {
    EXPECT_TRUE(amio_flush(nullptr, /*timeout_ms=*/1) == AMIO_ERR_NULL_HANDLE, "amio_flush(NULL) did not return AMIO_ERR_NULL_HANDLE");
    EXPECT_TRUE(amio_close(nullptr) == AMIO_ERR_NULL_HANDLE, "amio_close(NULL) did not return AMIO_ERR_NULL_HANDLE");
    EXPECT_TRUE(amio_wait(nullptr, /*timeout_ms=*/1) == AMIO_ERR_NULL_HANDLE, "amio_wait(NULL) did not return AMIO_ERR_NULL_HANDLE");
    EXPECT_TRUE(amio_release_view(nullptr) == AMIO_ERR_NULL_HANDLE, "amio_release_view(NULL) did not return AMIO_ERR_NULL_HANDLE");
    amio_shape_t shape{};
    EXPECT_TRUE(amio_view_shape(nullptr, &shape) == AMIO_ERR_NULL_HANDLE, "amio_view_shape(NULL) did not return AMIO_ERR_NULL_HANDLE");
}

// -------------------------------------------------------------------
// amio_flush / close / wait / release_view with garbage non-NULL
// handles must return AMIO_ERR_INVALID_HANDLE -- the cordon must
// not dereference the fabricated pointer.
// -------------------------------------------------------------------
void test_flush_close_wait_release_garbage_handle() {
    auto bogus_dataset = reinterpret_cast<amio_dataset_handle>(static_cast<std::uintptr_t>(0x1234567812345678ull));
    auto bogus_io = reinterpret_cast<amio_io_handle>(static_cast<std::uintptr_t>(0x8765432187654321ull));
    auto bogus_view = reinterpret_cast<amio_view_handle>(static_cast<std::uintptr_t>(0xABCDEF01ABCDEF01ull));

    EXPECT_TRUE(amio_flush(bogus_dataset, 0) == AMIO_ERR_INVALID_HANDLE, "amio_flush(garbage) did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(amio_close(bogus_dataset) == AMIO_ERR_INVALID_HANDLE, "amio_close(garbage) did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(amio_wait(bogus_io, 0) == AMIO_ERR_INVALID_HANDLE, "amio_wait(garbage) did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(amio_release_view(bogus_view) == AMIO_ERR_INVALID_HANDLE, "amio_release_view(garbage) did not return AMIO_ERR_INVALID_HANDLE");
    amio_shape_t shape{};
    EXPECT_TRUE(amio_view_shape(bogus_view, &shape) == AMIO_ERR_INVALID_HANDLE, "amio_view_shape(garbage) did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(amio_view_shape(bogus_view, nullptr) == AMIO_ERR_INVALID_INPUT,
                "amio_view_shape(garbage, NULL) did not return AMIO_ERR_INVALID_INPUT");
}

}  // namespace

int main() {
    test_init_rejects_null_arguments();
    test_finalize_rejects_null_and_garbage_handles();
    test_write_argument_validation();
    test_read_argument_validation();
    test_flush_close_wait_release_null_handle();
    test_flush_close_wait_release_garbage_handle();

    std::fprintf(stdout, "test_amio_api_handle_validation: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
