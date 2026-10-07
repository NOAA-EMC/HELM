// test_read_capacity_guard.cpp -- Unit tests for the read-path
// buffer-capacity guard and prefetch failure surfacing (task 10,
// design §5 / Error Handling, Property 8).
//
// These tests drive the real amio::detail::read coordinator (and the
// PrefetchQueue underneath it) with a mock Backend_Driver, mirroring
// the ReadFixture pattern in test_read_bbox_validation.cpp.  The
// Worker_Pool is null so the look-ahead fetch runs synchronously on
// the calling thread, making the outcome deterministic.
//
// Cases:
//   - A driver that reports a payload exceeding dst.capacity_bytes
//     (used_bytes > capacity_bytes) -> read fails with
//     AMIO_ERR_BACKEND_FAILURE and yields no view (Property 8 central
//     backstop, Req 4.4).
//   - A driver that THROWS when the payload would exceed capacity
//     (mirroring the real NetCDF/Zarr guards which throw before
//     writing) -> cordon maps to AMIO_ERR_BACKEND_FAILURE, no view
//     (Req 4.4).
//   - A driver whose fetch fails -> the failure is surfaced on the
//     read for that timestep, and is surfaced again on the next read
//     of the same timestep (failed_ map behavior, Req 6.2, 6.3).
//   - A normal in-capacity read still succeeds and yields a view
//     (guards do not regress the happy path).
//
// Validates: R4.4, R6.2, R6.3

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "amio/amio.h"

// Private headers for test setup.
#include "c_boundary/amio_core.hpp"
#include "c_boundary/handle_table.hpp"
#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"
#include "workers/worker_pool.hpp"

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

#define EXPECT_EQ(a, b, ctx)                                                                                             \
    do {                                                                                                                 \
        if ((a) != (b)) {                                                                                                \
            char buf[256];                                                                                               \
            std::snprintf(buf, sizeof(buf), "%s: expected %d, got %d", (ctx), static_cast<int>(b), static_cast<int>(a)); \
            report_failure(#a " == " #b, __FILE__, __LINE__, buf);                                                       \
        } else {                                                                                                         \
            ++g_result.passed;                                                                                           \
        }                                                                                                                \
    } while (0)

// Variable shape every mock describes: rank-1 [16] F32, single
// timestep.  Payload = 16 * 4 = 64 bytes, comfortably inside the
// fixture's 1024-byte staging buffers.
amio::detail::VariableInfo make_info() {
    amio::detail::VariableInfo info{};
    info.found = true;
    info.dtype = AMIO_DTYPE_F32;
    info.shape.rank = 1;
    info.shape.extents[0] = 16;
    info.total_timesteps = 1;
    return info;
}

// ---------------------------------------------------------------
// OverCapacityMockDriver -- reports a payload LARGER than the buffer
// capacity by setting used_bytes past capacity_bytes WITHOUT actually
// writing out of bounds.  Exercises the central post-read backstop in
// PrefetchQueue::sync_fetch (Req 4.4 / Property 8): a driver that
// would over-write is detected and the fetch is failed.
// ---------------------------------------------------------------
class OverCapacityMockDriver : public amio::detail::Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const amio::detail::StagingBuffer &, const amio::detail::VarMeta &) override {}

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta &, std::int64_t,
              const std::optional<amio::detail::BoundingBox> &) override {
        // Report MORE bytes than the buffer can hold.  We deliberately
        // do NOT write past dst.capacity_bytes (that would be UB); the
        // central guard keys off the reported used_bytes.
        dst.used_bytes = dst.capacity_bytes + 64;
    }

    void flush() override {}
    void close() override {}

    amio::detail::VariableInfo describe_variable(const std::string &) override {
        return make_info();
    }
};

// ---------------------------------------------------------------
// ThrowingCapacityMockDriver -- throws when asked to read, mirroring
// the real NetCDF/Zarr drivers that throw BEFORE writing when the
// computed payload exceeds dst.capacity_bytes.  The worker-pool /
// sync_fetch cordon maps the throw to AMIO_ERR_BACKEND_FAILURE.
// ---------------------------------------------------------------
class ThrowingCapacityMockDriver : public amio::detail::Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const amio::detail::StagingBuffer &, const amio::detail::VarMeta &) override {}

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta &, std::int64_t,
              const std::optional<amio::detail::BoundingBox> &) override {
        // Simulate the driver-side capacity guard: required payload
        // exceeds capacity, so throw before touching the buffer.
        throw std::runtime_error("mock driver: payload exceeds buffer capacity");
    }

    void flush() override {}
    void close() override {}

    amio::detail::VariableInfo describe_variable(const std::string &) override {
        return make_info();
    }
};

// ---------------------------------------------------------------
// GoodMockDriver -- fills the buffer within capacity (happy path).
// ---------------------------------------------------------------
class GoodMockDriver : public amio::detail::Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const amio::detail::StagingBuffer &, const amio::detail::VarMeta &) override {}

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta &meta, std::int64_t,
              const std::optional<amio::detail::BoundingBox> &) override {
        std::size_t bytes = amio::detail::element_size(meta.dtype);
        for (int d = 0; d < meta.shape.rank; ++d) {
            bytes *= static_cast<std::size_t>(meta.shape.extents[d]);
        }
        if (bytes > dst.capacity_bytes) {
            throw std::runtime_error("GoodMockDriver: unexpected over-capacity");
        }
        if (dst.data != nullptr) {
            std::memset(dst.data, 0, bytes);
        }
        dst.used_bytes = bytes;
    }

    void flush() override {}
    void close() override {}

    amio::detail::VariableInfo describe_variable(const std::string &) override {
        return make_info();
    }
};

// ---------------------------------------------------------------
// Harness: AMIO_Core + read-mode DatasetRecord with a real Staging_Pool
// and a null Worker_Pool (synchronous fetches).  Parameterized on the
// concrete mock driver type.
// ---------------------------------------------------------------
struct ReadFixture {
    std::unique_ptr<amio::detail::AMIO_Core> core;
    amio::detail::DatasetRecord *record = nullptr;  // owned by core->datasets

    template <typename DriverT>
    explicit ReadFixture(DriverT *driver) {
        core = std::make_unique<amio::detail::AMIO_Core>();
        core->staging_pool = std::make_unique<amio::detail::StagingPool>(/*buffer_count=*/8, /*buffer_capacity=*/1024, /*timeout_ms=*/5000);
        // worker_pool intentionally left null -> synchronous fetches.

        auto rec = std::make_unique<amio::detail::DatasetRecord>();
        rec->driver = std::unique_ptr<amio::detail::Backend_Driver>(driver);
        rec->mode = AMIO_MODE_READ;
        rec->dataset_id = core->next_dataset_id.fetch_add(1);
        rec->core = core.get();
        rec->dataset_config.prefetch.depth = 4;
        rec->dataset_config.prefetch.read_timeout_s = 60;

        record = rec.get();
        core->datasets[rec->dataset_id] = std::move(rec);
    }

    // Teardown mirrors finalize(): destroy dataset records (and their
    // PrefetchQueues, which release completed buffers) BEFORE the
    // Staging_Pool is destroyed.
    ~ReadFixture() {
        if (core) {
            core->datasets.clear();
        }
    }

    void release(amio_view_handle view) {
        if (view == nullptr) return;
        auto token = amio::detail::HandleTable::from_ptr(view);
        void *payload = nullptr;
        if (amio::detail::process_handle_table().lookup(token, amio::detail::HandleKind::View, &payload) == AMIO_OK) {
            amio::detail::release_view(payload);
        }
    }
};

// ---------------------------------------------------------------
// Test: a driver reporting used_bytes > capacity_bytes fails the read
// with AMIO_ERR_BACKEND_FAILURE and yields no view (Req 4.4 central
// backstop, Property 8).
// ---------------------------------------------------------------
void test_over_capacity_report_fails_read() {
    ReadFixture fx(new OverCapacityMockDriver{});
    amio_view_handle view = nullptr;

    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_ERR_BACKEND_FAILURE, "over-capacity payload should fail with BACKEND_FAILURE");
    EXPECT_TRUE(view == nullptr, "no view on over-capacity payload");

    // The over-capacity buffer must have been returned to the pool, so
    // all buffers are free again (nothing leaked / retained).
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), static_cast<int>(fx.core->staging_pool->total_buffer_count()),
              "over-capacity buffer should be released back to the pool");
}

// ---------------------------------------------------------------
// Test: a driver that throws on an over-capacity payload (the real
// driver guard pattern) maps to AMIO_ERR_BACKEND_FAILURE, no view.
// ---------------------------------------------------------------
void test_throwing_capacity_guard_fails_read() {
    ReadFixture fx(new ThrowingCapacityMockDriver{});
    amio_view_handle view = nullptr;

    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_ERR_BACKEND_FAILURE, "driver throw should map to BACKEND_FAILURE");
    EXPECT_TRUE(view == nullptr, "no view when the driver throws");

    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), static_cast<int>(fx.core->staging_pool->total_buffer_count()),
              "failed-fetch buffer should be released back to the pool");
}

// ---------------------------------------------------------------
// Test: a recorded fetch failure is surfaced on the read for that
// timestep, and is surfaced again on the NEXT read of the same
// timestep (Req 6.2, 6.3).
// ---------------------------------------------------------------
void test_failure_surfaced_on_repeated_read() {
    ReadFixture fx(new ThrowingCapacityMockDriver{});

    amio_view_handle view1 = nullptr;
    amio_status_t rc1 = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view1);
    EXPECT_EQ(rc1, AMIO_ERR_BACKEND_FAILURE, "first read of failing timestep surfaces BACKEND_FAILURE");
    EXPECT_TRUE(view1 == nullptr, "no view on first failing read");

    amio_view_handle view2 = nullptr;
    amio_status_t rc2 = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view2);
    EXPECT_EQ(rc2, AMIO_ERR_BACKEND_FAILURE, "next read of the same timestep surfaces BACKEND_FAILURE again");
    EXPECT_TRUE(view2 == nullptr, "no view on repeated failing read");
}

// ---------------------------------------------------------------
// Test: an in-capacity read still succeeds and returns a view (the
// guards do not regress the happy path).
// ---------------------------------------------------------------
void test_in_capacity_read_succeeds() {
    ReadFixture fx(new GoodMockDriver{});
    amio_view_handle view = nullptr;

    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_OK, "in-capacity read should succeed");
    EXPECT_TRUE(view != nullptr, "in-capacity read should return a view handle");
    fx.release(view);
}

}  // namespace

int main() {
    test_over_capacity_report_fails_read();
    test_throwing_capacity_guard_fails_read();
    test_failure_surfaced_on_repeated_read();
    test_in_capacity_read_succeeds();

    std::fprintf(stdout, "test_read_capacity_guard: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
