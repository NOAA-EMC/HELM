// test_read_release_view.cpp -- Unit tests for Memory_View release
// semantics (task 12, design §8, Req 7 / Property 7).
//
// Scope split (see task 12 note):
//   - Single-release SUCCESS and outstanding-view-at-close REJECTION
//     are already covered by test_read_close_guard.cpp (task 11), which
//     drives release through amio::detail::release_view and asserts the
//     Staging_Pool accounting and the close guard.  This file does NOT
//     duplicate that; it references it.
//   - This file pins the part NOT yet covered: DOUBLE-RELEASE rejection
//     through the PUBLIC `amio_release_view` C entry point, so the real
//     C-boundary handle-table generation check runs.  After the first
//     release bumps the View slot's generation, the second call presents
//     a now-stale token -> the kind_dispatch lookup in amio_api.cpp
//     returns AMIO_ERR_INVALID_HANDLE BEFORE the body executes (no
//     use-after-free) (Req 7.5).  A null handle -> AMIO_ERR_NULL_HANDLE
//     at the boundary; the detail body's own null check returns
//     AMIO_ERR_INVALID_HANDLE for a null payload (Req 7.4).
//
// The test reuses the ReadFixture / GoodMockDriver pattern from
// test_read_close_guard.cpp: a real AMIO_Core + read-mode DatasetRecord
// with a real Staging_Pool and a null Worker_Pool (synchronous fetches),
// so the look-ahead is deterministic.  The View handle returned by
// amio::detail::read is minted in the shared process_handle_table(), so
// the installed extern "C" amio_release_view validates and releases the
// very same token -- exercising the real stale-token path end-to-end
// without needing a live netcdf/zarr backend.
//
// Validates: R7.1, R7.2, R7.3, R7.4, R7.5

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

// ---------------------------------------------------------------
// GoodMockDriver -- fills the staging buffer within capacity and
// reports a rank-1 [4] F32 variable spanning `timesteps` records.
// (Mirrors the driver in test_read_close_guard.cpp.)
// ---------------------------------------------------------------
class GoodMockDriver : public amio::detail::Backend_Driver {
   public:
    explicit GoodMockDriver(std::int64_t timesteps) : timesteps_(timesteps) {}

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
        amio::detail::VariableInfo info{};
        info.found = true;
        info.dtype = AMIO_DTYPE_F32;
        info.shape.rank = 1;
        info.shape.extents[0] = 4;
        info.total_timesteps = timesteps_;
        return info;
    }

   private:
    std::int64_t timesteps_;
};

// ---------------------------------------------------------------
// Harness: AMIO_Core + read-mode DatasetRecord with a real
// Staging_Pool and a null Worker_Pool (synchronous fetches).
// ---------------------------------------------------------------
struct ReadFixture {
    std::unique_ptr<amio::detail::AMIO_Core> core;
    amio::detail::DatasetRecord *record = nullptr;  // owned by core->datasets

    explicit ReadFixture(std::int64_t timesteps) {
        core = std::make_unique<amio::detail::AMIO_Core>();
        core->staging_pool = std::make_unique<amio::detail::StagingPool>(/*buffer_count=*/8, /*buffer_capacity=*/1024, /*timeout_ms=*/5000);
        // worker_pool intentionally left null -> synchronous fetches.

        auto rec = std::make_unique<amio::detail::DatasetRecord>();
        rec->driver = std::make_unique<GoodMockDriver>(timesteps);
        rec->mode = AMIO_MODE_READ;
        rec->dataset_id = core->next_dataset_id.fetch_add(1);
        rec->core = core.get();
        rec->dataset_config.prefetch.depth = 1;  // minimise look-ahead noise
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
};

// ---------------------------------------------------------------
// Test: a single release through the PUBLIC amio_release_view returns
// the buffer to the pool, decrements outstanding_views exactly once,
// and invalidates the View token (Req 7.1, 7.2, 7.3).
// ---------------------------------------------------------------
void test_single_release_through_public_api_succeeds() {
    ReadFixture fx(/*timesteps=*/1);

    amio_view_handle view = nullptr;
    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_OK, "read should succeed and return a view");
    EXPECT_TRUE(view != nullptr, "read should return a non-null view handle");
    EXPECT_EQ(static_cast<int>(fx.record->outstanding_views.load()), 1, "one view outstanding after read (Req 7.2)");

    const int total = static_cast<int>(fx.core->staging_pool->total_buffer_count());
    EXPECT_TRUE(static_cast<int>(fx.core->staging_pool->free_buffer_count()) < total, "view retains a staging buffer (Req 7.2)");

    // Public C-boundary release: validates the View token, runs the
    // body, and bumps the slot generation.
    amio_status_t rel_rc = amio_release_view(view);
    EXPECT_EQ(rel_rc, AMIO_OK, "first release through public API succeeds (Req 7.3)");
    EXPECT_EQ(static_cast<int>(fx.record->outstanding_views.load()), 0, "outstanding_views decremented exactly once (Req 7.3)");
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), total, "buffer returned to the pool after release (Req 7.3)");
}

// ---------------------------------------------------------------
// Test: a SECOND amio_release_view for the same handle is rejected
// with AMIO_ERR_INVALID_HANDLE -- the stale token's slot generation no
// longer matches, so the C-boundary lookup fails before the body runs.
// The rejected call must NOT decrement outstanding_views again, proving
// the decrement happened exactly once (Req 7.3, 7.5).
// ---------------------------------------------------------------
void test_double_release_returns_invalid_handle() {
    ReadFixture fx(/*timesteps=*/1);

    amio_view_handle view = nullptr;
    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_OK, "read should succeed and return a view");
    EXPECT_TRUE(view != nullptr, "read should return a non-null view handle");

    amio_status_t first = amio_release_view(view);
    EXPECT_EQ(first, AMIO_OK, "first release succeeds");
    EXPECT_EQ(static_cast<int>(fx.record->outstanding_views.load()), 0, "one decrement after first release");

    // Second release of the SAME handle: stale token -> generation
    // mismatch in the handle table -> AMIO_ERR_INVALID_HANDLE (Req 7.5).
    amio_status_t second = amio_release_view(view);
    EXPECT_EQ(second, AMIO_ERR_INVALID_HANDLE, "double release -> AMIO_ERR_INVALID_HANDLE (Req 7.5)");

    // The rejected second release must not touch the dataset state.
    EXPECT_EQ(static_cast<int>(fx.record->outstanding_views.load()), 0, "outstanding_views unchanged by the rejected double release");
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), static_cast<int>(fx.core->staging_pool->total_buffer_count()),
              "no buffer double-released by the rejected second release");
}

// ---------------------------------------------------------------
// Test: amio_release_view(nullptr) -> AMIO_ERR_NULL_HANDLE at the
// C-boundary, and the detail body's own null check -> INVALID_HANDLE
// for a null payload (Req 7.4 maps null to the invalid-handle family).
// ---------------------------------------------------------------
void test_null_handle_rejected() {
    // Public boundary: a null handle is rejected with NULL_HANDLE
    // before any lookup.
    amio_status_t rc = amio_release_view(nullptr);
    EXPECT_EQ(rc, AMIO_ERR_NULL_HANDLE, "null handle -> AMIO_ERR_NULL_HANDLE at the boundary (Req 7.4)");

    // Defensive body check: a null payload reaching release_view
    // directly -> AMIO_ERR_INVALID_HANDLE (Req 7.4 invalid-handle family).
    amio_status_t body_rc = amio::detail::release_view(nullptr);
    EXPECT_EQ(body_rc, AMIO_ERR_INVALID_HANDLE, "null payload in body -> AMIO_ERR_INVALID_HANDLE (Req 7.4)");
}

}  // namespace

int main() {
    test_single_release_through_public_api_succeeds();
    test_double_release_returns_invalid_handle();
    test_null_handle_rejected();

    std::fprintf(stdout, "test_read_release_view: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
