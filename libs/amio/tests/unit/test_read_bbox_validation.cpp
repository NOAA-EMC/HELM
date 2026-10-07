// test_read_bbox_validation.cpp -- Unit tests for the selective-read
// bounding-box validation helper (task 9, design §9).
//
// The validate_bbox helper is file-local (static) to
// amio_core.cpp, so it is exercised here through the real
// amio::detail::read coordinator: a mock read driver reports a known
// variable shape via describe_variable, and read() validates the
// caller-supplied bounding box against that shape before scheduling a
// fetch.  The Staging_Pool is real and the Worker_Pool is null so the
// look-ahead fetches run synchronously on the calling thread, making
// the accept-case deterministic.
//
// Cases (Req 12):
//   - null bbox                                  -> AMIO_OK (full read)
//   - rank != variable rank                      -> INVALID_INPUT (12.2)
//   - stride < 1 in any dimension                -> INVALID_INPUT (12.4)
//   - negative offset                            -> INVALID_INPUT (12.3)
//   - extent < 1                                 -> INVALID_INPUT (12.3)
//   - offset + (extent-1)*stride >= var extent   -> INVALID_INPUT (12.3)
//   - valid in-range bbox                        -> AMIO_OK
//
// Validates: R12.1, R12.2, R12.3, R12.4

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
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
// BboxMockDriver -- reports a fixed variable shape and serves reads
// by filling the staging buffer.  describe_variable returns the
// shape the read coordinator validates the bbox against.
// ---------------------------------------------------------------
class BboxMockDriver : public amio::detail::Backend_Driver {
   public:
    explicit BboxMockDriver(amio::detail::VariableInfo info) : info_(info) {}

    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const amio::detail::StagingBuffer &, const amio::detail::VarMeta &) override {}

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta &, std::int64_t,
              const std::optional<amio::detail::BoundingBox> &) override {
        if (dst.data != nullptr && dst.capacity_bytes > 0) {
            std::memset(dst.data, 0, dst.capacity_bytes);
        }
        dst.used_bytes = dst.capacity_bytes;
    }

    void flush() override {}
    void close() override {}

    amio::detail::VariableInfo describe_variable(const std::string &) override {
        return info_;
    }

   private:
    amio::detail::VariableInfo info_;
};

// ---------------------------------------------------------------
// Harness: build an AMIO_Core + a read-mode DatasetRecord whose mock
// driver describes a rank-2, [4,4] F32 variable with a single
// timestep.  The Worker_Pool is intentionally null so prefetch
// fetches run synchronously on the calling thread.
// ---------------------------------------------------------------
struct ReadFixture {
    std::unique_ptr<amio::detail::AMIO_Core> core;
    amio::detail::DatasetRecord *record = nullptr;  // owned by core->datasets

    ReadFixture() {
        core = std::make_unique<amio::detail::AMIO_Core>();
        core->staging_pool = std::make_unique<amio::detail::StagingPool>(/*buffer_count=*/8, /*buffer_capacity=*/1024, /*timeout_ms=*/5000);
        // worker_pool intentionally left null -> synchronous fetches.

        amio::detail::VariableInfo info{};
        info.found = true;
        info.dtype = AMIO_DTYPE_F32;
        info.shape.rank = 2;
        info.shape.extents[0] = 4;
        info.shape.extents[1] = 4;
        info.total_timesteps = 1;

        auto rec = std::make_unique<amio::detail::DatasetRecord>();
        rec->driver = std::make_unique<BboxMockDriver>(info);
        rec->mode = AMIO_MODE_READ;
        rec->dataset_id = core->next_dataset_id.fetch_add(1);
        rec->core = core.get();
        rec->dataset_config.prefetch.depth = 4;
        rec->dataset_config.prefetch.read_timeout_s = 60;

        record = rec.get();
        core->datasets[rec->dataset_id] = std::move(rec);
    }

    // Teardown in the same order finalize() uses: clear the dataset
    // records (destroying each PrefetchQueue, which releases any
    // completed-but-unread staging buffers back to the pool) BEFORE the
    // Staging_Pool is destroyed.  AMIO_Core's default destructor would
    // otherwise tear members down in reverse declaration order
    // (staging_pool before datasets), releasing buffers into an
    // already-destroyed pool.
    ~ReadFixture() {
        if (core) {
            core->datasets.clear();
        }
    }

    // Release a view handle obtained from a successful read so the
    // staging buffer returns to the pool and outstanding_views is
    // decremented.
    void release(amio_view_handle view) {
        if (view == nullptr) return;
        auto token = amio::detail::HandleTable::from_ptr(view);
        void *payload = nullptr;
        if (amio::detail::process_handle_table().lookup(token, amio::detail::HandleKind::View, &payload) == AMIO_OK) {
            amio::detail::release_view(payload);
        }
    }
};

// Build a rank-2 bbox with the given fields.
amio_bbox_t make_bbox(std::int32_t rank, std::int64_t off0, std::int64_t off1, std::int64_t ext0, std::int64_t ext1, std::int64_t str0,
                      std::int64_t str1) {
    amio_bbox_t b{};
    b.rank = rank;
    b.offsets[0] = off0;
    b.offsets[1] = off1;
    b.extents[0] = ext0;
    b.extents[1] = ext1;
    b.strides[0] = str0;
    b.strides[1] = str1;
    return b;
}

// ---------------------------------------------------------------
// Test: null bbox -> full read -> AMIO_OK (Req 12 full-read case).
// ---------------------------------------------------------------
void test_null_bbox_is_full_read() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_OK, "null bbox should be accepted as a full read");
    EXPECT_TRUE(view != nullptr, "full read should return a view handle");
    fx.release(view);
}

// ---------------------------------------------------------------
// Test: rank mismatch -> AMIO_ERR_INVALID_INPUT (Req 12.2).
// ---------------------------------------------------------------
void test_rank_mismatch_rejected() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    // Variable rank is 2; supply a rank-1 box.
    amio_bbox_t b = make_bbox(/*rank=*/1, 0, 0, 4, 0, 1, 0);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_ERR_INVALID_INPUT, "bbox rank != variable rank should be rejected");
    EXPECT_TRUE(view == nullptr, "no view on rank mismatch");
}

// ---------------------------------------------------------------
// Test: stride < 1 -> AMIO_ERR_INVALID_INPUT (Req 12.4).
// ---------------------------------------------------------------
void test_stride_less_than_one_rejected() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    // strides[1] = 0 is invalid.
    amio_bbox_t b = make_bbox(2, 0, 0, 2, 2, 1, 0);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_ERR_INVALID_INPUT, "stride < 1 should be rejected");
    EXPECT_TRUE(view == nullptr, "no view on stride < 1");
}

// ---------------------------------------------------------------
// Test: negative offset -> AMIO_ERR_INVALID_INPUT (Req 12.3).
// ---------------------------------------------------------------
void test_negative_offset_rejected() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    amio_bbox_t b = make_bbox(2, -1, 0, 2, 2, 1, 1);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_ERR_INVALID_INPUT, "negative offset should be rejected");
    EXPECT_TRUE(view == nullptr, "no view on negative offset");
}

// ---------------------------------------------------------------
// Test: extent < 1 -> AMIO_ERR_INVALID_INPUT (Req 12.3).
// ---------------------------------------------------------------
void test_empty_extent_rejected() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    // extents[0] = 0 is invalid.
    amio_bbox_t b = make_bbox(2, 0, 0, 0, 2, 1, 1);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_ERR_INVALID_INPUT, "extent < 1 should be rejected");
    EXPECT_TRUE(view == nullptr, "no view on empty extent");
}

// ---------------------------------------------------------------
// Test: selection past the variable extent -> AMIO_ERR_INVALID_INPUT
// (Req 12.3).  offset + (extent-1)*stride >= variable_extent[d].
// ---------------------------------------------------------------
void test_out_of_range_selection_rejected() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    // Dim 0: offset 0, extent 3, stride 2 -> last index = 0 + 2*2 = 4,
    // which is >= variable extent 4 -> out of range.
    amio_bbox_t b = make_bbox(2, 0, 0, 3, 2, 2, 1);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_ERR_INVALID_INPUT, "selection past variable extent should be rejected");
    EXPECT_TRUE(view == nullptr, "no view on out-of-range selection");
}

// ---------------------------------------------------------------
// Test: a valid in-range bbox is accepted (Req 12.1).
// ---------------------------------------------------------------
void test_valid_bbox_accepted() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    // Dim 0: offset 0, extent 2, stride 2 -> last index = 2 < 4.
    // Dim 1: offset 1, extent 3, stride 1 -> last index = 3 < 4.
    amio_bbox_t b = make_bbox(2, 0, 1, 2, 3, 2, 1);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_OK, "valid in-range bbox should be accepted");
    EXPECT_TRUE(view != nullptr, "valid bbox read should return a view handle");
    fx.release(view);
}

// ---------------------------------------------------------------
// Test: a bbox whose selection exactly reaches the last valid index
// is accepted (boundary: offset + (extent-1)*stride == extent-1).
// ---------------------------------------------------------------
void test_boundary_selection_accepted() {
    ReadFixture fx;
    amio_view_handle view = nullptr;

    // Dim 0: offset 1, extent 2, stride 2 -> last index = 1 + 1*2 = 3 < 4.
    // Dim 1: offset 0, extent 4, stride 1 -> last index = 3 < 4.
    amio_bbox_t b = make_bbox(2, 1, 0, 2, 4, 2, 1);
    amio_status_t rc = amio::detail::read(fx.record, "var", 0, &b, &view);
    EXPECT_EQ(rc, AMIO_OK, "boundary selection touching the last index should be accepted");
    EXPECT_TRUE(view != nullptr, "boundary bbox read should return a view handle");
    fx.release(view);
}

}  // namespace

int main() {
    test_null_bbox_is_full_read();
    test_rank_mismatch_rejected();
    test_stride_less_than_one_rejected();
    test_negative_offset_rejected();
    test_empty_extent_rejected();
    test_out_of_range_selection_rejected();
    test_valid_bbox_accepted();
    test_boundary_selection_accepted();

    std::fprintf(stdout, "test_read_bbox_validation: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
