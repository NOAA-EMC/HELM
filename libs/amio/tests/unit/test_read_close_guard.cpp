// test_read_close_guard.cpp -- Unit tests for the outstanding-view
// guard at close and read-path prefetch cancellation (task 11,
// design §8, Property 7).
//
// These tests drive the real amio::detail::read / release_view /
// close_dataset coordinator (and the PrefetchQueue underneath it)
// with a mock Backend_Driver, mirroring the ReadFixture pattern in
// test_read_bbox_validation.cpp and test_read_capacity_guard.cpp.
// The Worker_Pool is null so the look-ahead fetches run synchronously
// on the calling thread, making the outcome deterministic.
//
// Cases:
//   - close while a Memory_View is outstanding -> AMIO_ERR_VIEWS_OUTSTANDING
//     and NO teardown (the driver stays open, the buffer stays held)
//     (Req 8.1).
//   - after releasing the view, close succeeds (Req 8.2).
//   - closing a read-mode dataset with completed-but-unread prefetched
//     buffers releases them back to the Staging_Pool via the queue's
//     cancel_pending() (Req 8.3): all buffers are free after close.
//   - closing a fresh read dataset with no reads succeeds.
//
// Validates: R8.1, R8.2, R8.3

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
// flush()/close() are no-ops; close tracks invocation so the
// outstanding-view guard (which must NOT close the driver) can be
// asserted.
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
    void close() override {
        closed = true;
    }

    amio::detail::VariableInfo describe_variable(const std::string &) override {
        amio::detail::VariableInfo info{};
        info.found = true;
        info.dtype = AMIO_DTYPE_F32;
        info.shape.rank = 1;
        info.shape.extents[0] = 4;
        info.total_timesteps = timesteps_;
        return info;
    }

    bool closed = false;

   private:
    std::int64_t timesteps_;
};

// ---------------------------------------------------------------
// Harness: AMIO_Core + read-mode DatasetRecord with a real Staging_Pool
// and a null Worker_Pool (synchronous fetches).  Keeps a back-pointer
// to the mock driver so tests can assert close() was / was not called.
// ---------------------------------------------------------------
struct ReadFixture {
    std::unique_ptr<amio::detail::AMIO_Core> core;
    amio::detail::DatasetRecord *record = nullptr;  // owned by core->datasets
    GoodMockDriver *driver = nullptr;               // owned by record->driver

    explicit ReadFixture(std::int64_t timesteps) {
        core = std::make_unique<amio::detail::AMIO_Core>();
        core->staging_pool = std::make_unique<amio::detail::StagingPool>(/*buffer_count=*/8, /*buffer_capacity=*/1024, /*timeout_ms=*/5000);
        // worker_pool intentionally left null -> synchronous fetches.

        auto *drv = new GoodMockDriver{timesteps};
        driver = drv;

        auto rec = std::make_unique<amio::detail::DatasetRecord>();
        rec->driver = std::unique_ptr<amio::detail::Backend_Driver>(drv);
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
// Test: close while a view is outstanding -> AMIO_ERR_VIEWS_OUTSTANDING
// and no teardown; after releasing the view, close succeeds (Req 8.1,
// 8.2).
// ---------------------------------------------------------------
void test_close_blocked_by_outstanding_view() {
    ReadFixture fx(/*timesteps=*/1);

    amio_view_handle view = nullptr;
    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_OK, "read should succeed and return a view");
    EXPECT_TRUE(view != nullptr, "read should return a view handle");
    EXPECT_EQ(static_cast<int>(fx.record->outstanding_views.load()), 1, "one view outstanding after read");

    // Close must be rejected before any teardown (Req 8.1).
    amio_status_t close_rc = amio::detail::close_dataset(fx.record);
    EXPECT_EQ(close_rc, AMIO_ERR_VIEWS_OUTSTANDING, "close with outstanding view -> VIEWS_OUTSTANDING");
    EXPECT_TRUE(!fx.driver->closed, "driver must NOT be closed while a view is outstanding");

    // Release the view, then close succeeds (Req 8.2).
    fx.release(view);
    EXPECT_EQ(static_cast<int>(fx.record->outstanding_views.load()), 0, "no views outstanding after release");

    close_rc = amio::detail::close_dataset(fx.record);
    EXPECT_EQ(close_rc, AMIO_OK, "close after releasing the view succeeds");
    EXPECT_TRUE(fx.driver->closed, "driver close() invoked once views are drained");

    // All staging buffers are free again (nothing leaked / retained).
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), static_cast<int>(fx.core->staging_pool->total_buffer_count()),
              "all buffers free after a clean close");
}

// ---------------------------------------------------------------
// Test: closing a read dataset with completed-but-unread prefetched
// buffers releases them back to the pool via cancel_pending (Req 8.3).
//
// With depth=4 and total_timesteps=4 the first read schedules four
// synchronous look-ahead fetches (timesteps 0..3), all of which land in
// the queue's completed map.  Reading timestep 0 consumes one buffer
// (held by the returned view); the other three remain completed-but-
// unread.  After releasing the view and closing, the pool must be fully
// free -- proving cancel_pending() reclaimed the unread buffers.
// ---------------------------------------------------------------
void test_close_releases_completed_unread_buffers() {
    ReadFixture fx(/*timesteps=*/4);

    amio_view_handle view = nullptr;
    amio_status_t rc = amio::detail::read(fx.record, "var", /*timestep=*/0, /*bbox=*/nullptr, &view);
    EXPECT_EQ(rc, AMIO_OK, "read should succeed");
    EXPECT_TRUE(view != nullptr, "read should return a view handle");

    // Four buffers were acquired by the look-ahead (ts 0..3); one is
    // held by the view, three are completed-but-unread.  So four are in
    // use and the rest are free.
    const int total = static_cast<int>(fx.core->staging_pool->total_buffer_count());
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), total - 4,
              "four buffers in use after first read (1 view + 3 prefetched)");

    // Release the view (returns the ts-0 buffer); three prefetched
    // buffers are still retained by the queue.
    fx.release(view);
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), total - 3,
              "three prefetched buffers still held by the queue after release");

    // Close: cancel_pending() must release the three completed-but-unread
    // buffers back to the pool (Req 8.3).
    amio_status_t close_rc = amio::detail::close_dataset(fx.record);
    EXPECT_EQ(close_rc, AMIO_OK, "close succeeds once the view is released");
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), total, "all prefetched buffers released back to the pool after close");
}

// ---------------------------------------------------------------
// Test: closing a fresh read dataset with no reads succeeds (no
// queues, no views) (Req 8.2).
// ---------------------------------------------------------------
void test_close_fresh_read_dataset_succeeds() {
    ReadFixture fx(/*timesteps=*/4);

    amio_status_t close_rc = amio::detail::close_dataset(fx.record);
    EXPECT_EQ(close_rc, AMIO_OK, "close on a never-read dataset succeeds");
    EXPECT_TRUE(fx.driver->closed, "driver close() invoked on a clean close");
    EXPECT_EQ(static_cast<int>(fx.core->staging_pool->free_buffer_count()), static_cast<int>(fx.core->staging_pool->total_buffer_count()),
              "no buffers retained when nothing was read");
}

}  // namespace

int main() {
    test_close_blocked_by_outstanding_view();
    test_close_releases_completed_unread_buffers();
    test_close_fresh_read_dataset_succeeds();

    std::fprintf(stdout, "test_read_close_guard: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
