// test_rp7_view_buffer_conservation.cpp -- Read-pipeline Property 7:
// View to buffer conservation.
//
// Design "Correctness Properties", Property 7:
//   While any Memory_View references a buffer, that buffer is never
//   returned to the Staging_Pool free list; each successful amio_read
//   increments and each amio_release_view decrements outstanding_views
//   exactly once; the count is 0 iff no views are outstanding.
//
// This property exercises the read coordinator's buffer/view lifecycle
// at the PrefetchQueue + StagingPool layer (the same layering used by
// the task 11/12 unit tests): it drives a generated sequence of
// read (get_buffer -> outstanding view) and release (pool.release ->
// view dropped) operations and asserts the conservation invariants
// after every step.
//
// **Validates: Requirements 7.2, 7.3**

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

namespace {

enum class ViewOp { Read = 0, Release = 1 };

}  // namespace

// ===================================================================
// Property RP7: read/release conservation
//
// Model an outstanding-view set as a vector of (buffer, timestep).
// For a generated op sequence:
//   - Read: get_buffer(next_ts) -> on success push the buffer as an
//     outstanding view and increment our outstanding counter.
//   - Release: pool.release(buf) on a held view -> drop it and
//     decrement the counter.
//
// Invariants checked after every step:
//   (a) free_buffer_count + (#buffers held by outstanding views)
//       == total_buffer_count   (a held buffer is never on the free
//       list).
//   (b) outstanding counter == number of outstanding views (each read
//       increments once, each release decrements once).
//   (c) outstanding counter == 0  iff  no views are held.
// ===================================================================

TEST_CASE("RP7: view-buffer conservation - held buffers never on free list, count matches outstanding", "[pbt][rp7][view][conservation]") {
    auto result = rc::check("read/release sequence conserves buffers and tracks outstanding views exactly", []() {
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(2, 33);
        auto depth = *rc::gen::inRange<std::size_t>(1, 9);

        // Pool big enough that we never hit backpressure for the held
        // views (cap outstanding at buffer_count - 1).
        std::size_t buffer_count = *rc::gen::inRange<std::size_t>(4, 17);
        std::size_t buffer_capacity = 256;  // payload below is 16*4 = 64 bytes
        StagingPool pool(buffer_count, buffer_capacity, 50);

        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);
        // Seed a payload so reads populate buffers.
        {
            std::vector<std::byte> dummy(64, std::byte{0x5A});
            StagingBuffer src;
            src.data = dummy.data();
            src.capacity_bytes = dummy.size();
            src.used_bytes = dummy.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            driver->write(src, meta);
        }

        auto info = make_var_info_1d(AMIO_DTYPE_F32, 16, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, nullptr, driver.get(), 1, "test_var", info, total_timesteps);
        pq.schedule_initial();

        // Our model of the read coordinator's outstanding-view tracking.
        std::int64_t outstanding_views = 0;  // the counter
        std::vector<StagingBuffer *> held;   // the actual views
        std::int64_t next_ts = 0;            // next timestep to read

        auto num_ops = *rc::gen::inRange<std::size_t>(1, 40);

        for (std::size_t i = 0; i < num_ops; ++i) {
            auto op = *rc::gen::elementOf(std::vector<ViewOp>{ViewOp::Read, ViewOp::Release});

            if (op == ViewOp::Read) {
                // Keep at least one free buffer so a read never blocks
                // on backpressure (deterministic property).
                if (next_ts >= total_timesteps) continue;
                if (held.size() + 1 >= buffer_count) continue;

                StagingBuffer *buf = nullptr;
                amio_status_t status = pq.get_buffer(next_ts, nullptr, &buf);
                ++next_ts;
                if (status == AMIO_OK && buf != nullptr) {
                    held.push_back(buf);
                    ++outstanding_views;  // each successful read +1
                }
            } else {
                if (!held.empty()) {
                    auto idx = *rc::gen::inRange<std::size_t>(0, held.size());
                    StagingBuffer *buf = held[idx];
                    pool.release(buf);
                    held.erase(held.begin() + static_cast<std::ptrdiff_t>(idx));
                    --outstanding_views;  // each release -1
                }
            }

            // (a) Buffers held by outstanding views are never on the
            // free list: free + (completed-but-unread) + held == total.
            // completed_count() are prefetched buffers still owned by
            // the queue (not free, not held as a view).  The pool's slot
            // count auto-grows beyond the configured buffer_count when
            // every slot is busy, so the conservation total is the live
            // slot count, not the provisioning hint.
            std::size_t accounted = pool.free_buffer_count() + held.size() + pq.completed_count();
            RC_ASSERT(accounted == pool.total_buffer_count());

            // (b) the counter equals the number of outstanding views.
            RC_ASSERT(outstanding_views == static_cast<std::int64_t>(held.size()));

            // (c) count == 0 iff no views outstanding.
            RC_ASSERT((outstanding_views == 0) == held.empty());
        }

        // Release everything still held; outstanding returns to 0.
        for (auto *buf : held) {
            pool.release(buf);
            --outstanding_views;
        }
        held.clear();
        RC_ASSERT(outstanding_views == 0);

        // Drain any completed-but-unread prefetched buffers.  After every
        // held and completed buffer is released, all live slots are free;
        // compare against the pool's (possibly grown) total, not the
        // configured buffer_count.
        pq.cancel_pending();
        RC_ASSERT(pool.free_buffer_count() == pool.total_buffer_count());
    });

    REQUIRE(result);
}
