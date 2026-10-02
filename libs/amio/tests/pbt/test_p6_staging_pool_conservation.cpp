// test_p6_staging_pool_conservation.cpp -- Property 6: Staging_Pool conservation invariant.
//
// For any sequence of (write, read, release_view, flush, close):
// after flush + all views released, free list count == original
// buffer_count; no buffer simultaneously in free list and
// outstanding task/view.
//
// Min 100 iterations.
//
// **Validates: Requirements R2.9, R3.10, R5.6, R5.9, R5.10**

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"
#include "staging/staging_pool.hpp"

// ===================================================================
// Operation enum for the random sequence of pool operations.
// ===================================================================

namespace {

enum class PoolOp : int { Acquire = 0, Release = 1, AddRef = 2, ReleaseRef = 3 };

}  // anonymous namespace

// ===================================================================
// RapidCheck generator for PoolOp.
// ===================================================================

namespace rc {

template <>
struct Arbitrary<PoolOp> {
    static Gen<PoolOp> arbitrary() {
        return gen::elementOf(std::vector<PoolOp>{PoolOp::Acquire, PoolOp::Release, PoolOp::AddRef, PoolOp::ReleaseRef});
    }
};

}  // namespace rc

// ===================================================================
// Property 6: Staging_Pool conservation invariant
//
// After all acquired buffers are released, the free list count must
// equal the original buffer_count.  At no point should a buffer be
// simultaneously on the free list and held by an outstanding
// acquire/view.
// ===================================================================

TEST_CASE("Property 6: Staging_Pool conservation - all buffers returned after release", "[pbt][p6][staging_pool][conservation]") {
    rc::check("after all acquires are released, free_count == buffer_count", []() {
        // Generate pool parameters.
        // Keep buffer_count small for fast iteration but meaningful.
        const auto buffer_count = *rc::gen::inRange<std::size_t>(1, 33);
        // Capacity between 64 and 4096 bytes.
        const auto buffer_capacity = *rc::gen::inRange<std::size_t>(64, 4097);

        // Use a short timeout since we don't want to block in tests.
        constexpr std::int64_t timeout_ms = 1;

        // Bounded mode: this property asserts conservation of a FIXED slot
        // set (free_count == buffer_count), which is the historical hard-
        // limit contract.  Auto-grow would legitimately change total count.
        amio::detail::StagingPool pool(buffer_count, buffer_capacity, timeout_ms, buffer_count, amio::detail::StagingPool::GrowMode::Bounded);

        // Verify initial state: all buffers are free.
        RC_ASSERT(pool.free_buffer_count() == buffer_count);
        RC_ASSERT(pool.total_buffer_count() == buffer_count);

        // Generate a random number of operations.
        const auto num_ops = *rc::gen::inRange<std::size_t>(1, 65);

        // Track acquired buffers (outstanding).
        std::vector<amio::detail::StagingBuffer *> outstanding;

        for (std::size_t i = 0; i < num_ops; ++i) {
            auto op = *rc::gen::arbitrary<PoolOp>();

            switch (op) {
                case PoolOp::Acquire: {
                    // Try to acquire a buffer with a random required size.
                    auto req_bytes = *rc::gen::inRange<std::size_t>(1, buffer_capacity + 1);
                    auto *buf = pool.acquire(req_bytes);
                    if (buf != nullptr) {
                        buf->used_bytes = req_bytes;
                        outstanding.push_back(buf);
                    }
                    // If nullptr, pool is exhausted (backpressure) - that's fine.
                    break;
                }
                case PoolOp::Release: {
                    // Release one outstanding buffer if any.
                    if (!outstanding.empty()) {
                        auto idx = *rc::gen::inRange<std::size_t>(0, outstanding.size());
                        pool.release(outstanding[idx]);
                        outstanding.erase(outstanding.begin() + static_cast<std::ptrdiff_t>(idx));
                    }
                    break;
                }
                case PoolOp::AddRef: {
                    // Add a reference to an outstanding buffer (simulates
                    // multi-view sharing on read path).
                    if (!outstanding.empty()) {
                        auto idx = *rc::gen::inRange<std::size_t>(0, outstanding.size());
                        pool.add_ref(outstanding[idx]);
                        // Push again to track the extra reference.
                        outstanding.push_back(outstanding[idx]);
                    }
                    break;
                }
                case PoolOp::ReleaseRef: {
                    // Release a reference (could be an added ref or original).
                    if (!outstanding.empty()) {
                        auto idx = *rc::gen::inRange<std::size_t>(0, outstanding.size());
                        pool.release(outstanding[idx]);
                        outstanding.erase(outstanding.begin() + static_cast<std::ptrdiff_t>(idx));
                    }
                    break;
                }
            }

            // Invariant: free_count + outstanding unique buffers <= buffer_count.
            // (outstanding may have duplicates due to add_ref)
            RC_ASSERT(pool.free_buffer_count() <= buffer_count);
        }

        // Release all remaining outstanding references.
        for (auto *buf : outstanding) {
            pool.release(buf);
        }
        outstanding.clear();

        // Conservation invariant: after all releases, free_count == buffer_count.
        RC_ASSERT(pool.free_buffer_count() == buffer_count);
    });
}

TEST_CASE("Property 6: Staging_Pool conservation - no buffer simultaneously free and acquired", "[pbt][p6][staging_pool][conservation][disjoint]") {
    rc::check("acquired buffers are never on the free list simultaneously", []() {
        const auto buffer_count = *rc::gen::inRange<std::size_t>(2, 17);
        const auto buffer_capacity = *rc::gen::inRange<std::size_t>(64, 4097);
        constexpr std::int64_t timeout_ms = 1;

        amio::detail::StagingPool pool(buffer_count, buffer_capacity, timeout_ms, buffer_count, amio::detail::StagingPool::GrowMode::Bounded);

        // Generate a sequence of acquire/release operations.
        const auto num_ops = *rc::gen::inRange<std::size_t>(4, 65);

        // Track acquired buffer pointers.
        std::vector<amio::detail::StagingBuffer *> acquired;

        for (std::size_t i = 0; i < num_ops; ++i) {
            // Bias towards acquire to stress the pool.
            bool do_acquire = *rc::gen::weightedElement<bool>({{3, true}, {2, false}});

            if (do_acquire) {
                auto req = *rc::gen::inRange<std::size_t>(1, buffer_capacity + 1);
                auto *buf = pool.acquire(req);
                if (buf != nullptr) {
                    buf->used_bytes = req;

                    // Verify this buffer is NOT already in our acquired set.
                    // (A buffer returned by acquire must not be one we already hold.)
                    for (auto *held : acquired) {
                        RC_ASSERT(held != buf);
                    }

                    acquired.push_back(buf);
                }
            } else {
                // Release a random acquired buffer.
                if (!acquired.empty()) {
                    auto idx = *rc::gen::inRange<std::size_t>(0, acquired.size());
                    pool.release(acquired[idx]);
                    acquired.erase(acquired.begin() + static_cast<std::ptrdiff_t>(idx));
                }
            }

            // Invariant: free_count + acquired.size() == buffer_count
            // (when no add_ref is used, each buffer is either free or acquired exactly once)
            RC_ASSERT(pool.free_buffer_count() + acquired.size() == buffer_count);
        }

        // Cleanup: release all.
        for (auto *buf : acquired) {
            pool.release(buf);
        }
        acquired.clear();

        RC_ASSERT(pool.free_buffer_count() == buffer_count);
    });
}

TEST_CASE("Property 6: Staging_Pool conservation - ref_count multi-view sharing", "[pbt][p6][staging_pool][conservation][refcount]") {
    rc::check("buffers with multiple refs only return to pool when all refs released", []() {
        const auto buffer_count = *rc::gen::inRange<std::size_t>(2, 17);
        const auto buffer_capacity = *rc::gen::inRange<std::size_t>(128, 4097);
        constexpr std::int64_t timeout_ms = 1;

        amio::detail::StagingPool pool(buffer_count, buffer_capacity, timeout_ms, buffer_count, amio::detail::StagingPool::GrowMode::Bounded);

        // Acquire some buffers.
        const auto num_acquire = *rc::gen::inRange<std::size_t>(1, buffer_count + 1);

        struct BufRef {
            amio::detail::StagingBuffer *buf;
            int refs;  // total refs held (including initial acquire)
        };

        std::vector<BufRef> held;
        for (std::size_t i = 0; i < num_acquire; ++i) {
            auto *buf = pool.acquire(1);
            RC_ASSERT(buf != nullptr);
            buf->used_bytes = 1;
            held.push_back({buf, 1});
        }

        // Add random extra references (simulating read-side multi-view).
        for (auto &h : held) {
            int extra_refs = *rc::gen::inRange(0, 4);
            for (int r = 0; r < extra_refs; ++r) {
                pool.add_ref(h.buf);
                h.refs++;
            }
        }

        // At this point, free_count == buffer_count - num_acquire.
        RC_ASSERT(pool.free_buffer_count() == buffer_count - num_acquire);

        // Release refs one by one; buffer should not return to pool
        // until all refs are released.
        for (auto &h : held) {
            // Release all but the last ref.
            for (int r = 0; r < h.refs - 1; ++r) {
                pool.release(h.buf);
                // Buffer should still NOT be on the free list.
                RC_ASSERT(pool.free_buffer_count() == buffer_count - held.size());
            }
            // Release the final ref -- buffer returns to pool.
            pool.release(h.buf);
        }

        // All buffers should be back.
        RC_ASSERT(pool.free_buffer_count() == buffer_count);
    });
}
