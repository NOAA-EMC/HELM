// staging_pool.hpp -- AMIO Staging_Pool buffer manager.
//
// This header is PRIVATE to the AMIO_Core build (`src/staging/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// The Staging_Pool manages a set of byte buffers sized from the
// manifest's `(buffer_count, buffer_capacity)` with a hard ceiling
// `max_buffer_count`.  Buffer STORAGE is committed lazily (a slot
// costs nothing until first acquired) and grows to fit any request,
// and the SLOT COUNT auto-grows on demand: when every slot is busy
// and the pool is below `max_buffer_count`, `acquire` appends a new
// slot instead of applying backpressure.  The configured
// `buffer_count` is therefore a provisioning hint, not a limit a
// host model can crash on: the steady-state footprint equals the
// concurrency the system actually runs, and slow-but-healthy IO can
// never starve a reader into AMIO_ERR_STAGING_BACKPRESSURE.
// It provides:
//
//   * Best-fit-or-larger acquisition over a free-list keyed on
//     capacity, with lazy commit and grow-to-required on the
//     selected buffer.
//
//   * Auto-grow on exhaustion (up to `max_buffer_count`, default
//     kMaxBufferCount).  A one-time WARN is emitted the first time
//     the pool grows past the configured count.  Only at the hard
//     ceiling does an exhausted pool fall back to the condition-
//     variable wait (up to the manifest staging timeout, 1
//     ms..60000 ms, default 5000 ms) and then signal
//     AMIO_ERR_STAGING_BACKPRESSURE (R2.6) -- at that point the
//     failure is a genuine stuck-IO / leaked-view bug, not a
//     sizing mistake.
//
//   * Buffer release on worker task completion (R3.10) and on
//     `release_view` for read paths (R5.6, R5.9).
//
//   * Memory ownership is exclusively the pool's; Memory_View
//     instances handed out are non-owning views.
//
// Thread safety
// -------------
// All public methods are safe to call concurrently from any thread.
// The implementation uses std::mutex + std::condition_variable.
//
// Validates: R1.3, R1.5, R2.2, R2.6, R2.9, R3.10, R5.6, R5.9

#ifndef AMIO_SRC_STAGING_STAGING_POOL_HPP
#define AMIO_SRC_STAGING_STAGING_POOL_HPP

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

// StagingBuffer -- header for a single pool-owned buffer.
//
// The `data` pointer is owned exclusively by the Staging_Pool.  It
// starts null and is allocated lazily on first acquire (and re-
// allocated, larger, if an acquire requests more than the current
// capacity).  Memory_View instances returned to the host are
// non-owning views over `data`.
//
// Fields:
//   data           - raw byte storage, owned by the pool
//   capacity_bytes - total allocated size of `data`
//   used_bytes     - payload size for the current task (0 when free)
//   ref_count      - for read-side multi-view sharing; when it
//                    drops to 0 the buffer is returned to the pool
//   seq            - generation counter, incremented on every
//                    acquire; used to detect stale references
struct StagingBuffer {
    std::byte *data = nullptr;
    std::size_t capacity_bytes = 0;
    std::size_t used_bytes = 0;
    int ref_count = 0;
    std::uint64_t seq = 0;
};

// StagingPool -- buffer pool with best-fit acquisition, lazy storage,
// grow-to-required, auto-grow slot count (up to a hard ceiling), and
// condition-variable backpressure only at the ceiling.
//
// Construction sets up the buffer *slots* (count + nominal capacity) but
// does NOT commit the per-buffer byte storage: backing memory is allocated
// lazily the first time a buffer is acquired, and only for buffers that are
// actually used.  This keeps a pool provisioned with a large `buffer_count`
// (needed to avoid read-path backpressure when several variables' prefetch
// queues are live at once) from eagerly reserving `count * capacity` bytes
// of resident memory that a small-variable read never touches.
//
// Grow-to-required: an acquire whose `required_bytes` exceeds the nominal
// capacity expands the smallest free buffer's storage to the requested size
// (and records the larger capacity on release), so callers may safely
// provision a small nominal capacity and still serve an occasional oversized
// slab without AMIO_ERR_STAGING_BACKPRESSURE.  Destruction frees all
// committed buffers.
//
// Auto-grow count: in the default (Grow) mode, when every existing slot is
// in use and the pool has fewer than `max_buffer_count` slots, `acquire`
// appends a fresh slot (committed lazily at the requested size) rather than
// waiting.  `buffer_count` thus acts as a hint and the hard ceiling bounds
// total resident staging memory at worst case `max_buffer_count *
// max(committed capacities)`.  In Bounded mode the pool keeps the original
// contract: an exhausted free list blocks on the condition variable and
// times out to AMIO_ERR_STAGING_BACKPRESSURE.
class StagingPool {
   public:
    // Configuration limits (from design.md / requirements).
    static constexpr std::size_t kMinBufferCount = 1;
    static constexpr std::size_t kMaxBufferCount = 4096;
    static constexpr std::size_t kMinBufferCapacity = 1;
    static constexpr std::size_t kMaxBufferCapacity = 1'073'741'824;  // 1 GiB

    static constexpr std::int64_t kMinTimeoutMs = 1;
    static constexpr std::int64_t kMaxTimeoutMs = 60'000;
    static constexpr std::int64_t kDefaultTimeoutMs = 5'000;

    // Slot-count policy.  Grow (default) auto-appends slots on exhaustion
    // up to max_buffer_count; Bounded keeps the historical hard limit.
    enum class GrowMode : bool { Bounded = false, Grow = true };

    // Construct a pool with `buffer_count` buffers, each of
    // `buffer_capacity` bytes, and a staging timeout of
    // `timeout_ms` milliseconds.  Auto-grow is ENABLED up to
    // `kMaxBufferCount` slots.
    //
    // Preconditions (enforced by assertions in debug builds;
    // callers are expected to validate via Config_Loader before
    // constructing):
    //   buffer_count    in [kMinBufferCount, kMaxBufferCount]
    //   buffer_capacity in [kMinBufferCapacity, kMaxBufferCapacity]
    //   timeout_ms      in [kMinTimeoutMs, kMaxTimeoutMs]
    StagingPool(std::size_t buffer_count, std::size_t buffer_capacity, std::int64_t timeout_ms = kDefaultTimeoutMs);

    // Construct a pool with an explicit slot-count policy.
    //   mode == Grow:    `max_buffer_count` is the hard ceiling for
    //                    auto-grown slots (must be >= buffer_count and
    //                    <= kMaxBufferCount).
    //   mode == Bounded: `max_buffer_count` is ignored; the pool never
    //                    exceeds `buffer_count` slots and an exhausted
    //                    pool times out to backpressure (R2.6).
    StagingPool(std::size_t buffer_count, std::size_t buffer_capacity, std::int64_t timeout_ms, std::size_t max_buffer_count,
                GrowMode mode = GrowMode::Grow);

    // Non-copyable, non-movable (owns raw memory).
    StagingPool(const StagingPool &) = delete;
    StagingPool &operator=(const StagingPool &) = delete;
    StagingPool(StagingPool &&) = delete;
    StagingPool &operator=(StagingPool &&) = delete;

    ~StagingPool();

    // acquire -- obtain a buffer with capacity >= `required_bytes`.
    //
    // Uses best-fit-or-larger selection over the free list.  If no
    // suitable buffer is available, blocks the calling thread on a
    // condition variable for up to the configured staging timeout.
    //
    // Returns:
    //   Pointer to a StagingBuffer on success (ref_count set to 1,
    //   used_bytes set to `required_bytes`, seq incremented).
    //   nullptr on timeout (caller should return
    //   AMIO_ERR_STAGING_BACKPRESSURE).
    StagingBuffer *acquire(std::size_t required_bytes);

    // release -- return a buffer to the free list.
    //
    // Decrements ref_count.  When ref_count reaches 0, the buffer
    // is returned to the free list and waiting threads are notified.
    //
    // Precondition: `buf` must be a buffer owned by this pool with
    // ref_count > 0.
    void release(StagingBuffer *buf);

    // add_ref -- increment the reference count on a buffer.
    //
    // Used for read-side multi-view sharing: when multiple
    // Memory_Views reference the same prefetched buffer, each
    // outstanding view holds a reference.
    void add_ref(StagingBuffer *buf);

    // ----- Diagnostics / test helpers -----

    // total_buffer_count -- the number of buffers currently in the pool
    // (both free and in-use).  In Grow mode this starts at the configured
    // `buffer_count` and increases as slots are auto-appended.
    std::size_t total_buffer_count() const noexcept;

    // max_buffer_count -- the hard ceiling on total slots.  Equals
    // `buffer_count` in Bounded mode.
    std::size_t max_buffer_count() const noexcept;

    // initial_buffer_count -- the `buffer_count` the pool was constructed
    // with (the provisioning hint).  Diagnostics / tests only.
    std::size_t initial_buffer_count() const noexcept;

    // free_buffer_count -- the number of buffers currently on the
    // free list (available for acquisition).
    std::size_t free_buffer_count() const noexcept;

    // buffer_capacity -- the per-buffer capacity in bytes.
    std::size_t buffer_capacity() const noexcept;

    // committed_bytes -- total byte storage actually allocated across all
    // buffers.  Zero after construction (storage is committed lazily on
    // first acquire); grows as buffers are used or grown.  Diagnostics /
    // tests only.
    std::size_t committed_bytes() const noexcept;

    // allocated_buffer_count -- number of buffers whose storage has been
    // committed (data != nullptr).  Diagnostics / tests only.
    std::size_t allocated_buffer_count() const noexcept;

    // timeout_ms -- the configured staging timeout.
    std::int64_t timeout_ms() const noexcept;

   private:
    // Find the best-fit-or-larger buffer in the free list.
    // Returns the index into free_list_, or free_list_.size() if
    // none is suitable.  Caller must hold mu_.
    std::size_t find_best_fit(std::size_t required_bytes) const;

    // Remove a buffer at the given free-list index and return it.
    // Caller must hold mu_.
    StagingBuffer *remove_from_free_list(std::size_t free_idx);

    // Ensure buffer `buf_idx` (on the free list) has committed storage of
    // at least `required_bytes`, allocating or growing it as needed.
    // Caller must hold mu_.  Updates buffers_[buf_idx].data and
    // .capacity_bytes.
    void ensure_storage(std::size_t buf_idx, std::size_t required_bytes);

    mutable std::mutex mu_;
    std::condition_variable cv_;

    // All buffers owned by the pool.  In Grow mode this may expand (by one
    // slot at a time, under mu_, up to max_buffer_count_) when an acquire
    // finds the free list empty.  Storage is committed lazily per slot.
    std::vector<StagingBuffer> buffers_;

    // Raw memory backing (one contiguous allocation per buffer).
    std::vector<std::unique_ptr<std::byte[]>> storage_;

    // Free list: indices into buffers_ that are currently available.
    // Kept sorted by capacity for efficient best-fit search.
    std::vector<std::size_t> free_list_;

    // Configuration.
    std::size_t buffer_count_;      // initial slot count (provisioning hint)
    std::size_t max_buffer_count_;  // hard ceiling on total slots
    std::size_t buffer_capacity_;
    std::int64_t timeout_ms_;
    GrowMode grow_mode_;
    bool grow_warned_ = false;  // one-time WARN past the configured count
};

}  // namespace amio::detail

#endif  // AMIO_SRC_STAGING_STAGING_POOL_HPP
