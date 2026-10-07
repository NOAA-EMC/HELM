// staging_pool.cpp -- AMIO Staging_Pool buffer manager implementation.
//
// Implements the buffer pool described in design.md §3
// (Staging Pool & Worker Pool).  Buffer *slots* are established during
// construction, but each buffer's byte storage is committed lazily on
// first acquire and grown on demand when an acquire requests more than
// the nominal capacity.  This keeps a pool provisioned with a large
// `buffer_count` (needed to avoid read-path backpressure when several
// variables' prefetch queues are live simultaneously) from eagerly
// reserving `count * capacity` bytes that a small-variable read never
// touches.
//
// In the default Grow mode, when an acquire finds the free list empty
// and the pool is below its hard ceiling (`max_buffer_count`), a new
// slot is appended on the spot: `buffer_count` is a provisioning hint,
// not a crash trigger.  Backpressure (condition-variable wait then
// AMIO_ERR_STAGING_BACKPRESSURE, R2.6) survives only at the ceiling or
// in Bounded mode, where it signals a genuine stuck-IO / leaked-view
// bug rather than a sizing mistake.
//
// Validates: R1.3, R1.5, R2.2, R2.6, R2.9, R3.10, R5.6, R5.9

#include "staging/staging_pool.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>

namespace amio::detail {

StagingPool::StagingPool(std::size_t buffer_count, std::size_t buffer_capacity, std::int64_t timeout_ms)
    : StagingPool(buffer_count, buffer_capacity, timeout_ms, kMaxBufferCount, GrowMode::Grow) {}

StagingPool::StagingPool(std::size_t buffer_count, std::size_t buffer_capacity, std::int64_t timeout_ms, std::size_t max_buffer_count, GrowMode mode)
    : buffer_count_(buffer_count), buffer_capacity_(buffer_capacity), timeout_ms_(timeout_ms), grow_mode_(mode) {
    assert(buffer_count >= kMinBufferCount && buffer_count <= kMaxBufferCount);
    assert(buffer_capacity >= kMinBufferCapacity && buffer_capacity <= kMaxBufferCapacity);
    assert(timeout_ms >= kMinTimeoutMs && timeout_ms <= kMaxTimeoutMs);
    assert(max_buffer_count >= buffer_count);
    assert(max_buffer_count <= kMaxBufferCount);

    // Bounded mode: the configured count IS the hard limit.
    max_buffer_count_ = (mode == GrowMode::Bounded) ? buffer_count : max_buffer_count;

    // Establish the buffer slots WITHOUT committing their byte storage:
    // `data` stays null and `capacity_bytes` stays 0 until the buffer is
    // first acquired (see ensure_storage).  The pool's nominal capacity is
    // still reported by buffer_capacity() and is the size an un-grown
    // buffer is provisioned with on first acquire.
    //
    // CRITICAL: reserve() to the ceiling so that auto-grow emplace_back()
    // NEVER reallocates the vectors -- callers hold raw StagingBuffer*
    // pointers into buffers_, and release() derives the slot index via
    // pointer arithmetic on buffers_.data().  Only the *headers* live in
    // these vectors (byte storage is a separate lazy unique_ptr), so the
    // reservation is a few hundred KiB at worst.
    buffers_.reserve(max_buffer_count_);
    storage_.reserve(max_buffer_count_);
    buffers_.resize(buffer_count);
    storage_.resize(buffer_count);
    free_list_.reserve(max_buffer_count_);

    for (std::size_t i = 0; i < buffer_count; ++i) {
        buffers_[i].data = nullptr;
        buffers_[i].capacity_bytes = 0;
        buffers_[i].used_bytes = 0;
        buffers_[i].ref_count = 0;
        buffers_[i].seq = 0;

        // All buffers start on the free list.
        free_list_.push_back(i);
    }

    // Sort free list by capacity (all zero here; the structure supports
    // heterogeneous capacities once buffers have grown).
    std::sort(free_list_.begin(), free_list_.end(),
              [this](std::size_t a, std::size_t b) { return buffers_[a].capacity_bytes < buffers_[b].capacity_bytes; });
}

StagingPool::~StagingPool() {
    // storage_ unique_ptrs handle deallocation automatically.
    // buffers_ data pointers become dangling but that's fine since
    // the pool is being destroyed.
}

StagingBuffer *StagingPool::acquire(std::size_t required_bytes) {
    std::unique_lock<std::mutex> lock(mu_);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

    while (true) {
        std::size_t idx = find_best_fit(required_bytes);
        if (idx < free_list_.size()) {
            // Commit (or grow) this buffer's byte storage before handing it
            // out.  Safe under mu_; the buffer is free so no existing view
            // references its storage.
            ensure_storage(free_list_[idx], required_bytes);
            return remove_from_free_list(idx);
        }

        // Free list empty.  In Grow mode below the hard ceiling, append a
        // fresh slot NOW rather than waiting: an exhausted pool is a
        // concurrency signal, not an error, and slow-but-healthy IO must
        // never starve a reader into AMIO_ERR_STAGING_BACKPRESSURE.  The
        // new slot's storage is committed by the ensure_storage() call on
        // the success path above (one more loop iteration).
        if (grow_mode_ == GrowMode::Grow && buffers_.size() < max_buffer_count_) {
            buffers_.emplace_back();  // data == nullptr, cap == 0: lazy slot
            storage_.emplace_back();
            free_list_.push_back(buffers_.size() - 1);
            if (!grow_warned_ && buffers_.size() > buffer_count_) {
                grow_warned_ = true;
                std::cerr << "[AMIO WARN] staging pool grew past configured buffer_count=" << buffer_count_ << " (now " << buffers_.size()
                          << " slots, ceiling " << max_buffer_count_
                          << "). This is informational: the pool auto-sizes to actual read/write concurrency. "
                             "Raise driver.amio_staging_buffer_count to silence it for steady-state concurrency."
                          << std::endl;
            }
            continue;  // retry: the new slot is on the free list
        }

        // No suitable buffer available (Bounded mode, or at the hard
        // ceiling); wait for one to be released.
        auto status = cv_.wait_until(lock, deadline);
        if (status == std::cv_status::timeout) {
            // One final check after wakeup -- a spurious wakeup or
            // a release that happened just before the timeout could
            // have made a buffer available.
            idx = find_best_fit(required_bytes);
            if (idx < free_list_.size()) {
                ensure_storage(free_list_[idx], required_bytes);
                return remove_from_free_list(idx);
            }
            // Timeout: return nullptr to signal backpressure.  In Grow
            // mode this only happens at the hard ceiling, so it means the
            // pool is genuinely saturated (stuck IO or leaked views), not
            // merely mis-provisioned.
            return nullptr;
        }
        // Spurious wakeup or legitimate notification -- loop back
        // and try to find a suitable buffer.
    }
}

void StagingPool::release(StagingBuffer *buf) {
    assert(buf != nullptr);

    std::lock_guard<std::mutex> lock(mu_);

    assert(buf->ref_count > 0);
    --buf->ref_count;

    if (buf->ref_count == 0) {
        // Return buffer to the free list.
        buf->used_bytes = 0;

        // Find the buffer's index in the buffers_ vector.
        auto buf_idx = static_cast<std::size_t>(buf - buffers_.data());
        assert(buf_idx < buffers_.size());

        // Insert into free list maintaining sorted order by capacity.
        auto it = std::lower_bound(free_list_.begin(), free_list_.end(), buf_idx,
                                   [this](std::size_t a, std::size_t b) { return buffers_[a].capacity_bytes < buffers_[b].capacity_bytes; });
        free_list_.insert(it, buf_idx);

        // Notify one waiting thread that a buffer is available.
        cv_.notify_one();
    }
}

void StagingPool::add_ref(StagingBuffer *buf) {
    assert(buf != nullptr);

    std::lock_guard<std::mutex> lock(mu_);
    assert(buf->ref_count > 0);
    ++buf->ref_count;
}

std::size_t StagingPool::total_buffer_count() const noexcept {
    // In Grow mode buffers_ may expand under mu_; take the lock so the
    // reading is consistent (this accessor is not on a hot path).
    std::lock_guard<std::mutex> lock(mu_);
    return buffers_.size();
}

std::size_t StagingPool::max_buffer_count() const noexcept {
    return max_buffer_count_;
}

std::size_t StagingPool::initial_buffer_count() const noexcept {
    return buffer_count_;
}

std::size_t StagingPool::free_buffer_count() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    return free_list_.size();
}

std::size_t StagingPool::buffer_capacity() const noexcept {
    return buffer_capacity_;
}

std::size_t StagingPool::committed_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t total = 0;
    for (const auto &b : buffers_) {
        total += b.capacity_bytes;
    }
    return total;
}

std::size_t StagingPool::allocated_buffer_count() const noexcept {
    std::lock_guard<std::mutex> lock(mu_);
    std::size_t n = 0;
    for (const auto &b : buffers_) {
        if (b.data != nullptr) ++n;
    }
    return n;
}

std::int64_t StagingPool::timeout_ms() const noexcept {
    return timeout_ms_;
}

std::size_t StagingPool::find_best_fit(std::size_t required_bytes) const {
    // Free list is sorted by committed capacity (ascending); uncommitted
    // buffers have capacity_bytes == 0 and sort first.  Selection policy:
    //
    //   Pass A (reuse): the smallest already-committed buffer whose capacity
    //     >= required_bytes.  No allocation.
    //   Pass B (allocate): an uncommitted buffer, which ensure_storage will
    //     provision at max(required_bytes, nominal).  This is the lazy path:
    //     it only commits memory for buffers that are actually used.
    //   Pass C (grow): all free buffers are committed but too small; grow the
    //     largest one (last in ascending order) to required_bytes, which
    //     minimises the growth delta and keeps smaller buffers pristine.
    //
    // If there is no free buffer at all, return the sentinel: acquire then
    // either auto-grows a new slot (Grow mode, below the ceiling) or
    // blocks/timeouts (Bounded mode or at the ceiling, R2.6).
    for (std::size_t i = 0; i < free_list_.size(); ++i) {
        const StagingBuffer &b = buffers_[free_list_[i]];
        if (b.data != nullptr && b.capacity_bytes >= required_bytes) {
            return i;
        }
    }
    for (std::size_t i = 0; i < free_list_.size(); ++i) {
        if (buffers_[free_list_[i]].data == nullptr) {
            return i;
        }
    }
    if (!free_list_.empty()) {
        return free_list_.size() - 1;
    }
    return free_list_.size();  // sentinel: no free buffer available
}

void StagingPool::ensure_storage(std::size_t buf_idx, std::size_t required_bytes) {
    assert(buf_idx < buffers_.size());
    StagingBuffer &b = buffers_[buf_idx];
    std::size_t target = required_bytes > buffer_capacity_ ? required_bytes : buffer_capacity_;
    if (b.data != nullptr && b.capacity_bytes >= target) {
        return;  // already committed large enough (Pass A reuse)
    }
    // (Re)allocate.  The buffer is on the free list (ref_count == 0), so no
    // outstanding Memory_View references the old storage; replacing it here
    // is safe.  Caller holds mu_.
    storage_[buf_idx] = std::make_unique<std::byte[]>(target);
    b.data = storage_[buf_idx].get();
    b.capacity_bytes = target;
}

StagingBuffer *StagingPool::remove_from_free_list(std::size_t free_idx) {
    assert(free_idx < free_list_.size());

    std::size_t buf_idx = free_list_[free_idx];
    free_list_.erase(free_list_.begin() + static_cast<std::ptrdiff_t>(free_idx));

    StagingBuffer *buf = &buffers_[buf_idx];
    buf->ref_count = 1;
    buf->seq += 1;
    // used_bytes is set by the caller after acquire returns.
    return buf;
}

}  // namespace amio::detail
