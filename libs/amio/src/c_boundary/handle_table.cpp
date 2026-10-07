// handle_table.cpp -- implementation of the C-Boundary handle table.
//
// See handle_table.hpp for the contract.  Validates: R10.5, R10.6,
// R10.7, R10.8, R12.2.

#include "c_boundary/handle_table.hpp"

#include <cstdint>
#include <limits>
#include <mutex>

namespace amio::detail {

HandleTable::Token HandleTable::insert(HandleKind kind, void *payload) {
    std::lock_guard<std::mutex> guard(mu_);

    std::uint32_t slot_index;

    if (!free_list_.empty()) {
        // Recycle a previously released slot.  The slot's generation
        // counter was already incremented inside `release()`, so it
        // is currently in a "live with new generation" state -- we
        // just need to flip the live flag back on and stash the
        // new payload + kind.
        slot_index = free_list_.back();
        free_list_.pop_back();
    } else {
        // Allocate a brand-new slot.  Generation starts at
        // `kMinGeneration` so the token is never zero.
        // We protect against the table growing past 2^32 - 1 entries
        // (which would overflow the slot index field of the token);
        // in practice AMIO will never come anywhere near this limit,
        // but a hard guard keeps the conversion well-defined.
        if (slots_.size() >= std::numeric_limits<std::uint32_t>::max()) {
            // No room left in the slot index space.  We cannot mint
            // a fresh token, so return zero -- the caller will
            // surface this as AMIO_ERR_BACKEND_FAILURE.  In practice
            // this branch is unreachable.
            return 0;
        }
        slot_index = static_cast<std::uint32_t>(slots_.size());
        slots_.emplace_back();
        slots_.back().generation = kMinGeneration;
    }

    Slot &s = slots_[slot_index];
    s.live = true;
    s.kind = kind;
    s.payload = payload;
    // generation already holds the next-to-issue value (set on
    // construction or on the prior `release()`).
    return pack(slot_index, s.generation);
}

amio_status_t HandleTable::lookup(Token token, HandleKind expected, void **out_payload) const noexcept {
    if (out_payload == nullptr) {
        // Misuse from inside AMIO_Core; treat conservatively.
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_payload = nullptr;

    if (token == 0) {
        return AMIO_ERR_NULL_HANDLE;
    }

    const std::uint32_t slot_index = unpack_slot(token);
    const std::uint32_t generation = unpack_generation(token);

    std::lock_guard<std::mutex> guard(mu_);

    if (slot_index >= slots_.size()) {
        // Token's slot index is out of range -- caller fabricated
        // a garbage handle, or freed a handle from a different
        // (smaller) table.  Reject without dereferencing.
        return AMIO_ERR_INVALID_HANDLE;
    }

    const Slot &s = slots_[slot_index];

    // Generation mismatch covers the reuse-after-free case: the
    // slot has been released and possibly re-issued under a new
    // generation, so the caller's old token is stale.
    if (!s.live || s.generation != generation) {
        return AMIO_ERR_INVALID_HANDLE;
    }

    if (s.kind != expected) {
        // Kind mismatch -- e.g. amio_close called on a view handle.
        // Treated as a stale handle so the caller cannot infer the
        // table layout from the error code alone.
        return AMIO_ERR_INVALID_HANDLE;
    }

    *out_payload = s.payload;
    return AMIO_OK;
}

amio_status_t HandleTable::release(Token token, HandleKind expected) noexcept {
    if (token == 0) {
        return AMIO_ERR_NULL_HANDLE;
    }

    const std::uint32_t slot_index = unpack_slot(token);
    const std::uint32_t generation = unpack_generation(token);

    std::lock_guard<std::mutex> guard(mu_);

    if (slot_index >= slots_.size()) {
        return AMIO_ERR_INVALID_HANDLE;
    }

    Slot &s = slots_[slot_index];

    if (!s.live || s.generation != generation || s.kind != expected) {
        return AMIO_ERR_INVALID_HANDLE;
    }

    // Mark the slot as released and bump the generation so any
    // outstanding token referring to it is now stale.  Generation
    // wraparound is theoretically possible after 2^32 releases of
    // the same slot, so we skip 0 to keep "token == 0" reserved as
    // the universal sentinel for "no handle".
    s.live = false;
    s.payload = nullptr;
    s.generation += 1;
    if (s.generation == 0) {
        s.generation = kMinGeneration;
    }
    free_list_.push_back(slot_index);
    return AMIO_OK;
}

std::size_t HandleTable::size_for_test() const noexcept {
    std::lock_guard<std::mutex> guard(mu_);
    return slots_.size() - free_list_.size();
}

}  // namespace amio::detail
