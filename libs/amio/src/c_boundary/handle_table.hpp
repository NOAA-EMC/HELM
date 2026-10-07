// handle_table.hpp -- AMIO C-Boundary opaque-handle table.
//
// This header is PRIVATE to the AMIO_Core build (`src/c_boundary/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// Every AMIO_C_API entry point that operates on an opaque `void*`
// handle (`amio_core_handle`, `amio_dataset_handle`, `amio_io_handle`,
// `amio_view_handle`) routes the handle through this table to:
//
//   * Reject NULL handles      -> AMIO_ERR_NULL_HANDLE      (R10.6)
//   * Reject stale / freed handles, including reuse-after-free that
//     happens to land on a recycled slot, via a per-slot generation
//     counter -> AMIO_ERR_INVALID_HANDLE                    (R10.7)
//   * Detect kind mismatches (e.g. a write-dataset handle passed to
//     a read entry point) without dereferencing the payload pointer
//                                                          (R10.6)
//   * Look up the underlying C++ payload (an AMIO_Core, dataset,
//     pending-I/O record, or view ref) so the C-Boundary can call
//     into the private C++ API.
//
// The table never dereferences a host-supplied pointer.  Handles
// are 64-bit tokens reinterpret_cast<>'d to `void*`; the high 32
// bits hold a slot index, the low 32 bits hold the slot's current
// generation counter.  A stale handle whose slot has been recycled
// presents a generation that no longer matches the slot, so the
// lookup returns `nullptr` and the C-Boundary returns
// AMIO_ERR_INVALID_HANDLE without touching freed memory.
//
// Thread safety
// -------------
// All public methods are safe to call concurrently from any thread.
// The implementation uses a single std::mutex; lookup is the hot
// path and the critical section is bounded to a vector index +
// two scalar comparisons, so contention is minimal in practice.
// (A reader-writer or lock-free implementation is a follow-up
// optimization but is not required for correctness.)
//
// Validates: R10.5, R10.6, R10.7, R10.8, R12.2

#ifndef AMIO_SRC_C_BOUNDARY_HANDLE_TABLE_HPP
#define AMIO_SRC_C_BOUNDARY_HANDLE_TABLE_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

// HandleKind -- discriminates the four opaque handle types that the
// public C99 surface exposes.  Storing the kind alongside the
// payload lets the C-Boundary detect "wrong kind of handle" misuse
// (e.g. a view handle passed to amio_finalize) and report it as
// AMIO_ERR_INVALID_HANDLE without dereferencing the payload.
enum class HandleKind : std::uint8_t {
    Core = 1,
    Dataset = 2,
    Io = 3,
    View = 4,
};

// HandleTable -- a slab of generation-stamped slots that hands out
// 64-bit token handles whose value remains stable until release().
//
// Token layout (most significant bit on the left):
//
//   63                           32 31                            0
//   +------------------------------+------------------------------+
//   |      slot index (32 bits)    |   generation counter (32 b)  |
//   +------------------------------+------------------------------+
//
// The generation counter is incremented on every release.  A handle
// whose generation no longer matches the slot's current generation
// is treated as stale and rejected.  The table never reuses
// generation 0, so a token computed from a default-constructed slot
// is always invalid.
//
// The token is reinterpreted as `void*` at the FFI boundary; on
// every supported platform sizeof(void*) >= 8, but we still guard
// against narrower pointers with a static_assert below.
class HandleTable {
   public:
    using Token = std::uint64_t;

    static constexpr std::uint32_t kMinGeneration = 1;

    HandleTable() = default;

    HandleTable(const HandleTable &) = delete;
    HandleTable &operator=(const HandleTable &) = delete;
    HandleTable(HandleTable &&) = delete;
    HandleTable &operator=(HandleTable &&) = delete;

    // insert -- bind `payload` to a fresh handle of kind `kind`.
    //
    // Returns the freshly minted token, never zero.  The token
    // remains stable until a matching `release()` call increments
    // the slot's generation.  `payload` is owned by the caller; the
    // table only stores the pointer.
    Token insert(HandleKind kind, void *payload);

    // lookup -- validate a token and return its payload + kind.
    //
    // Returns:
    //   AMIO_OK                with `*out_payload` set on success.
    //   AMIO_ERR_NULL_HANDLE   if `token == 0`.
    //   AMIO_ERR_INVALID_HANDLE for any token that is malformed,
    //                           refers to an unknown slot, or whose
    //                           generation no longer matches the
    //                           slot's current generation (i.e.
    //                           reuse-after-free).
    //
    // When `expected != HandleKind{0}` the kind stored in the slot
    // is also checked; mismatches yield AMIO_ERR_INVALID_HANDLE so
    // the C-Boundary can refuse cross-kind misuse without ever
    // dereferencing the payload pointer.
    amio_status_t lookup(Token token, HandleKind expected, void **out_payload) const noexcept;

    // release -- mark `token` as released and bump the slot's
    //            generation so subsequent lookups return
    //            AMIO_ERR_INVALID_HANDLE.
    //
    // Returns:
    //   AMIO_OK                 on success.
    //   AMIO_ERR_NULL_HANDLE    if `token == 0`.
    //   AMIO_ERR_INVALID_HANDLE if the token is stale, malformed,
    //                           or already released.
    //
    // The released slot is returned to the free list and may be
    // re-allocated by a future insert().  The released payload is
    // *not* dereferenced -- the caller is responsible for any
    // resource cleanup before calling release.
    amio_status_t release(Token token, HandleKind expected) noexcept;

    // size_for_test -- diagnostics-only count of currently live
    // slots.  Inexpensive; used by the unit tests.  Does not lock
    // the slot but does take the table mutex so the result is
    // self-consistent.
    std::size_t size_for_test() const noexcept;

    // ---- Token packing helpers --------------------------------------
    //
    // Exposed at namespace level (and as `static` members) so the
    // C-Boundary can reinterpret_cast the token to/from `void*` at
    // the FFI seam in exactly one place.  Keeping these in the
    // header lets the public-API translation unit avoid a function
    // call on the hot validation path.

    static constexpr Token pack(std::uint32_t slot, std::uint32_t generation) noexcept {
        return (static_cast<Token>(slot) << 32) | static_cast<Token>(generation);
    }

    static constexpr std::uint32_t unpack_slot(Token t) noexcept {
        return static_cast<std::uint32_t>(t >> 32);
    }

    static constexpr std::uint32_t unpack_generation(Token t) noexcept {
        return static_cast<std::uint32_t>(t & 0xFFFFFFFFu);
    }

    // void*  <->  Token round-trip.  These are the *only* place in
    // the codebase that should reinterpret the FFI handle pointer.
    static void *to_ptr(Token t) noexcept {
        return reinterpret_cast<void *>(static_cast<std::uintptr_t>(t));
    }

    static Token from_ptr(void *p) noexcept {
        return static_cast<Token>(reinterpret_cast<std::uintptr_t>(p));
    }

   private:
    struct Slot {
        std::uint32_t generation = 0;  // 0 means "never used"
        bool live = false;
        HandleKind kind = HandleKind::Core;
        void *payload = nullptr;
    };

    mutable std::mutex mu_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_list_;  // slot indices ready for reuse
};

// On every supported platform `void*` is at least 64 bits wide.
// Keeping the static_assert here ensures a 32-bit ABI build (which
// is not supported by AMIO) fails loudly at compile time rather
// than silently truncating handle tokens at the FFI boundary.
static_assert(sizeof(void *) >= sizeof(std::uint64_t),
              "AMIO requires sizeof(void*) >= 8; "
              "32-bit pointer ABIs are unsupported.");

}  // namespace amio::detail

#endif  // AMIO_SRC_C_BOUNDARY_HANDLE_TABLE_HPP
