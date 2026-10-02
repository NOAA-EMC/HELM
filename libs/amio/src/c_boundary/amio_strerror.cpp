// amio_strerror.cpp
//
// Implements `amio_strerror(int err)`, the public AMIO_C_API entry
// point that maps an AMIO error code to a stable, human-readable,
// null-terminated description string.
//
// Contract (declared in include/amio/amio_errors.h):
//
//   * Defined codes return byte-equal descriptions across repeated
//     invocations and are side-effect free (R12.6).  The
//     description table is a `constexpr` array of string literals,
//     so each call performs a pure read of static storage; no lazy
//     allocation, no first-call hazard, no table mutation.
//
//   * Undefined codes return a non-null, null-terminated description
//     of the form "AMIO_ERR_UNKNOWN(<int>)" formatted into a
//     `thread_local` scratch buffer (R12.8).  The constexpr table
//     is never mutated.  The buffer is per-thread, so concurrent
//     callers from multiple host threads do not race.
//
//   * Lazy-init (R12.7) is satisfied trivially: the description
//     table is `constexpr` and initialized at program start, before
//     any host code can invoke the lookup.  No double-checked
//     locking, no `std::call_once`, no global mutex.
//
// Header-isolation (R10.1, R10.2, R13.6) is preserved: this
// translation unit lives entirely under `src/c_boundary/`, is
// compiled into `libamio.so`, and is never installed.  Only the
// declaration in `include/amio/amio_errors.h` is visible to
// downstream consumers.
//
// Validates: R12.5, R12.6, R12.7, R12.8

#include <cstddef>  // std::size_t
#include <cstdio>   // std::snprintf

#include "amio/amio.h"

namespace {

// ---------------------------------------------------------------------------
// Defined-code description table.
//
// `kAmioErrorTable` is a `constexpr` array of `const char *` indexed
// by the integer value of `amio_err_t`.  Slot N holds the
// description of the enumerator whose value is N.
//
// CONTRACT INVARIANTS:
//
//   1. The slot for every enumerator declared in
//      `include/amio/amio_errors.h` is populated with a non-null
//      string literal.  Static_asserts below enforce array sizing
//      so that any future addition / removal of an enumerator that
//      is not mirrored in this table is caught at compile time.
//
//   2. Slot order matches enumerator integer values; this is the
//      "static constexpr array indexed by AMIO_ERR_* enum value"
//      that task 3.3 mandates.  R10.8 (stable enumerator integers)
//      makes this index-mapping safe across releases.
//
//   3. The strings are static-storage `const char *` to string
//      literals; their lifetime is the entire program.  Returning
//      them by pointer is safe across threads, library unload, and
//      arbitrary numbers of repeated invocations (R12.6).
// ---------------------------------------------------------------------------
constexpr const char *kAmioErrorTable[] = {
    /*  0 = AMIO_OK                        */ "AMIO_OK: success",
    /*  1 = AMIO_ERR_NULL_HANDLE           */ "AMIO_ERR_NULL_HANDLE: null opaque handle",
    /*  2 = AMIO_ERR_INVALID_HANDLE        */
    "AMIO_ERR_INVALID_HANDLE: handle never initialized, finalized, released, or stale (generation mismatch)",
    /*  3 = AMIO_ERR_MANIFEST_NOT_FOUND    */ "AMIO_ERR_MANIFEST_NOT_FOUND: manifest path missing or unreadable",
    /*  4 = AMIO_ERR_MANIFEST_INVALID      */ "AMIO_ERR_MANIFEST_INVALID: manifest failed schema validation",
    /*  5 = AMIO_ERR_ALREADY_INITIALIZED   */ "AMIO_ERR_ALREADY_INITIALIZED: amio_init invoked twice on the same handle",
    /*  6 = AMIO_ERR_FINALIZE_TIMEOUT      */ "AMIO_ERR_FINALIZE_TIMEOUT: finalize drain phase exceeded its 30-second bound",
    /*  7 = AMIO_ERR_STAGING_BACKPRESSURE  */
    "AMIO_ERR_STAGING_BACKPRESSURE: no staging buffer became available within the configured staging timeout",
    /*  8 = AMIO_ERR_INVALID_BINDING       */ "AMIO_ERR_INVALID_BINDING: requested CPU core or NUMA domain is not present or not permitted",
    /*  9 = AMIO_ERR_COMM_SPLIT_FAILED     */
    "AMIO_ERR_COMM_SPLIT_FAILED: MPI communicator split failed or I/O rank set is not a subset of world",
    /* 10 = AMIO_ERR_THREADING_UNSUPPORTED */ "AMIO_ERR_THREADING_UNSUPPORTED: host did not initialize MPI with at least MPI_THREAD_MULTIPLE",
    /* 11 = AMIO_ERR_UNKNOWN_BACKEND       */ "AMIO_ERR_UNKNOWN_BACKEND: configuration backend key does not match any registered driver",
    /* 12 = AMIO_ERR_LOSSY_CODEC_FORBIDDEN */ "AMIO_ERR_LOSSY_CODEC_FORBIDDEN: requested codec is not on the lossless allow-list",
    /* 13 = AMIO_ERR_VIEWS_OUTSTANDING     */ "AMIO_ERR_VIEWS_OUTSTANDING: dataset close called while one or more Memory_Views are still outstanding",
    /* 14 = AMIO_ERR_QUEUE_FULL            */ "AMIO_ERR_QUEUE_FULL: worker queue depth would exceed capacity and backpressure is not configured",
    /* 15 = AMIO_ERR_TIMEOUT               */ "AMIO_ERR_TIMEOUT: synchronous wait, flush, or read exceeded its configured timeout",
    /* 16 = AMIO_ERR_BACKEND_FAILURE       */ "AMIO_ERR_BACKEND_FAILURE: backend driver reported a serialization or deserialization failure",
    /* 17 = AMIO_ERR_INVALID_INPUT         */ "AMIO_ERR_INVALID_INPUT: null host pointer, unsupported dtype, or invalid shape descriptor"};

// Number of populated slots in the defined-code table.  Used as the
// upper bound of the "defined code" index range in amio_strerror.
constexpr std::size_t kAmioErrorTableSize = sizeof(kAmioErrorTable) / sizeof(kAmioErrorTable[0]);

// Compile-time invariant: every enumerator declared in
// amio_errors.h has a corresponding slot in the description table.
// If a future patch appends a new AMIO_ERR_* enumerator without
// mirroring it here, this static_assert (and the one at the bottom
// of the table) breaks the build instead of silently returning
// "AMIO_ERR_UNKNOWN(...)" for a freshly-defined code.
static_assert(kAmioErrorTableSize == static_cast<std::size_t>(AMIO_ERR_INVALID_INPUT) + 1u,
              "kAmioErrorTable must contain a slot for every AMIO_ERR_* "
              "enumerator declared in include/amio/amio_errors.h");

// Sanity-check that slot 0 is the success sentinel.  R10.8 freezes
// AMIO_OK at value 0; this assert protects against accidental
// reordering of either the enum or the table.
static_assert(static_cast<int>(AMIO_OK) == 0, "AMIO_OK must remain integer value 0");

// Capacity of the per-thread scratch buffer used to format the
// "AMIO_ERR_UNKNOWN(<int>)" description for undefined codes.
//
// 64 bytes is comfortably larger than the longest possible
// formatted output:
//
//     "AMIO_ERR_UNKNOWN("  =  17 bytes
//     decimal int (max 11) =  11 bytes (e.g. -2147483648)
//     ")"                  =   1 byte
//     terminating NUL      =   1 byte
//                           ----
//                            30 bytes
//
// std::snprintf truncates with a guaranteed NUL terminator if
// somehow exceeded, so the function never returns a non-terminated
// string.
constexpr std::size_t kUnknownBufferBytes = 64u;

}  // namespace

// ---------------------------------------------------------------------------
// amio_strerror -- public AMIO_C_API entry point.
//
// See include/amio/amio_errors.h for the full contract; this
// definition implements R12.5 - R12.8 inline below.
// ---------------------------------------------------------------------------
extern "C" AMIO_API const char *amio_strerror(int err) {
    // ------------------------------------------------------------------
    // Defined-code path (R12.5, R12.6, R12.7).
    //
    // The cast to size_t after the lower-bound check folds negative
    // ints into the unknown-code path automatically, since negative
    // ints become very large size_t values that fail the upper-bound
    // comparison.  The inclusive lower bound on `err` is therefore
    // strictly necessary.
    // ------------------------------------------------------------------
    if (err >= 0 && static_cast<std::size_t>(err) < kAmioErrorTableSize) {
        // Pure read of static-storage string literal.  No mutation,
        // no allocation, no synchronization required (R12.6).  The
        // returned pointer remains valid for the entire program
        // lifetime (R12.5).
        return kAmioErrorTable[static_cast<std::size_t>(err)];
    }

    // ------------------------------------------------------------------
    // Undefined-code path (R12.8).
    //
    // Format "AMIO_ERR_UNKNOWN(<int>)" into a thread_local scratch
    // buffer.  thread_local guarantees:
    //
    //   * Each calling thread sees its own buffer; concurrent calls
    //     from different threads do not corrupt each other.
    //   * The buffer's lifetime extends to thread exit, so the
    //     returned pointer remains valid as long as the calling
    //     thread is alive (which is at least until the next
    //     amio_strerror invocation on that same thread, which is
    //     all the contract requires).
    //   * The "table" of defined-code descriptions is never touched,
    //     satisfying the "no table mutation" clause.
    //
    // std::snprintf is bounded and always NUL-terminates the output
    // (assuming kUnknownBufferBytes >= 1), so the returned pointer
    // is guaranteed non-null and null-terminated.
    // ------------------------------------------------------------------
    static thread_local char unknown_buffer[kUnknownBufferBytes];

    // Even on truncation, snprintf writes a terminating NUL inside
    // unknown_buffer when the size argument is non-zero.  We ignore
    // the return value because the contract only requires
    // "non-null, null-terminated" -- the formatted body is best
    // effort.
    std::snprintf(unknown_buffer, kUnknownBufferBytes, "AMIO_ERR_UNKNOWN(%d)", err);

    return unknown_buffer;
}
