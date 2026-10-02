/**
 * @file amio_errors.h
 * @brief AMIO public error code enumeration (C99 surface).
 *
 * Implements the stable, ABI-frozen `AMIO_ERR_*` enumeration that
 * every public AMIO_C_API entry point returns. Integer values are
 * fixed for the library lifetime (R10.8, R12.5); new codes are
 * appended only -- existing values MUST NOT be renumbered or
 * reassigned across releases.
 *
 * Header-isolation contract (R10.1, R10.2, R13.6):
 *   - Includes only `<stdint.h>` and the AMIO_API export-macro header.
 *   - No C++ types, no third-party headers, no std:: symbols.
 *   - All declarations wrapped in `extern "C"` for C++ inclusion (R10.3).
 *   - Compiles cleanly under
 *         `gcc -std=c99 -pedantic -Werror -c amio_errors.h`
 */

#ifndef AMIO_AMIO_ERRORS_H
#define AMIO_AMIO_ERRORS_H

#include <stdint.h>

#include "amio/amio_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum amio_err_e
 * @brief Public error code enumeration for all AMIO_C_API entry points.
 *
 * The integer values below are FROZEN. Adding a new code is permitted
 * only by appending it with the next unused integer. Renumbering,
 * removing, or reordering any existing code is an ABI break and is
 * forbidden across minor and patch releases.
 */
typedef enum amio_err_e {
    AMIO_OK = 0,                         /**< Success */
    AMIO_ERR_NULL_HANDLE = 1,            /**< NULL handle passed (R1.9, R10.6) */
    AMIO_ERR_INVALID_HANDLE = 2,         /**< Stale or wrong-kind handle (R1.9, R10.6, R10.7) */
    AMIO_ERR_MANIFEST_NOT_FOUND = 3,     /**< Manifest file not found (R1.4) */
    AMIO_ERR_MANIFEST_INVALID = 4,       /**< Manifest parse/validation error (R1.5, R11.4) */
    AMIO_ERR_ALREADY_INITIALIZED = 5,    /**< amio_init called twice without finalize (R1.6) */
    AMIO_ERR_FINALIZE_TIMEOUT = 6,       /**< Drain timed out during finalize (R1.8) */
    AMIO_ERR_STAGING_BACKPRESSURE = 7,   /**< Staging pool exhausted (R2.6) */
    AMIO_ERR_INVALID_BINDING = 8,        /**< Invalid thread/core binding (R3.3) */
    AMIO_ERR_COMM_SPLIT_FAILED = 9,      /**< MPI communicator split failed (R3.6) */
    AMIO_ERR_THREADING_UNSUPPORTED = 10, /**< MPI threading level insufficient (R3.8) */
    AMIO_ERR_UNKNOWN_BACKEND = 11,       /**< Backend name not in factory registry (R4.6) */
    AMIO_ERR_LOSSY_CODEC_FORBIDDEN = 12, /**< Lossy codec rejected by policy (R7.5, R8.4) */
    AMIO_ERR_VIEWS_OUTSTANDING = 13,     /**< Cannot close: unreleased views (R5.10) */
    AMIO_ERR_QUEUE_FULL = 14,            /**< Write queue at capacity (R6.9) */
    AMIO_ERR_TIMEOUT = 15,               /**< Operation timed out (R3.9, R5.5, R6.10) */
    AMIO_ERR_BACKEND_FAILURE = 16,       /**< Backend driver reported an error (R6.6, R7.6) */
    AMIO_ERR_INVALID_INPUT = 17          /**< Invalid argument (NULL, out-of-range, etc.) (R2.10) */
    /*
     * Extension space reserved.  Append future codes with the next
     * unused integer (18, 19, ...).  Do NOT reorder, do NOT remove,
     * do NOT renumber.
     */
} amio_err_t;

/**
 * @typedef amio_status_t
 * @brief Fixed-width integer type for AMIO error codes across the FFI boundary.
 *
 * The enumeration amio_err_t is the canonical type, but this fixed-width
 * typedef is provided so that Fortran iso_c_binding wrappers and external
 * bindings can declare the return type independently of the enum representation.
 */
typedef int32_t amio_status_t;

/**
 * @brief Map an AMIO error code to a stable, human-readable description.
 *
 * @param err  The error code integer to look up. May be any integer value,
 *             not just defined AMIO_ERR_* enumerators.
 *
 * @return A non-null pointer to a null-terminated C string with static
 *         storage duration. Callers MUST NOT free or modify the returned
 *         string.
 *
 * Contract (R12.5, R12.6, R12.7, R12.8):
 *   1. The return value is ALWAYS a non-null pointer to a null-terminated
 *      C string with static storage duration.
 *   2. For any defined AMIO_ERR_* enumerator, repeated invocations return
 *      byte-equal descriptions. The call is side-effect free (R12.6).
 *   3. For any integer outside the defined set, the return value is
 *      `"AMIO_ERR_UNKNOWN(<int>)"` where `<int>` is the decimal
 *      representation of `err` (R12.8).
 *   4. The function is thread-safe: defined-code lookups are pure reads;
 *      unknown-code formatting uses a thread_local scratch buffer.
 *
 * @code
 * amio_status_t rc = amio_init("manifest.yaml", &core);
 * if (rc != AMIO_OK) {
 *     fprintf(stderr, "AMIO error: %s\n", amio_strerror(rc));
 * }
 * @endcode
 */
AMIO_API const char *amio_strerror(int err);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AMIO_AMIO_ERRORS_H */
