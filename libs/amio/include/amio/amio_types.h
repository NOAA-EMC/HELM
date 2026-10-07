/**
 * @file amio_types.h
 * @brief AMIO public ABI types (C99 surface).
 *
 * Declares the data-type tags, shape descriptor, and opaque handle
 * typedefs that every AMIO_C_API entry point consumes or returns.
 *
 * Header-isolation contract (R10.1, R10.2, R13.6):
 *   - Includes only `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`, and the
 *     companion mdspan forward-declaration header.
 *   - No C++ types, no template, namespace, reference, or std:: symbol.
 *   - No third-party (eckit / TensorStore / netCDF / HDF5 / nceplibs-g2c)
 *     header.
 *   - All declarations wrapped in `extern "C"` for C++ inclusion (R10.3).
 *   - Compiles cleanly under
 *         `gcc -std=c99 -pedantic -Werror -c amio_types.h`
 */

#ifndef AMIO_AMIO_TYPES_H
#define AMIO_AMIO_TYPES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "amio/amio_mdspan_fwd.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def AMIO_MAX_RANK
 * @brief Maximum supported rank for any AMIO Memory_View shape descriptor.
 *
 * Anchored by R2.1 ("a shape descriptor of 1 to 7 dimensions") and
 * fixed for the library lifetime. Hosts MUST NOT pass `rank > 7`
 * or `rank < 1`; doing so yields AMIO_ERR_INVALID_INPUT (R2.10).
 */
#define AMIO_MAX_RANK 7

/**
 * @enum amio_dtype_e
 * @brief Element-type tag carried alongside every payload crossing the FFI boundary.
 *
 * Integer values are FROZEN; new tags are appended only. The set below
 * is the full Phase 1 surface.
 *
 * Width and signedness agree with the corresponding Fortran
 * iso_c_binding kinds declared in amio_mod (R10.4, R1.12):
 *   - F32 corresponds to c_float
 *   - F64 corresponds to c_double
 *   - I8  corresponds to c_int8_t
 *   - I16 corresponds to c_int16_t
 *   - I32 corresponds to c_int32_t
 *   - I64 corresponds to c_int64_t
 */
typedef enum amio_dtype_e {
    AMIO_DTYPE_F32 = 0, /**< 32-bit IEEE 754 float (c_float) */
    AMIO_DTYPE_F64 = 1, /**< 64-bit IEEE 754 double (c_double) */
    AMIO_DTYPE_I8 = 2,  /**< 8-bit signed integer */
    AMIO_DTYPE_I16 = 3, /**< 16-bit signed integer */
    AMIO_DTYPE_I32 = 4, /**< 32-bit signed integer */
    AMIO_DTYPE_I64 = 5, /**< 64-bit signed integer */
    AMIO_DTYPE_U8 = 6,  /**< 8-bit unsigned integer */
    AMIO_DTYPE_U16 = 7, /**< 16-bit unsigned integer */
    AMIO_DTYPE_U32 = 8, /**< 32-bit unsigned integer */
    AMIO_DTYPE_U64 = 9  /**< 64-bit unsigned integer */
    /* Extension space reserved; do not renumber existing tags. */
} amio_dtype_t;

/**
 * @struct amio_shape_s
 * @brief N-dimensional shape descriptor for Memory_View payloads.
 *
 * @var amio_shape_s::rank
 *   Number of valid leading entries in extents/strides, in [1, AMIO_MAX_RANK].
 *   Values outside this range are rejected with AMIO_ERR_INVALID_INPUT (R2.10).
 *
 * @var amio_shape_s::extents
 *   Per-dimension extent in elements, row-major. Entries beyond `rank`
 *   MUST be zero. Each in-range extent MUST be strictly positive.
 *
 * @var amio_shape_s::strides
 *   Per-dimension stride in *elements* (not bytes). A stride entry of 0
 *   marks the dimension as contiguous and asks AMIO to derive the stride
 *   from `extents` row-major. Entries beyond `rank` MUST be zero.
 *
 * The total payload byte count is:
 *     `sizeof(dtype) * product(extents[0..rank-1])`
 * and is bounded by the AMIO manifest's per-buffer capacity (1 byte to 1 GiB).
 */
typedef struct amio_shape_s {
    int32_t rank;                   /**< Number of dimensions [1, AMIO_MAX_RANK] */
    int64_t extents[AMIO_MAX_RANK]; /**< Per-dimension size in elements */
    int64_t strides[AMIO_MAX_RANK]; /**< Per-dimension stride in elements (0 = contiguous) */
} amio_shape_t;

/**
 * @typedef amio_core_handle
 * @brief Opaque handle to an AMIO_Core context.
 *
 * Owner of Staging_Pool, Worker_Pool, Backend_Factory, and Prefetch_Queue.
 * Returned by amio_init(), invalidated by amio_finalize().
 * Host code MUST NOT dereference this pointer.
 */
typedef void *amio_core_handle;

/**
 * @typedef amio_dataset_handle
 * @brief Opaque handle to an open dataset (read or write).
 *
 * Returned by amio_open_dataset(), invalidated by amio_close_dataset().
 * Host code MUST NOT dereference this pointer.
 */
typedef void *amio_dataset_handle;

/**
 * @typedef amio_io_handle
 * @brief Opaque handle to a pending asynchronous I/O operation.
 *
 * Returned by amio_write() / amio_read(), invalidated on completion or error.
 * Host code MUST NOT dereference this pointer.
 */
typedef void *amio_io_handle;

/**
 * @typedef amio_view_handle
 * @brief Opaque handle to an outstanding read-side Memory_View.
 *
 * References a Staging_Pool buffer. Released by amio_release_view()
 * (R5.6, R5.9, R5.10).
 * Host code MUST NOT dereference this pointer.
 */
typedef void *amio_view_handle;

/**
 * @enum amio_io_kind_e
 * @brief Distinguishes write and read dataset handles at the FFI boundary.
 *
 * Misuse of a write handle on a read entry point (and vice versa) is
 * detected as AMIO_ERR_INVALID_HANDLE without dereferencing the handle.
 */
typedef enum amio_io_kind_e {
    AMIO_IO_KIND_WRITE = 0, /**< Dataset opened for writing */
    AMIO_IO_KIND_READ = 1   /**< Dataset opened for reading */
} amio_io_kind_t;

/**
 * @def AMIO_MODE_WRITE
 * @brief Dataset open mode constant for write access.
 */
#define AMIO_MODE_WRITE 0

/**
 * @def AMIO_MODE_READ
 * @brief Dataset open mode constant for read access.
 */
#define AMIO_MODE_READ 1

/**
 * @struct amio_bbox_s
 * @brief Optional bounding-box / stride descriptor for selective reads (R5.7).
 *
 * Passed as a pointer that may be NULL to request a full-payload read.
 * When non-NULL, every entry of `extents` and `offsets` beyond `rank`
 * MUST be zero, and the bounding box MUST lie wholly inside the
 * dataset's variable shape.
 *
 * @var amio_bbox_s::rank
 *   Number of dimensions for the bounding box.
 * @var amio_bbox_s::offsets
 *   Per-dimension start offset in elements.
 * @var amio_bbox_s::extents
 *   Per-dimension count in elements.
 * @var amio_bbox_s::strides
 *   Per-dimension stride in elements.
 */
typedef struct amio_bbox_s {
    int32_t rank;                   /**< Number of dimensions */
    int64_t offsets[AMIO_MAX_RANK]; /**< Per-dimension start offset */
    int64_t extents[AMIO_MAX_RANK]; /**< Per-dimension count */
    int64_t strides[AMIO_MAX_RANK]; /**< Per-dimension stride */
} amio_bbox_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AMIO_AMIO_TYPES_H */
