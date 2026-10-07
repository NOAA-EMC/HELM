/**
 * @file amio.h
 * @brief AMIO public umbrella header (C99 surface).
 *
 * This is the single entry point that downstream consumers
 * (Fortran iso_c_binding, C, C++) include to access the AMIO_C_API.
 * It composes the four leaf headers that together carry the full
 * Phase 1 public ABI:
 *
 *   - amio_export.h — AMIO_API public-symbol export macro (R10.8)
 *   - amio_errors.h — stable AMIO_ERR_* enumeration + amio_strerror (R12.5-R12.8)
 *   - amio_types.h  — amio_dtype_t, amio_shape_t, amio_bbox_t, opaque handles (R10.5-R10.7)
 *   - amio_mdspan_fwd.h — opaque forward declarations of mdspan descriptors (R10.1, R10.2)
 *
 * Header-isolation contract (R10.1, R10.2, R10.3, R13.6):
 *   - Transitively includes only: `<stdint.h>`, `<stddef.h>`, `<stdbool.h>`
 *   - MUST NOT include any C++ standard library header or third-party header
 *   - All declarations wrapped in `extern "C"` for C++ inclusion (R10.3)
 *   - Compilable with: `gcc -std=c99 -pedantic -Werror -c amio.h`
 *     and with: `g++ -std=c++20 -c amio.h`
 */

#ifndef AMIO_AMIO_H
#define AMIO_AMIO_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "amio/amio_errors.h"
#include "amio/amio_export.h"
#include "amio/amio_mdspan_fwd.h"
#include "amio/amio_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @name ABI Version
 *  @brief AMIO public ABI version macros.
 *
 *  The major version pins the integer values of amio_err_t and the
 *  layout of every public ABI struct. Bumping the major version is
 *  an ABI break (R10.8).
 *  @{
 */
#define AMIO_ABI_VERSION_MAJOR 0 /**< ABI major version */
#define AMIO_ABI_VERSION_MINOR 2 /**< ABI minor version */
#define AMIO_ABI_VERSION_PATCH 0 /**< ABI patch version */
/** @} */

/* -----------------------------------------------------------------
 * AMIO_C_API entry points
 * ----------------------------------------------------------------- */

/**
 * @brief Initialize AMIO from a configuration manifest.
 *
 * Creates an AMIO_Core context that owns the Staging_Pool, Worker_Pool,
 * Backend_Factory, and Prefetch_Queue.
 *
 * @param[in]  manifest_path  Non-NULL, NUL-terminated path to a YAML/JSON manifest.
 * @param[out] out_core       Non-NULL pointer; on success receives a freshly-minted
 *                            opaque core handle. On failure set to NULL.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_INVALID_INPUT — NULL arguments
 *   - AMIO_ERR_MANIFEST_NOT_FOUND — manifest file does not exist (R1.4)
 *   - AMIO_ERR_MANIFEST_INVALID — manifest parse/validation error (R1.5)
 *   - AMIO_ERR_INVALID_BINDING — invalid thread binding (R3.3)
 *   - AMIO_ERR_COMM_SPLIT_FAILED — MPI communicator split failed (R3.6)
 *   - AMIO_ERR_THREADING_UNSUPPORTED — MPI threading level insufficient (R3.8)
 *   - AMIO_ERR_BACKEND_FAILURE — any other initialization failure
 *
 * @code
 * amio_core_handle core = NULL;
 * amio_status_t rc = amio_init("amio_manifest.yaml", &core);
 * if (rc != AMIO_OK) {
 *     fprintf(stderr, "Init failed: %s\n", amio_strerror(rc));
 * }
 * @endcode
 */
AMIO_API amio_status_t amio_init(const char *manifest_path, amio_core_handle *out_core);

/**
 * @brief Drain pending I/O, release resources, and invalidate the core handle.
 *
 * After this call, the core handle is invalid and MUST NOT be reused.
 *
 * @param[in] core  A valid, initialized AMIO_Core handle.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — core is NULL
 *   - AMIO_ERR_INVALID_HANDLE — stale or never-initialized handle
 *   - AMIO_ERR_FINALIZE_TIMEOUT — drain timed out (R1.8)
 *   - AMIO_ERR_BACKEND_FAILURE — drain raised an exception
 */
AMIO_API amio_status_t amio_finalize(amio_core_handle core);

/**
 * @brief Open a dataset for reading or writing.
 *
 * Uses the Backend_Factory to instantiate the appropriate driver based
 * on the dataset configuration manifest.
 *
 * @param[in]  core         A valid, initialized AMIO_Core handle.
 * @param[in]  config_path  NUL-terminated path to a YAML/JSON dataset config.
 * @param[in]  mode         AMIO_MODE_WRITE or AMIO_MODE_READ.
 * @param[out] out_dataset  Non-NULL pointer; on success receives a dataset handle.
 *                          On failure set to NULL.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — core is NULL
 *   - AMIO_ERR_INVALID_HANDLE — core stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — NULL arguments or invalid mode
 *   - AMIO_ERR_UNKNOWN_BACKEND — backend name not in registry (R4.6)
 *   - AMIO_ERR_BACKEND_FAILURE — driver open failed
 */
AMIO_API amio_status_t amio_open_dataset(amio_core_handle core, const char *config_path, int32_t mode, amio_dataset_handle *out_dataset);

/**
 * @brief Initialize AMIO from an in-memory manifest string.
 *
 * Behaves exactly like amio_init(), except the manifest is supplied as a
 * NUL-terminated string rather than read from a file path.
 *
 * @param[in]  manifest_content  Non-NULL manifest text in the given format.
 * @param[in]  format            "yaml" or "json".
 * @param[out] out_core          Non-NULL pointer; on success receives a freshly-minted
 *                               opaque core handle. On failure set to NULL.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_INVALID_INPUT — NULL manifest_content/format/out_core
 *   - AMIO_ERR_MANIFEST_INVALID — manifest parse/validation error (R1.5)
 *   - AMIO_ERR_BACKEND_FAILURE — pool construction failure
 *
 * @code
 * amio_core_handle core = NULL;
 * amio_status_t rc = amio_init_from_string(manifest_text, "yaml", &core);
 * if (rc != AMIO_OK) {
 *     fprintf(stderr, "Init failed: %s\n", amio_strerror(rc));
 * }
 * @endcode
 */
AMIO_API amio_status_t amio_init_from_string(const char *manifest_content, const char *format, amio_core_handle *out_core);

/**
 * @brief Open a dataset from an in-memory config string.
 *
 * Behaves exactly like amio_open_dataset(), except the dataset config is
 * supplied as a NUL-terminated string rather than read from a file path.
 *
 * @param[in]  core            A valid, initialized AMIO_Core handle.
 * @param[in]  config_content  Non-NULL config text in the given format.
 * @param[in]  format          "yaml" or "json".
 * @param[in]  mode            AMIO_MODE_WRITE or AMIO_MODE_READ.
 * @param[out] out_dataset     Non-NULL pointer; on success receives a dataset handle.
 *                             On failure set to NULL.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — core is NULL
 *   - AMIO_ERR_INVALID_HANDLE — core stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — NULL config_content/format/out_dataset, or invalid mode
 *   - AMIO_ERR_MANIFEST_INVALID — config parse/validation error
 *   - AMIO_ERR_UNKNOWN_BACKEND — backend name not in registry (R4.6)
 *   - AMIO_ERR_BACKEND_FAILURE — driver open failed
 */
AMIO_API amio_status_t amio_open_dataset_from_string(amio_core_handle core, const char *config_content, const char *format, int32_t mode,
                                                     amio_dataset_handle *out_dataset);

/**
 * @brief Flush pending writes and close a dataset, releasing driver resources.
 *
 * Equivalent to calling amio_flush() followed by releasing the driver.
 * After this call, the dataset handle is invalidated.
 *
 * @param[in] dataset  A valid dataset handle.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — dataset is NULL
 *   - AMIO_ERR_INVALID_HANDLE — dataset stale or wrong kind
 *   - AMIO_ERR_VIEWS_OUTSTANDING — unreleased read views (R5.10)
 *   - AMIO_ERR_BACKEND_FAILURE — flush or close failed
 */
AMIO_API amio_status_t amio_close_dataset(amio_dataset_handle dataset);

/**
 * @brief Snapshot host data into a staging buffer and enqueue an async write.
 *
 * The host data is copied into the Staging_Pool synchronously, so the
 * caller may safely modify or free `host_data` after this call returns.
 * The actual I/O to the backend proceeds asynchronously.
 *
 * @param[in]  dataset    A write dataset handle from amio_open_dataset().
 * @param[in]  var_name   NUL-terminated variable identifier.
 * @param[in]  host_data  Pointer to the host-side payload.
 * @param[in]  dtype      Element type tag describing the payload.
 * @param[in]  shape      Shape descriptor (rank, extents, strides).
 * @param[out] out_io     On success, receives the per-write I/O handle (R2.7).
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — dataset is NULL
 *   - AMIO_ERR_INVALID_HANDLE — dataset stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — invalid arguments (R2.10)
 *   - AMIO_ERR_STAGING_BACKPRESSURE — staging pool exhausted (R2.6)
 *   - AMIO_ERR_QUEUE_FULL — write queue at capacity (R6.9)
 *   - AMIO_ERR_BACKEND_FAILURE — backend error
 */
AMIO_API amio_status_t amio_write(amio_dataset_handle dataset, const char *var_name, const void *host_data, amio_dtype_t dtype,
                                  const amio_shape_t *shape, amio_io_handle *out_io);

/**
 * @brief Return a Memory_View for a prefetched variable/timestep.
 *
 * @param[in]  dataset   A read dataset handle from amio_open_dataset().
 * @param[in]  var_name  NUL-terminated variable identifier.
 * @param[in]  timestep  Timestep index to read.
 * @param[in]  bbox      Optional bounding box (NULL for full payload).
 * @param[out] out_view  On success, receives the view handle. Caller MUST
 *                       release with amio_release_view() (R5.6, R5.9).
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — dataset is NULL
 *   - AMIO_ERR_INVALID_HANDLE — dataset stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — invalid arguments (R2.10)
 *   - AMIO_ERR_TIMEOUT — prefetch timed out (R5.5)
 *   - AMIO_ERR_BACKEND_FAILURE — read failed (R5.8)
 */
AMIO_API amio_status_t amio_read(amio_dataset_handle dataset, const char *var_name, int64_t timestep, const amio_bbox_t *bbox,
                                 amio_view_handle *out_view);

/**
 * @brief Query a variable's shape and record count without reading data.
 *
 * Returns the variable's per-timestep shape (the same shape amio_read()
 * validates a bbox against) and its total timestep count.  Stages NO
 * payload, so a caller can size a selective read (and its record loop)
 * without probing amio_read() at out-of-range timesteps.
 *
 * @param[in]  dataset             A read-mode dataset handle.
 * @param[in]  var_name            NUL-terminated variable identifier.
 * @param[out] out_shape           Receives the variable's shape descriptor.
 * @param[out] out_total_timesteps Receives the number of temporal records
 *                                 (1 for a non-time-varying variable).
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — dataset is NULL
 *   - AMIO_ERR_INVALID_HANDLE — dataset stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — NULL var_name/outputs or write-mode dataset
 *   - AMIO_ERR_BACKEND_FAILURE — variable absent / driver cannot describe it
 */
AMIO_API amio_status_t amio_describe(amio_dataset_handle dataset, const char *var_name, amio_shape_t *out_shape, int64_t *out_total_timesteps);

/**
 * @brief Read a variable's text attribute (e.g. "units", "calendar").
 *
 * Two-call pattern: pass @p out_buf == NULL to query the length via
 * @p out_len, then call again with a buffer of @p buf_cap bytes. The written
 * string is NUL-terminated and truncated to fit. Pass an empty or NULL
 * @p var_name to read a global attribute.
 *
 * @param[in]  dataset    A read dataset handle from amio_open_dataset().
 * @param[in]  var_name   Variable name, or ""/NULL for a global attribute.
 * @param[in]  attr_name  NUL-terminated attribute name.
 * @param[out] out_buf    Destination buffer, or NULL to size only.
 * @param[in]  buf_cap    Capacity of @p out_buf in bytes; must be non-zero when
 *                        @p out_buf is non-NULL.
 * @param[out] out_len    Receives the attribute length in bytes (excl. NUL).
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_INVALID_INPUT — NULL attr_name/out_len, a non-NULL @p out_buf with
 *     @p buf_cap == 0, or a write-mode dataset
 *   - AMIO_ERR_NULL_HANDLE / AMIO_ERR_INVALID_HANDLE — bad dataset handle
 *   - AMIO_ERR_BACKEND_FAILURE — attribute absent or backend cannot provide it
 */
AMIO_API amio_status_t amio_get_var_attribute_text(amio_dataset_handle dataset, const char *var_name, const char *attr_name, char *out_buf,
                                                   size_t buf_cap, size_t *out_len);

/**
 * @brief Read a variable's numeric attribute (e.g. "scale_factor", "add_offset").
 *
 * The stored value is converted to double regardless of its on-disk type.
 * Only the first element of a multi-valued attribute is returned. Pass an
 * empty or NULL @p var_name to read a global attribute.
 *
 * @param[in]  dataset    A read dataset handle from amio_open_dataset().
 * @param[in]  var_name   Variable name, or ""/NULL for a global attribute.
 * @param[in]  attr_name  NUL-terminated attribute name.
 * @param[out] out_value  Receives the attribute value.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_INVALID_INPUT — NULL attr_name/out_value, or a write-mode dataset
 *   - AMIO_ERR_NULL_HANDLE / AMIO_ERR_INVALID_HANDLE — bad dataset handle
 *   - AMIO_ERR_BACKEND_FAILURE — attribute absent, non-numeric, or unavailable
 */
AMIO_API amio_status_t amio_get_var_attribute_double(amio_dataset_handle dataset, const char *var_name, const char *attr_name, double *out_value);

/**
 * @brief Block until all pending writes for a dataset complete or timeout.
 *
 * @param[in] dataset     A write dataset handle.
 * @param[in] timeout_ms  Timeout in milliseconds [0, 86400000]. 0 = indefinite.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — dataset is NULL
 *   - AMIO_ERR_INVALID_HANDLE — dataset stale or wrong kind
 *   - AMIO_ERR_TIMEOUT — timeout elapsed (R6.10)
 *   - AMIO_ERR_BACKEND_FAILURE — one or more writes failed (R6.6)
 */
AMIO_API amio_status_t amio_flush(amio_dataset_handle dataset, int64_t timeout_ms);

/**
 * @brief Flush all pending writes and release dataset resources.
 *
 * @param[in] dataset  A valid dataset handle.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — dataset is NULL
 *   - AMIO_ERR_INVALID_HANDLE — dataset stale or wrong kind
 *   - AMIO_ERR_VIEWS_OUTSTANDING — unreleased read views (R5.10)
 *   - AMIO_ERR_BACKEND_FAILURE — flush or close failed (R6.6)
 */
AMIO_API amio_status_t amio_close(amio_dataset_handle dataset);

/**
 * @brief Block until an asynchronous I/O operation completes or times out.
 *
 * @param[in] io          An I/O handle from amio_write() or amio_read().
 * @param[in] timeout_ms  Timeout in milliseconds [0, 86400000]. 0 = indefinite.
 *
 * @return AMIO_OK on success (operation completed), or one of:
 *   - AMIO_ERR_NULL_HANDLE — io is NULL
 *   - AMIO_ERR_INVALID_HANDLE — io stale or wrong kind
 *   - AMIO_ERR_TIMEOUT — timeout elapsed (R3.9)
 *   - AMIO_ERR_BACKEND_FAILURE — operation failed
 */
AMIO_API amio_status_t amio_wait(amio_io_handle io, int64_t timeout_ms);

/**
 * @brief Retrieve the data pointer and byte count from a read view.
 *
 * The returned pointer is valid until amio_release_view() is called on
 * this view handle. The data resides in host (CPU) memory.
 *
 * @param[in]  view       A valid view handle from amio_read().
 * @param[out] out_data   On success, receives a pointer to the view's data.
 * @param[out] out_size   On success, receives the number of valid bytes.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — view is NULL
 *   - AMIO_ERR_INVALID_HANDLE — view stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — out_data or out_size is NULL
 */
AMIO_API amio_status_t amio_view_data(amio_view_handle view, const void **out_data, size_t *out_size);

/**
 * @brief Retrieve the shape descriptor of an outstanding read view.
 *
 * @param[in]  view       A valid view handle from amio_read().
 * @param[out] out_shape  Non-NULL pointer; receives the shape descriptor.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — view is NULL
 *   - AMIO_ERR_INVALID_HANDLE — view stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — out_shape is NULL
 */
AMIO_API amio_status_t amio_view_shape(amio_view_handle view, amio_shape_t *out_shape);

/**
 * @brief Retrieve the element type of an outstanding read view.
 *
 * The buffer returned by amio_view_data() holds elements of this type, as
 * reported by the backend for the variable that was read. Callers must consult
 * this before interpreting the payload: the byte size alone cannot distinguish
 * (for example) 32-bit integers from single-precision floats.
 *
 * @param[in]  view       A valid view handle from amio_read().
 * @param[out] out_dtype  Non-NULL pointer; receives the element type.
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — view is NULL
 *   - AMIO_ERR_INVALID_HANDLE — view stale or wrong kind
 *   - AMIO_ERR_INVALID_INPUT — out_dtype is NULL
 */
AMIO_API amio_status_t amio_view_dtype(amio_view_handle view, amio_dtype_t *out_dtype);

/**
 * @brief Release a read-side Memory_View, returning its buffer to the pool.
 *
 * Once the last reference is dropped, the buffer returns to the
 * Staging_Pool free list.
 *
 * @param[in] view  A view handle from amio_read().
 *
 * @return AMIO_OK on success, or one of:
 *   - AMIO_ERR_NULL_HANDLE — view is NULL
 *   - AMIO_ERR_INVALID_HANDLE — view stale, wrong kind, or already released
 */
AMIO_API amio_status_t amio_release_view(amio_view_handle view);

/*
 * Note: amio_strerror is declared in amio_errors.h (its natural home
 * alongside the AMIO_ERR_* enumeration). Including this umbrella header
 * transitively pulls that declaration into the consumer's translation unit.
 */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AMIO_AMIO_H */
