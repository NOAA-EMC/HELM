/**
 * @file amio_evolution.h
 * @brief Additive, versioned C99 interfaces for the AMIO step engine.
 *
 * This header preserves the legacy declarations and layouts in amio.h.
 * Descriptors are size/version tagged, use fixed-width C types, and are
 * extended only by appending fields while incrementing the descriptor version.
 */
#ifndef AMIO_AMIO_EVOLUTION_H
#define AMIO_AMIO_EVOLUTION_H

#include <stddef.h>
#include <stdint.h>

#include "amio/amio_errors.h"
#include "amio/amio_export.h"
#include "amio/amio_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 1u

typedef struct amio_context_handle_impl *amio_context_handle;
typedef struct amio_schema_handle_impl *amio_schema_handle;
typedef struct amio_step_handle_impl *amio_step_handle;
typedef struct amio_operation_handle_impl *amio_operation_handle;
typedef struct amio_step_lease_handle_impl *amio_step_lease_handle;
typedef struct amio_validation_report_handle_impl *amio_validation_report_handle;

typedef enum amio_execution_mode_e {
    AMIO_EXECUTION_PERSISTENT_DIRECT = 1,
    AMIO_EXECUTION_PERSISTENT_AGGREGATED = 2,
    AMIO_EXECUTION_MPI_STREAM = 3
} amio_execution_mode_t;

typedef enum amio_progress_mode_e { AMIO_PROGRESS_CALLER = 1, AMIO_PROGRESS_BACKGROUND = 2 } amio_progress_mode_t;

/** Effective local role reported by amio_query_context_role(). */
typedef enum amio_rank_role_e { AMIO_ROLE_INACTIVE = 0, AMIO_ROLE_PRODUCER = 1, AMIO_ROLE_WRITER = 2, AMIO_ROLE_CONSUMER = 3 } amio_rank_role_t;

typedef enum amio_evolution_format_e { AMIO_FORMAT_NETCDF4 = 1, AMIO_FORMAT_ZARR3 = 2, AMIO_FORMAT_GRIB2 = 3 } amio_evolution_format_t;

typedef enum amio_attribute_type_e {
    AMIO_ATTRIBUTE_BOOL = 1,
    AMIO_ATTRIBUTE_I64 = 2,
    AMIO_ATTRIBUTE_U64 = 3,
    AMIO_ATTRIBUTE_F64 = 4,
    AMIO_ATTRIBUTE_UTF8 = 5
} amio_attribute_type_t;

/** Input arrays and configuration bytes are copied before this call returns. */
typedef struct amio_context_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    uint32_t execution_mode;
    uint32_t progress_mode;
    uint32_t rank_count;
    uint32_t producer_count;
    uint32_t writer_count;
    uint32_t consumer_count;
    const int32_t *producers;
    const int32_t *writers;
    const int32_t *consumers;
    int64_t fortran_communicator;
    const char *canonical_configuration;
    uint64_t configuration_bytes;
    uint64_t reserved[2];
} amio_context_descriptor_t;

typedef struct amio_descriptor_header_s {
    uint32_t struct_size;
    uint32_t version;
} amio_descriptor_header_t;

/** Exclusive persistent collection location and finite local resource budgets.
 *
 * @param budget_bytes The local retained payload/snapshot byte ceiling.
 * @param budget_descriptors The local payload/snapshot descriptor ceiling.
 * @param control_budget_bytes The reserved local control/metadata byte ceiling;
 *        registered schema retention is charged here for the dataset lifetime.
 * @param control_budget_descriptors The reserved local control/metadata
 *        descriptor ceiling; each distinct registered schema consumes one.
 */
typedef struct amio_dataset_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    const char *location;
    uint64_t location_bytes;
    uint32_t format;
    uint32_t reserved32;
    uint64_t budget_bytes;
    uint64_t budget_descriptors;
    uint64_t control_budget_bytes;
    uint64_t control_budget_descriptors;
    uint64_t reserved[2];
} amio_dataset_descriptor_t;

typedef struct amio_dimension_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    const char *name;
    uint64_t name_bytes;
    uint64_t extent;
    const char *coordinate_variable;
    uint64_t coordinate_variable_bytes;
    uint64_t reserved[2];
} amio_dimension_descriptor_t;

typedef struct amio_attribute_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    const char *name;
    uint64_t name_bytes;
    uint32_t type;
    uint32_t reserved32;
    uint64_t element_count;
    const void *value;
    uint64_t value_bytes;
    uint64_t reserved[2];
} amio_attribute_descriptor_t;

typedef struct amio_variable_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    const char *name;
    uint64_t name_bytes;
    uint32_t source_dtype;
    uint32_t stored_dtype;
    uint32_t required;
    uint32_t dimension_count;
    const amio_dimension_descriptor_t *dimensions;
    uint32_t attribute_count;
    uint32_t reserved32;
    const amio_attribute_descriptor_t *attributes;
    uint64_t reserved[2];
} amio_variable_descriptor_t;

typedef struct amio_schema_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    uint64_t variable_count;
    const amio_variable_descriptor_t *variables;
    uint64_t reserved[2];
} amio_schema_descriptor_t;

/** Begin a persistent step with one previously registered immutable schema. */
typedef struct amio_step_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    uint64_t step_id;
    double model_time;
    amio_schema_handle schema;
    uint64_t reserved[2];
} amio_step_descriptor_t;

/** Rank-local box for one field in a distributed persistent NetCDF step. */
typedef struct amio_field_partition_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    const char *field;
    uint64_t field_bytes;
    uint32_t rank;
    uint32_t reserved32;
    int64_t offsets[AMIO_MAX_RANK];
    int64_t extents[AMIO_MAX_RANK];
    uint64_t reserved[2];
} amio_field_partition_descriptor_t;

/** Versioned memory description for an owned field snapshot. Device pointers are
 * accepted only by builds with an initialized compatible private runtime. */
typedef struct amio_memory_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    uint32_t memory_kind; /* 0 host, 1 device */
    uint32_t ownership;   /* 0 copy, 1 borrowed (currently unsupported) */
    uint32_t device_id;
    uint32_t rank;
    uint32_t producer_writes_synchronized;
    uint32_t reserved32;
    uint64_t element_bytes;
    uint64_t extents[AMIO_MAX_RANK];
    uint64_t strides[AMIO_MAX_RANK];
    uint64_t reserved[2];
} amio_memory_descriptor_t;

typedef enum amio_operation_outcome_e {
    AMIO_OPERATION_PENDING = 0,
    AMIO_OPERATION_SUCCEEDED = 1,
    AMIO_OPERATION_FAILED = 2
} amio_operation_outcome_t;

typedef struct amio_operation_status_s {
    uint32_t struct_size;
    uint32_t version;
    int32_t outcome;
    uint32_t milestones;
    int32_t terminal_status;
    uint32_t reserved32;
} amio_operation_status_t;

#define AMIO_EVOLUTION_ID_CAPACITY 128u
#define AMIO_EVOLUTION_COMMIT_CAPACITY 256u
typedef struct amio_catalog_cursor_s {
    uint32_t struct_size;
    uint32_t version;
    char dataset_uuid[AMIO_EVOLUTION_ID_CAPACITY];
    char generation[AMIO_EVOLUTION_COMMIT_CAPACITY];
    char next_commit[AMIO_EVOLUTION_COMMIT_CAPACITY];
    uint64_t reserved[2];
} amio_catalog_cursor_t;

typedef struct amio_step_summary_s {
    uint32_t struct_size;
    uint32_t version;
    uint64_t step_id;
    uint32_t present_field_count;
    uint32_t required_field_count;
    char schema_digest[17];
    uint64_t reserved[2];
} amio_step_summary_t;

typedef struct amio_step_selection_s {
    uint32_t struct_size;
    uint32_t version;
    uint32_t rank;
    uint32_t reserved32;
    int64_t offsets[AMIO_MAX_RANK];
    int64_t extents[AMIO_MAX_RANK];
    int64_t strides[AMIO_MAX_RANK];
    uint64_t reserved[2];
} amio_step_selection_t;

/** One contiguous producer contribution for a statically configured stream. */
typedef struct amio_stream_contribution_s {
    uint32_t struct_size;
    uint32_t version;
    uint64_t step_id;
    uint64_t sequence;
    uint32_t field_id;
    uint32_t rank;
    uint64_t element_bytes;
    uint8_t schema_digest[32];
    uint64_t offsets[AMIO_MAX_RANK];
    uint64_t extents[AMIO_MAX_RANK];
    uint64_t reserved[2];
} amio_stream_contribution_t;

/** Consumer supplies the announced step/schema before bounded receive allocation. */
typedef struct amio_stream_receive_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    uint64_t step_id;
    uint8_t schema_digest[32];
    uint32_t absent_optional_count;
    uint32_t fields_per_producer;
    const char *const *absent_optional_names;
    const uint64_t *absent_optional_name_bytes;
    uint64_t reserved[2];
} amio_stream_receive_descriptor_t;

typedef enum amio_stream_acquire_state_e {
    AMIO_STREAM_AVAILABLE = 0,
    AMIO_STREAM_SKIPPED = 1,
    AMIO_STREAM_TIMEOUT = 2,
    AMIO_STREAM_END_OF_STREAM = 3,
    AMIO_STREAM_FAILED = 4
} amio_stream_acquire_state_t;

typedef struct amio_stream_acquire_result_s {
    uint32_t struct_size;
    uint32_t version;
    int32_t state;
    int32_t failure_status;
    uint64_t skipped_first;
    uint64_t skipped_last;
    amio_step_lease_handle lease;
    uint64_t reserved[2];
} amio_stream_acquire_result_t;

typedef struct amio_stream_read_descriptor_s {
    uint32_t struct_size;
    uint32_t version;
    uint32_t producer_rank;
    uint32_t field_id;
    uint32_t rank;
    uint32_t reserved32;
    uint64_t element_bytes;
    uint64_t offsets[AMIO_MAX_RANK];
    uint64_t extents[AMIO_MAX_RANK];
    const char *logical_field_name;
    uint64_t logical_field_name_bytes;
    uint64_t reserved[2];
} amio_stream_read_descriptor_t;

enum { AMIO_OPERATION_SOURCE_RELEASED = 1u, AMIO_OPERATION_BACKEND_COMPLETE = 2u, AMIO_OPERATION_READER_VISIBLE = 4u, AMIO_OPERATION_DURABLE = 8u };

typedef enum amio_validation_check_e {
    AMIO_VALIDATE_COMPLETION_CHAIN = 1,
    AMIO_VALIDATE_REQUIRED_FIELDS = 2,
    AMIO_VALIDATE_PAYLOAD_PRESENCE = 3,
    AMIO_VALIDATE_INTEGRITY = 4,
    AMIO_VALIDATE_ORPHAN_OBJECTS = 5
} amio_validation_check_t;

typedef enum amio_validation_status_e {
    AMIO_VALIDATION_PASS = 0,
    AMIO_VALIDATION_FAIL = 1,
    AMIO_VALIDATION_UNSUPPORTED = 2
} amio_validation_status_t;

typedef struct amio_validation_request_s {
    uint32_t struct_size;
    uint32_t version;
    uint64_t check_mask;
    uint64_t cursor;
    uint32_t max_results;
    uint32_t reserved32;
    uint64_t reserved[2];
} amio_validation_request_t;

typedef struct amio_validation_result_s {
    uint32_t struct_size;
    uint32_t version;
    int32_t check;
    int32_t status;
    uint64_t object_id;
    int32_t detail_code;
    uint32_t reserved32;
} amio_validation_result_t;

/**
 * Validate one bounded page of a dataset. Every returned check has an
 * individual PASS, FAIL, or UNSUPPORTED status. A zero-capacity/null results
 * pair queries the page size; unsupported checks never count as success.
 */
AMIO_API amio_status_t amio_validate_dataset(amio_dataset_handle dataset, const amio_validation_request_t *request, amio_validation_result_t *results,
                                             size_t result_capacity, size_t *out_result_count, uint64_t *out_next_cursor);

/** Collectively create a context and duplicate its caller-owned communicator.
 * All members must provide a complete descriptor and the same usable caller
 * communicator. Once the communicator is known, descriptor and role errors
 * are agreed before context communicators are created.
 */
AMIO_API amio_status_t amio_init_ex(const amio_context_descriptor_t *descriptor, amio_context_handle *out_context);

/** Query a stable JSON capability description using size-query then fill. */
AMIO_API amio_status_t amio_query_capabilities(amio_context_handle context, char *buffer, size_t buffer_capacity, size_t *out_required_bytes);

/** Query this rank's effective role and rank within its role group; local operation, no MPI collective. */
AMIO_API amio_status_t amio_query_context_role(amio_context_handle context, uint32_t *out_role, uint32_t *out_group_rank);

/**
 * Query a bounded, process-local diagnostic snapshot as versioned JSON.
 * This operation never performs an MPI collective. Pass a null buffer and
 * zero capacity to obtain the required byte count, then provide that buffer
 * including room for the trailing NUL byte. The snapshot includes this
 * process's effective execution topology and bounded resource counters. JSON
 * API version 2 names the per-scope counter, histogram, and resource arrays;
 * the resource high-water values include retained step-read cache occupancy
 * under the prefetch-bytes dimension.
 */
AMIO_API amio_status_t amio_query_diagnostics(amio_context_handle context, char *buffer, size_t buffer_capacity, size_t *out_required_bytes);

/**
 * Collectively sum context-wide diagnostic counters and return version 2 JSON.
 * All members of the context communicator must call the fill operation in the
 * same order. A null/zero size query is local and does not enter MPI.
 */
AMIO_API amio_status_t amio_reduce_diagnostics(amio_context_handle context, char *buffer, size_t buffer_capacity, size_t *out_required_bytes);

/** Release a context handle and its duplicated communicator. */
AMIO_API amio_status_t amio_release_context(amio_context_handle context);
/** Execute up to max_callbacks of caller-driven service/progress work locally. */
AMIO_API amio_status_t amio_progress(amio_context_handle context, uint32_t max_callbacks, uint32_t *out_completed);
/** Run the dedicated aggregation service loop on a writer context. */
AMIO_API amio_status_t amio_run_service(amio_context_handle context);

/** Submit one contiguous stream field contribution; producer bytes are retained until release acknowledgment. */
AMIO_API amio_status_t amio_stream_submit(amio_context_handle context, const amio_stream_contribution_t *contribution, const void *data,
                                          size_t data_bytes);
/** Wait for the mapped consumer to release this step before producer reuse. */
AMIO_API amio_status_t amio_stream_wait_release(amio_context_handle context, uint64_t step_id);
/** Wait up to a bounded interval; OPERATION_PENDING preserves retained ownership for retry. */
AMIO_API amio_status_t amio_stream_try_wait_release(amio_context_handle context, uint64_t step_id, uint64_t timeout_milliseconds);
/** Receive the announced producer round and retain its completed step on a consumer rank. */
AMIO_API amio_status_t amio_stream_receive_step(amio_context_handle context, const amio_stream_receive_descriptor_t *descriptor);
/** Acquire the next locally received stream step; timeout is bounded and local. */
AMIO_API amio_status_t amio_stream_acquire(amio_context_handle context, uint64_t timeout_milliseconds, amio_stream_acquire_result_t *out_result);
/** Read a bounded region from a lease-backed stream field contribution. */
AMIO_API amio_status_t amio_stream_read(amio_step_lease_handle lease, const amio_stream_read_descriptor_t *descriptor, void *buffer,
                                        size_t buffer_capacity, size_t *out_bytes);
/** Release a stream view and acknowledge release to the statically mapped producers. */
AMIO_API amio_status_t amio_stream_release(amio_step_lease_handle lease);
/** Drain and send producer EOS/disconnect, or receive terminal consumer status. */
AMIO_API amio_status_t amio_stream_finish(amio_context_handle context, int32_t graceful);
/** Cancel a drained producer stream; consumers observe a FAILED terminal outcome. */
AMIO_API amio_status_t amio_stream_cancel(amio_context_handle context);

/**
 * Open a new exclusive persistent collection. NetCDF-4 direct mode supports
 * collective multi-rank writes when AMIO was built with parallel NetCDF.
 * Zarr v3 persistent writes remain single-rank.
 */
AMIO_API amio_status_t amio_open_dataset_ex(amio_context_handle context, const amio_dataset_descriptor_t *descriptor,
                                            amio_dataset_handle *out_dataset);
/** Close an evolution dataset after all steps reach a terminal state. */
AMIO_API amio_status_t amio_close_dataset_ex(amio_dataset_handle dataset);
/** Copy, validate, and register an immutable typed schema on the dataset. */
AMIO_API amio_status_t amio_define_schema(amio_dataset_handle dataset, const amio_schema_descriptor_t *descriptor, amio_schema_handle *out_schema);
/** Release the caller's schema handle; datasets retain registered schemas. */
AMIO_API amio_status_t amio_release_schema(amio_schema_handle schema);

/** Begin a step, copy a complete field snapshot, then publish it atomically. */
AMIO_API amio_status_t amio_begin_step(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor, amio_step_handle *out_step);
/**
 * Begin a multi-rank direct NetCDF step with exactly one rank-local box per
 * schema field. The call is collective. A box with any zero extent denotes
 * that this rank contributes no values for the field. Nonempty boxes must
 * tile each present field's global shape without gaps or overlap at end-step.
 * Optional fields may be omitted by every rank.
 */
AMIO_API amio_status_t amio_begin_step_partitioned(amio_dataset_handle dataset, const amio_step_descriptor_t *descriptor,
                                                   const amio_field_partition_descriptor_t *partitions, size_t partition_count,
                                                   amio_step_handle *out_step);
AMIO_API amio_status_t amio_put_field(amio_step_handle step, const char *field, size_t field_bytes, const void *data, size_t data_bytes,
                                      amio_operation_handle *out_operation);
/** Submit a contiguous synchronized host/device field using an owned snapshot. */
AMIO_API amio_status_t amio_put_field_memory(amio_step_handle step, const char *field, size_t field_bytes, const amio_memory_descriptor_t *memory,
                                             const void *data, size_t data_bytes, amio_operation_handle *out_operation);
/** Cancel an open step before payload publication; distributed steps require every rank to call collectively. */
AMIO_API amio_status_t amio_cancel_step(amio_step_handle step);
AMIO_API amio_status_t amio_end_step(amio_step_handle step);
AMIO_API amio_status_t amio_query_operation(amio_operation_handle operation, amio_operation_status_t *out_status);
AMIO_API amio_status_t amio_wait_operation(amio_operation_handle operation, uint32_t milestone, uint64_t timeout_ms);
AMIO_API amio_status_t amio_release_operation(amio_operation_handle operation);
AMIO_API amio_status_t amio_list_steps(amio_dataset_handle dataset, const amio_catalog_cursor_t *cursor, uint32_t max_results,
                                       amio_step_summary_t *results, size_t result_capacity, size_t *out_result_count,
                                       amio_catalog_cursor_t *out_next_cursor, int *out_has_more);
AMIO_API amio_status_t amio_describe_step(amio_dataset_handle dataset, uint64_t step_id, amio_step_summary_t *out_summary);
AMIO_API amio_status_t amio_read_step_field(amio_dataset_handle dataset, uint64_t step_id, const char *field, size_t field_bytes,
                                            const amio_step_selection_t *selection, void *data, size_t data_capacity, size_t *out_required_bytes);

/** Begin, submit a complete field snapshot, and publish a logical step. */

/** Drop the caller's producer-step observation reference; does not seal/cancel. */
AMIO_API amio_status_t amio_release_step_handle(amio_step_handle step);

/** Acquire a process-local lease on a published persistent step. Reads copy into caller buffers. */
AMIO_API amio_status_t amio_acquire_step_lease(amio_dataset_handle dataset, uint64_t step_id, amio_step_lease_handle *out_lease);

/** Release a consumer lease. Dataset close is rejected while any lease is active. */
AMIO_API amio_status_t amio_release_step_lease(amio_step_lease_handle lease);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AMIO_AMIO_EVOLUTION_H */
