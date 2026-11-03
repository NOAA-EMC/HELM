!------------------------------------------------------------------
! amio_mod.f90
!
! Full Fortran 2003 iso_c_binding interface module for the AMIO
! asynchronous I/O library.  Provides:
!   - type(c_ptr) aliases for every opaque handle
!   - bind(C) derived types for amio_shape_t and amio_bbox_t
!   - integer parameter constants for error codes, dtypes, modes
!   - interface blocks with bind(C, name="amio_*") for every
!     AMIO_C_API entry point
!   - Convenience subroutines that hide the C void* signature
!     behind Fortran assumed-shape arrays, computing rank, extents,
!     and dtype tags via c_loc() and array intrinsics
!
! Width and signedness match the C declarations in include/amio/
! exactly.  Only iso_c_binding kinds are used.
!
! Validates: R10.4, R10.5, R1.12
!------------------------------------------------------------------
module amio_mod
    use, intrinsic :: iso_c_binding, only: &
        c_ptr, c_int, c_int8_t, c_int32_t, c_int64_t, c_size_t, &
        c_char, c_bool, c_float, c_double, c_loc
    implicit none
    private

    ! ---------------------------------------------------------------
    ! Public API surface
    ! ---------------------------------------------------------------
    public :: AMIO_MAX_RANK
    public :: AMIO_MODE_WRITE, AMIO_MODE_READ

    ! Error codes
    public :: AMIO_OK
    public :: AMIO_ERR_NULL_HANDLE
    public :: AMIO_ERR_INVALID_HANDLE
    public :: AMIO_ERR_MANIFEST_NOT_FOUND
    public :: AMIO_ERR_MANIFEST_INVALID
    public :: AMIO_ERR_ALREADY_INITIALIZED
    public :: AMIO_ERR_FINALIZE_TIMEOUT
    public :: AMIO_ERR_STAGING_BACKPRESSURE
    public :: AMIO_ERR_INVALID_BINDING
    public :: AMIO_ERR_COMM_SPLIT_FAILED
    public :: AMIO_ERR_THREADING_UNSUPPORTED
    public :: AMIO_ERR_UNKNOWN_BACKEND
    public :: AMIO_ERR_LOSSY_CODEC_FORBIDDEN
    public :: AMIO_ERR_VIEWS_OUTSTANDING
    public :: AMIO_ERR_QUEUE_FULL
    public :: AMIO_ERR_TIMEOUT
    public :: AMIO_ERR_BACKEND_FAILURE
    public :: AMIO_ERR_INVALID_INPUT
    public :: AMIO_ERR_INVALID_POLICY
    public :: AMIO_ERR_UNSUPPORTED, AMIO_ERR_BUFFER_TOO_SMALL
    public :: AMIO_ERR_SCHEMA_MISMATCH, AMIO_ERR_STEP_FAILED
    public :: AMIO_EVOLUTION_DESCRIPTOR_VERSION_1
    public :: AMIO_VALIDATE_COMPLETION_CHAIN, AMIO_VALIDATE_REQUIRED_FIELDS
    public :: AMIO_VALIDATE_PAYLOAD_PRESENCE, AMIO_VALIDATE_INTEGRITY, AMIO_VALIDATE_ORPHAN_OBJECTS
    public :: AMIO_VALIDATION_PASS, AMIO_VALIDATION_FAIL, AMIO_VALIDATION_UNSUPPORTED

    ! Data type tags
    public :: AMIO_DTYPE_F32
    public :: AMIO_DTYPE_F64
    public :: AMIO_DTYPE_I8
    public :: AMIO_DTYPE_I16
    public :: AMIO_DTYPE_I32
    public :: AMIO_DTYPE_I64
    public :: AMIO_DTYPE_U8
    public :: AMIO_DTYPE_U16
    public :: AMIO_DTYPE_U32
    public :: AMIO_DTYPE_U64

    ! Derived types
    public :: amio_write_region_t, amio_write_region
    public :: amio_shape_t
    public :: amio_bbox_t

    ! Interface procedures
    public :: amio_init
    public :: amio_finalize
    public :: amio_open_dataset
    public :: amio_close_dataset
    public :: amio_write
    public :: amio_read
    public :: amio_flush
    public :: amio_close
    public :: amio_wait
    public :: amio_view_data
    public :: amio_release_view
    public :: amio_strerror
    public :: amio_validation_request_t, amio_validation_result_t
    public :: amio_validate_dataset, amio_release_step_handle, amio_release_step_lease
    public :: amio_acquire_step_lease
    public :: amio_context_descriptor_t, amio_init_ex, amio_query_capabilities, amio_query_diagnostics
    public :: amio_query_context_role, amio_reduce_diagnostics, amio_release_context
    public :: amio_progress
    public :: amio_run_service
    public :: amio_stream_contribution_t, amio_stream_receive_descriptor_t, amio_stream_acquire_result_t
    public :: amio_stream_read_descriptor_t, amio_stream_submit, amio_stream_wait_release
    public :: amio_stream_try_wait_release
    public :: amio_stream_receive_step, amio_stream_acquire, amio_stream_read, amio_stream_release, amio_stream_finish
    public :: amio_stream_cancel
    public :: AMIO_STREAM_AVAILABLE, AMIO_STREAM_SKIPPED, AMIO_STREAM_TIMEOUT
    public :: AMIO_STREAM_END_OF_STREAM, AMIO_STREAM_FAILED
    integer(c_int32_t), parameter :: AMIO_STREAM_AVAILABLE = 0, AMIO_STREAM_SKIPPED = 1
    integer(c_int32_t), parameter :: AMIO_STREAM_TIMEOUT = 2, AMIO_STREAM_END_OF_STREAM = 3, AMIO_STREAM_FAILED = 4
    public :: AMIO_ROLE_INACTIVE, AMIO_ROLE_PRODUCER, AMIO_ROLE_WRITER, AMIO_ROLE_CONSUMER
    integer(c_int32_t), parameter :: AMIO_ROLE_INACTIVE = 0, AMIO_ROLE_PRODUCER = 1
    integer(c_int32_t), parameter :: AMIO_ROLE_WRITER = 2, AMIO_ROLE_CONSUMER = 3
    public :: AMIO_FORMAT_NETCDF4, AMIO_FORMAT_ZARR3, AMIO_FORMAT_GRIB2
    public :: AMIO_ATTRIBUTE_BOOL, AMIO_ATTRIBUTE_I64, AMIO_ATTRIBUTE_U64, AMIO_ATTRIBUTE_F64, AMIO_ATTRIBUTE_UTF8
    public :: AMIO_OPERATION_PENDING, AMIO_OPERATION_SUCCEEDED, AMIO_OPERATION_FAILED
    public :: AMIO_OPERATION_SOURCE_RELEASED, AMIO_OPERATION_BACKEND_COMPLETE
    public :: AMIO_OPERATION_READER_VISIBLE, AMIO_OPERATION_DURABLE
    public :: amio_dataset_descriptor_t, amio_dimension_descriptor_t, amio_attribute_descriptor_t
    public :: amio_variable_descriptor_t, amio_schema_descriptor_t, amio_step_descriptor_t
    public :: amio_memory_descriptor_t, amio_put_field_memory
    public :: amio_operation_status_t, amio_catalog_cursor_t, amio_step_summary_t, amio_step_selection_t
    public :: amio_open_dataset_ex, amio_close_dataset_ex, amio_define_schema, amio_release_schema
    public :: amio_begin_step, amio_put_field, amio_cancel_step, amio_end_step, amio_query_operation, amio_wait_operation
    public :: amio_release_operation, amio_list_steps, amio_describe_step, amio_read_step_field
    public :: AMIO_EXECUTION_PERSISTENT_DIRECT, AMIO_EXECUTION_PERSISTENT_AGGREGATED, AMIO_EXECUTION_MPI_STREAM
    public :: AMIO_PROGRESS_CALLER, AMIO_PROGRESS_BACKGROUND

    ! Convenience subroutines (assumed-shape array wrappers)
    public :: amio_write_f32_1d
    public :: amio_write_f32_2d
    public :: amio_write_f32_3d
    public :: amio_write_f64_1d
    public :: amio_write_f64_2d
    public :: amio_write_f64_3d
    public :: amio_write_i32_1d
    public :: amio_write_i32_2d
    public :: amio_write_i32_3d

    ! Generic interface resolving type/rank automatically
    public :: amio_write_array

    ! ---------------------------------------------------------------
    ! Constants
    ! ---------------------------------------------------------------

    !> Maximum supported rank for shape descriptors
    integer(c_int32_t), parameter :: AMIO_MAX_RANK = 7

    !> Dataset open mode constants
    integer(c_int32_t), parameter :: AMIO_MODE_WRITE = 0
    integer(c_int32_t), parameter :: AMIO_MODE_READ = 1

    ! ---------------------------------------------------------------
    ! Error codes (amio_status_t = int32_t)
    ! Values are ABI-frozen and match include/amio/amio_errors.h
    ! ---------------------------------------------------------------
    integer(c_int32_t), parameter :: AMIO_OK = 0
    integer(c_int32_t), parameter :: AMIO_ERR_NULL_HANDLE = 1
    integer(c_int32_t), parameter :: AMIO_ERR_INVALID_HANDLE = 2
    integer(c_int32_t), parameter :: AMIO_ERR_MANIFEST_NOT_FOUND = 3
    integer(c_int32_t), parameter :: AMIO_ERR_MANIFEST_INVALID = 4
    integer(c_int32_t), parameter :: AMIO_ERR_ALREADY_INITIALIZED = 5
    integer(c_int32_t), parameter :: AMIO_ERR_FINALIZE_TIMEOUT = 6
    integer(c_int32_t), parameter :: AMIO_ERR_STAGING_BACKPRESSURE = 7
    integer(c_int32_t), parameter :: AMIO_ERR_INVALID_BINDING = 8
    integer(c_int32_t), parameter :: AMIO_ERR_COMM_SPLIT_FAILED = 9
    integer(c_int32_t), parameter :: AMIO_ERR_THREADING_UNSUPPORTED = 10
    integer(c_int32_t), parameter :: AMIO_ERR_UNKNOWN_BACKEND = 11
    integer(c_int32_t), parameter :: AMIO_ERR_LOSSY_CODEC_FORBIDDEN = 12
    integer(c_int32_t), parameter :: AMIO_ERR_VIEWS_OUTSTANDING = 13
    integer(c_int32_t), parameter :: AMIO_ERR_QUEUE_FULL = 14
    integer(c_int32_t), parameter :: AMIO_ERR_TIMEOUT = 15
    integer(c_int32_t), parameter :: AMIO_ERR_BACKEND_FAILURE = 16
    integer(c_int32_t), parameter :: AMIO_ERR_INVALID_INPUT = 17
    integer(c_int32_t), parameter :: AMIO_ERR_INVALID_POLICY = 26
    integer(c_int32_t), parameter :: AMIO_ERR_UNSUPPORTED = 18
    integer(c_int32_t), parameter :: AMIO_ERR_BUFFER_TOO_SMALL = 29
    integer(c_int32_t), parameter :: AMIO_ERR_SCHEMA_MISMATCH = 20
    integer(c_int32_t), parameter :: AMIO_ERR_STEP_FAILED = 23
    integer(c_int32_t), parameter :: AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 = 1
    integer(c_int32_t), parameter :: AMIO_VALIDATE_COMPLETION_CHAIN = 1
    integer(c_int32_t), parameter :: AMIO_VALIDATE_REQUIRED_FIELDS = 2
    integer(c_int32_t), parameter :: AMIO_VALIDATE_PAYLOAD_PRESENCE = 3
    integer(c_int32_t), parameter :: AMIO_VALIDATE_INTEGRITY = 4
    integer(c_int32_t), parameter :: AMIO_VALIDATE_ORPHAN_OBJECTS = 5
    integer(c_int32_t), parameter :: AMIO_VALIDATION_PASS = 0
    integer(c_int32_t), parameter :: AMIO_VALIDATION_FAIL = 1
    integer(c_int32_t), parameter :: AMIO_VALIDATION_UNSUPPORTED = 2
    integer(c_int32_t), parameter :: AMIO_EXECUTION_PERSISTENT_DIRECT = 1
    integer(c_int32_t), parameter :: AMIO_EXECUTION_PERSISTENT_AGGREGATED = 2
    integer(c_int32_t), parameter :: AMIO_EXECUTION_MPI_STREAM = 3
    integer(c_int32_t), parameter :: AMIO_PROGRESS_CALLER = 1
    integer(c_int32_t), parameter :: AMIO_PROGRESS_BACKGROUND = 2
    integer(c_int32_t), parameter :: AMIO_FORMAT_NETCDF4 = 1, AMIO_FORMAT_ZARR3 = 2, AMIO_FORMAT_GRIB2 = 3
    integer(c_int32_t), parameter :: AMIO_ATTRIBUTE_BOOL = 1, AMIO_ATTRIBUTE_I64 = 2, AMIO_ATTRIBUTE_U64 = 3
    integer(c_int32_t), parameter :: AMIO_ATTRIBUTE_F64 = 4, AMIO_ATTRIBUTE_UTF8 = 5
    integer(c_int32_t), parameter :: AMIO_OPERATION_PENDING = 0, AMIO_OPERATION_SUCCEEDED = 1, AMIO_OPERATION_FAILED = 2
    integer(c_int32_t), parameter :: AMIO_OPERATION_SOURCE_RELEASED = 1, AMIO_OPERATION_BACKEND_COMPLETE = 2
    integer(c_int32_t), parameter :: AMIO_OPERATION_READER_VISIBLE = 4, AMIO_OPERATION_DURABLE = 8

    ! ---------------------------------------------------------------
    ! Data type tags (amio_dtype_t = enum/int in C)
    ! ---------------------------------------------------------------
    integer(c_int), parameter :: AMIO_DTYPE_F32 = 0
    integer(c_int), parameter :: AMIO_DTYPE_F64 = 1
    integer(c_int), parameter :: AMIO_DTYPE_I8 = 2
    integer(c_int), parameter :: AMIO_DTYPE_I16 = 3
    integer(c_int), parameter :: AMIO_DTYPE_I32 = 4
    integer(c_int), parameter :: AMIO_DTYPE_I64 = 5
    integer(c_int), parameter :: AMIO_DTYPE_U8 = 6
    integer(c_int), parameter :: AMIO_DTYPE_U16 = 7
    integer(c_int), parameter :: AMIO_DTYPE_U32 = 8
    integer(c_int), parameter :: AMIO_DTYPE_U64 = 9

    ! ---------------------------------------------------------------
    ! Derived types matching C structs (bind(C))
    ! ---------------------------------------------------------------

    !> Fortran equivalent of C amio_shape_t
    type, bind(C) :: amio_shape_t
        integer(c_int32_t) :: rank
        integer(c_int64_t) :: extents(7)
        integer(c_int64_t) :: strides(7)
    end type amio_shape_t

    !> Global destination in C row-major dimension order; offsets are zero-based.
    type, bind(C) :: amio_write_region_t
        integer(c_int32_t) :: rank
        integer(c_int64_t) :: global_extents(7)
        integer(c_int64_t) :: offsets(7)
    end type amio_write_region_t

    !> Fortran equivalent of C amio_bbox_t
    type, bind(C) :: amio_bbox_t
        integer(c_int32_t) :: rank
        integer(c_int64_t) :: offsets(7)
        integer(c_int64_t) :: extents(7)
        integer(c_int64_t) :: strides(7)
    end type amio_bbox_t

    type, bind(C) :: amio_validation_request_t
        integer(c_int32_t) :: struct_size, version
        integer(c_int64_t) :: check_mask, cursor
        integer(c_int32_t) :: max_results, reserved32
        integer(c_int64_t) :: reserved(2)
    end type amio_validation_request_t

    type, bind(C) :: amio_validation_result_t
        integer(c_int32_t) :: struct_size, version, check, status
        integer(c_int64_t) :: object_id
        integer(c_int32_t) :: detail_code, reserved32
    end type amio_validation_result_t

    type, bind(C) :: amio_context_descriptor_t
        integer(c_int32_t) :: struct_size, version, execution_mode, progress_mode
        integer(c_int32_t) :: rank_count, producer_count, writer_count, consumer_count
        type(c_ptr) :: producers, writers, consumers
        integer(c_int64_t) :: fortran_communicator
        type(c_ptr) :: canonical_configuration
        integer(c_int64_t) :: configuration_bytes
        integer(c_int64_t) :: reserved(2)
    end type amio_context_descriptor_t

    ! budget_bytes/descriptors bound retained local payloads. The control
    ! budgets bound local schema metadata and control descriptors; each
    ! distinct registered schema consumes one control descriptor.
    type, bind(C) :: amio_dataset_descriptor_t
        integer(c_int32_t) :: struct_size, version
        type(c_ptr) :: location
        integer(c_int64_t) :: location_bytes
        integer(c_int32_t) :: format, reserved32
        integer(c_int64_t) :: budget_bytes, budget_descriptors, control_budget_bytes, control_budget_descriptors
        integer(c_int64_t) :: reserved(2)
    end type amio_dataset_descriptor_t

    type, bind(C) :: amio_dimension_descriptor_t
        integer(c_int32_t) :: struct_size, version
        type(c_ptr) :: name
        integer(c_int64_t) :: name_bytes, extent
        type(c_ptr) :: coordinate_variable
        integer(c_int64_t) :: coordinate_variable_bytes, reserved(2)
    end type amio_dimension_descriptor_t

    type, bind(C) :: amio_attribute_descriptor_t
        integer(c_int32_t) :: struct_size, version
        type(c_ptr) :: name
        integer(c_int64_t) :: name_bytes
        integer(c_int32_t) :: attribute_type, reserved32
        integer(c_int64_t) :: element_count
        type(c_ptr) :: value
        integer(c_int64_t) :: value_bytes, reserved(2)
    end type amio_attribute_descriptor_t

    type, bind(C) :: amio_variable_descriptor_t
        integer(c_int32_t) :: struct_size, version
        type(c_ptr) :: name
        integer(c_int64_t) :: name_bytes
        integer(c_int32_t) :: source_dtype, stored_dtype, required, dimension_count
        type(c_ptr) :: dimensions
        integer(c_int32_t) :: attribute_count, reserved32
        type(c_ptr) :: attributes
        integer(c_int64_t) :: reserved(2)
    end type amio_variable_descriptor_t

    type, bind(C) :: amio_schema_descriptor_t
        integer(c_int32_t) :: struct_size, version
        integer(c_int64_t) :: variable_count
        type(c_ptr) :: variables
        integer(c_int64_t) :: reserved(2)
    end type amio_schema_descriptor_t

    type, bind(C) :: amio_step_descriptor_t
        integer(c_int32_t) :: struct_size, version
        integer(c_int64_t) :: step_id
        real(c_double) :: model_time
        type(c_ptr) :: schema
        integer(c_int64_t) :: reserved(2)
    end type amio_step_descriptor_t

    type, bind(C) :: amio_memory_descriptor_t
        integer(c_int32_t) :: struct_size, version, memory_kind, ownership
        integer(c_int32_t) :: device_id, rank, producer_writes_synchronized, reserved32
        integer(c_int64_t) :: element_bytes
        integer(c_int64_t) :: extents(7), strides(7), reserved(2)
    end type amio_memory_descriptor_t

    type, bind(C) :: amio_operation_status_t
        integer(c_int32_t) :: struct_size, version, outcome, milestones, terminal_status, reserved32
    end type amio_operation_status_t

    type, bind(C) :: amio_catalog_cursor_t
        integer(c_int32_t) :: struct_size, version
        character(c_char) :: dataset_uuid(128), generation(256), next_commit(256)
        integer(c_int64_t) :: reserved(2)
    end type amio_catalog_cursor_t

    type, bind(C) :: amio_step_summary_t
        integer(c_int32_t) :: struct_size, version
        integer(c_int64_t) :: step_id
        integer(c_int32_t) :: present_field_count, required_field_count
        character(c_char) :: schema_digest(17)
        integer(c_int64_t) :: reserved(2)
    end type amio_step_summary_t

    type, bind(C) :: amio_step_selection_t
        integer(c_int32_t) :: struct_size, version, rank, reserved32
        integer(c_int64_t) :: offsets(7), extents(7), strides(7), reserved(2)
    end type amio_step_selection_t

    type, bind(C) :: amio_stream_contribution_t
        integer(c_int32_t) :: struct_size, version
        integer(c_int64_t) :: step_id, sequence
        integer(c_int32_t) :: field_id, rank
        integer(c_int64_t) :: element_bytes
        integer(c_int8_t) :: schema_digest(32)
        integer(c_int64_t) :: offsets(7), extents(7), reserved(2)
    end type amio_stream_contribution_t

    type, bind(C) :: amio_stream_receive_descriptor_t
        integer(c_int32_t) :: struct_size, version
        integer(c_int64_t) :: step_id
        integer(c_int8_t) :: schema_digest(32)
        integer(c_int32_t) :: absent_optional_count, fields_per_producer
        type(c_ptr) :: absent_optional_names, absent_optional_name_bytes
        integer(c_int64_t) :: reserved(2)
    end type amio_stream_receive_descriptor_t

    type, bind(C) :: amio_stream_acquire_result_t
        integer(c_int32_t) :: struct_size, version, state, failure_status
        integer(c_int64_t) :: skipped_first, skipped_last
        type(c_ptr) :: lease
        integer(c_int64_t) :: reserved(2)
    end type amio_stream_acquire_result_t

    type, bind(C) :: amio_stream_read_descriptor_t
        integer(c_int32_t) :: struct_size, version, producer_rank, field_id, rank, reserved32
        integer(c_int64_t) :: element_bytes, offsets(7), extents(7)
        type(c_ptr) :: logical_field_name
        integer(c_int64_t) :: logical_field_name_bytes, reserved(2)
    end type amio_stream_read_descriptor_t

    ! ---------------------------------------------------------------
    ! Interface blocks for AMIO_C_API entry points
    ! ---------------------------------------------------------------

    interface

        function amio_open_dataset_ex(context, descriptor, out_dataset) result(status) bind(C, name="amio_open_dataset_ex")
            import :: c_ptr, c_int32_t, amio_dataset_descriptor_t
            type(c_ptr), value, intent(in) :: context
            type(amio_dataset_descriptor_t), intent(in) :: descriptor
            type(c_ptr), intent(out) :: out_dataset
            integer(c_int32_t) :: status
        end function amio_open_dataset_ex

        function amio_close_dataset_ex(dataset) result(status) bind(C, name="amio_close_dataset_ex")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: dataset
            integer(c_int32_t) :: status
        end function amio_close_dataset_ex

        function amio_define_schema(dataset, descriptor, out_schema) result(status) bind(C, name="amio_define_schema")
            import :: c_ptr, c_int32_t, amio_schema_descriptor_t
            type(c_ptr), value, intent(in) :: dataset
            type(amio_schema_descriptor_t), intent(in) :: descriptor
            type(c_ptr), intent(out) :: out_schema
            integer(c_int32_t) :: status
        end function amio_define_schema

        function amio_release_schema(schema) result(status) bind(C, name="amio_release_schema")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: schema
            integer(c_int32_t) :: status
        end function amio_release_schema

        function amio_begin_step(dataset, descriptor, out_step) result(status) bind(C, name="amio_begin_step")
            import :: c_ptr, c_int32_t, amio_step_descriptor_t
            type(c_ptr), value, intent(in) :: dataset
            type(amio_step_descriptor_t), intent(in) :: descriptor
            type(c_ptr), intent(out) :: out_step
            integer(c_int32_t) :: status
        end function amio_begin_step

        function amio_put_field(step, field, field_bytes, data, data_bytes, out_operation) result(status) &
            bind(C, name="amio_put_field")
            import :: c_ptr, c_size_t, c_int32_t
            type(c_ptr), value, intent(in) :: step, field, data
            integer(c_size_t), value, intent(in) :: field_bytes, data_bytes
            type(c_ptr), intent(out) :: out_operation
            integer(c_int32_t) :: status
        end function amio_put_field

        function amio_put_field_memory(step, field, field_bytes, memory, data, data_bytes, out_operation) result(status) &
            bind(C, name="amio_put_field_memory")
            import :: c_ptr, c_size_t, c_int32_t, amio_memory_descriptor_t
            type(c_ptr), value, intent(in) :: step, field, data
            integer(c_size_t), value, intent(in) :: field_bytes, data_bytes
            type(amio_memory_descriptor_t), intent(in) :: memory
            type(c_ptr), intent(out) :: out_operation
            integer(c_int32_t) :: status
        end function amio_put_field_memory

        function amio_cancel_step(step) result(status) bind(C, name="amio_cancel_step")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: step
            integer(c_int32_t) :: status
        end function amio_cancel_step

        function amio_end_step(step) result(status) bind(C, name="amio_end_step")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: step
            integer(c_int32_t) :: status
        end function amio_end_step

        function amio_query_operation(operation, out_status) result(status) bind(C, name="amio_query_operation")
            import :: c_ptr, c_int32_t, amio_operation_status_t
            type(c_ptr), value, intent(in) :: operation
            type(amio_operation_status_t), intent(out) :: out_status
            integer(c_int32_t) :: status
        end function amio_query_operation

        function amio_wait_operation(operation, milestone, timeout_ms) result(status) bind(C, name="amio_wait_operation")
            import :: c_ptr, c_int32_t, c_int64_t
            type(c_ptr), value, intent(in) :: operation
            integer(c_int32_t), value, intent(in) :: milestone
            integer(c_int64_t), value, intent(in) :: timeout_ms
            integer(c_int32_t) :: status
        end function amio_wait_operation

        function amio_release_operation(operation) result(status) bind(C, name="amio_release_operation")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: operation
            integer(c_int32_t) :: status
        end function amio_release_operation

        function amio_list_steps(dataset, cursor, max_results, results, result_capacity, out_result_count, &
                                 out_next_cursor, out_has_more) result(status) bind(C, name="amio_list_steps")
            import :: c_ptr, c_int, c_int32_t, c_size_t, amio_catalog_cursor_t
            type(c_ptr), value, intent(in) :: dataset, cursor, results
            integer(c_int32_t), value, intent(in) :: max_results
            integer(c_size_t), value, intent(in) :: result_capacity
            integer(c_size_t), intent(out) :: out_result_count
            type(amio_catalog_cursor_t), intent(out) :: out_next_cursor
            integer(c_int), intent(out) :: out_has_more
            integer(c_int32_t) :: status
        end function amio_list_steps

        function amio_describe_step(dataset, step_id, out_summary) result(status) bind(C, name="amio_describe_step")
            import :: c_ptr, c_int64_t, c_int32_t, amio_step_summary_t
            type(c_ptr), value, intent(in) :: dataset
            integer(c_int64_t), value, intent(in) :: step_id
            type(amio_step_summary_t), intent(out) :: out_summary
            integer(c_int32_t) :: status
        end function amio_describe_step

        function amio_read_step_field(dataset, step_id, field, field_bytes, selection, data, data_capacity, &
                                      out_required_bytes) result(status) bind(C, name="amio_read_step_field")
            import :: c_ptr, c_int64_t, c_size_t, c_int32_t
            type(c_ptr), value, intent(in) :: dataset, field, selection, data
            integer(c_int64_t), value, intent(in) :: step_id
            integer(c_size_t), value, intent(in) :: field_bytes, data_capacity
            integer(c_size_t), intent(out) :: out_required_bytes
            integer(c_int32_t) :: status
        end function amio_read_step_field

        function amio_init_ex(descriptor, out_context) result(status) bind(C, name="amio_init_ex")
            import :: c_ptr, c_int32_t, amio_context_descriptor_t
            type(amio_context_descriptor_t), intent(in) :: descriptor
            type(c_ptr), intent(out) :: out_context
            integer(c_int32_t) :: status
        end function amio_init_ex

        function amio_query_capabilities(context, buffer, buffer_capacity, out_required_bytes) result(status) &
            bind(C, name="amio_query_capabilities")
            import :: c_ptr, c_size_t, c_int32_t
            type(c_ptr), value, intent(in) :: context, buffer
            integer(c_size_t), value, intent(in) :: buffer_capacity
            integer(c_size_t), intent(out) :: out_required_bytes
            integer(c_int32_t) :: status
        end function amio_query_capabilities

        !> Query this rank's effective role and its local role-group rank; no collective is entered.
        function amio_query_context_role(context, out_role, out_group_rank) result(status) &
            bind(C, name="amio_query_context_role")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int32_t), intent(out) :: out_role, out_group_rank
            integer(c_int32_t) :: status
        end function amio_query_context_role

        !> Return a bounded local diagnostics JSON snapshot with effective context topology.
        !! No MPI collective is entered.
        !! C signature:
        !!   amio_status_t amio_query_diagnostics(amio_context_handle, char *, size_t, size_t *);
        function amio_query_diagnostics(context, buffer, buffer_capacity, out_required_bytes) result(status) &
            bind(C, name="amio_query_diagnostics")
            import :: c_ptr, c_size_t, c_int32_t
            type(c_ptr), value, intent(in) :: context, buffer
            integer(c_size_t), value, intent(in) :: buffer_capacity
            integer(c_size_t), intent(out) :: out_required_bytes
            integer(c_int32_t) :: status
        end function amio_query_diagnostics

        !> Collectively sum context counters. Only the size query is local.
        function amio_reduce_diagnostics(context, buffer, buffer_capacity, out_required_bytes) result(status) &
            bind(C, name="amio_reduce_diagnostics")
            import :: c_ptr, c_size_t, c_int32_t
            type(c_ptr), value, intent(in) :: context, buffer
            integer(c_size_t), value, intent(in) :: buffer_capacity
            integer(c_size_t), intent(out) :: out_required_bytes
            integer(c_int32_t) :: status
        end function amio_reduce_diagnostics

        !> Run a bounded quantum of caller-driven progress on this rank only.
        function amio_progress(context, max_callbacks, out_completed) result(status) bind(C, name="amio_progress")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int32_t), value, intent(in) :: max_callbacks
            integer(c_int32_t), intent(out) :: out_completed
            integer(c_int32_t) :: status
        end function amio_progress

        !> Enter the dedicated service operation on a configured aggregation writer.
        function amio_run_service(context) result(status) bind(C, name="amio_run_service")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int32_t) :: status
        end function amio_run_service

        function amio_stream_submit(context, contribution, data, data_bytes) result(status) &
            bind(C, name="amio_stream_submit")
            import :: c_ptr, c_size_t, c_int32_t, amio_stream_contribution_t
            type(c_ptr), value, intent(in) :: context, data
            type(amio_stream_contribution_t), intent(in) :: contribution
            integer(c_size_t), value, intent(in) :: data_bytes
            integer(c_int32_t) :: status
        end function amio_stream_submit

        function amio_stream_wait_release(context, step_id) result(status) bind(C, name="amio_stream_wait_release")
            import :: c_ptr, c_int64_t, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int64_t), value, intent(in) :: step_id
            integer(c_int32_t) :: status
        end function amio_stream_wait_release

        function amio_stream_try_wait_release(context, step_id, timeout_milliseconds) result(status) &
            bind(C, name="amio_stream_try_wait_release")
            import :: c_ptr, c_int64_t, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int64_t), value, intent(in) :: step_id, timeout_milliseconds
            integer(c_int32_t) :: status
        end function amio_stream_try_wait_release

        function amio_stream_receive_step(context, descriptor) result(status) bind(C, name="amio_stream_receive_step")
            import :: c_ptr, c_int32_t, amio_stream_receive_descriptor_t
            type(c_ptr), value, intent(in) :: context
            type(amio_stream_receive_descriptor_t), intent(in) :: descriptor
            integer(c_int32_t) :: status
        end function amio_stream_receive_step

        function amio_stream_acquire(context, timeout_milliseconds, out_result) result(status) &
            bind(C, name="amio_stream_acquire")
            import :: c_ptr, c_int64_t, c_int32_t, amio_stream_acquire_result_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int64_t), value, intent(in) :: timeout_milliseconds
            type(amio_stream_acquire_result_t), intent(out) :: out_result
            integer(c_int32_t) :: status
        end function amio_stream_acquire

        function amio_stream_read(lease, descriptor, buffer, buffer_capacity, out_bytes) result(status) &
            bind(C, name="amio_stream_read")
            import :: c_ptr, c_size_t, c_int32_t, amio_stream_read_descriptor_t
            type(c_ptr), value, intent(in) :: lease, buffer
            type(amio_stream_read_descriptor_t), intent(in) :: descriptor
            integer(c_size_t), value, intent(in) :: buffer_capacity
            integer(c_size_t), intent(out) :: out_bytes
            integer(c_int32_t) :: status
        end function amio_stream_read

        function amio_stream_release(lease) result(status) bind(C, name="amio_stream_release")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: lease
            integer(c_int32_t) :: status
        end function amio_stream_release

        function amio_stream_finish(context, graceful) result(status) bind(C, name="amio_stream_finish")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int32_t), value, intent(in) :: graceful
            integer(c_int32_t) :: status
        end function amio_stream_finish

        function amio_stream_cancel(context) result(status) bind(C, name="amio_stream_cancel")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int32_t) :: status
        end function amio_stream_cancel

        function amio_release_context(context) result(status) bind(C, name="amio_release_context")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: context
            integer(c_int32_t) :: status
        end function amio_release_context

        function amio_validate_dataset(dataset, request, results, result_capacity, out_result_count, &
                                       out_next_cursor) result(status) &
            bind(C, name="amio_validate_dataset")
            import :: c_ptr, c_size_t, c_int64_t, c_int32_t
            import :: amio_validation_request_t
            type(c_ptr), value, intent(in) :: dataset
            type(amio_validation_request_t), intent(in) :: request
            type(c_ptr), value, intent(in) :: results
            integer(c_size_t), value, intent(in) :: result_capacity
            integer(c_size_t), intent(out) :: out_result_count
            integer(c_int64_t), intent(out) :: out_next_cursor
            integer(c_int32_t) :: status
        end function amio_validate_dataset

        function amio_release_step_handle(step) result(status) bind(C, name="amio_release_step_handle")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: step
            integer(c_int32_t) :: status
        end function amio_release_step_handle

        function amio_acquire_step_lease(dataset, step_id, out_lease) result(status) &
            bind(C, name="amio_acquire_step_lease")
            import :: c_ptr, c_int32_t, c_int64_t
            type(c_ptr), value, intent(in) :: dataset
            integer(c_int64_t), value, intent(in) :: step_id
            type(c_ptr), intent(out) :: out_lease
            integer(c_int32_t) :: status
        end function amio_acquire_step_lease

        function amio_release_step_lease(lease) result(status) bind(C, name="amio_release_step_lease")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: lease
            integer(c_int32_t) :: status
        end function amio_release_step_lease

        !> amio_init -- create an AMIO_Core context from a manifest.
        !! C signature:
        !!   amio_status_t amio_init(const char *manifest_path,
        !!                           amio_core_handle *out_core);
        function amio_init(manifest_path, out_core) result(status) &
            bind(C, name="amio_init")
            import :: c_char, c_ptr, c_int32_t
            character(kind=c_char), intent(in) :: manifest_path(*)
            type(c_ptr), intent(out) :: out_core
            integer(c_int32_t) :: status
        end function amio_init

        !> amio_finalize -- drain pending I/O and release core context.
        !! C signature:
        !!   amio_status_t amio_finalize(amio_core_handle core);
        function amio_finalize(core) result(status) &
            bind(C, name="amio_finalize")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: core
            integer(c_int32_t) :: status
        end function amio_finalize

        !> amio_open_dataset -- open a dataset for reading or writing.
        !! C signature:
        !!   amio_status_t amio_open_dataset(amio_core_handle core,
        !!                                   const char *config_path,
        !!                                   int32_t mode,
        !!                                   amio_dataset_handle *out_dataset);
        function amio_open_dataset(core, config_path, mode, out_dataset) &
            result(status) bind(C, name="amio_open_dataset")
            import :: c_ptr, c_char, c_int32_t
            type(c_ptr), value, intent(in) :: core
            character(kind=c_char), intent(in) :: config_path(*)
            integer(c_int32_t), value, intent(in) :: mode
            type(c_ptr), intent(out) :: out_dataset
            integer(c_int32_t) :: status
        end function amio_open_dataset

        !> amio_close_dataset -- flush and close a dataset.
        !! C signature:
        !!   amio_status_t amio_close_dataset(amio_dataset_handle dataset);
        function amio_close_dataset(dataset) result(status) &
            bind(C, name="amio_close_dataset")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: dataset
            integer(c_int32_t) :: status
        end function amio_close_dataset

        !> amio_write -- snapshot host data and enqueue async write.
        !! C signature:
        !!   amio_status_t amio_write(amio_dataset_handle dataset,
        !!                            const char *var_name,
        !!                            const void *host_data,
        !!                            amio_dtype_t dtype,
        !!                            const amio_shape_t *shape,
        !!                            amio_io_handle *out_io);
        function amio_write(dataset, var_name, host_data, dtype, shape, &
                            out_io) result(status) bind(C, name="amio_write")
            import :: c_ptr, c_char, c_int, c_int32_t
            import :: amio_shape_t
            type(c_ptr), value, intent(in) :: dataset
            character(kind=c_char), intent(in) :: var_name(*)
            type(c_ptr), value, intent(in) :: host_data
            integer(c_int), value, intent(in) :: dtype
            type(amio_shape_t), intent(in) :: shape
            type(c_ptr), intent(out) :: out_io
            integer(c_int32_t) :: status
        end function amio_write

        function amio_write_region(dataset, var_name, host_data, dtype, shape, &
                                   region, out_io) result(status) bind(C, name="amio_write_region")
            import :: c_ptr, c_char, c_int, c_int32_t, amio_shape_t, amio_write_region_t
            type(c_ptr), value, intent(in) :: dataset
            character(kind=c_char), intent(in) :: var_name(*)
            type(c_ptr), value, intent(in) :: host_data
            integer(c_int), value, intent(in) :: dtype
            type(amio_shape_t), intent(in) :: shape
            type(amio_write_region_t), intent(in) :: region
            type(c_ptr), intent(out) :: out_io
            integer(c_int32_t) :: status
        end function amio_write_region

        !> amio_read -- return a Memory_View for a prefetched timestep.
        !! C signature:
        !!   amio_status_t amio_read(amio_dataset_handle dataset,
        !!                           const char *var_name,
        !!                           int64_t timestep,
        !!                           const amio_bbox_t *bbox,
        !!                           amio_view_handle *out_view);
        function amio_read(dataset, var_name, timestep, bbox, out_view) &
            result(status) bind(C, name="amio_read")
            import :: c_ptr, c_char, c_int64_t, c_int32_t
            import :: amio_bbox_t
            type(c_ptr), value, intent(in) :: dataset
            character(kind=c_char), intent(in) :: var_name(*)
            integer(c_int64_t), value, intent(in) :: timestep
            type(c_ptr), value, intent(in) :: bbox
            type(c_ptr), intent(out) :: out_view
            integer(c_int32_t) :: status
        end function amio_read

        !> amio_flush -- block until all pending writes complete.
        !! C signature:
        !!   amio_status_t amio_flush(amio_dataset_handle dataset,
        !!                            int64_t timeout_ms);
        function amio_flush(dataset, timeout_ms) result(status) &
            bind(C, name="amio_flush")
            import :: c_ptr, c_int64_t, c_int32_t
            type(c_ptr), value, intent(in) :: dataset
            integer(c_int64_t), value, intent(in) :: timeout_ms
            integer(c_int32_t) :: status
        end function amio_flush

        !> amio_close -- flush and release dataset resources.
        !! C signature:
        !!   amio_status_t amio_close(amio_dataset_handle dataset);
        function amio_close(dataset) result(status) &
            bind(C, name="amio_close")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: dataset
            integer(c_int32_t) :: status
        end function amio_close

        !> amio_wait -- block until an async I/O operation completes.
        !! C signature:
        !!   amio_status_t amio_wait(amio_io_handle io,
        !!                           int64_t timeout_ms);
        function amio_wait(io, timeout_ms) result(status) &
            bind(C, name="amio_wait")
            import :: c_ptr, c_int64_t, c_int32_t
            type(c_ptr), value, intent(in) :: io
            integer(c_int64_t), value, intent(in) :: timeout_ms
            integer(c_int32_t) :: status
        end function amio_wait

        !> amio_view_data -- retrieve data pointer and size from a view.
        !! C signature:
        !!   amio_status_t amio_view_data(amio_view_handle view,
        !!                                const void **out_data,
        !!                                size_t *out_size);
        function amio_view_data(view, out_data, out_size) result(status) &
            bind(C, name="amio_view_data")
            import :: c_ptr, c_size_t, c_int32_t
            type(c_ptr), value, intent(in) :: view
            type(c_ptr), intent(out) :: out_data
            integer(c_size_t), intent(out) :: out_size
            integer(c_int32_t) :: status
        end function amio_view_data

        !> amio_release_view -- release a read-side Memory_View.
        !! C signature:
        !!   amio_status_t amio_release_view(amio_view_handle view);
        function amio_release_view(view) result(status) &
            bind(C, name="amio_release_view")
            import :: c_ptr, c_int32_t
            type(c_ptr), value, intent(in) :: view
            integer(c_int32_t) :: status
        end function amio_release_view

        !> amio_strerror -- map error code to human-readable string.
        !! C signature:
        !!   const char *amio_strerror(int err);
        !! Returns a c_ptr to a null-terminated C string.
        function amio_strerror(err) result(msg_ptr) &
            bind(C, name="amio_strerror")
            import :: c_int, c_ptr
            integer(c_int), value, intent(in) :: err
            type(c_ptr) :: msg_ptr
        end function amio_strerror

    end interface

    ! ---------------------------------------------------------------
    ! Generic interface: amio_write_array
    !
    ! Resolves to the correct specific subroutine based on the
    ! element type (real(c_float), real(c_double), integer(c_int32_t))
    ! and rank (1D, 2D, 3D) of the passed assumed-shape array.
    ! ---------------------------------------------------------------
    interface amio_write_array
        module procedure amio_write_f32_1d
        module procedure amio_write_f32_2d
        module procedure amio_write_f32_3d
        module procedure amio_write_f64_1d
        module procedure amio_write_f64_2d
        module procedure amio_write_f64_3d
        module procedure amio_write_i32_1d
        module procedure amio_write_i32_2d
        module procedure amio_write_i32_3d
    end interface amio_write_array

contains

    ! ===============================================================
    ! Convenience subroutines
    !
    ! Each subroutine:
    !   1. Uses c_loc() to get the data pointer from the Fortran array
    !   2. Computes rank from the array's intrinsic rank
    !   3. Computes extents from size(array, dim=d) for each dimension
    !   4. Sets the appropriate AMIO_DTYPE_* tag
    !   5. Builds an amio_shape_t with strides=0 (contiguous)
    !   6. Calls the raw amio_write interface
    ! ===============================================================

    !> Write a 1D single-precision (real32) array.
    subroutine amio_write_f32_1d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        real(c_float), intent(in), target       :: array(:)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 1_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_F32, shp, out_io)
    end subroutine amio_write_f32_1d

    !> Write a 2D single-precision (real32) array.
    subroutine amio_write_f32_2d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        real(c_float), intent(in), target       :: array(:, :)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 2_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)
        shp%extents(2) = int(size(array, 2), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1), lbound(array, 2)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_F32, shp, out_io)
    end subroutine amio_write_f32_2d

    !> Write a 3D single-precision (real32) array.
    subroutine amio_write_f32_3d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        real(c_float), intent(in), target       :: array(:, :, :)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 3_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)
        shp%extents(2) = int(size(array, 2), c_int64_t)
        shp%extents(3) = int(size(array, 3), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1), lbound(array, 2), &
                               lbound(array, 3)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_F32, shp, out_io)
    end subroutine amio_write_f32_3d

    !> Write a 1D double-precision (real64) array.
    subroutine amio_write_f64_1d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        real(c_double), intent(in), target      :: array(:)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 1_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_F64, shp, out_io)
    end subroutine amio_write_f64_1d

    !> Write a 2D double-precision (real64) array.
    subroutine amio_write_f64_2d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        real(c_double), intent(in), target      :: array(:, :)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 2_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)
        shp%extents(2) = int(size(array, 2), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1), lbound(array, 2)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_F64, shp, out_io)
    end subroutine amio_write_f64_2d

    !> Write a 3D double-precision (real64) array.
    subroutine amio_write_f64_3d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        real(c_double), intent(in), target      :: array(:, :, :)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 3_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)
        shp%extents(2) = int(size(array, 2), c_int64_t)
        shp%extents(3) = int(size(array, 3), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1), lbound(array, 2), &
                               lbound(array, 3)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_F64, shp, out_io)
    end subroutine amio_write_f64_3d

    !> Write a 1D 32-bit integer array.
    subroutine amio_write_i32_1d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        integer(c_int32_t), intent(in), target  :: array(:)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 1_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_I32, shp, out_io)
    end subroutine amio_write_i32_1d

    !> Write a 2D 32-bit integer array.
    subroutine amio_write_i32_2d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        integer(c_int32_t), intent(in), target  :: array(:, :)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 2_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)
        shp%extents(2) = int(size(array, 2), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1), lbound(array, 2)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_I32, shp, out_io)
    end subroutine amio_write_i32_2d

    !> Write a 3D 32-bit integer array.
    subroutine amio_write_i32_3d(dataset, var_name, array, out_io, status)
        type(c_ptr), value, intent(in)          :: dataset
        character(kind=c_char), intent(in)      :: var_name(*)
        integer(c_int32_t), intent(in), target  :: array(:, :, :)
        type(c_ptr), intent(out)                :: out_io
        integer(c_int32_t), intent(out)         :: status
        type(amio_shape_t) :: shp
        type(c_ptr)        :: data_ptr

        shp%rank = 3_c_int32_t
        shp%extents = 0_c_int64_t
        shp%strides = 0_c_int64_t
        shp%extents(1) = int(size(array, 1), c_int64_t)
        shp%extents(2) = int(size(array, 2), c_int64_t)
        shp%extents(3) = int(size(array, 3), c_int64_t)

        data_ptr = c_loc(array(lbound(array, 1), lbound(array, 2), &
                               lbound(array, 3)))
        status = amio_write(dataset, var_name, data_ptr, AMIO_DTYPE_I32, shp, out_io)
    end subroutine amio_write_i32_3d

end module amio_mod
