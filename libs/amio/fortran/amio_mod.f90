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
        c_ptr, c_int, c_int32_t, c_int64_t, c_size_t, &
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

    !> Fortran equivalent of C amio_bbox_t
    type, bind(C) :: amio_bbox_t
        integer(c_int32_t) :: rank
        integer(c_int64_t) :: offsets(7)
        integer(c_int64_t) :: extents(7)
        integer(c_int64_t) :: strides(7)
    end type amio_bbox_t

    ! ---------------------------------------------------------------
    ! Interface blocks for AMIO_C_API entry points
    ! ---------------------------------------------------------------

    interface

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
