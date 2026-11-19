!> @file example_region_batch_write.f90
!> @brief Example: schema-prepared batched NetCDF-4 region writes from Fortran.
!>
!> Two MPI ranks open one shared NetCDF-4 dataset, declare a fixed global
!> schema with amio_prepare_write_schema(), and submit one ordered batch of
!> region writes through amio_write_region_batch() (see
!> docs/AMIO-parallel-writes.md).  Each field is one-dimensional here, so the
!> Fortran payload maps directly onto the C row-major convention without
!> dimension reordering.  Rank r owns global elements [4r, 4r+3] of both
!> fields; the batch completes through amio_wait() and amio_flush().
!>
!> Run (via CTest):
!>   mpiexec -n 2 ./example_region_batch_write_f90

program example_region_batch_write
    use, intrinsic :: iso_c_binding, only: &
        c_char, c_int, c_int32_t, c_int64_t, c_size_t, c_sizeof, c_float, c_ptr, c_null_ptr, c_null_char, c_loc
    use mpi
    use amio_mod
    implicit none

    integer, parameter :: PER_RANK = 4       ! elements written by every rank
    character(len=*), parameter :: MANIFEST = 'examples/manifests/netcdf4_region_batch_manifest.yaml'

    type(amio_write_variable_t), target :: schema(2)
    type(amio_region_write_t) :: writes(2)
    type(amio_shape_t), target :: local_shape
    type(amio_write_region_t), target :: local_region(2)
    character(kind=c_char), target :: temp_name(12), hum_name(12)
    real(c_float), target :: temperature(PER_RANK), humidity(PER_RANK)
    type(c_ptr) :: core, dataset, io
    integer :: rank, nranks, ierr, provided, i
    integer(c_int32_t) :: status
    integer(c_int64_t) :: timeout_ms

    call MPI_Init_thread(MPI_THREAD_MULTIPLE, provided, ierr)
    if (ierr /= MPI_SUCCESS .or. provided < MPI_THREAD_MULTIPLE) error stop 'MPI_THREAD_MULTIPLE is required'
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
    if (ierr /= MPI_SUCCESS) error stop 'MPI_Comm_rank/size failed'
    if (nranks /= 2) error stop 'example expects exactly two MPI ranks'

    ! Verify the Fortran derived type matches the C amio_write_variable_t ABI (4+4+8+4+4+8*7+8*2 bytes).
    if (c_sizeof(schema(1)) /= 96_c_size_t) error stop 'amio_write_variable_t ABI size mismatch'

    call set_c_string(temp_name, 'temperature')
    call set_c_string(hum_name, 'humidity')

    ! Step 1: Initialize AMIO and collectively open the shared NetCDF-4 output.
    core = c_null_ptr
    status = amio_init(MANIFEST//c_null_char, core)
    call check_status(status, 'amio_init')
    dataset = c_null_ptr
    status = amio_open_dataset(core, MANIFEST//c_null_char, AMIO_MODE_WRITE, dataset)
    call check_status(status, 'amio_open_dataset')

    ! Step 2: Prepare the fixed global schema before the first write.  Every rank
    ! provides the same ordered variable definitions; writes_started is set only
    ! collectively inside amio_write_region_batch.
    do i = 1, 2
        schema(i)%struct_size = int(c_sizeof(schema(i)), c_int32_t)
        schema(i)%version = AMIO_WRITE_VARIABLE_VERSION
        if (i == 1) then
            schema(i)%name = c_loc(temp_name(1))
        else
            schema(i)%name = c_loc(hum_name(1))
        end if
        schema(i)%dtype = AMIO_DTYPE_F32
        schema(i)%rank = 1
        schema(i)%global_extents = 0_c_int64_t
        schema(i)%global_extents(1) = int(nranks * PER_RANK, c_int64_t)
        schema(i)%reserved = 0_c_int64_t
    end do
    status = amio_prepare_write_schema(dataset, schema, int(size(schema, kind=c_size_t), c_size_t))
    call check_status(status, 'amio_prepare_write_schema')

    ! Step 3: Fill rank-local payloads (contiguous, row-major compatible).
    do i = 1, PER_RANK
        temperature(i) = 300.0_c_float + real(PER_RANK * rank + i, c_float)
        humidity(i) = real(PER_RANK * rank + i, c_float)
    end do

    ! Step 4: Submit one ordered batch covering both fields in schema order.
    local_shape%rank = 1
    local_shape%extents = 0_c_int64_t
    local_shape%strides = 0_c_int64_t
    local_shape%extents(1) = int(PER_RANK, c_int64_t)
    local_shape%strides(1) = 1_c_int64_t
    do i = 1, 2
        local_region(i)%rank = 1
        local_region(i)%global_extents = 0_c_int64_t
        local_region(i)%global_extents(1) = int(nranks * PER_RANK, c_int64_t)
        local_region(i)%offsets = 0_c_int64_t
        local_region(i)%offsets(1) = int(PER_RANK * rank, c_int64_t)
    end do
    writes(1)%var_name = c_loc(temp_name(1))
    writes(1)%host_data = c_loc(temperature(1))
    writes(1)%dtype = AMIO_DTYPE_F32
    writes(1)%shape = c_loc(local_shape)
    writes(1)%region = c_loc(local_region(1))
    writes(2)%var_name = c_loc(hum_name(1))
    writes(2)%host_data = c_loc(humidity(1))
    writes(2)%dtype = AMIO_DTYPE_F32
    writes(2)%shape = c_loc(local_shape)
    writes(2)%region = c_loc(local_region(2))

    io = c_null_ptr
    status = amio_write_region_batch(dataset, writes, int(size(writes, kind=c_size_t), c_size_t), io)
    call check_status(status, 'amio_write_region_batch')

    ! Step 5: Wait for batch completion and flush.  amio_wait times out in milliseconds.
    timeout_ms = 30000_c_int64_t
    status = amio_wait(io, timeout_ms)
    call check_status(status, 'amio_wait')
    status = amio_flush(dataset, timeout_ms)
    call check_status(status, 'amio_flush')

    ! Step 6: Close and finalize.
    status = amio_close_dataset(dataset)
    call check_status(status, 'amio_close_dataset')
    status = amio_finalize(core)
    call check_status(status, 'amio_finalize')

    call MPI_Finalize(ierr)

    if (rank == 0) print '(A)', 'AMIO Fortran batched region write example completed successfully.'

contains

    subroutine set_c_string(destination, source)
        character(kind=c_char), intent(out) :: destination(:)
        character(len=*), intent(in) :: source
        integer :: j
        destination = c_null_char
        do j = 1, min(len(source), size(destination) - 1)
            destination(j) = source(j:j)
        end do
    end subroutine set_c_string

    subroutine check_status(code, operation_name)
        integer(c_int32_t), intent(in) :: code
        character(len=*), intent(in) :: operation_name
        if (code /= AMIO_OK) then
            print *, trim(operation_name), ' failed with status ', code
            call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
        end if
    end subroutine check_status

end program example_region_batch_write
