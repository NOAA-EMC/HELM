program example_aggregated_steps
    use iso_c_binding, only: c_ptr, c_null_ptr, c_sizeof, c_int32_t, c_int64_t, c_char, c_loc, c_associated
    use mpi
    use amio_mod
    implicit none
    type(amio_context_descriptor_t) :: descriptor
    type(c_ptr) :: context
    integer(c_int32_t), allocatable, target :: producers(:)
    integer(c_int32_t), target :: writer(1)
    character(kind=c_char), target :: configuration(2) = [character(kind=c_char) :: '{', '}']
    integer(c_int32_t) :: status, completed
    integer :: ierr, provided, rank, size, allocation_status

    call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
    if (ierr /= MPI_SUCCESS .or. provided < MPI_THREAD_FUNNELED) call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    if (size < 2) call MPI_Abort(MPI_COMM_WORLD, 2, ierr)
    allocate (producers(size - 1), stat=allocation_status)
    if (allocation_status /= 0) then
        call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
        error stop 'could not allocate producer ranks'
    end if
    producers = [(int(ierr, c_int32_t), ierr=0, size - 2)]
    writer = int(size - 1, c_int32_t)
    descriptor = amio_context_descriptor_t( &
                 int(c_sizeof(descriptor), c_int32_t), 1_c_int32_t, &
                 AMIO_EXECUTION_PERSISTENT_AGGREGATED, AMIO_PROGRESS_CALLER, int(size, c_int32_t), &
                 int(size - 1, c_int32_t), 1_c_int32_t, 0_c_int32_t, &
                 c_loc(producers(1)), c_loc(writer(1)), c_null_ptr, &
                 int(MPI_COMM_WORLD, c_int64_t), c_loc(configuration(1)), &
                 2_c_int64_t, [0_c_int64_t, 0_c_int64_t])
    context = c_null_ptr
    status = amio_init_ex(descriptor, context)
    if (status == AMIO_OK) then
        if (rank == size - 1) then
            status = amio_run_service(context)
            if (status /= AMIO_ERR_UNSUPPORTED) call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
            print *, 'writer service reports unsupported until aggregation lifecycle integration'
            status = AMIO_OK
        else if (rank < size - 1) then
            status = amio_progress(context, 8_c_int32_t, completed)
        end if
        if (c_associated(context)) status = amio_release_context(context)
    end if
    if (status /= AMIO_OK) call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
    call MPI_Finalize(ierr)
end program example_aggregated_steps
