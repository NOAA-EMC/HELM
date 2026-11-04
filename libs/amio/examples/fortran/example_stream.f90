program example_stream
    use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr, c_sizeof, c_int32_t, c_int64_t, c_int8_t, &
                                                                              c_float, c_char, c_loc, c_associated, c_size_t
    use mpi
    use amio_mod
    implicit none

    type(amio_context_descriptor_t) :: context_descriptor
    type(amio_stream_contribution_t) :: contribution
    type(amio_stream_receive_descriptor_t) :: receive
    type(amio_stream_acquire_result_t) :: acquired
    type(amio_stream_read_descriptor_t) :: selection
    type(c_ptr) :: context
    integer(c_int32_t), target :: producers(4) = [0_c_int32_t, 1_c_int32_t, 2_c_int32_t, 3_c_int32_t]
    integer(c_int32_t), target :: consumers(2) = [4_c_int32_t, 5_c_int32_t]
    character(kind=c_char), target :: configuration(2) = [character(kind=c_char) :: '{', '}']
    integer(c_int8_t) :: digest(32)
    real(c_float), target :: value
    integer(c_int32_t) :: status
    integer(c_size_t) :: read_bytes
    integer :: ierr, provided, rank, world_size, producer

    call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
    if (ierr /= MPI_SUCCESS .or. provided < MPI_THREAD_FUNNELED) call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, world_size, ierr)
    if (world_size /= 6) call MPI_Abort(MPI_COMM_WORLD, 2, ierr)

    context_descriptor%struct_size = int(c_sizeof(context_descriptor), c_int32_t)
    context_descriptor%version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1
    context_descriptor%execution_mode = AMIO_EXECUTION_MPI_STREAM
    context_descriptor%progress_mode = AMIO_PROGRESS_CALLER
    context_descriptor%rank_count = 6
    context_descriptor%producer_count = 4
    context_descriptor%writer_count = 0
    context_descriptor%consumer_count = 2
    context_descriptor%producers = c_loc(producers(1))
    context_descriptor%writers = c_null_ptr
    context_descriptor%consumers = c_loc(consumers(1))
    context_descriptor%fortran_communicator = int(MPI_COMM_WORLD, c_int64_t)
    context_descriptor%canonical_configuration = c_loc(configuration(1))
    context_descriptor%configuration_bytes = 2
    context_descriptor%reserved = 0_c_int64_t
    context = c_null_ptr
    status = amio_init_ex(context_descriptor, context)
    if (status /= AMIO_OK) call MPI_Abort(MPI_COMM_WORLD, 3, ierr)
    digest = int(108, c_int8_t)

    if (rank < 4) then
        value = real(5000 + rank, c_float)
        contribution%struct_size = int(c_sizeof(contribution), c_int32_t)
        contribution%version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1
        contribution%step_id = 1
        contribution%sequence = 1
        contribution%field_id = 7
        contribution%rank = 1
        contribution%element_bytes = int(c_sizeof(value), c_int64_t)
        contribution%schema_digest = digest
        contribution%offsets = 0_c_int64_t
        contribution%extents = 0_c_int64_t
        contribution%offsets(1) = int(rank, c_int64_t)
        contribution%extents(1) = 1
        contribution%reserved = 0_c_int64_t
        status = amio_stream_submit(context, contribution, c_loc(value), c_sizeof(value))
        if (status == AMIO_OK) status = amio_stream_wait_release(context, 1_c_int64_t)
        if (status == AMIO_OK) status = amio_stream_finish(context, 1_c_int32_t)
    else
        acquired%struct_size = int(c_sizeof(acquired), c_int32_t)
        acquired%version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1
        status = amio_stream_acquire(context, 0_c_int64_t, acquired)
        if (status == AMIO_OK .and. acquired%state /= AMIO_STREAM_TIMEOUT) status = AMIO_ERR_STEP_FAILED
        receive%struct_size = int(c_sizeof(receive), c_int32_t)
        receive%version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1
        receive%step_id = 1
        receive%schema_digest = digest
        receive%absent_optional_count = 0
        receive%fields_per_producer = 1
        receive%absent_optional_names = c_null_ptr
        receive%absent_optional_name_bytes = c_null_ptr
        receive%reserved = 0_c_int64_t
        if (status == AMIO_OK) status = amio_stream_receive_step(context, receive)
        if (status == AMIO_OK) status = amio_stream_acquire(context, 0_c_int64_t, acquired)
        if (status == AMIO_OK .and. acquired%state /= AMIO_STREAM_AVAILABLE) status = AMIO_ERR_STEP_FAILED
        do producer = 0, 3
            if (status /= AMIO_OK .or. 4 + mod(producer, 2) /= rank) cycle
            selection%struct_size = int(c_sizeof(selection), c_int32_t)
            selection%version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1
            selection%producer_rank = int(producer, c_int32_t)
            selection%field_id = 7
            selection%rank = 1
            selection%reserved32 = 0
            selection%element_bytes = int(c_sizeof(value), c_int64_t)
            selection%logical_field_name = c_null_ptr
            selection%logical_field_name_bytes = 0
            selection%offsets = 0_c_int64_t
            selection%extents = 0_c_int64_t
            selection%offsets(1) = int(producer, c_int64_t)
            selection%extents(1) = 1
            selection%reserved = 0_c_int64_t
            value = 0.0_c_float
            status = amio_stream_read(acquired%lease, selection, c_loc(value), c_sizeof(value), read_bytes)
            if (status == AMIO_OK .and. (read_bytes /= c_sizeof(value) .or. value /= real(5000 + producer, c_float))) &
                status = AMIO_ERR_SCHEMA_MISMATCH
        end do
        if (status == AMIO_OK) status = amio_stream_release(acquired%lease)
        if (status == AMIO_OK) status = amio_stream_finish(context, 1_c_int32_t)
        if (status == AMIO_OK) then
            status = amio_stream_acquire(context, 0_c_int64_t, acquired)
            if (status == AMIO_OK .and. acquired%state /= AMIO_STREAM_END_OF_STREAM) status = AMIO_ERR_STEP_FAILED
        end if
    end if
    if (c_associated(context)) then
        if (status == AMIO_OK) status = amio_release_context(context)
    end if
    if (status /= AMIO_OK) call MPI_Abort(MPI_COMM_WORLD, 4, ierr)
    call MPI_Finalize(ierr)
end program example_stream
