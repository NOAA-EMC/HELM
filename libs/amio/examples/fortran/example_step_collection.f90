program example_step_collection
    use, intrinsic :: iso_c_binding, only: c_char, c_int, c_int32_t, c_int64_t, c_size_t, c_sizeof, &
                                                                              c_float, c_double, c_ptr, c_null_ptr, c_null_char, c_loc
    use mpi
    use amio_mod
    implicit none

    type(amio_context_descriptor_t) :: context_desc
    type(amio_dataset_descriptor_t) :: dataset_desc
    type(amio_dimension_descriptor_t), target :: dimension
    type(amio_variable_descriptor_t), target :: variable
    type(amio_schema_descriptor_t) :: schema_desc
    type(amio_step_descriptor_t) :: step_desc
    type(amio_operation_status_t) :: operation_status
    type(amio_step_summary_t) :: summary
    type(c_ptr) :: context, dataset, schema, step, operation, lease
    character(kind=c_char), target :: config(2) = [character(kind=c_char) :: '{', '}']
    character(kind=c_char), target :: location(128), dimension_name(2), field_name(12)
    character(len=64) :: location_text
    real(c_float), target :: temperature(4) = [280.0_c_float, 281.0_c_float, 282.0_c_float, 283.0_c_float]
    integer :: ierr, provided, clock_count
    integer(c_int32_t) :: status

    call MPI_Init_thread(MPI_THREAD_FUNNELED, provided, ierr)
    if (ierr /= MPI_SUCCESS .or. provided < MPI_THREAD_FUNNELED) stop 1

    location = c_null_char
    call system_clock(clock_count)
    write (location_text, '("./amio-step-",i0)') clock_count
    call set_c_string(location, trim(location_text))
    dimension_name = c_null_char
    call set_c_string(dimension_name, 'x')
    field_name = c_null_char
    call set_c_string(field_name, 'temperature')

    context_desc = amio_context_descriptor_t( &
                   int(c_sizeof(context_desc), c_int32_t), 1_c_int32_t, AMIO_EXECUTION_PERSISTENT_DIRECT, AMIO_PROGRESS_CALLER, &
                   1_c_int32_t, 0_c_int32_t, 0_c_int32_t, 0_c_int32_t, c_null_ptr, c_null_ptr, c_null_ptr, &
                   int(MPI_COMM_SELF, c_int64_t), c_loc(config(1)), 2_c_int64_t, [0_c_int64_t, 0_c_int64_t])
    context = c_null_ptr
    status = amio_init_ex(context_desc, context)
    call check_status(status, 'amio_init_ex')

    dataset_desc = amio_dataset_descriptor_t( &
                   int(c_sizeof(dataset_desc), c_int32_t), 1_c_int32_t, c_loc(location(1)), int(len_trim(location_text), c_int64_t), &
                   AMIO_FORMAT_NETCDF4, 0_c_int32_t, 1048576_c_int64_t, 64_c_int64_t, 4096_c_int64_t, 8_c_int64_t, &
                   [0_c_int64_t, 0_c_int64_t])
    dataset = c_null_ptr
    status = amio_open_dataset_ex(context, dataset_desc, dataset)
    call check_status(status, 'amio_open_dataset_ex')

    dimension = amio_dimension_descriptor_t(int(c_sizeof(dimension), c_int32_t), 1_c_int32_t, c_loc(dimension_name(1)), &
                                            1_c_int64_t, 4_c_int64_t, c_null_ptr, 0_c_int64_t, [0_c_int64_t, 0_c_int64_t])
    variable = amio_variable_descriptor_t(int(c_sizeof(variable), c_int32_t), 1_c_int32_t, c_loc(field_name(1)), &
                                          11_c_int64_t, AMIO_DTYPE_F32, AMIO_DTYPE_F32, 1_c_int32_t, 1_c_int32_t, c_loc(dimension), &
                                          0_c_int32_t, 0_c_int32_t, c_null_ptr, [0_c_int64_t, 0_c_int64_t])
    schema_desc = amio_schema_descriptor_t(int(c_sizeof(schema_desc), c_int32_t), 1_c_int32_t, 1_c_int64_t, c_loc(variable), &
                                           [0_c_int64_t, 0_c_int64_t])
    schema = c_null_ptr
    status = amio_define_schema(dataset, schema_desc, schema)
    call check_status(status, 'amio_define_schema')

    step_desc = amio_step_descriptor_t(int(c_sizeof(step_desc), c_int32_t), 1_c_int32_t, 0_c_int64_t, 12.5_c_double, schema, &
                                       [0_c_int64_t, 0_c_int64_t])
    step = c_null_ptr
    status = amio_begin_step(dataset, step_desc, step)
    call check_status(status, 'amio_begin_step')
    operation = c_null_ptr
    status = amio_put_field(step, c_loc(field_name(1)), 11_c_size_t, c_loc(temperature(1)), &
                            int(size(temperature, kind=c_size_t) * c_sizeof(temperature(1)), c_size_t), operation)
    call check_status(status, 'amio_put_field')
    status = amio_wait_operation(operation, AMIO_OPERATION_SOURCE_RELEASED, 0_c_int64_t)
    call check_status(status, 'wait for source release')
    status = amio_end_step(step)
    call check_status(status, 'amio_end_step')
    status = amio_query_operation(operation, operation_status)
    call check_status(status, 'amio_query_operation')
    if (operation_status%outcome /= AMIO_OPERATION_SUCCEEDED) error stop 'step operation did not succeed'
    status = amio_describe_step(dataset, 0_c_int64_t, summary)
    call check_status(status, 'amio_describe_step')
    lease = c_null_ptr
    status = amio_acquire_step_lease(dataset, summary%step_id, lease)
    call check_status(status, 'amio_acquire_step_lease')
    status = amio_close_dataset_ex(dataset)
    if (status /= AMIO_ERR_INVALID_POLICY) error stop 'dataset close must reject an outstanding step lease'
    status = amio_release_step_lease(lease)
    call check_status(status, 'amio_release_step_lease')
    print *, 'Published step ', summary%step_id, ' with ', summary%present_field_count, ' field(s).'

    status = amio_release_operation(operation)
    call check_status(status, 'amio_release_operation')
    status = amio_release_step_handle(step)
    call check_status(status, 'amio_release_step_handle')
    status = amio_release_schema(schema)
    call check_status(status, 'amio_release_schema')
    status = amio_close_dataset_ex(dataset)
    call check_status(status, 'amio_close_dataset_ex')
    status = amio_release_context(context)
    call check_status(status, 'amio_release_context')
    call MPI_Finalize(ierr)

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

end program example_step_collection
