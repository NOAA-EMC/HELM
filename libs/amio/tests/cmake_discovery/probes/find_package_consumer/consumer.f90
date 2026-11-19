program amio_fortran_consumer_check
    use, intrinsic :: iso_c_binding, only: c_null_ptr, c_ptr, c_size_t, c_int32_t, c_int64_t, c_sizeof
    use amio_mod
    implicit none

    type(amio_region_write_t) :: writes(1)
    type(amio_write_variable_t) :: variables(1)
    type(c_ptr) :: pending
    integer(c_int32_t) :: status

    ! ABI presence check: the batch write path needs the schema-preparation bindings too.
    if (c_sizeof(variables(1)) /= 96_c_size_t) stop 3
    if (AMIO_WRITE_VARIABLE_VERSION /= 1_c_int32_t) stop 4

    if (command_argument_count() > 0) then
        variables(1)%struct_size = int(c_sizeof(variables(1)), c_int32_t)
        variables(1)%version = AMIO_WRITE_VARIABLE_VERSION
        variables(1)%name = c_null_ptr
        variables(1)%dtype = AMIO_DTYPE_F32
        variables(1)%rank = 1
        variables(1)%global_extents = 0_c_int64_t
        variables(1)%global_extents(1) = 4_c_int64_t
        variables(1)%reserved = 0_c_int64_t
        status = amio_prepare_write_schema(c_null_ptr, variables, 1_c_size_t)
        if (status /= AMIO_ERR_NULL_HANDLE) stop 1
        status = amio_write_region_batch(c_null_ptr, writes, 0_c_size_t, pending)
        if (status /= AMIO_ERR_NULL_HANDLE) stop 2
    end if
end program amio_fortran_consumer_check
