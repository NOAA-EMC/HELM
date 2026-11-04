program amio_fortran_consumer_check
    use, intrinsic :: iso_c_binding, only: c_null_ptr, c_ptr, c_size_t, c_int32_t
    use amio_mod
    implicit none

    type(amio_region_write_t) :: writes(1)
    type(c_ptr) :: pending
    integer(c_int32_t) :: status

    if (command_argument_count() > 0) then
        status = amio_write_region_batch(c_null_ptr, writes, 0_c_size_t, pending)
        if (status /= AMIO_ERR_NULL_HANDLE) stop 1
    end if
end program amio_fortran_consumer_check
