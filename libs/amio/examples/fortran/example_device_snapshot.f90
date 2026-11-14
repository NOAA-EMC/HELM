module example_device_snapshot
    use, intrinsic :: iso_c_binding, only: c_ptr, c_char, c_int32_t, c_int64_t, c_size_t, c_float, c_loc, c_sizeof
    use amio_mod, only: amio_memory_descriptor_t, amio_put_field_memory
    implicit none
contains
    function submit_host_snapshot(step, values, operation) result(status)
        type(c_ptr), value :: step
        real(c_float), target, contiguous, intent(in) :: values(:)
        type(c_ptr), intent(out) :: operation
        integer(c_int32_t) :: status
        type(amio_memory_descriptor_t) :: memory
        character(kind=c_char, len=5), target :: field

        ! Synchronize model/device writes before calling this routine. The host
        ! call snapshots into AMIO-owned storage; values may be reused on return.
        field = 'field'
        memory%struct_size = int(c_sizeof(memory), c_int32_t)
        memory%version = 1
        memory%memory_kind = 0 ! host
        memory%ownership = 0   ! owned snapshot
        memory%device_id = 0
        memory%rank = 1
        memory%producer_writes_synchronized = 1
        memory%reserved32 = 0
        memory%element_bytes = c_sizeof(values(1))
        memory%extents = 0
        memory%strides = 0
        memory%extents(1) = size(values, kind=c_int64_t)
        memory%strides(1) = memory%element_bytes
        memory%reserved = 0
        status = amio_put_field_memory(step, c_loc(field), 5_c_size_t, memory, c_loc(values(1)), &
                                       size(values, kind=c_size_t) * int(memory%element_bytes, c_size_t), operation)
    end function submit_host_snapshot

    function submit_device_snapshot(step, device_values, count, operation) result(status)
        type(c_ptr), value :: step, device_values
        integer(c_int64_t), value :: count
        type(c_ptr), intent(out) :: operation
        integer(c_int32_t) :: status
        type(amio_memory_descriptor_t) :: memory
        character(kind=c_char, len=5), target :: field

        ! Initialize the private device runtime and synchronize producer writes
        ! before calling. This profile reports device_memory unsupported, so the
        ! call is rejected before device_values is dereferenced.
        field = 'field'
        memory%struct_size = int(c_sizeof(memory), c_int32_t)
        memory%version = 1
        memory%memory_kind = 1 ! device
        memory%ownership = 0
        memory%device_id = 0
        memory%rank = 1
        memory%producer_writes_synchronized = 1
        memory%reserved32 = 0
        memory%element_bytes = 4
        memory%extents = 0
        memory%strides = 0
        memory%extents(1) = count
        memory%strides(1) = memory%element_bytes
        memory%reserved = 0
        status = amio_put_field_memory(step, c_loc(field), 5_c_size_t, memory, device_values, &
                                       int(count * memory%element_bytes, c_size_t), operation)
    end function submit_device_snapshot
end module example_device_snapshot
