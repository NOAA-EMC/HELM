!> @file example_netcdf4_write.f90
!> @brief Example: Write a 3D temperature field to NetCDF-4 using AMIO (Fortran).
!>
!> This example demonstrates the complete AMIO lifecycle from Fortran for
!> writing a 3D atmospheric temperature field (100 x 50 x 25: lon x lat x level)
!> to a NetCDF-4 file using the netcdf4 backend driver.
!>
!> Steps:
!>   1. Initialize AMIO with a manifest targeting the netcdf4 backend
!>   2. Open a dataset for writing
!>   3. Populate a 3D real(4) array with synthetic temperature data
!>   4. Write the array using the amio_write_array convenience subroutine
!>   5. Flush to ensure all writes complete
!>   6. Close the dataset and finalize AMIO
!>
!> Build (via CMake):
!>   Linked against AMIO::amio_fortran
!>
!> Run:
!>   ./example_netcdf4_write_f90

program example_netcdf4_write
    use, intrinsic :: iso_c_binding, only: &
        c_ptr, c_null_ptr, c_int32_t, c_int64_t, c_float, c_char, c_null_char
    use amio_mod
    implicit none

    ! Grid dimensions
    integer, parameter :: NX = 100   ! longitude points
    integer, parameter :: NY = 50    ! latitude points
    integer, parameter :: NZ = 25    ! vertical levels

    ! AMIO handles
    type(c_ptr) :: core
    type(c_ptr) :: dataset
    type(c_ptr) :: io_handle
    integer(c_int32_t) :: rc

    ! Data array
    real(c_float), allocatable, target :: temperature(:, :, :)

    ! Local variables
    integer :: i, j, k
    real(c_float) :: lat, lon, altitude_km, base_temp, temp
    real(c_float), parameter :: PI = 3.14159265_c_float

    write (*, '(A)') 'AMIO NetCDF-4 Write Example (Fortran)'
    write (*, '(A)') '======================================'
    write (*, '(A,I0,A,I0,A,I0,A)') &
        'Writing 3D temperature field (', NX, ' x ', NY, ' x ', NZ, ') to NetCDF-4'
    write (*, *)

    ! -------------------------------------------------------------------
    ! Step 1: Initialize AMIO with the netcdf4 manifest.
    !
    ! Note: Fortran strings passed to C must be null-terminated using
    ! c_null_char. The amio_init function expects a C string.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 1: Initializing AMIO...'
    rc = amio_init('examples/manifests/netcdf4_manifest.yaml'//c_null_char, core)
    call check_amio(rc, 'amio_init')
    write (*, '(A)') '  AMIO initialized successfully.'

    ! -------------------------------------------------------------------
    ! Step 2: Open a dataset for writing.
    !
    ! AMIO_MODE_WRITE creates a new file (or overwrites if configured).
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 2: Opening dataset for writing...'
    rc = amio_open_dataset(core, &
                           'examples/manifests/netcdf4_manifest.yaml'//c_null_char, &
                           AMIO_MODE_WRITE, dataset)
    call check_amio(rc, 'amio_open_dataset')
    write (*, '(A)') '  Dataset opened.'

    ! -------------------------------------------------------------------
    ! Step 3: Allocate and fill the temperature array.
    !
    ! Generate a realistic temperature field that decreases with altitude
    ! (standard atmosphere lapse rate) and varies with latitude.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 3: Generating synthetic temperature data...'
    allocate (temperature(NX, NY, NZ))

    do k = 1, NZ
        altitude_km = real(k - 1, c_float) * 0.5_c_float
        do j = 1, NY
            lat = -90.0_c_float + real(j - 1, c_float) * (180.0_c_float / real(NY - 1, c_float))
            base_temp = 288.15_c_float - 30.0_c_float * abs(lat) / 90.0_c_float
            do i = 1, NX
                lon = real(i - 1, c_float) * (360.0_c_float / real(NX, c_float))
                temp = base_temp - 6.5_c_float * altitude_km
                temp = temp + 2.0_c_float * sin(lon * PI / 180.0_c_float)
                temperature(i, j, k) = temp
            end do
        end do
    end do

    write (*, '(A,F6.1,A,F6.1,A)') '  Data generated: T range ~ [', &
        minval(temperature), ', ', maxval(temperature), '] K'

    ! -------------------------------------------------------------------
    ! Step 4: Write the 3D array using the convenience subroutine.
    !
    ! amio_write_array is a generic interface that automatically:
    !   - Determines the data type (AMIO_DTYPE_F32 for real(c_float))
    !   - Computes the rank (3 for a 3D array)
    !   - Extracts extents from the array shape
    !   - Obtains the data pointer via c_loc()
    !   - Calls the raw amio_write C function
    !
    ! This is much simpler than manually constructing amio_shape_t.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 4: Writing temperature field...'
    call amio_write_f32_3d(dataset, 'temperature'//c_null_char, &
                           temperature, io_handle, rc)
    call check_amio(rc, 'amio_write_array(temperature)')
    write (*, '(A)') '  Write enqueued (async I/O handle obtained).'

    ! -------------------------------------------------------------------
    ! Step 5: Flush to ensure all pending writes complete.
    !
    ! timeout_ms = 0 means wait indefinitely.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 5: Flushing pending writes...'
    rc = amio_flush(dataset, 0_c_int64_t)
    call check_amio(rc, 'amio_flush')
    write (*, '(A)') '  All writes completed successfully.'

    ! -------------------------------------------------------------------
    ! Step 6: Close the dataset and finalize AMIO.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 6: Closing dataset and finalizing...'
    rc = amio_close_dataset(dataset)
    call check_amio(rc, 'amio_close_dataset')

    rc = amio_finalize(core)
    call check_amio(rc, 'amio_finalize')
    write (*, '(A)') '  AMIO finalized.'

    ! Clean up
    deallocate (temperature)

    write (*, *)
    write (*, '(A)') 'Done! Temperature field written to NetCDF-4.'

contains

    !> @brief Check an AMIO return code and stop on failure.
    subroutine check_amio(status, context)
        integer(c_int32_t), intent(in) :: status
        character(len=*), intent(in) :: context

        if (status /= AMIO_OK) then
            write (*, '(A,A,A,I0,A)') 'AMIO error in ', context, &
                ' (code ', status, ')'
            error stop 'AMIO operation failed'
        end if
    end subroutine check_amio

end program example_netcdf4_write
