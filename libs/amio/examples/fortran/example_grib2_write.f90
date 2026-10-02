!> @file example_grib2_write.f90
!> @brief Example: Write a 2D 500hPa geopotential height field to GRIB2 using AMIO (Fortran).
!>
!> This example demonstrates writing a 2D single-precision array
!> (500 hPa geopotential height, 360 x 181: lon x lat) to a GRIB2 file
!> using the grib2 backend driver (nceplibs-g2c) from Fortran.
!>
!> Build (via CMake):
!>   Linked against AMIO::amio_fortran
!>
!> Run:
!>   ./example_grib2_write_f90

program example_grib2_write
    use, intrinsic :: iso_c_binding, only: &
        c_ptr, c_null_ptr, c_int32_t, c_int64_t, c_float, c_char, c_null_char
    use amio_mod
    implicit none

    ! Grid dimensions: 1-degree global grid
    integer, parameter :: NLON = 360   ! longitude points
    integer, parameter :: NLAT = 181   ! latitude points (90N to 90S inclusive)

    ! AMIO handles
    type(c_ptr) :: core
    type(c_ptr) :: dataset
    type(c_ptr) :: io_handle
    integer(c_int32_t) :: rc

    ! Data array
    real(c_float), allocatable, target :: hgt(:, :)

    ! Local variables
    integer :: i, j
    real(c_float) :: lat, lon, base_height, wave_amp
    real(c_float), parameter :: PI = 3.14159265_c_float

    write (*, '(A)') 'AMIO GRIB2 Write Example (Fortran)'
    write (*, '(A)') '==================================='
    write (*, '(A,I0,A,I0,A)') &
        'Writing 2D 500hPa geopotential height (', NLON, ' x ', NLAT, ') to GRIB2'
    write (*, *)

    ! -------------------------------------------------------------------
    ! Step 1: Initialize AMIO with the grib2 manifest.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 1: Initializing AMIO (GRIB2 backend)...'
    rc = amio_init('examples/manifests/grib2_manifest.yaml'//c_null_char, core)
    call check_amio(rc, 'amio_init')
    write (*, '(A)') '  AMIO initialized.'

    ! -------------------------------------------------------------------
    ! Step 2: Open a dataset for writing.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 2: Opening GRIB2 dataset for writing...'
    rc = amio_open_dataset(core, &
                           'examples/manifests/grib2_manifest.yaml'//c_null_char, &
                           AMIO_MODE_WRITE, dataset)
    call check_amio(rc, 'amio_open_dataset')
    write (*, '(A)') '  Dataset opened.'

    ! -------------------------------------------------------------------
    ! Step 3: Allocate and fill the geopotential height array.
    !
    ! Generate a realistic 500 hPa height field with:
    !   - Higher values in the tropics (~5880 gpm)
    !   - Lower values at the poles (~5100 gpm)
    !   - Rossby wave pattern in midlatitudes
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 3: Generating synthetic 500hPa height data...'
    allocate (hgt(NLON, NLAT))

    do j = 1, NLAT
        ! Latitude from 90N to 90S
        lat = 90.0_c_float - real(j - 1, c_float)
        ! Base height: higher in tropics, lower at poles
        base_height = 5600.0_c_float + 280.0_c_float * cos(lat * PI / 180.0_c_float)
        do i = 1, NLON
            lon = real(i - 1, c_float)
            hgt(i, j) = base_height
            ! Add Rossby wave pattern in midlatitudes
            if (abs(lat) > 20.0_c_float .and. abs(lat) < 70.0_c_float) then
                wave_amp = 120.0_c_float * sin((abs(lat) - 20.0_c_float) * PI / 50.0_c_float)
                ! Wavenumber 5 pattern
                hgt(i, j) = hgt(i, j) + wave_amp * sin(5.0_c_float * lon * PI / 180.0_c_float)
                ! Wavenumber 3 pattern (weaker)
                hgt(i, j) = hgt(i, j) + 0.4_c_float * wave_amp * &
                            cos(3.0_c_float * lon * PI / 180.0_c_float + 1.2_c_float)
            end if
            ! Slight hemispheric asymmetry
            if (lat < 0.0_c_float) then
                hgt(i, j) = hgt(i, j) - 20.0_c_float
            end if
        end do
    end do

    write (*, '(A,F7.1,A,F7.1,A)') '  Height data generated: range ~ [', &
        minval(hgt), ', ', maxval(hgt), '] gpm'

    ! -------------------------------------------------------------------
    ! Step 4: Write the 2D geopotential height array.
    !
    ! amio_write_array detects real(c_float) rank-2 and uses AMIO_DTYPE_F32.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 4: Writing geopotential height field...'
    call amio_write_f32_2d(dataset, 'geopotential_height'//c_null_char, &
                           hgt, io_handle, rc)
    call check_amio(rc, 'amio_write_array(geopotential_height)')
    write (*, '(A)') '  Write enqueued.'

    ! -------------------------------------------------------------------
    ! Step 5: Flush to ensure the write completes.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 5: Flushing...'
    rc = amio_flush(dataset, 0_c_int64_t)
    call check_amio(rc, 'amio_flush')
    write (*, '(A)') '  Flush complete.'

    ! -------------------------------------------------------------------
    ! Step 6: Close and finalize.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 6: Closing dataset and finalizing...'
    rc = amio_close_dataset(dataset)
    call check_amio(rc, 'amio_close_dataset')

    rc = amio_finalize(core)
    call check_amio(rc, 'amio_finalize')
    write (*, '(A)') '  Done.'

    deallocate (hgt)

    write (*, *)
    write (*, '(A)') '500hPa geopotential height written to GRIB2.'

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

end program example_grib2_write
