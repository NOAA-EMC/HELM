!> @file example_zarr_write.f90
!> @brief Example: Write a 2D sea surface temperature field to Zarr v3 using AMIO (Fortran).
!>
!> This example demonstrates writing a 2D double-precision array
!> (sea surface temperature, 360 x 180: lon x lat) to a Zarr v3 store
!> using the NCZarr fallback mode from Fortran.
!>
!> Build (via CMake):
!>   Linked against AMIO::amio_fortran
!>
!> Run:
!>   ./example_zarr_write_f90

program example_zarr_write
    use, intrinsic :: iso_c_binding, only: &
        c_ptr, c_null_ptr, c_int32_t, c_int64_t, c_double, c_char, c_null_char
    use amio_mod
    implicit none

    ! Grid dimensions: 1-degree global ocean grid
    integer, parameter :: NLON = 360   ! longitude points
    integer, parameter :: NLAT = 180   ! latitude points

    ! AMIO handles
    type(c_ptr) :: core
    type(c_ptr) :: dataset
    type(c_ptr) :: io_handle
    integer(c_int32_t) :: rc

    ! Data array (double precision for SST)
    real(c_double), allocatable, target :: sst(:, :)

    ! Local variables
    integer :: i, j
    real(c_double) :: lat, lon, base_sst

    write (*, '(A)') 'AMIO Zarr v3 (NCZarr) Write Example (Fortran)'
    write (*, '(A)') '=============================================='
    write (*, '(A,I0,A,I0,A)') &
        'Writing 2D SST field (', NLON, ' x ', NLAT, ') to Zarr v3 store'
    write (*, *)

    ! -------------------------------------------------------------------
    ! Step 1: Initialize AMIO with the zarr3 manifest.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 1: Initializing AMIO (Zarr v3 / NCZarr mode)...'
    rc = amio_init('examples/manifests/zarr3_manifest.yaml'//c_null_char, core)
    call check_amio(rc, 'amio_init')
    write (*, '(A)') '  AMIO initialized.'

    ! -------------------------------------------------------------------
    ! Step 2: Open a dataset for writing.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 2: Opening Zarr dataset for writing...'
    rc = amio_open_dataset(core, &
                           'examples/manifests/zarr3_manifest.yaml'//c_null_char, &
                           AMIO_MODE_WRITE, dataset)
    call check_amio(rc, 'amio_open_dataset')
    write (*, '(A)') '  Dataset opened.'

    ! -------------------------------------------------------------------
    ! Step 3: Allocate and fill the SST array.
    !
    ! Generate a realistic SST pattern with warm equatorial waters
    ! and cold polar waters.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 3: Generating synthetic SST data...'
    allocate (sst(NLON, NLAT))

    do j = 1, NLAT
        lat = -90.0_c_double + real(j - 1, c_double) + 0.5_c_double
        ! Base SST: warm at equator, cold at poles
        base_sst = 300.0_c_double - 29.0_c_double * (lat * lat) / (90.0_c_double * 90.0_c_double)
        do i = 1, NLON
            lon = real(i - 1, c_double) + 0.5_c_double
            sst(i, j) = base_sst
            ! Add small longitudinal variation
            sst(i, j) = sst(i, j) + 0.5_c_double * sin(lon * 0.1_c_double) * &
                        cos(lat * 0.15_c_double)
            ! Clamp to physical range
            sst(i, j) = max(271.15_c_double, min(308.15_c_double, sst(i, j)))
        end do
    end do

    write (*, '(A,F7.2,A,F7.2,A)') '  SST data generated: range ~ [', &
        minval(sst), ', ', maxval(sst), '] K'

    ! -------------------------------------------------------------------
    ! Step 4: Write the 2D SST array using the convenience subroutine.
    !
    ! amio_write_array detects real(c_double) and uses AMIO_DTYPE_F64.
    ! -------------------------------------------------------------------
    write (*, '(A)') 'Step 4: Writing SST field...'
    call amio_write_f64_2d(dataset, 'sea_surface_temperature'//c_null_char, &
                           sst, io_handle, rc)
    call check_amio(rc, 'amio_write_array(sst)')
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

    deallocate (sst)

    write (*, *)
    write (*, '(A)') 'SST field written to Zarr v3 store.'

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

end program example_zarr_write
