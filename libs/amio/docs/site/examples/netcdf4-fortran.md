# NetCDF-4 Write Example (Fortran)

This example demonstrates the complete AMIO lifecycle from Fortran for writing
a 3D atmospheric temperature field (100 × 50 × 25: lon × lat × level) to a
NetCDF-4 file using the `netcdf4` backend driver.

## What It Does

1. Initializes AMIO with a manifest targeting the NetCDF-4 backend
2. Opens a dataset for writing
3. Populates a 3D `real(c_float)` array with synthetic temperature data
4. Writes the array using the `amio_write_array` convenience subroutine
5. Flushes to ensure all writes complete
6. Closes the dataset and finalizes AMIO

## Source Code

```fortran title="examples/fortran/example_netcdf4_write.f90"
--8<-- "examples/fortran/example_netcdf4_write.f90"
```

??? note "Inline source (if includes are not configured)"

    ```fortran
    program example_netcdf4_write
        use, intrinsic :: iso_c_binding
        use amio_mod
        implicit none

        integer, parameter :: NX = 100, NY = 50, NZ = 25
        type(c_ptr) :: core, dataset, io_handle
        integer(c_int32_t) :: rc
        real(c_float), allocatable, target :: temperature(:,:,:)

        ! Initialize AMIO
        rc = amio_init('examples/manifests/netcdf4_manifest.yaml' // c_null_char, core)

        ! Open dataset
        rc = amio_open_dataset(core, &
            'examples/manifests/netcdf4_manifest.yaml' // c_null_char, &
            AMIO_MODE_WRITE, dataset)

        ! Generate and write data
        allocate(temperature(NX, NY, NZ))
        ! ... fill with synthetic data ...

        call amio_write_array(dataset, 'temperature' // c_null_char, &
                              temperature, io_handle, rc)

        ! Flush and cleanup
        rc = amio_flush(dataset, 0_c_int64_t)
        rc = amio_close_dataset(dataset)
        rc = amio_finalize(core)
        deallocate(temperature)
    end program example_netcdf4_write
    ```

## Key Points

!!! info "Convenience Subroutine"
    `amio_write_array` is a generic interface that automatically determines
    the data type (`AMIO_DTYPE_F32` for `real(c_float)`), computes the rank
    (3 for a 3D array), extracts extents from the array shape, and calls the
    raw `amio_write` C function.

!!! warning "Null Termination"
    Fortran strings passed to C must be null-terminated using `c_null_char`:
    ```fortran
    rc = amio_init('manifest.yaml' // c_null_char, core)
    ```

## Build

Via CMake (linked against `AMIO::amio_fortran`):

```bash
cmake --build build --target example_netcdf4_write_f90
./build/examples/fortran/example_netcdf4_write_f90
```
