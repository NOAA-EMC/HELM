# GRIB2 Write Example (Fortran)

This example demonstrates writing a 2D 500 hPa geopotential height field
(360 × 181: lon × lat) to a GRIB2 file using the `grib2` backend driver
(nceplibs-g2c) from Fortran.

## What It Does

1. Initializes AMIO with a manifest targeting the `grib2` backend
2. Opens a dataset for writing
3. Populates a 2D `real(c_float)` array with synthetic geopotential height
4. Writes the array using `amio_write_array`
5. Flushes and closes

## Source Code

```fortran title="examples/fortran/example_grib2_write.f90"
--8<-- "examples/fortran/example_grib2_write.f90"
```

??? note "Inline source (if includes are not configured)"

    ```fortran
    program example_grib2_write
        use, intrinsic :: iso_c_binding
        use amio_mod
        implicit none

        integer, parameter :: NLON = 360, NLAT = 181
        type(c_ptr) :: core, dataset, io_handle
        integer(c_int32_t) :: rc
        real(c_float), allocatable, target :: hgt(:,:)

        rc = amio_init('examples/manifests/grib2_manifest.yaml' // c_null_char, core)
        rc = amio_open_dataset(core, &
            'examples/manifests/grib2_manifest.yaml' // c_null_char, &
            AMIO_MODE_WRITE, dataset)

        allocate(hgt(NLON, NLAT))
        ! ... fill with synthetic geopotential height data ...

        call amio_write_array(dataset, 'geopotential_height' // c_null_char, &
                              hgt, io_handle, rc)

        rc = amio_flush(dataset, 0_c_int64_t)
        rc = amio_close_dataset(dataset)
        rc = amio_finalize(core)
        deallocate(hgt)
    end program example_grib2_write
    ```

## Key Points

!!! info "Single Precision"
    The geopotential height field uses `real(c_float)` (single precision).
    `amio_write_array` detects this and uses `AMIO_DTYPE_F32` automatically.

!!! note "Rossby Wave Pattern"
    The synthetic data includes a realistic Rossby wave pattern in
    midlatitudes (wavenumber 4–6) with hemispheric asymmetry, producing
    height values in the range ~5100–5880 gpm.

## Build

Via CMake (linked against `AMIO::amio_fortran`):

```bash
cmake --build build --target example_grib2_write_f90
./build/examples/fortran/example_grib2_write_f90
```
