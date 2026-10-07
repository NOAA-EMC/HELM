# Zarr v3 Write Example (Fortran)

This example demonstrates writing a 2D sea surface temperature (SST) field
(360 × 180: lon × lat) to a Zarr v3 store using the NCZarr fallback mode
from Fortran.

## What It Does

1. Initializes AMIO with a manifest targeting the `zarr3` backend
2. Opens a dataset for writing
3. Populates a 2D `real(c_double)` array with synthetic SST data
4. Writes the array using `amio_write_array` (auto-detects `AMIO_DTYPE_F64`)
5. Flushes and closes

## Source Code

```fortran title="examples/fortran/example_zarr_write.f90"
--8<-- "examples/fortran/example_zarr_write.f90"
```

??? note "Inline source (if includes are not configured)"

    ```fortran
    program example_zarr_write
        use, intrinsic :: iso_c_binding
        use amio_mod
        implicit none

        integer, parameter :: NLON = 360, NLAT = 180
        type(c_ptr) :: core, dataset, io_handle
        integer(c_int32_t) :: rc
        real(c_double), allocatable, target :: sst(:,:)

        rc = amio_init('examples/manifests/zarr3_manifest.yaml' // c_null_char, core)
        rc = amio_open_dataset(core, &
            'examples/manifests/zarr3_manifest.yaml' // c_null_char, &
            AMIO_MODE_WRITE, dataset)

        allocate(sst(NLON, NLAT))
        ! ... fill with synthetic SST data ...

        call amio_write_array(dataset, 'sea_surface_temperature' // c_null_char, &
                              sst, io_handle, rc)

        rc = amio_flush(dataset, 0_c_int64_t)
        rc = amio_close_dataset(dataset)
        rc = amio_finalize(core)
        deallocate(sst)
    end program example_zarr_write
    ```

## Key Points

!!! info "Double Precision Detection"
    The `amio_write_array` generic interface detects `real(c_double)` arrays
    and automatically uses `AMIO_DTYPE_F64`. No manual dtype specification
    is needed.

## Build

Via CMake (linked against `AMIO::amio_fortran`):

```bash
cmake --build build --target example_zarr_write_f90
./build/examples/fortran/example_zarr_write_f90
```
