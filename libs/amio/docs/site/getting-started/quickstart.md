# Quick Start

This page shows the minimal steps to write a multidimensional array through AMIO.

## 1. Create a Manifest

AMIO is configured through a YAML manifest that specifies the backend driver,
staging pool size, and output path:

```yaml
# manifest.yaml
amio:
  version: "0.1.0"
  staging:
    buffer_count: 8
    buffer_size_bytes: 52428800   # 50 MiB per buffer
  workers:
    thread_count: 2
    pin_threads: false
  backend:
    name: "netcdf4"
    options:
      parallel_io: false
      compression:
        algorithm: "zlib"
        level: 4
      chunking:
        auto: true
  output:
    path: "output.nc"
    overwrite: true
```

## 2. Write Data (C)

```c
#include <amio/amio.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
    amio_core_handle core = NULL;
    amio_dataset_handle dataset = NULL;
    amio_io_handle io = NULL;

    // Initialize AMIO
    amio_status_t rc = amio_init("manifest.yaml", &core);
    if (rc != AMIO_OK) {
        fprintf(stderr, "Init failed: %s\n", amio_strerror(rc));
        return 1;
    }

    // Open dataset for writing
    rc = amio_open_dataset(core, "manifest.yaml", AMIO_MODE_WRITE, &dataset);

    // Prepare a 2D array (100 x 50)
    float data[100 * 50];
    for (int i = 0; i < 100 * 50; i++) data[i] = (float)i * 0.1f;

    // Describe the shape
    amio_shape_t shape;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 2;
    shape.extents[0] = 100;
    shape.extents[1] = 50;

    // Write (async — data is copied into staging pool immediately)
    rc = amio_write(dataset, "temperature", data, AMIO_DTYPE_F32, &shape, &io);

    // Flush all pending writes
    rc = amio_flush(dataset, 0);

    // Cleanup
    rc = amio_close_dataset(dataset);
    rc = amio_finalize(core);

    return 0;
}
```

## 3. Write Data (Fortran)

```fortran
program quickstart
    use, intrinsic :: iso_c_binding
    use amio_mod
    implicit none

    type(c_ptr) :: core, dataset, io_handle
    integer(c_int32_t) :: rc
    real(c_float), allocatable, target :: data(:,:)

    ! Initialize AMIO
    rc = amio_init("manifest.yaml" // c_null_char, core)

    ! Open dataset
    rc = amio_open_dataset(core, "manifest.yaml" // c_null_char, &
                           AMIO_MODE_WRITE, dataset)

    ! Create and fill a 100x50 array
    allocate(data(100, 50))
    data = 273.15  ! Fill with temperature values

    ! Write using the convenience subroutine (auto-detects type and shape)
    call amio_write_array(dataset, "temperature" // c_null_char, &
                          data, io_handle, rc)

    ! Flush and cleanup
    rc = amio_flush(dataset, 0_c_int64_t)
    rc = amio_close_dataset(dataset)
    rc = amio_finalize(core)

    deallocate(data)
end program quickstart
```

## 4. Build and Run

```bash
# C example
cc -o quickstart quickstart.c -lamio
./quickstart

# Fortran example
gfortran -o quickstart_f quickstart.f90 -lamio -lamio_fortran
./quickstart_f
```

## AMIO Lifecycle

The typical AMIO workflow follows this pattern:

1. **`amio_init`** — Load manifest, create staging pool and worker threads
2. **`amio_open_dataset`** — Open a file for reading or writing
3. **`amio_write` / `amio_read`** — Transfer data (async for writes)
4. **`amio_flush`** — Wait for pending writes to complete
5. **`amio_close_dataset`** — Close the file and release driver resources
6. **`amio_finalize`** — Drain workers and free all AMIO resources

!!! note "Host pointer safety"
    After `amio_write` returns, the host buffer is safe to reuse or free.
    AMIO deep-copies the data into its staging pool synchronously before
    returning control to the caller.

## Next Steps

- [C API Guide](../user-guide/c-api.md) — Full lifecycle and error handling
- [Fortran API Guide](../user-guide/fortran-api.md) — Convenience subroutines
- [Configuration](../user-guide/configuration.md) — Manifest format reference
- [Examples](../examples/netcdf4-c.md) — Complete, compilable examples
