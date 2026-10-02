# NetCDF-4 Write Example (C)

This example demonstrates the complete AMIO lifecycle for writing a 3D
atmospheric temperature field (100 × 50 × 25: lon × lat × level) to a
NetCDF-4 file using the `netcdf4` backend driver.

## What It Does

1. Initializes AMIO with a manifest targeting the NetCDF-4 backend
2. Opens a dataset for writing
3. Populates a 3D float array with synthetic temperature data
4. Writes the array through AMIO's async pipeline
5. Flushes to ensure all writes complete
6. Closes the dataset and finalizes AMIO

## Manifest

```yaml title="examples/manifests/netcdf4_manifest.yaml"
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
    path: "output_netcdf4.nc"
    overwrite: true
```

## Source Code

```c title="examples/c/example_netcdf4_write.c"
--8<-- "examples/c/example_netcdf4_write.c"
```

??? note "Inline source (if includes are not configured)"

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <amio/amio.h>

    #define NX 100
    #define NY  50
    #define NZ  25

    static void check_amio(amio_status_t rc, const char *context)
    {
        if (rc != AMIO_OK) {
            fprintf(stderr, "AMIO error in %s: %s (code %d)\n",
                    context, amio_strerror(rc), (int)rc);
            exit(EXIT_FAILURE);
        }
    }

    int main(void)
    {
        amio_core_handle core = NULL;
        amio_dataset_handle dataset = NULL;
        amio_io_handle io = NULL;

        // Step 1: Initialize AMIO
        amio_status_t rc = amio_init(
            "examples/manifests/netcdf4_manifest.yaml", &core);
        check_amio(rc, "amio_init");

        // Step 2: Open dataset
        rc = amio_open_dataset(core,
            "examples/manifests/netcdf4_manifest.yaml",
            AMIO_MODE_WRITE, &dataset);
        check_amio(rc, "amio_open_dataset");

        // Step 3: Generate data
        float *temperature = malloc(NX * NY * NZ * sizeof(float));
        // ... fill with synthetic temperature data ...

        // Step 4: Write
        amio_shape_t shape = {0};
        shape.rank = 3;
        shape.extents[0] = NX;
        shape.extents[1] = NY;
        shape.extents[2] = NZ;

        rc = amio_write(dataset, "temperature", temperature,
                        AMIO_DTYPE_F32, &shape, &io);
        check_amio(rc, "amio_write");

        // Step 5: Flush
        rc = amio_flush(dataset, 0);
        check_amio(rc, "amio_flush");

        // Step 6: Cleanup
        rc = amio_close_dataset(dataset);
        rc = amio_finalize(core);
        free(temperature);

        return EXIT_SUCCESS;
    }
    ```

## Key Points

!!! info "Asynchronous Write Semantics"
    `amio_write` copies the data into the staging pool **synchronously**, then
    enqueues the backend write **asynchronously**. The caller can safely free
    or reuse the `temperature` buffer immediately after `amio_write` returns.

!!! tip "Shape Descriptor"
    Setting `strides` to all zeros tells AMIO the array is contiguous in
    row-major order. AMIO will derive the strides automatically from the
    extents.

## Build

```bash
cc -o example_netcdf4_write example_netcdf4_write.c -lamio -lm
./example_netcdf4_write
```

Or via CMake (when `AMIO_BUILD_EXAMPLES=ON`):

```bash
cmake --build build --target example_netcdf4_write
./build/examples/c/example_netcdf4_write
```
