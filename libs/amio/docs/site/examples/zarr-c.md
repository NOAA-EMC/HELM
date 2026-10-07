# Zarr v3 Write Example (C)

This example demonstrates writing a 2D sea surface temperature (SST) field
(360 × 180: lon × lat) to a Zarr v3 store using the NCZarr fallback mode.

## What It Does

1. Initializes AMIO with a manifest targeting the `zarr3` backend
2. Opens a dataset for writing
3. Populates a 2D double-precision array with synthetic SST data
4. Writes the array through AMIO's async pipeline
5. Flushes and closes

## Manifest

```yaml title="examples/manifests/zarr3_manifest.yaml"
amio:
  version: "0.1.0"
  staging:
    buffer_count: 4
    buffer_size_bytes: 20971520   # 20 MiB per buffer
  workers:
    thread_count: 2
    pin_threads: false
  backend:
    name: "zarr3"
    options:
      mode: "nczarr"
      compression:
        algorithm: "zstd"
        level: 3
      chunking:
        auto: true
  output:
    path: "output_zarr.zarr"
    overwrite: true
```

## Source Code

```c title="examples/c/example_zarr_write.c"
--8<-- "examples/c/example_zarr_write.c"
```

??? note "Inline source (if includes are not configured)"

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <amio/amio.h>

    #define NLON 360
    #define NLAT 180

    int main(void)
    {
        amio_core_handle core = NULL;
        amio_dataset_handle dataset = NULL;
        amio_io_handle io = NULL;

        amio_status_t rc = amio_init(
            "examples/manifests/zarr3_manifest.yaml", &core);

        rc = amio_open_dataset(core,
            "examples/manifests/zarr3_manifest.yaml",
            AMIO_MODE_WRITE, &dataset);

        double *sst = malloc(NLON * NLAT * sizeof(double));
        // ... fill with synthetic SST data ...

        amio_shape_t shape = {0};
        shape.rank = 2;
        shape.extents[0] = NLON;
        shape.extents[1] = NLAT;

        rc = amio_write(dataset, "sea_surface_temperature", sst,
                        AMIO_DTYPE_F64, &shape, &io);

        rc = amio_flush(dataset, 0);
        rc = amio_close_dataset(dataset);
        rc = amio_finalize(core);
        free(sst);

        return EXIT_SUCCESS;
    }
    ```

## Key Points

!!! info "NCZarr Fallback"
    When TensorStore is not available at build time, AMIO uses the system
    netCDF-c library's NCZarr implementation. This mode supports local file
    output but not cloud storage or Zarr v3 sharding.

!!! tip "Double Precision"
    This example uses `AMIO_DTYPE_F64` (double) for the SST field. The shape
    descriptor is the same regardless of element type — only the `dtype`
    argument to `amio_write` changes.

## Build

```bash
cc -o example_zarr_write example_zarr_write.c -lamio -lm
./example_zarr_write
```
