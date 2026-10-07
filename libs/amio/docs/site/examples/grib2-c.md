# GRIB2 Write Example (C)

This example demonstrates writing a 2D 500 hPa geopotential height field
(360 × 181: lon × lat) to a GRIB2 file using the `grib2` backend driver
(nceplibs-g2c).

## What It Does

1. Initializes AMIO with a manifest targeting the `grib2` backend
2. Opens a dataset for writing
3. Populates a 2D float array with synthetic geopotential height data
4. Writes the array through AMIO's async pipeline
5. Flushes and closes

## Manifest

```yaml title="examples/manifests/grib2_manifest.yaml"
amio:
  version: "0.1.0"
  staging:
    buffer_count: 4
    buffer_size_bytes: 10485760   # 10 MiB per buffer
  workers:
    thread_count: 1
    pin_threads: false
  backend:
    name: "grib2"
    options:
      packing: "complex_second_order"
      discipline: 0
      compression:
        algorithm: "jpeg2000"
        target_ratio: 20
  output:
    path: "output_grib2.grib2"
    overwrite: true
```

## Source Code

```c title="examples/c/example_grib2_write.c"
--8<-- "examples/c/example_grib2_write.c"
```

??? note "Inline source (if includes are not configured)"

    ```c
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <amio/amio.h>

    #define NLON 360
    #define NLAT 181

    int main(void)
    {
        amio_core_handle core = NULL;
        amio_dataset_handle dataset = NULL;
        amio_io_handle io = NULL;

        amio_status_t rc = amio_init(
            "examples/manifests/grib2_manifest.yaml", &core);

        rc = amio_open_dataset(core,
            "examples/manifests/grib2_manifest.yaml",
            AMIO_MODE_WRITE, &dataset);

        float *hgt = malloc(NLON * NLAT * sizeof(float));
        // ... fill with synthetic geopotential height data ...

        amio_shape_t shape = {0};
        shape.rank = 2;
        shape.extents[0] = NLON;
        shape.extents[1] = NLAT;

        rc = amio_write(dataset, "geopotential_height", hgt,
                        AMIO_DTYPE_F32, &shape, &io);

        rc = amio_flush(dataset, 0);
        rc = amio_close_dataset(dataset);
        rc = amio_finalize(core);
        free(hgt);

        return EXIT_SUCCESS;
    }
    ```

## Key Points

!!! info "WMO Variable Naming"
    For GRIB2, the variable name (e.g., `"geopotential_height"`) is mapped
    through a WMO code table to the appropriate GRIB2 parameter category
    and number by the backend driver.

!!! tip "GRIB2 Compression"
    The GRIB2 backend supports lossless compression via JPEG2000 or
    Adaptive Entropy Coding (AEC/libaec). Lossy codecs are rejected
    with `AMIO_ERR_LOSSY_CODEC_FORBIDDEN`.

## Build

```bash
cc -o example_grib2_write example_grib2_write.c -lamio -lm
./example_grib2_write
```
