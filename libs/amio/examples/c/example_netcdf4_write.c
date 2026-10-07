/**
 * @file example_netcdf4_write.c
 * @brief Example: Write a 3D temperature field to NetCDF-4 using AMIO.
 *
 * This example demonstrates the complete AMIO lifecycle for writing a
 * 3D atmospheric temperature field (100 x 50 x 25: lon x lat x level)
 * to a NetCDF-4 file using the netcdf4 backend driver.
 *
 * Steps:
 *   1. Initialize AMIO with a manifest targeting the netcdf4 backend
 *   2. Open a dataset for writing
 *   3. Populate a 3D float array with synthetic temperature data
 *   4. Write the array through AMIO's async pipeline
 *   5. Flush to ensure all writes complete
 *   6. Close the dataset and finalize AMIO
 *
 * Build:
 *   cc -o example_netcdf4_write example_netcdf4_write.c -lamio
 *
 * Run:
 *   ./example_netcdf4_write
 */

#include <amio/amio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Grid dimensions: longitude x latitude x vertical levels */
#define NX 100 /* longitude points */
#define NY 50  /* latitude points */
#define NZ 25  /* vertical levels */

/**
 * @brief Check an AMIO return code and exit on failure.
 */
static void check_amio(amio_status_t rc, const char *context) {
    if (rc != AMIO_OK) {
        fprintf(stderr, "AMIO error in %s: %s (code %d)\n", context, amio_strerror(rc), (int)rc);
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Generate synthetic temperature data.
 *
 * Creates a realistic-looking temperature field that decreases with
 * altitude (standard atmosphere lapse rate ~6.5 K/km) and varies
 * with latitude (warmer at equator, cooler at poles).
 */
static void fill_temperature(float *data, int nx, int ny, int nz) {
    for (int k = 0; k < nz; k++) {
        /* Approximate pressure level altitude (km) */
        float altitude_km = (float)k * 0.5f;
        for (int j = 0; j < ny; j++) {
            /* Latitude from -90 to +90 */
            float lat = -90.0f + (float)j * (180.0f / (float)(ny - 1));
            /* Base temperature varies with latitude (warmer at equator) */
            float base_temp = 288.15f - 30.0f * fabsf(lat) / 90.0f;
            for (int i = 0; i < nx; i++) {
                /* Standard atmosphere lapse rate */
                float temp = base_temp - 6.5f * altitude_km;
                /* Small longitudinal variation */
                float lon = (float)i * (360.0f / (float)nx);
                temp += 2.0f * sinf(lon * 3.14159f / 180.0f);
                data[k * ny * nx + j * nx + i] = temp;
            }
        }
    }
}

int main(void) {
    amio_status_t rc;
    amio_core_handle core = NULL;
    amio_dataset_handle dataset = NULL;
    amio_io_handle io = NULL;

    printf("AMIO NetCDF-4 Write Example\n");
    printf("===========================\n");
    printf("Writing 3D temperature field (%d x %d x %d) to NetCDF-4\n\n", NX, NY, NZ);

    /* ---------------------------------------------------------------
     * Step 1: Initialize AMIO with the netcdf4 manifest.
     *
     * The manifest specifies the backend driver, staging pool size,
     * worker thread count, and output path.
     * --------------------------------------------------------------- */
    printf("Step 1: Initializing AMIO...\n");
    rc = amio_init("examples/manifests/netcdf4_manifest.yaml", &core);
    check_amio(rc, "amio_init");
    printf("  AMIO initialized successfully.\n");

    /* ---------------------------------------------------------------
     * Step 2: Open a dataset for writing.
     *
     * The dataset configuration path can be the same manifest or a
     * separate dataset-specific config. AMIO_MODE_WRITE creates a
     * new file (or overwrites if configured).
     * --------------------------------------------------------------- */
    printf("Step 2: Opening dataset for writing...\n");
    rc = amio_open_dataset(core, "examples/manifests/netcdf4_manifest.yaml", AMIO_MODE_WRITE, &dataset);
    check_amio(rc, "amio_open_dataset");
    printf("  Dataset opened.\n");

    /* ---------------------------------------------------------------
     * Step 3: Allocate and fill the temperature array.
     * --------------------------------------------------------------- */
    printf("Step 3: Generating synthetic temperature data...\n");
    float *temperature = (float *)malloc(NX * NY * NZ * sizeof(float));
    if (!temperature) {
        fprintf(stderr, "Failed to allocate temperature array\n");
        return EXIT_FAILURE;
    }
    fill_temperature(temperature, NX, NY, NZ);
    printf("  Data generated: T range ~ [%.1f, %.1f] K\n", temperature[0], temperature[NX * NY * NZ - 1]);

    /* ---------------------------------------------------------------
     * Step 4: Write the 3D array through AMIO.
     *
     * We construct an amio_shape_t describing the array layout:
     *   - rank = 3
     *   - extents = {NX, NY, NZ} (row-major: fastest-varying first)
     *   - strides = {0, 0, 0} (contiguous, AMIO derives strides)
     *
     * amio_write() copies the data into the staging pool synchronously,
     * then enqueues the backend write asynchronously. The caller can
     * safely free or reuse the host buffer after this call returns.
     * --------------------------------------------------------------- */
    printf("Step 4: Writing temperature field...\n");
    amio_shape_t shape;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 3;
    shape.extents[0] = NX;
    shape.extents[1] = NY;
    shape.extents[2] = NZ;
    /* strides = 0 means contiguous (AMIO derives row-major strides) */

    rc = amio_write(dataset, "temperature", temperature, AMIO_DTYPE_F32, &shape, &io);
    check_amio(rc, "amio_write(temperature)");
    printf("  Write enqueued (async I/O handle obtained).\n");

    /* ---------------------------------------------------------------
     * Step 5: Flush to ensure all pending writes complete.
     *
     * timeout_ms = 0 means wait indefinitely. In production code,
     * you might set a reasonable timeout (e.g., 30000 ms).
     * --------------------------------------------------------------- */
    printf("Step 5: Flushing pending writes...\n");
    rc = amio_flush(dataset, 0);
    check_amio(rc, "amio_flush");
    printf("  All writes completed successfully.\n");

    /* ---------------------------------------------------------------
     * Step 6: Close the dataset and finalize AMIO.
     *
     * amio_close_dataset() flushes any remaining writes and releases
     * the backend driver resources. amio_finalize() joins worker
     * threads and frees the staging pool.
     * --------------------------------------------------------------- */
    printf("Step 6: Closing dataset and finalizing...\n");
    rc = amio_close_dataset(dataset);
    check_amio(rc, "amio_close_dataset");

    rc = amio_finalize(core);
    check_amio(rc, "amio_finalize");
    printf("  AMIO finalized.\n");

    /* Clean up host memory */
    free(temperature);

    printf("\nDone! Temperature field written to NetCDF-4.\n");
    return EXIT_SUCCESS;
}
