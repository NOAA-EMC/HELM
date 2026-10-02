/**
 * @file example_zarr_write.c
 * @brief Example: Write a 2D sea surface temperature field to Zarr v3 using AMIO.
 *
 * This example demonstrates writing a 2D double-precision array
 * (sea surface temperature, 360 x 180: lon x lat) to a Zarr v3 store
 * using the NCZarr fallback mode.
 *
 * Steps:
 *   1. Initialize AMIO with a manifest targeting the zarr3 backend
 *   2. Open a dataset for writing
 *   3. Populate a 2D double array with synthetic SST data
 *   4. Write the array through AMIO's async pipeline
 *   5. Flush and close
 *   6. Finalize AMIO
 *
 * Build:
 *   cc -o example_zarr_write example_zarr_write.c -lamio -lm
 *
 * Run:
 *   ./example_zarr_write
 */

#include <amio/amio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Grid dimensions: 1-degree global ocean grid */
#define NLON 360 /* longitude points (0 to 359) */
#define NLAT 180 /* latitude points (-90 to +89) */

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
 * @brief Generate synthetic sea surface temperature data.
 *
 * Creates a realistic SST pattern:
 *   - Warm equatorial waters (~300 K / 27°C)
 *   - Cold polar waters (~271 K / -2°C)
 *   - Western boundary current warm anomalies
 *   - Land mask represented as NaN (not-a-number)
 */
static void fill_sst(double *data, int nlon, int nlat) {
    for (int j = 0; j < nlat; j++) {
        double lat = -90.0 + (double)j + 0.5; /* cell center */
        /* Base SST: warm at equator, cold at poles */
        double base_sst = 300.0 - 29.0 * (lat * lat) / (90.0 * 90.0);
        for (int i = 0; i < nlon; i++) {
            double lon = (double)i + 0.5; /* cell center */
            double sst = base_sst;
            /* Add western boundary current warm anomaly (Gulf Stream-like) */
            if (lon > 280.0 && lon < 320.0 && lat > 25.0 && lat < 45.0) {
                sst += 3.0 * exp(-0.01 * (lon - 300.0) * (lon - 300.0));
            }
            /* Add small random-like perturbation using sin */
            sst += 0.5 * sin(lon * 0.1) * cos(lat * 0.15);
            /* Clamp to physical range (freezing point to ~35°C) */
            if (sst < 271.15) sst = 271.15;
            if (sst > 308.15) sst = 308.15;
            data[j * nlon + i] = sst;
        }
    }
}

int main(void) {
    amio_status_t rc;
    amio_core_handle core = NULL;
    amio_dataset_handle dataset = NULL;
    amio_io_handle io = NULL;

    printf("AMIO Zarr v3 (NCZarr) Write Example\n");
    printf("====================================\n");
    printf("Writing 2D SST field (%d x %d) to Zarr v3 store\n\n", NLON, NLAT);

    /* ---------------------------------------------------------------
     * Step 1: Initialize AMIO with the zarr3 manifest.
     *
     * The zarr3 manifest configures NCZarr fallback mode, which uses
     * the system netCDF-c library's NCZarr implementation to write
     * Zarr v3-compatible stores.
     * --------------------------------------------------------------- */
    printf("Step 1: Initializing AMIO (Zarr v3 / NCZarr mode)...\n");
    rc = amio_init("examples/manifests/zarr3_manifest.yaml", &core);
    check_amio(rc, "amio_init");
    printf("  AMIO initialized.\n");

    /* ---------------------------------------------------------------
     * Step 2: Open a dataset for writing.
     * --------------------------------------------------------------- */
    printf("Step 2: Opening Zarr dataset for writing...\n");
    rc = amio_open_dataset(core, "examples/manifests/zarr3_manifest.yaml", AMIO_MODE_WRITE, &dataset);
    check_amio(rc, "amio_open_dataset");
    printf("  Dataset opened.\n");

    /* ---------------------------------------------------------------
     * Step 3: Allocate and fill the SST array.
     * --------------------------------------------------------------- */
    printf("Step 3: Generating synthetic SST data...\n");
    double *sst = (double *)malloc(NLON * NLAT * sizeof(double));
    if (!sst) {
        fprintf(stderr, "Failed to allocate SST array\n");
        return EXIT_FAILURE;
    }
    fill_sst(sst, NLON, NLAT);
    printf("  SST data generated: range ~ [271.2, 308.1] K\n");

    /* ---------------------------------------------------------------
     * Step 4: Write the 2D SST array.
     *
     * Shape descriptor for a 2D array:
     *   - rank = 2
     *   - extents = {NLON, NLAT}
     *   - strides = {0, 0} (contiguous)
     *   - dtype = AMIO_DTYPE_F64 (double precision)
     * --------------------------------------------------------------- */
    printf("Step 4: Writing SST field...\n");
    amio_shape_t shape;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 2;
    shape.extents[0] = NLON;
    shape.extents[1] = NLAT;

    rc = amio_write(dataset, "sea_surface_temperature", sst, AMIO_DTYPE_F64, &shape, &io);
    check_amio(rc, "amio_write(sea_surface_temperature)");
    printf("  Write enqueued.\n");

    /* ---------------------------------------------------------------
     * Step 5: Flush to ensure the write completes.
     * --------------------------------------------------------------- */
    printf("Step 5: Flushing...\n");
    rc = amio_flush(dataset, 0);
    check_amio(rc, "amio_flush");
    printf("  Flush complete.\n");

    /* ---------------------------------------------------------------
     * Step 6: Close and finalize.
     * --------------------------------------------------------------- */
    printf("Step 6: Closing dataset and finalizing...\n");
    rc = amio_close_dataset(dataset);
    check_amio(rc, "amio_close_dataset");

    rc = amio_finalize(core);
    check_amio(rc, "amio_finalize");
    printf("  Done.\n");

    free(sst);

    printf("\nSST field written to Zarr v3 store.\n");
    return EXIT_SUCCESS;
}
