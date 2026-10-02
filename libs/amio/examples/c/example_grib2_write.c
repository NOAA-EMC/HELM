/**
 * @file example_grib2_write.c
 * @brief Example: Write a 2D 500hPa geopotential height field to GRIB2 using AMIO.
 *
 * This example demonstrates writing a 2D float array representing
 * the 500 hPa geopotential height field (360 x 181: lon x lat) to
 * a GRIB2 file using the grib2 backend driver (nceplibs-g2c).
 *
 * Steps:
 *   1. Initialize AMIO with a manifest targeting the grib2 backend
 *   2. Open a dataset for writing
 *   3. Populate a 2D float array with synthetic geopotential height
 *   4. Write the array through AMIO's async pipeline
 *   5. Flush and close
 *   6. Finalize AMIO
 *
 * Build:
 *   cc -o example_grib2_write example_grib2_write.c -lamio -lm
 *
 * Run:
 *   ./example_grib2_write
 */

#include <amio/amio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Grid dimensions: 1-degree global grid */
#define NLON 360 /* longitude points (0 to 359) */
#define NLAT 181 /* latitude points (90N to 90S, inclusive) */

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
 * @brief Generate synthetic 500 hPa geopotential height data.
 *
 * Creates a realistic-looking 500 hPa height field:
 *   - Higher values in the tropics (~5880 m)
 *   - Lower values at the poles (~5100 m)
 *   - Rossby wave pattern in midlatitudes
 *   - Units: geopotential meters (gpm)
 */
static void fill_geopotential_height(float *data, int nlon, int nlat) {
    for (int j = 0; j < nlat; j++) {
        /* Latitude from 90N to 90S */
        float lat = 90.0f - (float)j;
        /* Base height: higher in tropics, lower at poles */
        float base_height = 5600.0f + 280.0f * cosf(lat * 3.14159f / 180.0f);
        for (int i = 0; i < nlon; i++) {
            float lon = (float)i;
            float height = base_height;
            /* Add Rossby wave pattern in midlatitudes (wavenumber 4-6) */
            if (fabsf(lat) > 20.0f && fabsf(lat) < 70.0f) {
                float wave_amp = 120.0f * sinf((fabsf(lat) - 20.0f) * 3.14159f / 50.0f);
                /* Wavenumber 5 pattern */
                height += wave_amp * sinf(5.0f * lon * 3.14159f / 180.0f);
                /* Wavenumber 3 pattern (weaker) */
                height += 0.4f * wave_amp * cosf(3.0f * lon * 3.14159f / 180.0f + 1.2f);
            }
            /* Slight asymmetry between hemispheres */
            if (lat < 0.0f) {
                height -= 20.0f;
            }
            data[j * nlon + i] = height;
        }
    }
}

int main(void) {
    amio_status_t rc;
    amio_core_handle core = NULL;
    amio_dataset_handle dataset = NULL;
    amio_io_handle io = NULL;

    printf("AMIO GRIB2 Write Example\n");
    printf("========================\n");
    printf("Writing 2D 500hPa geopotential height (%d x %d) to GRIB2\n\n", NLON, NLAT);

    /* ---------------------------------------------------------------
     * Step 1: Initialize AMIO with the grib2 manifest.
     *
     * The grib2 manifest configures the nceplibs-g2c backend with
     * JPEG2000 compression and complex second-order packing.
     * --------------------------------------------------------------- */
    printf("Step 1: Initializing AMIO (GRIB2 backend)...\n");
    rc = amio_init("examples/manifests/grib2_manifest.yaml", &core);
    check_amio(rc, "amio_init");
    printf("  AMIO initialized.\n");

    /* ---------------------------------------------------------------
     * Step 2: Open a dataset for writing.
     * --------------------------------------------------------------- */
    printf("Step 2: Opening GRIB2 dataset for writing...\n");
    rc = amio_open_dataset(core, "examples/manifests/grib2_manifest.yaml", AMIO_MODE_WRITE, &dataset);
    check_amio(rc, "amio_open_dataset");
    printf("  Dataset opened.\n");

    /* ---------------------------------------------------------------
     * Step 3: Allocate and fill the geopotential height array.
     * --------------------------------------------------------------- */
    printf("Step 3: Generating synthetic 500hPa height data...\n");
    float *hgt = (float *)malloc(NLON * NLAT * sizeof(float));
    if (!hgt) {
        fprintf(stderr, "Failed to allocate height array\n");
        return EXIT_FAILURE;
    }
    fill_geopotential_height(hgt, NLON, NLAT);
    printf("  Height data generated: range ~ [5100, 5880] gpm\n");

    /* ---------------------------------------------------------------
     * Step 4: Write the 2D geopotential height array.
     *
     * For GRIB2, the variable name typically follows WMO conventions.
     * The backend driver maps "geopotential_height" to the appropriate
     * GRIB2 parameter category and number.
     * --------------------------------------------------------------- */
    printf("Step 4: Writing geopotential height field...\n");
    amio_shape_t shape;
    memset(&shape, 0, sizeof(shape));
    shape.rank = 2;
    shape.extents[0] = NLON;
    shape.extents[1] = NLAT;

    rc = amio_write(dataset, "geopotential_height", hgt, AMIO_DTYPE_F32, &shape, &io);
    check_amio(rc, "amio_write(geopotential_height)");
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

    free(hgt);

    printf("\n500hPa geopotential height written to GRIB2.\n");
    return EXIT_SUCCESS;
}
