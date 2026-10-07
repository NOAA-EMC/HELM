/**
 * @file example_netcdf4_read.c
 * @brief Example: Read a variable from a NetCDF-4 file using AMIO.
 */

#include <amio/amio.h>
#include <stdio.h>
#include <stdlib.h>

static void check_amio(amio_status_t rc, const char *context) {
    if (rc != AMIO_OK) {
        fprintf(stderr, "AMIO error in %s: %s (code %d)\n", context, amio_strerror(rc), (int)rc);
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    amio_status_t rc;
    amio_core_handle core = NULL;
    amio_dataset_handle dataset = NULL;

    printf("AMIO NetCDF-4 Read Example\n");
    printf("==========================\n\n");

    /* Initialize AMIO */
    rc = amio_init("examples/manifests/netcdf4_manifest.yaml", &core);
    check_amio(rc, "amio_init");

    /* Open dataset for reading */
    rc = amio_open_dataset(core, "examples/manifests/netcdf4_manifest.yaml", AMIO_MODE_READ, &dataset);
    check_amio(rc, "amio_open_dataset(READ)");

    /* Read temperature at timestep 0 (full field, no bounding box) */
    amio_view_handle view = NULL;
    rc = amio_read(dataset, "temperature", 0, NULL, &view);
    check_amio(rc, "amio_read");

    /* Access the data through the view */
    const void *data = NULL;
    size_t nbytes = 0;
    rc = amio_view_data(view, &data, &nbytes);
    check_amio(rc, "amio_view_data");

    printf("Read %zu bytes of temperature data.\n", nbytes);
    printf("First 5 values: ");
    const float *temp = (const float *)data;
    for (int i = 0; i < 5 && i < (int)(nbytes / sizeof(float)); i++) {
        printf("%.2f ", temp[i]);
    }
    printf("\n");

    /* Release the view (returns staging buffer to the pool) */
    rc = amio_release_view(view);
    check_amio(rc, "amio_release_view");

    /* Close and finalize */
    rc = amio_close_dataset(dataset);
    check_amio(rc, "amio_close_dataset");

    rc = amio_finalize(core);
    check_amio(rc, "amio_finalize");

    printf("\nDone!\n");
    return EXIT_SUCCESS;
}
