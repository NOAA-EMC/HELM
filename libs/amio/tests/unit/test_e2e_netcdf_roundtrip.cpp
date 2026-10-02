// test_e2e_netcdf_roundtrip.cpp -- End-to-end public-C-API round-trip
// test for the NetCDF-4 backend.
//
// Exercises the FULL public C API lifecycle for a write → read round
// trip with the real netcdf4 driver:
//
//   amio_init → amio_open_dataset(WRITE) → amio_write → amio_flush
//     → amio_close_dataset → amio_open_dataset(READ) → amio_read
//     → verify → amio_release_view → amio_close_dataset → amio_finalize
//
// Byte-equality verification strategy:
// The public C API does not expose a function to retrieve the raw data
// pointer from an amio_view_handle (the handle table and ViewRecord are
// hidden symbols in the shared library).  To verify byte-for-byte
// equality of the round-tripped data, this test uses netCDF C API
// directly (nc_open / nc_get_var_float) to read the written .nc file
// and compares against the original source array.  This confirms that
// driver->write actually persisted the data to disk correctly.
//
// The amio_read path is also tested: we assert it returns AMIO_OK (not
// AMIO_ERR_STAGING_BACKPRESSURE) and a non-null view handle, confirming
// the read pipeline serves from staging.
//
// Requires: netCDF (parallel HDF5 build) + MPI.  The test initializes
// MPI_THREAD_MULTIPLE as required by the NetCDF-4 parallel driver.
//
// Validates: write-path unstub (driver->write), read-path fidelity,
//            full C API lifecycle.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

// Suppress deprecated C++ MPI bindings.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <netcdf.h>

#include "amio/amio.h"

/* ----------------------------------------------------------------
 * Minimal test harness
 * ---------------------------------------------------------------- */

static int g_passed = 0;
static int g_failed = 0;

static void report_failure(const char *expr, const char *file, int line, const char *ctx) {
    fprintf(stderr, "FAIL %s:%d: %s   (%s)\n", file, line, expr, ctx);
    ++g_failed;
}

#define EXPECT_TRUE(cond, ctx)                                \
    do {                                                      \
        if (!(cond)) {                                        \
            report_failure(#cond, __FILE__, __LINE__, (ctx)); \
        } else {                                              \
            ++g_passed;                                       \
        }                                                     \
    } while (0)

#define EXPECT_EQ(a, b, ctx)                                                                                             \
    do {                                                                                                                 \
        if ((a) != (b)) {                                                                                                \
            char buf[256];                                                                                               \
            std::snprintf(buf, sizeof(buf), "%s: expected %d, got %d", (ctx), static_cast<int>(b), static_cast<int>(a)); \
            report_failure(#a " == " #b, __FILE__, __LINE__, buf);                                                       \
        } else {                                                                                                         \
            ++g_passed;                                                                                                  \
        }                                                                                                                \
    } while (0)

/* ----------------------------------------------------------------
 * Constants
 * ---------------------------------------------------------------- */

static const char *MANIFEST_PATH = "/tmp/amio_test_e2e_netcdf_roundtrip.yaml";
static const char *OUTPUT_PATH = "/tmp/amio_test_e2e_netcdf_roundtrip.nc";

/* ----------------------------------------------------------------
 * Helper: write the manifest YAML to /tmp
 * ---------------------------------------------------------------- */

static int write_manifest(void) {
    std::ofstream ofs(MANIFEST_PATH);
    if (!ofs) return -1;

    ofs << "backend: netcdf4\n";
    ofs << "path: " << OUTPUT_PATH << "\n";
    ofs << "output_path: " << OUTPUT_PATH << "\n";
    ofs << "data_model: classic\n";
    ofs << "staging_pool:\n";
    ofs << "  buffer_count: 8\n";
    ofs << "  buffer_capacity_bytes: 65536\n";
    ofs << "worker_pool:\n";
    ofs << "  threads: 2\n";
    ofs << "prefetch:\n";
    ofs << "  depth: 4\n";
    ofs << "  read_timeout_s: 60\n";
    ofs << "staging_timeout_ms: 10000\n";
    ofs << "codec:\n";
    ofs << "  active_codec: blosc\n";
    ofs << "  lossless_allow_list:\n";
    ofs << "    - blosc\n";
    ofs.close();
    return 0;
}

/* ----------------------------------------------------------------
 * Helper: clean up temp files
 * ---------------------------------------------------------------- */

static void cleanup(void) {
    std::remove(MANIFEST_PATH);
    std::remove(OUTPUT_PATH);
}

/* ----------------------------------------------------------------
 * Helper: verify written data via netCDF C API (nc_open + nc_get_var)
 *
 * Opens the .nc file directly and reads the "temperature" variable,
 * comparing against the original source array for byte equality.
 * This confirms that driver->write actually persisted the data.
 * ---------------------------------------------------------------- */
static void verify_with_netcdf_c_api(const float source[10][10]) {
    int ncid = -1;
    int rc = nc_open(OUTPUT_PATH, NC_NOWRITE, &ncid);
    if (rc != NC_NOERR) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "nc_open failed: %s", nc_strerror(rc));
        report_failure("nc_open == NC_NOERR", __FILE__, __LINE__, buf);
        return;
    }
    ++g_passed;  // nc_open succeeded

    // Look up the variable.
    int varid = -1;
    rc = nc_inq_varid(ncid, "temperature", &varid);
    if (rc != NC_NOERR) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "nc_inq_varid failed: %s", nc_strerror(rc));
        report_failure("nc_inq_varid == NC_NOERR", __FILE__, __LINE__, buf);
        nc_close(ncid);
        return;
    }
    ++g_passed;  // variable found

    // Read the data.
    float read_data[10][10];
    std::memset(read_data, 0, sizeof(read_data));
    rc = nc_get_var_float(ncid, varid, &read_data[0][0]);
    if (rc != NC_NOERR) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "nc_get_var_float failed: %s", nc_strerror(rc));
        report_failure("nc_get_var_float == NC_NOERR", __FILE__, __LINE__, buf);
        nc_close(ncid);
        return;
    }
    ++g_passed;  // read succeeded

    // Byte-for-byte comparison.
    int cmp = std::memcmp(read_data, source, sizeof(read_data));
    EXPECT_EQ(cmp, 0, "netCDF C API: read data matches written source byte-for-byte");

    if (cmp != 0) {
        // Print first mismatch for diagnostics.
        const float *src_flat = &source[0][0];
        const float *read_flat = &read_data[0][0];
        for (int k = 0; k < 100; ++k) {
            if (src_flat[k] != read_flat[k]) {
                fprintf(stderr, "  First mismatch at element %d: wrote %f, read %f\n", k, static_cast<double>(src_flat[k]),
                        static_cast<double>(read_flat[k]));
                break;
            }
        }
    }

    nc_close(ncid);
}

/* ----------------------------------------------------------------
 * The round-trip test
 * ---------------------------------------------------------------- */

static void test_write_read_roundtrip() {
    // Clean up any leftover files.
    cleanup();

    // Write the manifest.
    EXPECT_TRUE(write_manifest() == 0, "write manifest to /tmp");

    // ---- Known source data: 10x10 float array with distinct values ----
    float source[10][10];
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            source[i][j] = static_cast<float>(i * 10 + j) * 1.5f + 0.25f;
        }
    }

    // ==================================================================
    // WRITE PATH: init → open(WRITE) → write → flush → close → finalize
    // ==================================================================

    amio_core_handle core = nullptr;
    amio_status_t rc = amio_init(MANIFEST_PATH, &core);
    EXPECT_EQ(rc, AMIO_OK, "amio_init (write phase)");
    if (rc != AMIO_OK || core == nullptr) {
        fprintf(stdout, "NOTE: amio_init failed with %d (%s), skipping test\n", static_cast<int>(rc), amio_strerror(rc));
        cleanup();
        return;
    }

    amio_dataset_handle write_ds = nullptr;
    rc = amio_open_dataset(core, MANIFEST_PATH, AMIO_MODE_WRITE, &write_ds);
    EXPECT_EQ(rc, AMIO_OK, "amio_open_dataset(WRITE)");
    if (rc != AMIO_OK || write_ds == nullptr) {
        fprintf(stdout, "NOTE: open_dataset(WRITE) failed with %d (%s), skipping\n", static_cast<int>(rc), amio_strerror(rc));
        amio_finalize(core);
        cleanup();
        return;
    }

    // Write the 10x10 float array.
    amio_shape_t shape;
    std::memset(&shape, 0, sizeof(shape));
    shape.rank = 2;
    shape.extents[0] = 10;
    shape.extents[1] = 10;

    amio_io_handle io = nullptr;
    rc = amio_write(write_ds, "temperature", source, AMIO_DTYPE_F32, &shape, &io);
    EXPECT_EQ(rc, AMIO_OK, "amio_write(temperature 10x10 F32)");
    if (rc != AMIO_OK) {
        fprintf(stdout, "NOTE: amio_write failed with %d (%s)\n", static_cast<int>(rc), amio_strerror(rc));
        amio_close_dataset(write_ds);
        amio_finalize(core);
        cleanup();
        return;
    }

    // Flush to ensure the write completes (driver->write is called
    // BEFORE io_rec->completed.store(true), so flush blocks until
    // the data is persisted).
    rc = amio_flush(write_ds, 30000);
    EXPECT_EQ(rc, AMIO_OK, "amio_flush (write phase)");
    if (rc != AMIO_OK) {
        fprintf(stdout, "NOTE: amio_flush failed with %d (%s)\n", static_cast<int>(rc), amio_strerror(rc));
    }

    // Close the write dataset.
    rc = amio_close_dataset(write_ds);
    EXPECT_EQ(rc, AMIO_OK, "amio_close_dataset (write)");
    write_ds = nullptr;

    // Finalize write-phase runtime.
    rc = amio_finalize(core);
    EXPECT_EQ(rc, AMIO_OK, "amio_finalize (write phase)");
    core = nullptr;

    // ==================================================================
    // VERIFY VIA NETCDF C API: read the .nc file directly to confirm
    // driver->write persisted the data correctly (byte equality).
    // ==================================================================

    verify_with_netcdf_c_api(source);

    // ==================================================================
    // READ PATH: init → open(READ) → read → release → close → finalize
    //
    // Asserts the AMIO read pipeline serves from staging with AMIO_OK
    // (not AMIO_ERR_STAGING_BACKPRESSURE).
    // ==================================================================

    rc = amio_init(MANIFEST_PATH, &core);
    EXPECT_EQ(rc, AMIO_OK, "amio_init (read phase)");
    if (rc != AMIO_OK || core == nullptr) {
        fprintf(stdout, "NOTE: amio_init (read) failed with %d (%s), skipping\n", static_cast<int>(rc), amio_strerror(rc));
        cleanup();
        return;
    }

    amio_dataset_handle read_ds = nullptr;
    rc = amio_open_dataset(core, MANIFEST_PATH, AMIO_MODE_READ, &read_ds);
    EXPECT_EQ(rc, AMIO_OK, "amio_open_dataset(READ)");
    if (rc != AMIO_OK || read_ds == nullptr) {
        fprintf(stdout, "NOTE: open_dataset(READ) failed with %d (%s), skipping\n", static_cast<int>(rc), amio_strerror(rc));
        amio_finalize(core);
        cleanup();
        return;
    }

    // Read timestep 0 of "temperature", no bbox (full read).
    amio_view_handle view = nullptr;
    rc = amio_read(read_ds, "temperature", /*timestep=*/0, /*bbox=*/nullptr, &view);

    // Key assertions: NOT AMIO_ERR_STAGING_BACKPRESSURE, and returns AMIO_OK.
    EXPECT_TRUE(rc != AMIO_ERR_STAGING_BACKPRESSURE, "amio_read does NOT return AMIO_ERR_STAGING_BACKPRESSURE");
    EXPECT_EQ(rc, AMIO_OK, "amio_read returns AMIO_OK");
    EXPECT_TRUE(view != nullptr, "amio_read returns a non-null view handle");

    // Retrieve the view's data pointer and compare against source.
    if (view != nullptr) {
        const void *view_ptr = nullptr;
        size_t view_size = 0;
        rc = amio_view_data(view, &view_ptr, &view_size);
        EXPECT_EQ(rc, AMIO_OK, "amio_view_data returns AMIO_OK");
        EXPECT_TRUE(view_ptr != nullptr, "amio_view_data returns non-null pointer");
        EXPECT_EQ(static_cast<int>(view_size), static_cast<int>(sizeof(source)), "amio_view_data size matches source payload");
        if (view_ptr != nullptr && view_size == sizeof(source)) {
            int cmp = std::memcmp(view_ptr, source, sizeof(source));
            EXPECT_EQ(cmp, 0, "amio_view_data: read payload matches written source byte-for-byte");
        }
    }

    // Release the view.
    if (view != nullptr) {
        rc = amio_release_view(view);
        EXPECT_EQ(rc, AMIO_OK, "amio_release_view");
    }

    // Close the read dataset.
    rc = amio_close_dataset(read_ds);
    EXPECT_EQ(rc, AMIO_OK, "amio_close_dataset (read)");
    read_ds = nullptr;

    // Finalize.
    rc = amio_finalize(core);
    EXPECT_EQ(rc, AMIO_OK, "amio_finalize (read phase)");
    core = nullptr;

    // Clean up temp files.
    cleanup();
}

/* ----------------------------------------------------------------
 * Main -- initialize MPI (required by NetCDF-4 parallel driver)
 * ---------------------------------------------------------------- */

int main(int argc, char **argv) {
    int mpi_already = 0;
    MPI_Initialized(&mpi_already);
    if (!mpi_already) {
        int provided = 0;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    }

    test_write_read_roundtrip();

    fprintf(stdout, "test_e2e_netcdf_roundtrip: passed=%d failed=%d\n", g_passed, g_failed);

    int mpi_init_flag = 0;
    MPI_Initialized(&mpi_init_flag);
    int mpi_final_flag = 0;
    MPI_Finalized(&mpi_final_flag);
    if (mpi_init_flag && !mpi_final_flag) {
        MPI_Finalize();
    }

    return g_failed == 0 ? 0 : 1;
}
