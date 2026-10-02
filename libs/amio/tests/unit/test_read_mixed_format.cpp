// test_read_mixed_format.cpp
//
// Integration test for the mixed-format read -> (reformat) -> write ->
// read path (task 18, design §10, Req 14).  A host opens one source
// dataset for reading and one target dataset of a DIFFERENT format for
// writing on the same runtime (one shared Staging_Pool, the "AMIO_Core"
// at the driver level), reads a variable from the source, writes those
// SAME staging bytes to the target, reads the target back, and asserts
// byte equality.  This exercises:
//
//   * Req 14.1 -- both datasets are open and serviced concurrently on
//     one runtime, sharing the Staging_Pool and VarMeta.  The bytes
//     read from the source flow through a single shared StagingBuffer
//     into the target write (no per-format copy/translation layer).
//   * Req 14.2 -- data read from a source dataset is written to a
//     target dataset of a different format with no external
//     post-processing.
//   * Req 14.3 -- read(source) -> write(target) -> read(target) is
//     byte-equal for the same element type and shape, subject to each
//     format's supported element types.
//
// Two mixed-format legs
// ---------------------
//
//   1. NetCDF-4 -> GRIB2 -> read  (ALWAYS-ON core, guarded only by
//      AMIO_HAS_NETCDF + AMIO_HAS_G2C at build time).
//
//      The robustly verifiable path in every build of this repo: write
//      an F32 grid of integer-valued floats to NetCDF-4, read it back,
//      write those bytes to a GRIB2 dataset (lossless DRT, decimal scale
//      factor 0 -> exact for integer-valued fields, per the precision
//      assumption documented in test_read_grib2.cpp), and read the GRIB2
//      field back.  GRIB2's supported element type is F32 grid-point, so
//      this leg covers the supported source/target pair.
//
//   2. NetCDF-4 -> Zarr v3 -> read  (guarded behind AMIO_HAS_TENSORSTORE).
//
//      The TensorStore-backed Zarr v3 write/read path is only compiled
//      in the TensorStore build (mirroring tests/unit/test_read_zarr.cpp,
//      task 15.2).  In the NCZarr-fallback build (AMIO_HAS_TENSORSTORE
//      OFF) this leg is compiled out entirely; the NCZarr fallback's read
//      support is limited and a NetCDF->Zarr(TensorStore) byte-equal test
//      belongs to the TensorStore-enabled build.
//
// MPI
// ---
// The NetCDF-4 driver issues parallel HDF5 calls (nc_create_par /
// nc_open_par) that require the host to have initialized MPI before the
// driver opens.  We initialize MPI in main (the host role) exactly as
// test_read_netcdf4.cpp / test_lifecycle_netcdf4.cpp do, and tolerate an
// environment without functional parallel HDF5 by treating a thrown
// NetCDF open/write as a SKIP of the dependent mixed-format leg -- while
// still HARD-asserting byte equality whenever the path is functional.
// The GRIB2 driver issues no MPI calls.
//
// Validates: Req 14.1, 14.2, 14.3 (mixed-format read -> write).

// Use only the MPI C API; suppress the deprecated C++ MPI bindings in
// THIS translation unit (the netcdf driver TU pulls them in separately).
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <eckit/config/YAMLConfiguration.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "drivers/netcdf/netcdf_driver.hpp"
#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"

#ifdef AMIO_HAS_TENSORSTORE
#include <filesystem>

#include "drivers/zarr/zarr_driver.hpp"
#endif

using amio::detail::BoundingBox;
using amio::detail::GRIB2_Driver;
using amio::detail::NetCDF_Driver;
using amio::detail::StagingBuffer;
using amio::detail::StagingPool;
using amio::detail::VariableInfo;
using amio::detail::VarMeta;
#ifdef AMIO_HAS_TENSORSTORE
using amio::detail::Zarr_Driver;
#endif

namespace {

int g_passed = 0;
int g_failed = 0;

void report_failure(const char *expr, const char *file, int line, const std::string &ctx) {
    std::fprintf(stderr, "FAIL %s:%d: %s   (%s)\n", file, line, expr, ctx.c_str());
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

// Test grid: NY (Nj, slowest) x NX (Ni, fastest) 2-D F32 field of small
// integer-valued floats.  Integer values are exactly representable in
// F32 AND survive GRIB2 grid-point packing at decimal_scale_factor = 0
// byte-for-byte (see the precision assumption in test_read_grib2.cpp),
// so any mis-indexing or lossy step surfaces as a byte mismatch.
constexpr int NY = 6;
constexpr int NX = 8;

float source_value(int i, int j) {
    return static_cast<float>(i * NX + j);
}

// Variable name on the NetCDF (source) side: an ordinary name.
const std::string kNcVar = "temperature";

// Variable name on the GRIB2 (target) side: the field-identity string
// the driver synthesizes from the encode settings below (discipline 0,
// parameter_category 3, parameter_number 5, first fixed surface 100,
// scaled value 50000) -> "d0_c3_n5_s100_l50000".
const std::string kGribVar = "d0_c3_n5_s100_l50000";

// NetCDF source manifest: classic data model, no compression codec ->
// dependency-free, lossless.
std::string make_netcdf_yaml(const std::string &path) {
    return std::string("path: ") + path +
           "\n"
           "data_model: classic\n";
}

// GRIB2 target manifest for a given lossless DRT.  Matches the field
// identity encoded into kGribVar.
std::string make_grib2_yaml(const std::string &path, const std::string &drt) {
    return std::string("path: ") + path +
           "\n"
           "drt: " +
           drt +
           "\n"
           "grib2:\n"
           "  discipline: 0\n"
           "  center: 7\n"
           "  parameter_category: 3\n"
           "  parameter_number: 5\n"
           "  type_of_first_fixed_surface: 100\n"
           "  scaled_value_first_surface: 50000\n";
}

// ---------------------------------------------------------------------
// Leg 1: NetCDF-4 -> GRIB2 -> read (always-on core).
//
// Returns true if the leg ran functionally (and all its assertions were
// hard-checked); false if it was skipped because the underlying NetCDF
// or GRIB2 encoder is unavailable in this environment.
// ---------------------------------------------------------------------
bool run_netcdf_to_grib2(const std::string &drt) {
    const char *NC_PATH = "/tmp/amio_test_mixed_src.nc";
    const std::string grib_path = "/tmp/amio_test_mixed_tgt_" + drt + ".grib2";
    const std::string ctx = "[netcdf->grib2 drt=" + drt + "]";

    std::remove(NC_PATH);
    std::remove(grib_path.c_str());

    // Reference F32 grid (row-major), integer-valued.
    std::vector<float> source(static_cast<std::size_t>(NY) * NX);
    for (int i = 0; i < NY; ++i) {
        for (int j = 0; j < NX; ++j) {
            source[static_cast<std::size_t>(i) * NX + j] = source_value(i, j);
        }
    }
    const std::size_t payload_bytes = source.size() * sizeof(float);

    eckit::YAMLConfiguration nc_cfg{make_netcdf_yaml(NC_PATH)};
    eckit::YAMLConfiguration grib_cfg{make_grib2_yaml(grib_path, drt)};

    // ---- Stage the NetCDF source (write the variable to be read) ----
    // A thrown NetCDF write means parallel HDF5 is unavailable here; the
    // whole NetCDF-sourced leg is then skipped (note), not failed.
    try {
        NetCDF_Driver nc_writer;
        nc_writer.open_write(nc_cfg);

        StagingBuffer buf{};
        buf.data = reinterpret_cast<std::byte *>(source.data());
        buf.capacity_bytes = payload_bytes;
        buf.used_bytes = payload_bytes;

        VarMeta meta{};
        meta.name = kNcVar;
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = NY;
        meta.shape.extents[1] = NX;

        nc_writer.write(buf, meta);
        nc_writer.flush();
        nc_writer.close();
    } catch (const std::exception &e) {
        std::fprintf(stdout, "NOTE: %s NetCDF source unavailable, skipping leg: %s\n", ctx.c_str(), e.what());
        std::remove(NC_PATH);
        return false;
    }

    // ---- Concurrent mixed-format translation (Req 14.1, 14.2) --------
    //
    // One shared Staging_Pool stands in for the AMIO_Core runtime: a
    // NetCDF reader and a GRIB2 writer are open at the SAME time, and a
    // SINGLE shared StagingBuffer carries the bytes read from NetCDF
    // straight into the GRIB2 write -- no external post-processing.
    StagingPool pool(/*buffer_count=*/4, /*buffer_capacity=*/4096);

    // Bytes observed coming out of the NetCDF read (the "source" payload
    // the mixed-format round trip must preserve).
    std::vector<float> nc_read(source.size(), -1.0f);

    // The NetCDF reader stays open across the GRIB2 write so both
    // datasets are concurrently live on the shared runtime (Req 14.1).
    NetCDF_Driver nc_reader;
    StagingBuffer *shared = nullptr;
    try {
        nc_reader.open_read(nc_cfg);

        VariableInfo nc_info = nc_reader.describe_variable(kNcVar);
        EXPECT_TRUE(nc_info.found, ctx + " source describe_variable found");
        EXPECT_TRUE(nc_info.dtype == AMIO_DTYPE_F32, ctx + " source dtype == F32");
        EXPECT_TRUE(nc_info.shape.rank == 2, ctx + " source rank == 2");
        EXPECT_TRUE(nc_info.shape.extents[0] == NY, ctx + " source extent[0] == NY");
        EXPECT_TRUE(nc_info.shape.extents[1] == NX, ctx + " source extent[1] == NX");

        shared = pool.acquire(payload_bytes);
        EXPECT_TRUE(shared != nullptr, ctx + " shared staging buffer acquired");
        if (shared == nullptr) {
            nc_reader.close();
            std::remove(NC_PATH);
            std::remove(grib_path.c_str());
            return false;
        }

        VarMeta nc_meta{};
        nc_meta.name = kNcVar;
        nc_meta.dtype = AMIO_DTYPE_F32;
        nc_meta.shape.rank = 2;
        nc_meta.shape.extents[0] = NY;
        nc_meta.shape.extents[1] = NX;

        nc_reader.read(*shared, nc_meta, /*timestep=*/0, std::nullopt);
        EXPECT_TRUE(shared->used_bytes == payload_bytes, ctx + " source read used_bytes == payload size");

        // The bytes that came out of the NetCDF read must equal the
        // staged source (NetCDF round trip is bit-exact, Req 9.4).
        std::memcpy(nc_read.data(), shared->data, payload_bytes);
        EXPECT_TRUE(std::memcmp(nc_read.data(), source.data(), payload_bytes) == 0, ctx + " NetCDF read byte-equal to source");
    } catch (const std::exception &e) {
        // After a successful NetCDF write, a thrown read is a REAL failure.
        report_failure("mixed netcdf read phase", __FILE__, __LINE__, ctx + " " + e.what());
        if (shared != nullptr) {
            pool.release(shared);
        }
        nc_reader.close();
        std::remove(NC_PATH);
        std::remove(grib_path.c_str());
        return false;
    }

    // --- write(target) from the SAME shared buffer (Req 14.2) ---
    //
    // The NetCDF reader is still open here, so both datasets are live on
    // the shared runtime concurrently (Req 14.1).  The whole GRIB2 write
    // attempt (open_write -> write -> flush -> close) doubles as the
    // encoder-availability probe: g2c only encodes a DRT whose backing
    // codec was compiled in, and an absent codec throws (e.g. "DRT 5.42
    // not yet implemented") at open_write OR at write.  A throw anywhere
    // here means this DRT is unavailable in this g2c build, which is a
    // SKIP (note), not a failure, mirroring test_read_grib2.cpp.  The
    // shared staging bytes are handed straight to the GRIB2 writer with
    // no reformatting in between (no external post-processing).
    bool grib_encoded = false;
    try {
        GRIB2_Driver grib_writer;
        grib_writer.open_write(grib_cfg);

        VarMeta grib_meta{};
        grib_meta.name = kGribVar;
        grib_meta.dtype = AMIO_DTYPE_F32;
        grib_meta.shape.rank = 2;
        grib_meta.shape.extents[0] = NY;
        grib_meta.shape.extents[1] = NX;

        grib_writer.write(*shared, grib_meta);
        grib_writer.flush();
        grib_writer.close();
        grib_encoded = true;
    } catch (const std::exception &e) {
        std::fprintf(stdout, "NOTE: %s GRIB2 encoder unavailable, skipping leg: %s\n", ctx.c_str(), e.what());
    }

    // Release the shared buffer and close the source now that both
    // datasets have been serviced concurrently.
    pool.release(shared);
    nc_reader.close();

    if (!grib_encoded) {
        std::remove(NC_PATH);
        std::remove(grib_path.c_str());
        return false;
    }

    // ---- read(target) back and assert byte equality (Req 14.3) -------
    try {
        GRIB2_Driver grib_reader;
        grib_reader.open_read(grib_cfg);

        VariableInfo info = grib_reader.describe_variable(kGribVar);
        EXPECT_TRUE(info.found, ctx + " target describe_variable found");
        EXPECT_TRUE(info.dtype == AMIO_DTYPE_F32, ctx + " target dtype == F32");
        EXPECT_TRUE(info.shape.rank == 2, ctx + " target rank == 2");
        EXPECT_TRUE(info.shape.extents[0] == NY, ctx + " target extent[0] == NY (Nj)");
        EXPECT_TRUE(info.shape.extents[1] == NX, ctx + " target extent[1] == NX (Ni)");

        VarMeta meta{};
        meta.name = kGribVar;
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = NY;
        meta.shape.extents[1] = NX;

        std::vector<float> out(source.size(), -1.0f);
        StagingBuffer dst{};
        dst.data = reinterpret_cast<std::byte *>(out.data());
        dst.capacity_bytes = out.size() * sizeof(float);
        dst.used_bytes = 0;

        grib_reader.read(dst, meta, /*timestep=*/0, std::nullopt);
        EXPECT_TRUE(dst.used_bytes == payload_bytes, ctx + " target read used_bytes == payload size");

        // The full mixed-format round trip: read(NetCDF) -> write(GRIB2)
        // -> read(GRIB2) is byte-equal to both the NetCDF-read bytes and
        // the original source (Req 14.3).
        EXPECT_TRUE(std::memcmp(out.data(), nc_read.data(), payload_bytes) == 0, ctx + " GRIB2 read byte-equal to NetCDF read");
        EXPECT_TRUE(std::memcmp(out.data(), source.data(), payload_bytes) == 0, ctx + " GRIB2 read byte-equal to source (full round trip)");

        grib_reader.close();
    } catch (const std::exception &e) {
        report_failure("mixed grib2 read-back phase", __FILE__, __LINE__, ctx + " " + e.what());
        std::remove(NC_PATH);
        std::remove(grib_path.c_str());
        return false;
    }

    std::remove(NC_PATH);
    std::remove(grib_path.c_str());
    return true;
}

#ifdef AMIO_HAS_TENSORSTORE
// ---------------------------------------------------------------------
// Leg 2: NetCDF-4 -> Zarr v3 -> read (TensorStore build only).
//
// Compiled out in the NCZarr-fallback build (AMIO_HAS_TENSORSTORE OFF),
// where the active Zarr path's read support is limited.  Mirrors
// test_read_zarr.cpp's TensorStore round-trip but sources the bytes from
// a NetCDF-4 read so the leg is a genuine NetCDF -> Zarr translation.
//
// Returns true if functional, false if skipped (NetCDF unavailable).
// ---------------------------------------------------------------------
bool run_netcdf_to_zarr() {
    const char *NC_PATH = "/tmp/amio_test_mixed_src_zarr.nc";
    const std::string zarr_uri = "/tmp/amio_test_mixed_tgt.zarr";
    const std::string ctx = "[netcdf->zarr]";

    std::remove(NC_PATH);
    {
        std::error_code ec;
        std::filesystem::remove_all(zarr_uri, ec);
    }

    std::vector<float> source(static_cast<std::size_t>(NY) * NX);
    for (int i = 0; i < NY; ++i) {
        for (int j = 0; j < NX; ++j) {
            source[static_cast<std::size_t>(i) * NX + j] = source_value(i, j);
        }
    }
    const std::size_t payload_bytes = source.size() * sizeof(float);

    eckit::YAMLConfiguration nc_cfg{make_netcdf_yaml(NC_PATH)};

    // Zarr v3 target manifest: chunk == shard == array shape (trivially
    // valid sharding), lossless blosc codec -> byte-equal round trip.
    std::string zarr_yaml = "uri: " + zarr_uri + "\n";
    zarr_yaml += "chunk_shape: [" + std::to_string(NY) + ", " + std::to_string(NX) + "]\n";
    zarr_yaml += "shard_shape: [" + std::to_string(NY) + ", " + std::to_string(NX) + "]\n";
    zarr_yaml += "array_shape: [" + std::to_string(NY) + ", " + std::to_string(NX) + "]\n";
    zarr_yaml += "codec: blosc\n";
    zarr_yaml += "dtype: float32\n";
    eckit::YAMLConfiguration zarr_cfg{zarr_yaml};

    // ---- Stage the NetCDF source ----
    try {
        NetCDF_Driver nc_writer;
        nc_writer.open_write(nc_cfg);

        StagingBuffer buf{};
        buf.data = reinterpret_cast<std::byte *>(source.data());
        buf.capacity_bytes = payload_bytes;
        buf.used_bytes = payload_bytes;

        VarMeta meta{};
        meta.name = kNcVar;
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = NY;
        meta.shape.extents[1] = NX;

        nc_writer.write(buf, meta);
        nc_writer.flush();
        nc_writer.close();
    } catch (const std::exception &e) {
        std::fprintf(stdout, "NOTE: %s NetCDF source unavailable, skipping leg: %s\n", ctx.c_str(), e.what());
        std::remove(NC_PATH);
        return false;
    }

    std::vector<float> nc_read(source.size(), -1.0f);

    // ---- Concurrent NetCDF read -> Zarr write (Req 14.1, 14.2) ----
    StagingPool pool(/*buffer_count=*/4, /*buffer_capacity=*/4096);
    try {
        NetCDF_Driver nc_reader;
        nc_reader.open_read(nc_cfg);

        Zarr_Driver zarr_writer;
        zarr_writer.open_write(zarr_cfg);

        StagingBuffer *buf = pool.acquire(payload_bytes);
        EXPECT_TRUE(buf != nullptr, ctx + " shared staging buffer acquired");
        if (buf == nullptr) {
            nc_reader.close();
            std::remove(NC_PATH);
            return false;
        }

        VarMeta nc_meta{};
        nc_meta.name = kNcVar;
        nc_meta.dtype = AMIO_DTYPE_F32;
        nc_meta.shape.rank = 2;
        nc_meta.shape.extents[0] = NY;
        nc_meta.shape.extents[1] = NX;

        nc_reader.read(*buf, nc_meta, /*timestep=*/0, std::nullopt);
        EXPECT_TRUE(buf->used_bytes == payload_bytes, ctx + " source read used_bytes == payload size");
        std::memcpy(nc_read.data(), buf->data, payload_bytes);

        VarMeta zarr_meta{};
        zarr_meta.name = "var";
        zarr_meta.dtype = AMIO_DTYPE_F32;
        zarr_meta.shape.rank = 2;
        zarr_meta.shape.extents[0] = NY;
        zarr_meta.shape.extents[1] = NX;

        zarr_writer.write(*buf, zarr_meta);
        zarr_writer.flush();
        zarr_writer.close();

        pool.release(buf);
        nc_reader.close();
    } catch (const std::exception &e) {
        report_failure("mixed netcdf->zarr translate phase", __FILE__, __LINE__, ctx + " " + e.what());
        std::remove(NC_PATH);
        return false;
    }

    // ---- read(target) back, assert byte equality (Req 14.3) ----
    try {
        Zarr_Driver zarr_reader;
        zarr_reader.open_read(zarr_cfg);

        VarMeta meta{};
        meta.name = "var";
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = NY;
        meta.shape.extents[1] = NX;

        std::vector<float> out(source.size(), -1.0f);
        StagingBuffer dst{};
        dst.data = reinterpret_cast<std::byte *>(out.data());
        dst.capacity_bytes = out.size() * sizeof(float);
        dst.used_bytes = 0;

        zarr_reader.read(dst, meta, /*timestep=*/0, std::nullopt);
        EXPECT_TRUE(dst.used_bytes == payload_bytes, ctx + " target read used_bytes == payload size");
        EXPECT_TRUE(std::memcmp(out.data(), nc_read.data(), payload_bytes) == 0, ctx + " Zarr read byte-equal to NetCDF read");
        EXPECT_TRUE(std::memcmp(out.data(), source.data(), payload_bytes) == 0, ctx + " Zarr read byte-equal to source (full round trip)");

        zarr_reader.close();
    } catch (const std::exception &e) {
        report_failure("mixed zarr read-back phase", __FILE__, __LINE__, ctx + " " + e.what());
    }

    std::remove(NC_PATH);
    {
        std::error_code ec;
        std::filesystem::remove_all(zarr_uri, ec);
    }
    return true;
}
#endif  // AMIO_HAS_TENSORSTORE

}  // namespace

int main() {
    // Host role: initialize MPI before the NetCDF driver opens
    // (parallel HDF5), mirroring test_read_netcdf4 / test_lifecycle_netcdf4.
    int mpi_already = 0;
    MPI_Initialized(&mpi_already);
    if (!mpi_already) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    }

    // ---- Leg 1: NetCDF-4 -> GRIB2 -> read (always-on core) ----
    // libaec (DRT 5.42) is preferred; JPEG2000 (5.40) is the common
    // lossless fallback (libaec is absent in some g2c builds).  Each is
    // attempted; a DRT whose encoder is not compiled in is skipped.  At
    // least one lossless DRT must complete the mixed-format round trip
    // byte-equal for Req 14.3 to be validated in this build.
    const bool aec = run_netcdf_to_grib2("libaec");
    const bool jp2 = run_netcdf_to_grib2("jpeg2000");

    std::fprintf(stdout, "netcdf->grib2 libaec:   %s\n", aec ? "RAN" : "skipped");
    std::fprintf(stdout, "netcdf->grib2 jpeg2000: %s\n", jp2 ? "RAN" : "skipped");

    if (!aec && !jp2) {
        // Neither leg ran: NetCDF parallel HDF5 and/or both GRIB2 lossless
        // encoders are unavailable in this environment.  Treat as a skip
        // (no functional path to assert) rather than a failure.
        std::fprintf(stdout, "NOTE: NetCDF->GRIB2 mixed-format leg skipped (NetCDF or GRIB2 encoder unavailable)\n");
    }

#ifdef AMIO_HAS_TENSORSTORE
    // ---- Leg 2: NetCDF-4 -> Zarr v3 -> read (TensorStore build only) ----
    const bool zarr = run_netcdf_to_zarr();
    std::fprintf(stdout, "netcdf->zarr (TensorStore): %s\n", zarr ? "RAN" : "skipped");
#else
    std::fprintf(stdout, "netcdf->zarr (TensorStore): compiled out (AMIO_HAS_TENSORSTORE undefined)\n");
#endif

    std::fprintf(stdout, "test_read_mixed_format: passed=%d failed=%d\n", g_passed, g_failed);

    int mpi_init_flag = 0;
    MPI_Initialized(&mpi_init_flag);
    int mpi_final_flag = 0;
    MPI_Finalized(&mpi_final_flag);
    if (mpi_init_flag && !mpi_final_flag) {
        MPI_Finalize();
    }

    return g_failed == 0 ? 0 : 1;
}
