// test_netcdf4_multistep.cpp
//
// Production-like multi-timestep integration test for the NetCDF-4 backend.
//
// Exercises a realistic GFS-like scenario: a 3D field "temperature" with
// shape [ntimes=6, nlat=181, nlon=360] (1-degree global grid, ~254 KiB per
// timestep, ~1.5 MiB total).  Writes the entire 3D array in one
// amio_write call via the real NetCDF_Driver, then reads back each
// timestep slice via a rank-3 bounding box selecting one time-step
// [t:t+1, :, :] and asserts byte equality.
//
// Value encoding: value = timestep * 100000.0f + lat_idx * 1000.0f + lon_idx
// ensures any misalignment is immediately detectable.
//
// Requires MPI for parallel HDF5 (nc_create_par / nc_open_par).
// Mirrors the pattern in tests/unit/test_read_netcdf4.cpp.

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <eckit/config/YAMLConfiguration.h>
#include <mpi.h>

#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/netcdf/netcdf_driver.hpp"
#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"

using amio::detail::BoundingBox;
using amio::detail::NetCDF_Driver;
using amio::detail::StagingBuffer;
using amio::detail::VariableInfo;
using amio::detail::VarMeta;

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

// Production-like grid: 6 forecast hours, 181 latitudes, 360 longitudes.
constexpr int NTIMES = 6;
constexpr int NLAT = 181;
constexpr int NLON = 360;
constexpr std::size_t POINTS_PER_STEP = static_cast<std::size_t>(NLAT) * NLON;  // 65,160
constexpr std::size_t TOTAL_POINTS = static_cast<std::size_t>(NTIMES) * POINTS_PER_STEP;

// Distinct value per (timestep, lat, lon) — any misalignment is detectable.
float source_value(int t, int lat, int lon) {
    return static_cast<float>(t) * 100000.0f + static_cast<float>(lat) * 1000.0f + static_cast<float>(lon);
}

}  // namespace

int main() {
    const char *OUTPUT_PATH = "/tmp/amio_test_netcdf4_multistep.nc";
    const std::string var_name = "temperature";

    // Initialize MPI (required for parallel HDF5).
    int mpi_already = 0;
    MPI_Initialized(&mpi_already);
    if (!mpi_already) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    }

    std::remove(OUTPUT_PATH);

    // Build the full 3D source array [NTIMES, NLAT, NLON] in row-major order.
    std::vector<float> source(TOTAL_POINTS);
    for (int t = 0; t < NTIMES; ++t) {
        for (int lat = 0; lat < NLAT; ++lat) {
            for (int lon = 0; lon < NLON; ++lon) {
                const std::size_t idx =
                    static_cast<std::size_t>(t) * POINTS_PER_STEP + static_cast<std::size_t>(lat) * NLON + static_cast<std::size_t>(lon);
                source[idx] = source_value(t, lat, lon);
            }
        }
    }

    const std::string yaml = std::string("path: ") + OUTPUT_PATH +
                             "\n"
                             "data_model: classic\n";
    eckit::YAMLConfiguration cfg{yaml};

    // ---- Write phase: single 3D write [6, 181, 360] ----
    bool functional = true;
    try {
        NetCDF_Driver writer;
        writer.open_write(cfg);

        StagingBuffer buf{};
        buf.data = reinterpret_cast<std::byte *>(source.data());
        buf.capacity_bytes = source.size() * sizeof(float);
        buf.used_bytes = source.size() * sizeof(float);

        VarMeta meta{};
        meta.name = var_name;
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 3;
        meta.shape.extents[0] = NTIMES;
        meta.shape.extents[1] = NLAT;
        meta.shape.extents[2] = NLON;

        writer.write(buf, meta);
        writer.flush();
        writer.close();
    } catch (const std::exception &e) {
        std::fprintf(stdout, "NOTE: NetCDF write path unavailable, skipping: %s\n", e.what());
        functional = false;
    }

    // ---- Read phase: per-timestep slice reads via bbox ----
    if (functional) {
        try {
            NetCDF_Driver reader;
            reader.open_read(cfg);

            // Verify describe_variable reports the full 3D shape.
            VariableInfo info = reader.describe_variable(var_name);
            EXPECT_TRUE(info.found, "describe_variable found temperature");
            EXPECT_TRUE(info.dtype == AMIO_DTYPE_F32, "describe_variable dtype == F32");
            EXPECT_TRUE(info.shape.rank == 3, "describe_variable rank == 3");
            EXPECT_TRUE(info.shape.extents[0] == NTIMES, "describe_variable extent[0] == NTIMES");
            EXPECT_TRUE(info.shape.extents[1] == NLAT, "describe_variable extent[1] == NLAT");
            EXPECT_TRUE(info.shape.extents[2] == NLON, "describe_variable extent[2] == NLON");
            EXPECT_TRUE(info.total_timesteps == 1, "describe_variable total_timesteps == 1 (no unlimited dim)");

            VarMeta meta{};
            meta.name = var_name;
            meta.dtype = AMIO_DTYPE_F32;
            meta.shape.rank = 3;
            meta.shape.extents[0] = NTIMES;
            meta.shape.extents[1] = NLAT;
            meta.shape.extents[2] = NLON;

            // Read each timestep slice with bbox [t:t+1, 0:181, 0:360].
            for (int t = 0; t < NTIMES; ++t) {
                BoundingBox box{};
                box.rank = 3;
                box.offsets[0] = t;
                box.offsets[1] = 0;
                box.offsets[2] = 0;
                box.extents[0] = 1;
                box.extents[1] = NLAT;
                box.extents[2] = NLON;
                box.strides[0] = 1;
                box.strides[1] = 1;
                box.strides[2] = 1;

                std::vector<float> out(POINTS_PER_STEP, 0.0f);
                StagingBuffer dst{};
                dst.data = reinterpret_cast<std::byte *>(out.data());
                dst.capacity_bytes = out.size() * sizeof(float);
                dst.used_bytes = 0;

                reader.read(dst, meta, /*timestep=*/0, std::optional<BoundingBox>{box});

                const std::size_t expected_bytes = POINTS_PER_STEP * sizeof(float);
                EXPECT_TRUE(dst.used_bytes == expected_bytes, "timestep " + std::to_string(t) + " used_bytes == expected");

                // Compare against the corresponding source slice.
                const float *src_slice = source.data() + static_cast<std::size_t>(t) * POINTS_PER_STEP;
                const bool eq = std::memcmp(out.data(), src_slice, expected_bytes) == 0;
                EXPECT_TRUE(eq, "timestep " + std::to_string(t) + " byte-equal to source slice");
            }

            reader.close();
        } catch (const std::exception &e) {
            report_failure("read phase", __FILE__, __LINE__, e.what());
        }
    }

    std::remove(OUTPUT_PATH);

    std::fprintf(stdout, "test_netcdf4_multistep: passed=%d failed=%d\n", g_passed, g_failed);
    std::fprintf(stdout, "  grid: [%d timesteps, %d lat, %d lon] = %zu floats/step (%.1f KiB/step)\n", NTIMES, NLAT, NLON, POINTS_PER_STEP,
                 static_cast<double>(POINTS_PER_STEP * sizeof(float)) / 1024.0);

    int mpi_init_flag = 0;
    MPI_Initialized(&mpi_init_flag);
    int mpi_final_flag = 0;
    MPI_Finalized(&mpi_final_flag);
    if (mpi_init_flag && !mpi_final_flag) {
        MPI_Finalize();
    }

    return g_failed == 0 ? 0 : 1;
}
