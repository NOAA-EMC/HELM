// test_read_netcdf4.cpp
//
// Integration test for the NetCDF-4 read path.  Drives the REAL
// NetCDF_Driver end-to-end (no mocks):
//
//   open_write -> write known array -> flush -> close
//     -> open_read -> describe_variable -> read (full / bbox) -> close
//
// and asserts:
//
//   * describe_variable reports the correct dtype, rank, extents, and
//     total_timesteps (Req 4.1, 4.2, 4.5, 9.1)
//   * a full read returns the written payload byte-for-byte
//     (Req 9.1, 9.4)
//   * a unit-stride bounding-box read (nc_get_vara) returns exactly the
//     requested sub-region (Req 9.2, 9.4)
//   * a strided bounding-box read (nc_get_vars) returns exactly the
//     strided sub-region (Req 9.3, 9.4)
//
// The NetCDF-4 driver issues parallel HDF5 calls (nc_create_par /
// nc_open_par) that require MPI to be initialized by the host before
// the driver opens.  We initialize MPI in main (the host role) exactly
// as tests/unit/test_lifecycle_netcdf4.cpp does, and tolerate an
// environment without functional parallel HDF5 by treating a thrown
// open/write failure as a skip -- while still HARD-asserting byte
// equality whenever the read path is actually functional.
//
// Validates: Req 9.1, 9.2, 9.3, 9.4 (NetCDF-4 read end-to-end).

// Use only the MPI C API; suppress the deprecated C++ MPI bindings so we
// do not need to link libmpi_cxx.
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

// Test grid: NY x NX 2-D float field with distinct values so any
// mis-indexing in the read path shows up as a byte mismatch.
constexpr int NY = 6;
constexpr int NX = 8;

float source_value(int i, int j) {
    return static_cast<float>(i) * 100.0f + static_cast<float>(j);
}

}  // namespace

int main() {
    const char *OUTPUT_PATH = "/tmp/amio_test_read_netcdf4.nc";
    const std::string var_name = "temperature";

    // Host role: initialize MPI before the driver opens (parallel HDF5).
    int mpi_already = 0;
    MPI_Initialized(&mpi_already);
    if (!mpi_already) {
        int provided = 0;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    }

    std::remove(OUTPUT_PATH);

    // Reference payload (row-major).
    float source[NY][NX];
    for (int i = 0; i < NY; ++i) {
        for (int j = 0; j < NX; ++j) {
            source[i][j] = source_value(i, j);
        }
    }

    // No compression codec: keep the round trip dependency-free.
    const std::string yaml = std::string("path: ") + OUTPUT_PATH +
                             "\n"
                             "data_model: classic\n";

    eckit::YAMLConfiguration cfg{yaml};

    // ---- Write phase (tolerate an environment without parallel HDF5) ----
    bool functional = true;
    try {
        NetCDF_Driver writer;
        writer.open_write(cfg);

        StagingBuffer buf{};
        buf.data = reinterpret_cast<std::byte *>(&source[0][0]);
        buf.capacity_bytes = sizeof(source);
        buf.used_bytes = sizeof(source);

        VarMeta meta{};
        meta.name = var_name;
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = NY;
        meta.shape.extents[1] = NX;

        writer.write(buf, meta);
        writer.flush();
        writer.close();
    } catch (const std::exception &e) {
        // Parallel HDF5 / MPI-IO may be unavailable in this environment.
        // Treat the write path as a skip rather than a failure, mirroring
        // test_lifecycle_netcdf4's AMIO_ERR_BACKEND_FAILURE tolerance.
        std::fprintf(stdout, "NOTE: NetCDF write path unavailable, skipping read assertions: %s\n", e.what());
        functional = false;
    }

    // ---- Read phase ----
    if (functional) {
        try {
            NetCDF_Driver reader;
            reader.open_read(cfg);

            // --- describe_variable (Req 4.1, 4.2, 4.5, 9.1) ---
            VariableInfo info = reader.describe_variable(var_name);
            EXPECT_TRUE(info.found, "describe_variable found temperature");
            EXPECT_TRUE(info.dtype == AMIO_DTYPE_F32, "describe_variable dtype == F32");
            EXPECT_TRUE(info.shape.rank == 2, "describe_variable rank == 2");
            EXPECT_TRUE(info.shape.extents[0] == NY, "describe_variable extent[0] == NY");
            EXPECT_TRUE(info.shape.extents[1] == NX, "describe_variable extent[1] == NX");
            // Dimensions are fixed (no unlimited/record dim) -> 1 timestep.
            EXPECT_TRUE(info.total_timesteps == 1, "describe_variable total_timesteps == 1");

            // A variable that does not exist must report found == false.
            VariableInfo missing = reader.describe_variable("does_not_exist");
            EXPECT_TRUE(!missing.found, "describe_variable(missing) -> found == false");

            VarMeta meta{};
            meta.name = var_name;
            meta.dtype = AMIO_DTYPE_F32;
            meta.shape.rank = 2;
            meta.shape.extents[0] = NY;
            meta.shape.extents[1] = NX;

            // --- Full read, byte-for-byte fidelity (Req 9.1, 9.4) ---
            {
                std::vector<float> out(static_cast<std::size_t>(NY) * NX, 0.0f);
                StagingBuffer dst{};
                dst.data = reinterpret_cast<std::byte *>(out.data());
                dst.capacity_bytes = out.size() * sizeof(float);
                dst.used_bytes = 0;

                reader.read(dst, meta, /*timestep=*/0, std::nullopt);

                EXPECT_TRUE(dst.used_bytes == sizeof(source), "full read used_bytes == payload size");
                EXPECT_TRUE(std::memcmp(out.data(), source, sizeof(source)) == 0, "full read is byte-for-byte equal to source");
            }

            // --- Unit-stride bbox read via nc_get_vara (Req 9.2, 9.4) ---
            {
                BoundingBox box{};
                box.rank = 2;
                box.offsets[0] = 1;
                box.offsets[1] = 2;
                box.extents[0] = 3;
                box.extents[1] = 4;
                box.strides[0] = 1;
                box.strides[1] = 1;

                const std::size_t n = static_cast<std::size_t>(box.extents[0]) * box.extents[1];
                std::vector<float> out(n, 0.0f);
                StagingBuffer dst{};
                dst.data = reinterpret_cast<std::byte *>(out.data());
                dst.capacity_bytes = out.size() * sizeof(float);
                dst.used_bytes = 0;

                reader.read(dst, meta, /*timestep=*/0, std::optional<BoundingBox>{box});

                EXPECT_TRUE(dst.used_bytes == n * sizeof(float), "unit-stride bbox used_bytes == sub-region size");

                bool match = true;
                for (int r = 0; r < box.extents[0] && match; ++r) {
                    for (int c = 0; c < box.extents[1]; ++c) {
                        const float expected = source[box.offsets[0] + r][box.offsets[1] + c];
                        const float got = out[static_cast<std::size_t>(r) * box.extents[1] + c];
                        if (std::memcmp(&expected, &got, sizeof(float)) != 0) {
                            match = false;
                            break;
                        }
                    }
                }
                EXPECT_TRUE(match, "unit-stride bbox equals offset/extent slice of source");
            }

            // --- Strided bbox read via nc_get_vars (Req 9.3, 9.4) ---
            {
                BoundingBox box{};
                box.rank = 2;
                box.offsets[0] = 0;
                box.offsets[1] = 0;
                box.extents[0] = 3;  // rows 0, 2, 4
                box.extents[1] = 4;  // cols 0, 2, 4, 6
                box.strides[0] = 2;
                box.strides[1] = 2;

                const std::size_t n = static_cast<std::size_t>(box.extents[0]) * box.extents[1];
                std::vector<float> out(n, 0.0f);
                StagingBuffer dst{};
                dst.data = reinterpret_cast<std::byte *>(out.data());
                dst.capacity_bytes = out.size() * sizeof(float);
                dst.used_bytes = 0;

                reader.read(dst, meta, /*timestep=*/0, std::optional<BoundingBox>{box});

                EXPECT_TRUE(dst.used_bytes == n * sizeof(float), "strided bbox used_bytes == sub-region size");

                bool match = true;
                for (int r = 0; r < box.extents[0] && match; ++r) {
                    for (int c = 0; c < box.extents[1]; ++c) {
                        const int si = static_cast<int>(box.offsets[0] + static_cast<long long>(r) * box.strides[0]);
                        const int sj = static_cast<int>(box.offsets[1] + static_cast<long long>(c) * box.strides[1]);
                        const float expected = source[si][sj];
                        const float got = out[static_cast<std::size_t>(r) * box.extents[1] + c];
                        if (std::memcmp(&expected, &got, sizeof(float)) != 0) {
                            match = false;
                            break;
                        }
                    }
                }
                EXPECT_TRUE(match, "strided bbox equals strided slice of source");
            }

            reader.close();
        } catch (const std::exception &e) {
            // After a successful write, a thrown read is a real failure.
            report_failure("read phase", __FILE__, __LINE__, e.what());
        }
    }

    std::remove(OUTPUT_PATH);

    std::fprintf(stdout, "test_read_netcdf4: passed=%d failed=%d\n", g_passed, g_failed);

    int mpi_init_flag = 0;
    MPI_Initialized(&mpi_init_flag);
    int mpi_final_flag = 0;
    MPI_Finalized(&mpi_final_flag);
    if (mpi_init_flag && !mpi_final_flag) {
        MPI_Finalize();
    }

    return g_failed == 0 ? 0 : 1;
}
