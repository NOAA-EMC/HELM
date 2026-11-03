// test_grib2_multistep.cpp
//
// Production-like multi-timestep integration test for the GRIB2 backend.
//
// Exercises a realistic GFS-like scenario: 4 timesteps of a 181×360 F32
// field (1-degree global grid, 65,160 floats per timestep ≈ 254 KiB/step).
// Writes via the real GRIB2_Driver using the JPEG2000 DRT (lossless for
// integer-valued floats), then reads back each timestep and asserts byte
// equality.
//
// The GRIB2 driver's write() appends a new GRIB2 message per call
// (g2_create + g2_addgrid + g2_addfield + g2_gribend + fwrite), so
// calling write() 4 times with the same VarMeta produces 4 records for
// the same field identity.  At open_read, the record index finds 4
// records → total_timesteps = 4.
//
// Uses integer-valued source data for lossless byte equality (same
// precision assumption as tests/unit/test_read_grib2.cpp):
//   value = timestep * 100000 + lat_idx * 1000 + lon_idx
// All values are small integers exactly representable as F32 and survive
// the GRIB2 encode→decode round trip byte-for-byte at
// decimal_scale_factor=0.
//
// This test exercises the serial driver variant without MPI initialization.

#include <algorithm>
#include <array>
#include <conf/config.hpp>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"

using amio::detail::BoundingBox;
using amio::detail::GRIB2_Driver;
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

// Production-like grid: 4 timesteps, 181 latitudes, 360 longitudes.
constexpr int NTIMES = 4;
constexpr int NLAT = 181;
constexpr int NLON = 360;
constexpr std::size_t POINTS_PER_STEP = static_cast<std::size_t>(NLAT) * NLON;  // 65,160

// Distinct integer value per (timestep, lat, lon) — exactly representable
// as F32, survives lossless GRIB2 encode→decode byte-for-byte.
float source_value(int t, int lat, int lon) {
    return static_cast<float>(t * 100000 + lat * 1000 + lon);
}

// Field identity name matching the GRIB2 settings below:
// discipline 0, parameter_category 3, parameter_number 5,
// type_of_first_fixed_surface 100, scaled_value_first_surface 50000.
const std::string kVarName = "d0_c3_n5_s100_l50000";

}  // namespace

int main() {
    const char *OUTPUT_PATH = "/tmp/amio_test_grib2_multistep.grib2";
    const char *STREAM_PATH = "/tmp/amio_test_grib2_multistep_streamed.grib2";
    std::remove(OUTPUT_PATH);
    std::remove(STREAM_PATH);

    // Build source data: 4 timesteps, each [181, 360].
    std::vector<std::vector<float>> source(NTIMES, std::vector<float>(POINTS_PER_STEP));
    for (int t = 0; t < NTIMES; ++t) {
        for (int lat = 0; lat < NLAT; ++lat) {
            for (int lon = 0; lon < NLON; ++lon) {
                source[t][static_cast<std::size_t>(lat) * NLON + lon] = source_value(t, lat, lon);
            }
        }
    }

    const std::string yaml = std::string("path: ") + OUTPUT_PATH +
                             "\n"
                             "drt: libaec\n"
                             "grib2:\n"
                             "  discipline: 0\n"
                             "  center: 7\n"
                             "  parameter_category: 3\n"
                             "  parameter_number: 5\n"
                             "  type_of_first_fixed_surface: 100\n"
                             "  scaled_value_first_surface: 50000\n";
    conf::Config cfg = conf::Config::from_string(yaml);

    // ---- Write phase: 4 sequential write() calls → 4 GRIB2 messages ----
    bool functional = true;
    try {
        GRIB2_Driver writer;
        writer.open_write(cfg);

        for (int t = 0; t < NTIMES; ++t) {
            StagingBuffer buf{};
            buf.data = reinterpret_cast<std::byte *>(source[t].data());
            buf.capacity_bytes = source[t].size() * sizeof(float);
            buf.used_bytes = source[t].size() * sizeof(float);

            VarMeta meta{};
            meta.name = kVarName;
            meta.dtype = AMIO_DTYPE_F32;
            meta.shape.rank = 2;
            meta.shape.extents[0] = NLAT;
            meta.shape.extents[1] = NLON;

            writer.write(buf, meta);
        }
        writer.flush();
        writer.close();
    } catch (const std::exception &e) {
        report_failure("write phase", __FILE__, __LINE__, e.what());
        functional = false;
    }

    // Stream the actual g2c-encoded messages in small bounded chunks before
    // asking the native decoder to recover the original scientific values.
    if (functional) {
        try {
            std::ifstream input(OUTPUT_PATH, std::ios::binary);
            if (!input) throw std::runtime_error("cannot open encoded GRIB input");
            amio::detail::Grib2StreamWriter streamed(STREAM_PATH, 1024, NTIMES, 16 * 1024 * 1024);
            std::array<unsigned char, 1024> buffer{};
            std::string error;
            for (int t = 0; t < NTIMES; ++t) {
                input.read(reinterpret_cast<char *>(buffer.data()), 16);
                if (!input) throw std::runtime_error("cannot read GRIB indicator section");
                std::uint64_t length = 0;
                for (std::size_t i = 8; i < 16; ++i) length = (length << 8) | buffer[i];
                if (!streamed.begin_message(0, 0, length, error) || !streamed.append_chunk(0, std::as_bytes(std::span(buffer.data(), 16)), error))
                    throw std::runtime_error(error);
                std::uint64_t offset = 16;
                while (offset < length) {
                    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(buffer.size(), length - offset));
                    input.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(count));
                    if (!input || !streamed.append_chunk(offset, std::as_bytes(std::span(buffer.data(), count)), error))
                        throw std::runtime_error(error.empty() ? "cannot read encoded GRIB chunk" : error);
                    offset += count;
                }
            }
            if (!streamed.finish(error) || streamed.index().size() != NTIMES) throw std::runtime_error(error);
        } catch (const std::exception &e) {
            report_failure("stream phase", __FILE__, __LINE__, e.what());
            functional = false;
        }
    }

    // ---- Read phase: describe_variable + per-timestep full reads ----
    if (functional) {
        try {
            GRIB2_Driver reader;
            auto streamed_yaml = yaml;
            streamed_yaml.replace(streamed_yaml.find(OUTPUT_PATH), std::strlen(OUTPUT_PATH), STREAM_PATH);
            const auto streamed_cfg = conf::Config::from_string(streamed_yaml);
            reader.open_read(streamed_cfg);

            // Verify describe_variable reports shape and 4 timesteps.
            VariableInfo info = reader.describe_variable(kVarName);
            EXPECT_TRUE(info.found, "describe_variable found field");
            EXPECT_TRUE(info.dtype == AMIO_DTYPE_F32, "describe_variable dtype == F32");
            EXPECT_TRUE(info.shape.rank == 2, "describe_variable rank == 2");
            EXPECT_TRUE(info.shape.extents[0] == NLAT, "describe_variable extent[0] == NLAT (181)");
            EXPECT_TRUE(info.shape.extents[1] == NLON, "describe_variable extent[1] == NLON (360)");
            EXPECT_TRUE(info.total_timesteps == NTIMES, "describe_variable total_timesteps == 4");

            VarMeta meta{};
            meta.name = kVarName;
            meta.dtype = AMIO_DTYPE_F32;
            meta.shape.rank = 2;
            meta.shape.extents[0] = NLAT;
            meta.shape.extents[1] = NLON;

            // Read each timestep and assert byte equality.
            for (int t = 0; t < NTIMES; ++t) {
                std::vector<float> out(POINTS_PER_STEP, -1.0f);
                StagingBuffer dst{};
                dst.data = reinterpret_cast<std::byte *>(out.data());
                dst.capacity_bytes = out.size() * sizeof(float);
                dst.used_bytes = 0;

                reader.read(dst, meta, /*timestep=*/t, std::nullopt);

                const std::size_t expected_bytes = POINTS_PER_STEP * sizeof(float);
                EXPECT_TRUE(dst.used_bytes == expected_bytes, "timestep " + std::to_string(t) + " used_bytes == expected");

                const bool eq = std::memcmp(out.data(), source[t].data(), expected_bytes) == 0;
                EXPECT_TRUE(eq, "timestep " + std::to_string(t) + " byte-equal to source");
            }

            reader.close();
        } catch (const std::exception &e) {
            report_failure("read phase", __FILE__, __LINE__, e.what());
        }
    }

    std::remove(OUTPUT_PATH);
    std::remove(STREAM_PATH);

    std::fprintf(stdout, "test_grib2_multistep: passed=%d failed=%d\n", g_passed, g_failed);
    std::fprintf(stdout, "  grid: [%d timesteps, %d lat, %d lon] = %zu floats/step (%.1f KiB/step)\n", NTIMES, NLAT, NLON, POINTS_PER_STEP,
                 static_cast<double>(POINTS_PER_STEP * sizeof(float)) / 1024.0);

    return g_failed == 0 ? 0 : 1;
}
