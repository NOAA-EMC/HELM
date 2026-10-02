// test_read_zarr.cpp
//
// Integration test for the Zarr v3 READ path (task 15.2).  Drives the
// REAL Zarr_Driver in TensorStore mode (no mocks):
//
//   open_write -> write known array -> flush -> close
//     -> open_read -> describe_variable -> read (full / bbox / strided)
//        -> close
//
// and asserts:
//
//   * describe_variable reports the correct dtype, rank, extents, and
//     total_timesteps from ts_store_.domain()/dtype() (Req 4.1, 4.2,
//     4.5; design §3, task 15.1).  The Zarr timestep model treats the
//     leading dimension of a rank >= 2 store as the time axis, so a
//     stored [NY, NX] array reports total_timesteps == NY and a rank-1
//     per-timestep shape with extent[0] == NX.
//   * a full read returns the written payload byte-for-byte
//     (Req 10.4, 11.2, 11.3)
//   * a unit-stride bounding-box read restricts the TensorStore read
//     domain and returns exactly the requested sub-region, with
//     used_bytes equal to the sub-region size (Req 10.1, 10.3)
//   * a strided bounding-box read applies the stride to the read domain
//     and returns exactly the strided sub-region (Req 10.2, 10.3)
//   * a non-F32 (int32 and float64) write -> read round trip is
//     byte-equal, confirming the dtype dispatch transfers the real
//     element type rather than reinterpreting it as float32
//     (Req 11.2, 11.3)
//
// This test is only built when TensorStore is available
// (AMIO_HAS_TENSORSTORE); the build env that runs the NCZarr fallback
// suite skips the target entirely (see tests/unit/CMakeLists.txt).  It
// compiles the zarr driver + its deps directly into the test binary
// (mirroring how test_cf_attributes compiles the zarr driver in NCZarr
// mode) so the real TensorStore read/write/describe paths are
// exercised.
//
// Validates: Req 10.1, 10.2, 10.3, 11.2, 11.3 (Zarr read + round trip).

#include <eckit/config/YAMLConfiguration.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/zarr/zarr_driver.hpp"
#include "factory/backend_driver.hpp"
#include "staging/staging_pool.hpp"

using amio::detail::BoundingBox;
using amio::detail::StagingBuffer;
using amio::detail::VariableInfo;
using amio::detail::VarMeta;
using amio::detail::Zarr_Driver;

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

// Build a Zarr v3 dataset manifest (eckit YAML) for a 2-D [ny, nx]
// array.  chunk_shape == shard_shape == array_shape keeps the sharding
// constraint trivially satisfied (each chunk dim divides the shard
// dim).  The dtype string drives the stored element type; an empty
// string lets the driver default to float32.
std::string make_yaml(const std::string &uri, int ny, int nx, const std::string &dtype_str) {
    std::string yaml = "uri: " + uri + "\n";
    yaml += "chunk_shape: [" + std::to_string(ny) + ", " + std::to_string(nx) + "]\n";
    yaml += "shard_shape: [" + std::to_string(ny) + ", " + std::to_string(nx) + "]\n";
    yaml += "array_shape: [" + std::to_string(ny) + ", " + std::to_string(nx) + "]\n";
    yaml += "codec: blosc\n";  // lossless -> round trip is byte-equal
    if (!dtype_str.empty()) {
        yaml += "dtype: " + dtype_str + "\n";
    }
    return yaml;
}

// Write a known 2-D array of `nbytes` bytes (row-major, ny x nx of
// `dtype`) to a fresh Zarr store at `uri`.
void write_array(const std::string &uri, const std::string &dtype_str, amio_dtype_t dtype, int ny, int nx, const void *data, std::size_t nbytes) {
    std::error_code ec;
    std::filesystem::remove_all(uri, ec);

    eckit::YAMLConfiguration cfg{make_yaml(uri, ny, nx, dtype_str)};

    Zarr_Driver writer;
    writer.open_write(cfg);

    StagingBuffer buf{};
    buf.data = const_cast<std::byte *>(reinterpret_cast<const std::byte *>(data));
    buf.capacity_bytes = nbytes;
    buf.used_bytes = nbytes;

    VarMeta meta{};
    meta.name = "var";
    meta.dtype = dtype;
    meta.shape.rank = 2;
    meta.shape.extents[0] = ny;
    meta.shape.extents[1] = nx;

    writer.write(buf, meta);
    writer.flush();
    writer.close();
}

// ---- F32 grid used for describe / full / bbox / strided assertions ----
constexpr int NY = 6;
constexpr int NX = 8;

float f32_source_value(int i, int j) {
    return static_cast<float>(i) * 100.0f + static_cast<float>(j);
}

// Run the describe_variable + full-read + unit-stride + strided-read
// assertions against a stored F32 [NY, NX] array.
void run_f32_tests(const std::string &uri) {
    float source[NY][NX];
    for (int i = 0; i < NY; ++i) {
        for (int j = 0; j < NX; ++j) {
            source[i][j] = f32_source_value(i, j);
        }
    }

    write_array(uri, /*dtype_str=*/"float32", AMIO_DTYPE_F32, NY, NX, &source[0][0], sizeof(source));

    eckit::YAMLConfiguration cfg{make_yaml(uri, NY, NX, "float32")};
    Zarr_Driver reader;
    reader.open_read(cfg);

    // --- describe_variable (Req 4.1, 4.2, 4.5; design §3) ---
    // Leading-dim time model: a [NY, NX] store reports the leading
    // dimension as the time axis (total_timesteps == NY) and a rank-1
    // per-timestep shape of extent NX.
    {
        VariableInfo info = reader.describe_variable("var");
        EXPECT_TRUE(info.found, "describe_variable found var");
        EXPECT_TRUE(info.dtype == AMIO_DTYPE_F32, "describe_variable dtype == F32");
        EXPECT_TRUE(info.shape.rank == 1, "describe_variable per-timestep rank == 1");
        EXPECT_TRUE(info.shape.extents[0] == NX, "describe_variable extent[0] == NX");
        EXPECT_TRUE(info.total_timesteps == NY, "describe_variable total_timesteps == NY");
    }

    // VarMeta carrying the full 2-D shape: the read path sizes the
    // array view from meta.shape when there is no bbox (full read), and
    // from the bbox extents otherwise.
    VarMeta meta{};
    meta.name = "var";
    meta.dtype = AMIO_DTYPE_F32;
    meta.shape.rank = 2;
    meta.shape.extents[0] = NY;
    meta.shape.extents[1] = NX;

    // --- Full read, byte-for-byte fidelity (Req 10.4, 11.2, 11.3) ---
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

    // --- Unit-stride bbox read (Req 10.1, 10.3) ---
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

    // --- Strided bbox read (Req 10.2, 10.3) ---
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
}

// Generic non-F32 full write -> read round trip asserting byte
// equality.  `T` is the element type, `dtype` its AMIO tag, and
// `dtype_str` the Zarr/TensorStore element-type string.  Confirms the
// dtype dispatch transfers the real element type rather than
// reinterpreting it as float32 (Req 11.2, 11.3).
template <typename T>
void run_roundtrip(const std::string &uri, amio_dtype_t dtype, const std::string &dtype_str, const char *label) {
    constexpr int RNY = 4;
    constexpr int RNX = 5;

    std::vector<T> source(static_cast<std::size_t>(RNY) * RNX);
    for (int i = 0; i < RNY; ++i) {
        for (int j = 0; j < RNX; ++j) {
            // Distinct values across the grid so any mis-indexing or
            // dtype reinterpretation surfaces as a byte mismatch.
            source[static_cast<std::size_t>(i) * RNX + j] = static_cast<T>(i * 100 + j);
        }
    }
    const std::size_t nbytes = source.size() * sizeof(T);

    write_array(uri, dtype_str, dtype, RNY, RNX, source.data(), nbytes);

    eckit::YAMLConfiguration cfg{make_yaml(uri, RNY, RNX, dtype_str)};
    Zarr_Driver reader;
    reader.open_read(cfg);

    // describe_variable reports the real element type, not float32.
    {
        VariableInfo info = reader.describe_variable("var");
        EXPECT_TRUE(info.found, std::string(label) + " describe_variable found");
        EXPECT_TRUE(info.dtype == dtype, std::string(label) + " describe_variable dtype matches");
    }

    VarMeta meta{};
    meta.name = "var";
    meta.dtype = dtype;
    meta.shape.rank = 2;
    meta.shape.extents[0] = RNY;
    meta.shape.extents[1] = RNX;

    std::vector<T> out(source.size(), static_cast<T>(0));
    StagingBuffer dst{};
    dst.data = reinterpret_cast<std::byte *>(out.data());
    dst.capacity_bytes = nbytes;
    dst.used_bytes = 0;

    reader.read(dst, meta, /*timestep=*/0, std::nullopt);

    EXPECT_TRUE(dst.used_bytes == nbytes, std::string(label) + " round trip used_bytes == payload size");
    EXPECT_TRUE(std::memcmp(out.data(), source.data(), nbytes) == 0, std::string(label) + " round trip is byte-for-byte equal");

    reader.close();
}

}  // namespace

int main() {
    const std::string f32_uri = "/tmp/amio_test_read_zarr_f32.zarr";
    const std::string i32_uri = "/tmp/amio_test_read_zarr_i32.zarr";
    const std::string f64_uri = "/tmp/amio_test_read_zarr_f64.zarr";

    try {
        run_f32_tests(f32_uri);
        run_roundtrip<std::int32_t>(i32_uri, AMIO_DTYPE_I32, "int32", "int32");
        run_roundtrip<double>(f64_uri, AMIO_DTYPE_F64, "float64", "float64");
    } catch (const std::exception &e) {
        report_failure("zarr read path", __FILE__, __LINE__, e.what());
    }

    std::error_code ec;
    std::filesystem::remove_all(f32_uri, ec);
    std::filesystem::remove_all(i32_uri, ec);
    std::filesystem::remove_all(f64_uri, ec);

    std::fprintf(stdout, "test_read_zarr: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
