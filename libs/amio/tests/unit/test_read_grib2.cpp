// test_read_grib2.cpp
//
// Integration test for the GRIB2 read/decode path.  Drives the REAL
// GRIB2_Driver end-to-end (no mocks):
//
//   open_write -> write known F32 grid(s) -> flush -> close
//     -> open_read -> describe_variable -> read (full / bbox) -> close
//
// and asserts:
//
//   * describe_variable reports F32, the Nj x Ni shape, and a
//     total_timesteps equal to the number of encoded records
//     (Req 4.1, 4.2, 4.5, 13.1)
//   * a full read decodes the record to F32 grid-point values that are
//     byte-equal to the written field (Req 13.2, 13.3)
//   * a lossless DRT (libaec 5.42, JPEG2000 5.40) write->read round trip
//     of the same F32 field is byte-equal (Req 13.4)
//   * a unit-stride bounding-box read returns exactly the requested
//     decoded sub-region (Req 13.5)
//   * a strided bounding-box read returns exactly the strided
//     decoded sub-region (Req 13.5)
//
// Precision assumption for the lossless round trip
// ------------------------------------------------
// GRIB2 grid-point packing (DRT 5.42 CCSDS/AEC, 5.40 JPEG2000) is only
// bit-exact at a sufficient bit depth / decimal scale.  The write path's
// Data Representation templates use decimal_scale_factor = 0 and let g2c
// choose the bit width (number_of_bits = 0).  At that setting the packed
// representation is round((value - reference) * 2^E) with E = 0, which is
// exact for integer-valued fields.  The source grid therefore uses small
// integer values stored as float (0..~1047), which are exactly
// representable and survive the encode->decode round trip byte-for-byte.
//
// DRT availability is build-dependent: g2c only encodes a template whose
// backing codec (libaec, JPEG2000/jasper, PNG) was compiled in.  This
// test attempts both lossless DRTs, skips any whose encoder is absent in
// this g2c build (printing a NOTE), HARD-asserts decode / bbox / byte
// equality for every DRT that does encode, and requires at least one
// lossless DRT to complete a byte-equal round trip (Req 13.4).
//
// The GRIB2 driver does NOT use MPI (unlike the NetCDF-4 driver), so no
// MPI initialization is required here.
//
// Validates: Req 13.2, 13.3, 13.4, 13.5 (GRIB2 decode + bbox + lossless
// round trip).

#include <eckit/config/YAMLConfiguration.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Test grid: NY (Nj, latitudes/slowest) x NX (Ni, longitudes/fastest)
// 2-D float field of small integer values so any mis-indexing or lossy
// packing shows up as a byte mismatch.
constexpr int NY = 6;
constexpr int NX = 8;

// Distinct per-(timestep, row, col) integer value, stored as float.
float source_value(int t, int i, int j) {
    return static_cast<float>(t * 1000 + i * NX + j);
}

// The field-identity string the driver synthesizes from the encode
// settings used below: discipline 0, parameter_category 3,
// parameter_number 5, first fixed surface type 100, scaled value 50000.
// The read coordinator passes this exact name through to read().
const std::string kVarName = "d0_c3_n5_s100_l50000";

// Build the YAML manifest for a given lossless DRT and output path.
std::string make_yaml(const std::string &path, const std::string &drt) {
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

// Outcome of a single DRT write->read round trip.
struct RoundTripResult {
    bool encoded = false;     // the DRT's encoder is available in this g2c build
    bool byte_equal = false;  // full read matched the written field byte-for-byte
};

// Run a full write->read round trip for one DRT.  A DRT whose encoder is
// not compiled into this g2c build causes write to fail; that is treated
// as a SKIP (note) rather than a test failure, because DRT availability
// is a build property.  When the DRT does encode, decode / bbox / byte
// equality are all hard-asserted (lossless DRTs must round-trip
// byte-equal, Req 13.4).
RoundTripResult run_roundtrip(const std::string &drt, const char *output_path) {
    RoundTripResult result{};
    std::remove(output_path);

    // Two timesteps with distinct data to exercise record indexing.
    std::vector<std::vector<float>> source(2, std::vector<float>(static_cast<std::size_t>(NY) * NX));
    for (int t = 0; t < 2; ++t) {
        for (int i = 0; i < NY; ++i) {
            for (int j = 0; j < NX; ++j) {
                source[t][static_cast<std::size_t>(i) * NX + j] = source_value(t, i, j);
            }
        }
    }

    eckit::YAMLConfiguration cfg{make_yaml(output_path, drt)};
    const std::string ctx = "[drt=" + drt + "]";

    // ---- Write phase (a DRT absent from this g2c build -> skip) ----
    try {
        GRIB2_Driver writer;
        writer.open_write(cfg);
        for (int t = 0; t < 2; ++t) {
            StagingBuffer buf{};
            buf.data = reinterpret_cast<std::byte *>(source[t].data());
            buf.capacity_bytes = source[t].size() * sizeof(float);
            buf.used_bytes = source[t].size() * sizeof(float);

            VarMeta meta{};
            meta.name = kVarName;
            meta.dtype = AMIO_DTYPE_F32;
            meta.shape.rank = 2;
            meta.shape.extents[0] = NY;
            meta.shape.extents[1] = NX;

            writer.write(buf, meta);
        }
        writer.flush();
        writer.close();
        result.encoded = true;
    } catch (const std::exception &e) {
        std::fprintf(stdout, "NOTE: %s encoder unavailable in this g2c build, skipping (%s)\n", ctx.c_str(), e.what());
        std::remove(output_path);
        return result;  // encoded == false -> skipped
    }

    // ---- Read phase ----
    try {
        GRIB2_Driver reader;
        reader.open_read(cfg);

        // --- describe_variable (Req 4.1, 4.2, 4.5, 13.1) ---
        VariableInfo info = reader.describe_variable(kVarName);
        EXPECT_TRUE(info.found, ctx + " describe_variable found field");
        EXPECT_TRUE(info.dtype == AMIO_DTYPE_F32, ctx + " describe_variable dtype == F32");
        EXPECT_TRUE(info.shape.rank == 2, ctx + " describe_variable rank == 2");
        EXPECT_TRUE(info.shape.extents[0] == NY, ctx + " describe_variable extent[0] == NY (Nj)");
        EXPECT_TRUE(info.shape.extents[1] == NX, ctx + " describe_variable extent[1] == NX (Ni)");
        EXPECT_TRUE(info.total_timesteps == 2, ctx + " describe_variable total_timesteps == 2");

        // A field that does not exist must report found == false.
        VariableInfo missing = reader.describe_variable("d9_c9_n9_s9_l9");
        EXPECT_TRUE(!missing.found, ctx + " describe_variable(missing) -> found == false");

        VarMeta meta{};
        meta.name = kVarName;
        meta.dtype = AMIO_DTYPE_F32;
        meta.shape.rank = 2;
        meta.shape.extents[0] = NY;
        meta.shape.extents[1] = NX;

        // --- Full read per timestep, decode-to-F32 + lossless round
        //     trip byte equality (Req 13.2, 13.3, 13.4) ---
        bool all_equal = true;
        for (int t = 0; t < 2; ++t) {
            std::vector<float> out(static_cast<std::size_t>(NY) * NX, -1.0f);
            StagingBuffer dst{};
            dst.data = reinterpret_cast<std::byte *>(out.data());
            dst.capacity_bytes = out.size() * sizeof(float);
            dst.used_bytes = 0;

            reader.read(dst, meta, /*timestep=*/t, std::nullopt);

            EXPECT_TRUE(dst.used_bytes == out.size() * sizeof(float), ctx + " full read used_bytes == payload size");

            const bool eq = std::memcmp(out.data(), source[t].data(), out.size() * sizeof(float)) == 0;
            if (!eq) {
                all_equal = false;
            }
            EXPECT_TRUE(eq, ctx + " full read byte-equal to source (lossless round trip)");
        }
        result.byte_equal = all_equal;

        // --- Unit-stride bbox read (decoded sub-region) (Req 13.5) ---
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
            std::vector<float> out(n, -1.0f);
            StagingBuffer dst{};
            dst.data = reinterpret_cast<std::byte *>(out.data());
            dst.capacity_bytes = out.size() * sizeof(float);
            dst.used_bytes = 0;

            reader.read(dst, meta, /*timestep=*/0, std::optional<BoundingBox>{box});

            EXPECT_TRUE(dst.used_bytes == n * sizeof(float), ctx + " unit-stride bbox used_bytes == sub-region size");

            bool match = true;
            for (int r = 0; r < box.extents[0] && match; ++r) {
                for (int c = 0; c < box.extents[1]; ++c) {
                    const float expected = source_value(0, static_cast<int>(box.offsets[0] + r), static_cast<int>(box.offsets[1] + c));
                    const float got = out[static_cast<std::size_t>(r) * box.extents[1] + c];
                    if (std::memcmp(&expected, &got, sizeof(float)) != 0) {
                        match = false;
                        break;
                    }
                }
            }
            EXPECT_TRUE(match, ctx + " unit-stride bbox equals offset/extent slice");
        }

        // --- Strided bbox read (decoded strided sub-region) (Req 13.5) ---
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
            std::vector<float> out(n, -1.0f);
            StagingBuffer dst{};
            dst.data = reinterpret_cast<std::byte *>(out.data());
            dst.capacity_bytes = out.size() * sizeof(float);
            dst.used_bytes = 0;

            reader.read(dst, meta, /*timestep=*/0, std::optional<BoundingBox>{box});

            EXPECT_TRUE(dst.used_bytes == n * sizeof(float), ctx + " strided bbox used_bytes == sub-region size");

            bool match = true;
            for (int r = 0; r < box.extents[0] && match; ++r) {
                for (int c = 0; c < box.extents[1]; ++c) {
                    const int si = static_cast<int>(box.offsets[0] + static_cast<long long>(r) * box.strides[0]);
                    const int sj = static_cast<int>(box.offsets[1] + static_cast<long long>(c) * box.strides[1]);
                    const float expected = source_value(0, si, sj);
                    const float got = out[static_cast<std::size_t>(r) * box.extents[1] + c];
                    if (std::memcmp(&expected, &got, sizeof(float)) != 0) {
                        match = false;
                        break;
                    }
                }
            }
            EXPECT_TRUE(match, ctx + " strided bbox equals strided slice");
        }

        reader.close();
    } catch (const std::exception &e) {
        // After a successful write, a thrown read is a real failure.
        report_failure("grib2 read phase", __FILE__, __LINE__, ctx + " " + e.what());
        result.byte_equal = false;
    }

    std::remove(output_path);
    return result;
}

}  // namespace

int main() {
    // Attempt both lossless DRTs.  libaec (5.42) is preferred; JPEG2000
    // (5.40) is included.  Each is skipped if its encoder is not built
    // into this g2c build; any that encodes is fully asserted.
    const RoundTripResult aec = run_roundtrip("libaec", "/tmp/amio_test_read_grib2_aec.grib2");
    const RoundTripResult jp2 = run_roundtrip("jpeg2000", "/tmp/amio_test_read_grib2_jp2.grib2");

    std::fprintf(stdout, "libaec (DRT 5.42):   encoded=%s byte_equal=%s\n", aec.encoded ? "YES" : "NO (skipped)", aec.byte_equal ? "YES" : "NO");
    std::fprintf(stdout, "JPEG2000 (DRT 5.40): encoded=%s byte_equal=%s\n", jp2.encoded ? "YES" : "NO (skipped)", jp2.byte_equal ? "YES" : "NO");

    // At least one lossless DRT must be available and round-trip
    // byte-equal, otherwise Req 13.4 cannot be validated in this build.
    EXPECT_TRUE((aec.encoded && aec.byte_equal) || (jp2.encoded && jp2.byte_equal), "at least one lossless DRT round-trips byte-equal");

    std::fprintf(stdout, "test_read_grib2: passed=%d failed=%d\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
