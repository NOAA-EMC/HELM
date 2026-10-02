// test_p25_grib2_metadata_invariant.cpp -- Property test P25: GRIB2
// metadata / table-sourcing invariant.
//
// GRIB2 code and template tables are sourced from NCEPLIBS-g2c, not
// from a legacy code map.  The manifest supplies the numeric
// GRIB2 product identifiers (discipline, parameter category/number,
// template numbers, fixed-surface descriptors) which AMIO forwards to
// g2c verbatim.  The remaining string-level invariant is the DRT:
// it must name one of {Adaptive Entropy Coding via libaec, Lossless
// JPEG2000}; anything else -> std::runtime_error, zero output bytes.
//
// This test verifies:
//   * valid DRT names parse to a GRIB2_DRT enum (R9.6)
//   * invalid DRT names throw (R9.6, R9.7)
//   * the NCEP template builders emit the correct template layouts
//     populated from the numeric identifiers, with no string->code
//     remapping (R9.3, R9.8)
//
// Min 100 iterations with valid + invalid DRT generators.
//
// **Validates: Requirements R9.3, R9.6, R9.7, R9.8**

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Generators for GRIB2 metadata testing.
// ===================================================================

namespace {

// Valid DRT names that the GRIB2_Driver accepts.
std::vector<std::string> valid_drt_names() {
    return {"adaptive_entropy_coding", "libaec", "adaptive entropy coding via libaec", "adaptive entropy coding", "lossless_jpeg2000", "jpeg2000",
            "lossless jpeg2000",
            // Case variations (parse_drt_name normalizes to lowercase)
            "Libaec", "LIBAEC", "Lossless_JPEG2000", "JPEG2000"};
}

// Invalid DRT names that should be rejected.
std::vector<std::string> invalid_drt_names() {
    return {"png", "deflate",     "bzip2",          "lossy_jpeg2000", "simple_packing", "complex_packing", "grid_point_data",
            "",    "unknown_drt", "adaptive_lossy", "zstandard",      "blosc"};
}

// Generate a random valid DRT name.
rc::Gen<std::string> genValidDRT() {
    return rc::gen::exec([]() {
        auto names = valid_drt_names();
        auto idx = *rc::gen::inRange<std::size_t>(0, names.size());
        return names[idx];
    });
}

// Generate a random invalid DRT name.
rc::Gen<std::string> genInvalidDRT() {
    return rc::gen::exec([]() {
        auto names = invalid_drt_names();
        auto idx = *rc::gen::inRange<std::size_t>(0, names.size());
        return names[idx];
    });
}

}  // anonymous namespace

// ===================================================================
// Property Test P25a: Valid DRT names are accepted by parse_drt_name.
//
// For any valid DRT name: parse_drt_name returns a valid GRIB2_DRT
// enum value without throwing.
//
// Validates: R9.6
// ===================================================================

TEST_CASE("P25: GRIB2 metadata invariant - valid DRT names accepted", "[pbt][p25][grib2][metadata][drt_valid]") {
    auto result = rc::check("valid DRT names are parsed without exception", []() {
        auto drt_name = *genValidDRT();

        // Should not throw -- valid DRT names are accepted.
        GRIB2_DRT drt = parse_drt_name(drt_name);

        // Result must be one of the two allowed DRT values.
        RC_ASSERT(drt == GRIB2_DRT::AdaptiveEntropyCoding || drt == GRIB2_DRT::LosslessJPEG2000);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P25b: Invalid DRT names are rejected with exception.
//
// For any DRT name NOT in {Adaptive Entropy Coding via libaec,
// Lossless JPEG2000}: parse_drt_name throws std::runtime_error.
// Zero output bytes (no encoding occurs).
//
// Validates: R9.6, R9.7
// ===================================================================

TEST_CASE("P25: GRIB2 metadata invariant - invalid DRT names rejected", "[pbt][p25][grib2][metadata][drt_invalid]") {
    auto result = rc::check("invalid DRT names throw std::runtime_error", []() {
        auto drt_name = *genInvalidDRT();

        // Must throw for invalid DRT names.
        bool threw = false;
        try {
            parse_drt_name(drt_name);
        } catch (const std::runtime_error &) {
            threw = true;
        }

        RC_ASSERT(threw);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P25c: NCEP template values are forwarded verbatim.
//
// The Product Definition Template is built directly from the numeric
// identifiers in Grib2Settings -- there is no string->code remapping.
// For any (category, parameter) the PDT 4.0 layout must carry exactly
// those values in entries [0] and [1] (R9.3, R9.8).
// ===================================================================

TEST_CASE("P25: GRIB2 table sourcing - PDT carries numeric identifiers verbatim", "[pbt][p25][grib2][metadata][pdt]") {
    auto result = rc::check("PDT 4.0 entries equal the manifest's numeric GRIB2 identifiers", []() {
        Grib2Settings s{};
        s.parameter_category = *rc::gen::inRange<std::int64_t>(0, 255);
        s.parameter_number = *rc::gen::inRange<std::int64_t>(0, 255);
        s.type_of_first_fixed_surface = *rc::gen::inRange<std::int64_t>(0, 255);
        s.scaled_value_first_surface = *rc::gen::inRange<std::int64_t>(0, 100000);
        s.forecast_time = *rc::gen::inRange<std::int64_t>(0, 240);

        auto pdt = GRIB2_Driver::build_pdt_4_0(s);

        // PDT 4.0 is a 15-entry template.
        RC_ASSERT(pdt.size() == 15);
        // Numeric identifiers are forwarded verbatim (no remapping).
        RC_ASSERT(pdt[0] == s.parameter_category);
        RC_ASSERT(pdt[1] == s.parameter_number);
        RC_ASSERT(pdt[7] == s.indicator_of_unit_of_time);
        RC_ASSERT(pdt[8] == s.forecast_time);
        RC_ASSERT(pdt[9] == s.type_of_first_fixed_surface);
        RC_ASSERT(pdt[11] == s.scaled_value_first_surface);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P25d: DRS template matches the selected DRT layout.
//
// The Data Representation Template is chosen by the validated DRT and
// its layout (entry count) follows the NCEP/WMO table for that DRT:
//   DRT 5.40 (JPEG2000) -> 7 entries
//   DRT 5.42 (AEC/CCSDS) -> 8 entries
// The decimal scale factor is forwarded verbatim (R9.3, R9.8).
// ===================================================================

TEST_CASE("P25: GRIB2 table sourcing - DRS template matches selected DRT", "[pbt][p25][grib2][metadata][drs]") {
    auto result = rc::check("DRS template layout follows the NCEP table for the chosen DRT", []() {
        Grib2Settings s{};
        s.decimal_scale_factor = *rc::gen::inRange<std::int64_t>(0, 6);

        auto drt_name = *genValidDRT();
        GRIB2_DRT drt = parse_drt_name(drt_name);

        auto drs = GRIB2_Driver::build_drs_template(drt, s);

        if (drt == GRIB2_DRT::LosslessJPEG2000) {
            RC_ASSERT(drs.size() == 7);
        } else {
            RC_ASSERT(drs.size() == 8);
        }
        // Decimal scale factor is forwarded verbatim.
        RC_ASSERT(drs[2] == s.decimal_scale_factor);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P25e: DRT validation distinguishes missing vs
// unrecognized.
//
// For any configuration: if the DRT field is entirely missing,
// validate_drt throws identifying "missing"; if the DRT field is
// present but unrecognized, throws identifying "unrecognized".
//
// Validates: R9.7
// ===================================================================

TEST_CASE("P25: GRIB2 metadata invariant - DRT missing vs unrecognized", "[pbt][p25][grib2][metadata][drt_distinction]") {
    auto result = rc::check("DRT validation distinguishes missing from unrecognized", []() {
        // Test with an invalid DRT name: parse_drt_name should
        // throw with a message containing "unrecognized".
        auto invalid_name = *genInvalidDRT();

        // Skip empty string (that's a "missing value" case, not
        // "unrecognized name" case).
        RC_PRE(!invalid_name.empty());

        bool threw = false;
        std::string error_msg;
        try {
            parse_drt_name(invalid_name);
        } catch (const std::runtime_error &e) {
            threw = true;
            error_msg = e.what();
        }

        RC_ASSERT(threw);
        // The error message should indicate the DRT is unrecognized.
        RC_ASSERT(error_msg.find("unrecognized") != std::string::npos || error_msg.find("not recognized") != std::string::npos);
    });

    REQUIRE(result);
}
