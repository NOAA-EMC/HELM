// test_p29_pdt_builder_correctness.cpp -- Property test P29: PDT
// template builder index correctness.
//
// Feature: grib2-composition-templates, Property 1: PDT template
// builder index correctness
//
// For any valid Grib2Settings and any supported PDT number P in
// {0, 8, 40, 44, 45, 46, 48, 49}, calling build_pdt_P(settings) SHALL
// produce a std::vector<std::int64_t> whose .size() matches the
// WMO-specified entry count for PDT P, and whose composition-relevant
// indices hold the corresponding Grib2Settings field values.
//
// Generator strategy:
//   - pdt_number drawn from {0, 8, 40, 44, 45, 46, 48, 49}
//   - All integer fields drawn from reasonable WMO-range values (0–65535)
//
// Min 100 iterations via RC_PARAMS=max_success=100.
//
// **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4,
//              4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.1,
//              7.2, 8.1, 8.2, 8.3**

#include <cstdint>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Custom Arbitrary<Grib2Settings> generator.
//
// Produces random valid Grib2Settings with pdt_number drawn from the
// supported set and all integer fields in WMO-range [0, 65535].
// ===================================================================

namespace {

rc::Gen<Grib2Settings> genGrib2Settings() {
    return rc::gen::exec([]() {
        Grib2Settings s{};

        // Draw pdt_number from the supported set.
        static const std::vector<std::int64_t> supported_pdts = {0, 8, 40, 44, 45, 46, 48, 49};
        auto pdt_idx = *rc::gen::inRange<std::size_t>(0, supported_pdts.size());
        s.pdt_number = supported_pdts[pdt_idx];

        // All integer fields in WMO-range [0, 65535].
        s.parameter_category = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.parameter_number = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.type_of_first_fixed_surface = *rc::gen::inRange<std::int64_t>(0, 256);
        s.scale_factor_first_surface = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.scaled_value_first_surface = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.forecast_time = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.indicator_of_unit_of_time = *rc::gen::inRange<std::int64_t>(0, 256);

        // Composition-specific fields.
        s.chemical_constituent_type = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.aerosol_type = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.size_dist_param_first = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.size_dist_param_second = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.optical_property_type = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.wavelength_first_nm = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.wavelength_last_nm = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.ensemble_perturbation_number = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.statistical_process = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.time_range_unit = *rc::gen::inRange<std::int64_t>(0, 256);
        s.time_range_length = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.number_of_time_range_specs = *rc::gen::inRange<std::int64_t>(0, 65536);
        s.total_missing_from_statistical_process = *rc::gen::inRange<std::int64_t>(0, 65536);

        return s;
    });
}

}  // anonymous namespace

// ===================================================================
// Property Test P29: PDT template builder index correctness.
//
// For each supported PDT, verify:
//   1. The returned vector has the correct WMO-specified size.
//   2. Composition-relevant indices hold the corresponding
//      Grib2Settings field values.
// ===================================================================

TEST_CASE("Feature: grib2-composition-templates, Property 1: PDT template builder index correctness", "[pbt][p29][grib2][composition][pdt_builder]") {
    auto result = rc::check("PDT builder produces correct size and index values for all supported PDTs", []() {
        auto s = *genGrib2Settings();

        switch (s.pdt_number) {
            case 0: {
                auto t = GRIB2_Driver::build_pdt_4_0(s);
                // PDT 4.0: 15 entries.
                RC_ASSERT(t.size() == 15);
                // Key indices.
                RC_ASSERT(t[0] == s.parameter_category);
                RC_ASSERT(t[1] == s.parameter_number);
                RC_ASSERT(t[8] == s.forecast_time);
                RC_ASSERT(t[9] == s.type_of_first_fixed_surface);
                break;
            }
            case 8: {
                auto t = GRIB2_Driver::build_pdt_4_8(s);
                // PDT 4.8: 29 entries.
                RC_ASSERT(t.size() == 29);
                // Key indices.
                RC_ASSERT(t[0] == s.parameter_category);
                RC_ASSERT(t[1] == s.parameter_number);
                RC_ASSERT(t[8] == s.forecast_time);
                RC_ASSERT(t[23] == s.statistical_process);
                RC_ASSERT(t[26] == s.time_range_length);
                break;
            }
            case 40: {
                auto t = GRIB2_Driver::build_pdt_4_40(s);
                // PDT 4.40: 16 entries.
                RC_ASSERT(t.size() == 16);
                // Key indices.
                RC_ASSERT(t[0] == s.parameter_category);
                RC_ASSERT(t[1] == s.parameter_number);
                RC_ASSERT(t[2] == s.chemical_constituent_type);
                RC_ASSERT(t[9] == s.forecast_time);
                break;
            }
            case 44: {
                auto t = GRIB2_Driver::build_pdt_4_44(s);
                // PDT 4.44: 21 entries.
                RC_ASSERT(t.size() == 21);
                // Key indices.
                RC_ASSERT(t[2] == s.aerosol_type);
                RC_ASSERT(t[5] == s.size_dist_param_first);
                RC_ASSERT(t[7] == s.size_dist_param_second);
                RC_ASSERT(t[14] == s.forecast_time);
                break;
            }
            case 45: {
                auto t = GRIB2_Driver::build_pdt_4_45(s);
                // PDT 4.45: 24 entries.
                RC_ASSERT(t.size() == 24);
                // Key indices.
                RC_ASSERT(t[2] == s.aerosol_type);
                RC_ASSERT(t[5] == s.size_dist_param_first);
                RC_ASSERT(t[7] == s.size_dist_param_second);
                RC_ASSERT(t[16] == s.ensemble_perturbation_number);
                break;
            }
            case 46: {
                auto t = GRIB2_Driver::build_pdt_4_46(s);
                // PDT 4.46: 35 entries.
                RC_ASSERT(t.size() == 35);
                // Key indices.
                RC_ASSERT(t[2] == s.aerosol_type);
                RC_ASSERT(t[5] == s.size_dist_param_first);
                RC_ASSERT(t[7] == s.size_dist_param_second);
                RC_ASSERT(t[29] == s.statistical_process);
                RC_ASSERT(t[32] == s.time_range_length);
                break;
            }
            case 48: {
                auto t = GRIB2_Driver::build_pdt_4_48(s);
                // PDT 4.48: 26 entries.
                RC_ASSERT(t.size() == 26);
                // Key indices.
                RC_ASSERT(t[2] == s.aerosol_type);
                RC_ASSERT(t[8] == s.optical_property_type);
                RC_ASSERT(t[10] == s.wavelength_first_nm);
                RC_ASSERT(t[12] == s.wavelength_last_nm);
                break;
            }
            case 49: {
                auto t = GRIB2_Driver::build_pdt_4_49(s);
                // PDT 4.49: 29 entries.
                RC_ASSERT(t.size() == 29);
                // Key indices.
                RC_ASSERT(t[2] == s.aerosol_type);
                RC_ASSERT(t[8] == s.optical_property_type);
                RC_ASSERT(t[10] == s.wavelength_first_nm);
                RC_ASSERT(t[12] == s.wavelength_last_nm);
                RC_ASSERT(t[21] == s.ensemble_perturbation_number);
                break;
            }
            default:
                RC_FAIL("Unexpected pdt_number in generator");
                break;
        }
    });

    REQUIRE(result);
}
