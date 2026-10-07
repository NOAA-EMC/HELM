// test_p31_identity_roundtrip.cpp -- Property test P31:
// Field_Identity_Name encode-decode round-trip.
//
// Feature: grib2-composition-templates, Property 3: Field_Identity_Name
// encode-decode round-trip
//
// For any valid Grib2Settings with pdt_number in {0, 8, 40, 44, 45,
// 46, 48, 49}: the Field_Identity_Name computed from settings at
// encode time SHALL equal the Field_Identity_Name extracted by
// extract_composition_metadata from the template array produced by
// the corresponding build_pdt_P(settings).
//
// That is:
//   field_identity_name(settings) == field_identity_name(extract(build_pdt_P(settings), P))
//
// Generator strategy:
//   - pdt_number drawn from {0, 8, 40, 44, 45, 46, 48, 49}
//   - All integer fields drawn from reasonable WMO-range values (0–65535)
//
// Min 100 iterations via RC_PARAMS=max_success=100.
//
// **Validates: Requirements 2.5, 3.5, 4.4, 5.4, 6.4, 7.3, 8.4,
//              12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.8, 13.1,
//              13.2, 13.3, 13.4, 13.5, 14.3**

#include <cstdint>
#include <string>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Custom generator for Grib2Settings with pdt_number from supported set.
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
        s.discipline = 0;  // Fixed to 0 (meteorological) for identity
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

// PDT-specific surface type and value index lookup.
// Returns the indices of (type_of_first_fixed_surface, scaled_value_first_surface)
// in the PDT template array for the given PDT number.
std::pair<std::size_t, std::size_t> surface_indices(std::int64_t pdt_number) {
    switch (pdt_number) {
        case 0:
            return {9, 11};
        case 8:
            return {9, 11};
        case 40:
            return {10, 12};
        case 44:
            return {15, 17};
        case 45:
            return {18, 20};
        case 46:
            return {15, 17};
        case 48:
            return {20, 22};
        case 49:
            return {23, 25};
        default:
            return {9, 11};
    }
}

}  // anonymous namespace

// ===================================================================
// Property Test P31: Field_Identity_Name encode-decode round-trip.
//
// For each supported PDT:
//   1. Generate random Grib2Settings
//   2. Compute encode-time identity from settings
//   3. Build the PDT template array
//   4. Extract composition metadata from the template
//   5. Compute decode-time identity from extracted metadata
//   6. Assert the two identity strings are equal
// ===================================================================

TEST_CASE("Feature: grib2-composition-templates, Property 3: Field_Identity_Name encode-decode round-trip",
          "[pbt][p31][grib2][composition][identity_roundtrip]") {
    auto result = rc::check("Field_Identity_Name round-trips through build_pdt + extract_composition_metadata", []() {
        auto s = *genGrib2Settings();

        // Step 2: Compute the "encode-time" identity from settings.
        std::string encode_identity = GRIB2_Driver::field_identity_name(
            s.discipline, s.parameter_category, s.parameter_number, s.type_of_first_fixed_surface, s.scaled_value_first_surface, s.pdt_number,
            s.chemical_constituent_type, s.aerosol_type, s.optical_property_type, s.wavelength_first_nm, s.wavelength_last_nm,
            s.ensemble_perturbation_number, s.statistical_process);

        // Step 3: Build the PDT template array using the appropriate builder.
        std::vector<std::int64_t> pdt_template;
        switch (s.pdt_number) {
            case 0:
                pdt_template = GRIB2_Driver::build_pdt_4_0(s);
                break;
            case 8:
                pdt_template = GRIB2_Driver::build_pdt_4_8(s);
                break;
            case 40:
                pdt_template = GRIB2_Driver::build_pdt_4_40(s);
                break;
            case 44:
                pdt_template = GRIB2_Driver::build_pdt_4_44(s);
                break;
            case 45:
                pdt_template = GRIB2_Driver::build_pdt_4_45(s);
                break;
            case 46:
                pdt_template = GRIB2_Driver::build_pdt_4_46(s);
                break;
            case 48:
                pdt_template = GRIB2_Driver::build_pdt_4_48(s);
                break;
            case 49:
                pdt_template = GRIB2_Driver::build_pdt_4_49(s);
                break;
            default:
                RC_FAIL("Unexpected pdt_number");
                break;
        }

        // Step 4: Extract composition metadata from the template.
        CompositionMetadata meta = GRIB2_Driver::extract_composition_metadata(s.pdt_number, pdt_template);

        // Step 5: Compute the "decode-time" identity from extracted metadata.
        // Surface type and value come from known PDT-specific indices.
        auto [surf_type_idx, surf_value_idx] = surface_indices(s.pdt_number);
        std::int64_t decoded_surface_type = pdt_template[surf_type_idx];
        std::int64_t decoded_surface_value = pdt_template[surf_value_idx];

        // Parameter category and number are always at indices 0, 1.
        std::int64_t decoded_category = pdt_template[0];
        std::int64_t decoded_number = pdt_template[1];

        std::string decode_identity = GRIB2_Driver::field_identity_name(
            s.discipline, decoded_category, decoded_number, decoded_surface_type, decoded_surface_value, meta.pdt_number,
            meta.chemical_constituent_type, meta.aerosol_type, meta.optical_property_type, meta.wavelength_first_nm, meta.wavelength_last_nm,
            meta.ensemble_perturbation_number, meta.statistical_process);

        // Step 6: Assert the two identity strings are equal.
        RC_ASSERT(encode_identity == decode_identity);
    });

    REQUIRE(result);
}
