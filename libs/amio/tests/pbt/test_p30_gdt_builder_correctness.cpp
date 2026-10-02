// test_p30_gdt_builder_correctness.cpp -- Property test P30: GDT 3.40
// template builder index correctness.
//
// For any valid Grib2Settings with n_parallel > 0 and any positive grid
// dimensions (ni, nj), calling build_gdt_3_40(settings, ni, nj) SHALL
// produce a 19-entry vector where:
//   t[7]  == ni
//   t[8]  == nj
//   t[11] == s.lat_first
//   t[12] == s.lon_first
//   t[14] == s.lat_last
//   t[15] == s.lon_last
//   t[17] == s.n_parallel
//
// Min 100 iterations with random valid Grib2Settings.
//
// Feature: grib2-composition-templates, Property 2: GDT 3.40 template
// builder index correctness
//
// **Validates: Requirements 9.1, 9.2, 9.3**

#include <cstdint>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Property Test P30: GDT 3.40 template builder index correctness.
//
// For any valid Grib2Settings with n_parallel > 0 and positive (ni, nj),
// build_gdt_3_40 produces a 19-entry vector with the correct values at
// the WMO-defined indices for Gaussian latitude/longitude grids.
//
// Validates: Requirements 9.1, 9.2, 9.3
// ===================================================================

TEST_CASE("Feature: grib2-composition-templates, Property 2: GDT 3.40 template builder index correctness", "[pbt][p30][grib2][gdt][gdt_3_40]") {
    auto result = rc::check("build_gdt_3_40 places fields at correct WMO-defined indices", []() {
        // Generate random valid Grib2Settings with n_parallel > 0.
        Grib2Settings s{};
        s.n_parallel = *rc::gen::inRange<std::int64_t>(1, 641);  // 1–640

        // Grid dimensions: small positive values (4–720).
        auto ni = *rc::gen::inRange<std::int64_t>(4, 721);
        auto nj = *rc::gen::inRange<std::int64_t>(4, 721);

        // Latitude range: -90000000 to 90000000 (microdegrees).
        s.lat_first = *rc::gen::inRange<std::int64_t>(-90000000, 90000001);
        s.lon_first = *rc::gen::inRange<std::int64_t>(0, 360000001);
        s.lat_last = *rc::gen::inRange<std::int64_t>(-90000000, 90000001);
        s.lon_last = *rc::gen::inRange<std::int64_t>(0, 360000001);

        // Call the builder.
        auto t = GRIB2_Driver::build_gdt_3_40(s, ni, nj);

        // Verify the vector has exactly 19 entries (GDT 3.40 layout).
        RC_ASSERT(t.size() == 19);

        // Verify grid dimensions at indices 7 and 8.
        RC_ASSERT(t[7] == ni);
        RC_ASSERT(t[8] == nj);

        // Verify latitude/longitude of first grid point at indices 11, 12.
        RC_ASSERT(t[11] == s.lat_first);
        RC_ASSERT(t[12] == s.lon_first);

        // Verify latitude/longitude of last grid point at indices 14, 15.
        RC_ASSERT(t[14] == s.lat_last);
        RC_ASSERT(t[15] == s.lon_last);

        // Verify n_parallel (N - number of parallels between equator
        // and pole) at index 17.
        RC_ASSERT(t[17] == s.n_parallel);
    });

    REQUIRE(result);
}
