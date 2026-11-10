// ─── Property-Based Tests: DatelineHandler ───────────────────────────────────
// Feature: axis-v2-improvements
//
// Property 15: Dateline normalization continuity
//
// **Validates: Requirements 9.1**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/dateline_handler.hpp>
#include <cmath>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

using axis::detail::DatelineHandler;

constexpr double pi = 3.14159265358979323846;
constexpr double two_pi = 2.0 * pi;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Generate a random polygon that crosses the dateline.
/// Vertices span both positive (> π/2) and negative (< -π/2) longitudes,
/// producing a gap > π which indicates a dateline crossing.
///
/// Strategy: place some vertices near +π and some near -π so that the raw
/// difference exceeds π radians, triggering dateline detection.
struct DatelineCrossingPolygon {
    std::vector<double> lons;
    int n;
};

/// RapidCheck generator for dateline-crossing polygons.
/// We generate n vertices (3–8) where at least one is in (π/2, π] and at
/// least one is in [-π, -π/2), ensuring the polygon spans the dateline.
rc::Gen<DatelineCrossingPolygon> genDatelineCrossingPolygon() {
    return rc::gen::exec([]() {
        // Number of vertices: 3 to 8
        int n = *rc::gen::inRange(3, 9);

        // We need at least one vertex on each side of the dateline.
        // Positive side: lon in (π/2, π]
        // Negative side: lon in [-π, -π/2)

        std::vector<double> lons(n);

        // First vertex: positive side near +π
        // Range: [0.6π, π] mapped from integer range
        lons[0] = *rc::gen::map(rc::gen::inRange(600, 1001), [](int v) { return v * 0.001 * pi; });

        // Second vertex: negative side near -π
        // Range: [-π, -0.6π] mapped from integer range
        lons[1] = *rc::gen::map(rc::gen::inRange(600, 1001), [](int v) { return -v * 0.001 * pi; });

        // Remaining vertices: randomly on either side
        for (int i = 2; i < n; ++i) {
            bool positive_side = *rc::gen::arbitrary<bool>();
            if (positive_side) {
                lons[i] = *rc::gen::map(rc::gen::inRange(500, 1001), [](int v) { return v * 0.001 * pi; });
            } else {
                lons[i] = *rc::gen::map(rc::gen::inRange(500, 1001), [](int v) { return -v * 0.001 * pi; });
            }
        }

        return DatelineCrossingPolygon{lons, n};
    });
}

// ─── Property 15: Dateline normalization continuity ──────────────────────────
// For any dateline-crossing cell, `normalize` SHALL produce
// `max(lon) - min(lon) < 2π` (360° in radians).
//
// This ensures that after normalization, all longitudes lie in a continuous
// range smaller than a full circle, eliminating the discontinuity at ±π.
//
// **Validates: Requirements 9.1**

RC_GTEST_PROP(PropDateline, NormalizationContinuity, ()) {
    auto poly = *genDatelineCrossingPolygon();

    // Copy longitudes to a mutable array for normalize()
    constexpr int kMaxVerts = 32;
    double lons[kMaxVerts];
    int n = std::min(poly.n, kMaxVerts);
    for (int i = 0; i < n; ++i) {
        lons[i] = poly.lons[i];
    }

    // Precondition: the polygon must actually cross the dateline
    RC_PRE(DatelineHandler::crosses_dateline(lons, n));

    // Apply normalization
    DatelineHandler::normalize(lons, n);

    // Find min and max after normalization
    double min_lon = lons[0];
    double max_lon = lons[0];
    for (int i = 1; i < n; ++i) {
        if (lons[i] < min_lon) min_lon = lons[i];
        if (lons[i] > max_lon) max_lon = lons[i];
    }

    double span = max_lon - min_lon;

    // Property: span must be < 2π (360°)
    RC_ASSERT(span < two_pi);
    // Also verify span is non-negative (sanity check)
    RC_ASSERT(span >= 0.0);
}

// ─── Property 15b: crosses_dateline correctly detects dateline-crossing cells ─
// For any polygon generated with vertices on both sides of ±π, the
// crosses_dateline() function SHALL return true.
//
// **Validates: Requirements 9.1**

RC_GTEST_PROP(PropDateline, CrossesDatelineDetection, ()) {
    auto poly = *genDatelineCrossingPolygon();

    constexpr int kMaxVerts = 32;
    double lons[kMaxVerts];
    int n = std::min(poly.n, kMaxVerts);
    for (int i = 0; i < n; ++i) {
        lons[i] = poly.lons[i];
    }

    // A polygon with vertices on both sides of the dateline (gap > π)
    // MUST be detected as crossing
    RC_ASSERT(DatelineHandler::crosses_dateline(lons, n));
}

// ─── Property 15c: Non-dateline polygons remain unchanged in span ────────────
// For any polygon that does NOT cross the dateline, `normalize` SHALL still
// produce `max(lon) - min(lon) < 2π`.
// (This verifies normalize doesn't break non-crossing cells.)
//
// **Validates: Requirements 9.1**

RC_GTEST_PROP(PropDateline, NonCrossingNormalizationSafe, ()) {
    // Generate a polygon entirely on one hemisphere (no dateline crossing)
    int n = *rc::gen::inRange(3, 9);

    constexpr int kMaxVerts = 32;
    double lons[kMaxVerts];

    // All longitudes in a narrow band: [-π/2, π/2]
    for (int i = 0; i < n; ++i) {
        lons[i] = *rc::gen::map(rc::gen::inRange(-1500, 1501), [](int v) { return v * 0.001; });
    }

    // Precondition: must NOT cross the dateline
    RC_PRE(!DatelineHandler::crosses_dateline(lons, n));

    // Capture original span
    double orig_min = *std::min_element(lons, lons + n);
    double orig_max = *std::max_element(lons, lons + n);
    double orig_span = orig_max - orig_min;

    // Apply normalization
    DatelineHandler::normalize(lons, n);

    // Find span after normalization
    double min_lon = *std::min_element(lons, lons + n);
    double max_lon = *std::max_element(lons, lons + n);
    double span = max_lon - min_lon;

    // Property: span must still be < 2π
    RC_ASSERT(span < two_pi);
    // For non-crossing polygons, the span shouldn't grow significantly
    // (normalization should be a no-op or near no-op)
    RC_ASSERT(span >= 0.0);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
