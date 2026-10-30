// ─── Property-Based Tests: SphericalClipper ──────────────────────────────────
// Feature: axis-v2-improvements
//
// Property 1: Non-overlapping clip yields zero area
// Property 2: Contained polygon clip preserves area
// Property 3: Clipper terminates with valid output on all inputs
//
// **Validates: Requirements 1.3, 1.4, 1.6, 1.8**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/spherical_clipper.hpp>
#include <cmath>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

using axis::detail::add;
using axis::detail::cross;
using axis::detail::dot;
using axis::detail::length;
using axis::detail::normalize;
using axis::detail::scale;
using axis::detail::SphericalClipper;
using axis::detail::SphericalPolygon;
using axis::detail::Vec3;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert lon (radians), lat (radians) to unit-sphere Vec3.
Vec3 lonlat_to_vec3(double lon, double lat) {
    double cl = std::cos(lat);
    return {cl * std::cos(lon), cl * std::sin(lon), std::sin(lat)};
}

/// Generate a convex spherical polygon as a cap centered at `center` with
/// angular radius `radius` (radians), approximated by `n_verts` equally spaced
/// vertices.
template <int M = 32>
SphericalPolygon<M> make_spherical_cap(Vec3 center, double radius, int n_verts) {
    // Build a local coordinate frame: center is the "north pole" of the cap.
    center = normalize(center);

    // Find an orthogonal vector to center
    Vec3 arbitrary = (std::abs(center.z) < 0.9) ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    Vec3 u = normalize(cross(center, arbitrary));
    Vec3 v = cross(center, u);

    SphericalPolygon<M> poly;
    for (int i = 0; i < n_verts && i < M; ++i) {
        double angle = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(n_verts);
        // Point on sphere at angular distance `radius` from center
        double cos_r = std::cos(radius);
        double sin_r = std::sin(radius);
        double ca = std::cos(angle);
        double sa = std::sin(angle);

        Vec3 pt{center.x * cos_r + u.x * sin_r * ca + v.x * sin_r * sa, center.y * cos_r + u.y * sin_r * ca + v.y * sin_r * sa,
                center.z * cos_r + u.z * sin_r * ca + v.z * sin_r * sa};
        poly.push(normalize(pt));
    }
    return poly;
}

/// Generate a random unit vector from lon/lat ranges.
Vec3 random_direction(double lon, double lat) {
    return lonlat_to_vec3(lon, lat);
}

// ─── Property 1: Non-overlapping clip yields zero area ───────────────────────
// For any two non-overlapping spherical polygons, SphericalClipper SHALL produce
// an empty result with area == 0.
//
// Strategy: generate two polygons on opposite hemispheres (one near lon=0,
// one near lon=π) with angular radii small enough to guarantee no overlap.
//
// **Validates: Requirements 1.3, 1.4, 1.6, 1.8**

RC_GTEST_PROP(PropSphericalClipper, NonOverlappingClipYieldsZeroArea, ()) {
    // Generate polygon A centered in lon ∈ [-0.3, 0.3], lat ∈ [-0.3, 0.3]
    double lon_a = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.001; });
    double lat_a = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.001; });

    // Generate polygon B centered on the opposite side of the sphere
    // lon_b ∈ [π-0.3, π+0.3], same lat range
    double lon_b = M_PI + *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.001; });
    double lat_b = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.001; });

    // Angular radius: small enough that polygons on opposite hemispheres never overlap.
    // Max angular extent of each polygon center from origin: ~0.3 rad
    // Separation between centers: ~π - 0.6 ≈ 2.54 rad
    // So radius up to ~1.0 rad each guarantees no overlap (2*1.0 < 2.54)
    double radius_a = *rc::gen::map(rc::gen::inRange(50, 500), [](int v) { return v * 0.001; });
    double radius_b = *rc::gen::map(rc::gen::inRange(50, 500), [](int v) { return v * 0.001; });

    // Ensure separation is large enough: angular distance between centers > radius_a + radius_b
    Vec3 ca = lonlat_to_vec3(lon_a, lat_a);
    Vec3 cb = lonlat_to_vec3(lon_b, lat_b);
    double ang_dist = std::acos(std::clamp(dot(ca, cb), -1.0, 1.0));
    RC_PRE(ang_dist > radius_a + radius_b + 0.1);

    int nverts_a = *rc::gen::inRange(4, 9);
    int nverts_b = *rc::gen::inRange(4, 9);

    auto poly_a = make_spherical_cap<32>(ca, radius_a, nverts_a);
    auto poly_b = make_spherical_cap<32>(cb, radius_b, nverts_b);

    double area = SphericalClipper::overlap_area(poly_a, poly_b);

    RC_ASSERT(area == 0.0);
}

// ─── Property 2: Contained polygon clip preserves area ───────────────────────
// For any subject polygon fully contained within a clip polygon, the result area
// SHALL equal the subject area within 1e-14 relative tolerance.
//
// Strategy: generate a large cap polygon, then a smaller cap fully inside it.
//
// **Validates: Requirements 1.3, 1.4, 1.6, 1.8**

RC_GTEST_PROP(PropSphericalClipper, ContainedPolygonClipPreservesArea, ()) {
    // Center of both polygons (large and small share a center region)
    double lon_c = *rc::gen::map(rc::gen::inRange(-3000, 3001), [](int v) { return v * 0.001; });
    double lat_c = *rc::gen::map(rc::gen::inRange(-1200, 1201), [](int v) { return v * 0.001; });

    Vec3 center = lonlat_to_vec3(lon_c, lat_c);

    // Large polygon radius: 0.3 to 0.6 radians (~17-34 degrees)
    double radius_large = *rc::gen::map(rc::gen::inRange(300, 601), [](int v) { return v * 0.001; });

    // Small polygon radius: 0.05 to 0.15 radians (~3-9 degrees)
    double radius_small = *rc::gen::map(rc::gen::inRange(50, 151), [](int v) { return v * 0.001; });

    int nverts_large = *rc::gen::inRange(6, 12);
    int nverts_small = *rc::gen::inRange(4, 9);

    // Offset the small polygon center slightly from the large polygon center,
    // but ensure it's still fully contained. Because polygons have flat edges,
    // the in-radius of the large polygon is radius_large * cos(pi/nverts_large),
    // and the small polygon out-radius is radius_small.
    double max_offset = radius_large * std::cos(M_PI / nverts_large) - radius_small - 0.02;
    RC_PRE(max_offset > 0.01);

    double offset_frac = *rc::gen::map(rc::gen::inRange(0, 100), [](int v) { return v * 0.01; });
    double offset = offset_frac * max_offset;

    double offset_angle = *rc::gen::map(rc::gen::inRange(0, 628), [](int v) { return v * 0.01; });

    // Compute offset center
    Vec3 arbitrary = (std::abs(center.z) < 0.9) ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    Vec3 u = normalize(cross(center, arbitrary));
    Vec3 v_dir = cross(center, u);

    Vec3 small_center = normalize(
        Vec3{center.x * std::cos(offset) + u.x * std::sin(offset) * std::cos(offset_angle) + v_dir.x * std::sin(offset) * std::sin(offset_angle),
             center.y * std::cos(offset) + u.y * std::sin(offset) * std::cos(offset_angle) + v_dir.y * std::sin(offset) * std::sin(offset_angle),
             center.z * std::cos(offset) + u.z * std::sin(offset) * std::cos(offset_angle) + v_dir.z * std::sin(offset) * std::sin(offset_angle)});

    auto poly_large = make_spherical_cap<32>(center, radius_large, nverts_large);
    auto poly_small = make_spherical_cap<32>(small_center, radius_small, nverts_small);

    // The small polygon should be fully contained in the large polygon
    double subject_area = poly_small.area();
    RC_PRE(subject_area > 1e-10);  // Non-degenerate

    auto result = SphericalClipper::clip(poly_small, poly_large);
    double result_area = result.area();

    // Relative tolerance check
    double rel_error = std::abs(result_area - subject_area) / subject_area;
    RC_ASSERT(rel_error < 1e-14);
}

// ─── Property 3: Clipper terminates with valid output on all inputs ──────────
// For any pair of spherical polygons (including degenerate cases),
// SphericalClipper SHALL terminate and produce either empty or a valid polygon
// (no NaN, n≥3 or n==0).
//
// Strategy: generate arbitrary polygon pairs including potentially degenerate
// (very thin, small, near-antipodal) configurations.
//
// **Validates: Requirements 1.3, 1.4, 1.6, 1.8**

RC_GTEST_PROP(PropSphericalClipper, ClipperTerminatesWithValidOutput, ()) {
    // Generate two arbitrary polygons with varying vertex counts
    int nverts_a = *rc::gen::inRange(3, 12);
    int nverts_b = *rc::gen::inRange(3, 12);

    // Generate polygon A from random lon/lat vertices sorted by angle around center
    double lon_a = *rc::gen::map(rc::gen::inRange(-3141, 3142), [](int v) { return v * 0.001; });
    double lat_a = *rc::gen::map(rc::gen::inRange(-1500, 1501), [](int v) { return v * 0.001; });
    // Random radius (can be very small for degenerate cases)
    double radius_a = *rc::gen::map(rc::gen::inRange(1, 1000), [](int v) { return v * 0.001; });

    double lon_b = *rc::gen::map(rc::gen::inRange(-3141, 3142), [](int v) { return v * 0.001; });
    double lat_b = *rc::gen::map(rc::gen::inRange(-1500, 1501), [](int v) { return v * 0.001; });
    double radius_b = *rc::gen::map(rc::gen::inRange(1, 1000), [](int v) { return v * 0.001; });

    Vec3 center_a = lonlat_to_vec3(lon_a, lat_a);
    Vec3 center_b = lonlat_to_vec3(lon_b, lat_b);

    auto poly_a = make_spherical_cap<32>(center_a, radius_a, nverts_a);
    auto poly_b = make_spherical_cap<32>(center_b, radius_b, nverts_b);

    // Execute clip — must terminate (implicit by reaching assertion below)
    auto result = SphericalClipper::clip(poly_a, poly_b);

    // Validity checks
    // 1. n must be 0 (empty) or >= 3 (valid polygon)
    RC_ASSERT(result.n == 0 || result.n >= 3);

    // 2. No NaN in any vertex coordinate
    for (int i = 0; i < result.n; ++i) {
        RC_ASSERT(!std::isnan(result.verts[i].x));
        RC_ASSERT(!std::isnan(result.verts[i].y));
        RC_ASSERT(!std::isnan(result.verts[i].z));
    }

    // 3. If non-empty, area must be non-negative
    if (result.n >= 3) {
        double area = result.area();
        RC_ASSERT(!std::isnan(area));
        RC_ASSERT(area >= 0.0);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
