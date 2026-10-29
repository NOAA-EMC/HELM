// ─── Property-Based Tests: True Bilinear Interpolator ────────────────────────
// Feature: axis-v2-improvements
//
// Property 4: True bilinear affine exactness
// Property 5: Bilinear weights partition of unity
//
// **Validates: Requirements 2.3, 2.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/gnomonic_projector.hpp>
#include <axis/detail/spherical_clipper.hpp>
#include <cmath>

namespace {

using axis::detail::add;
using axis::detail::cross;
using axis::detail::dot;
using axis::detail::GnomonicProjector;
using axis::detail::length;
using axis::detail::normalize;
using axis::detail::scale;
using axis::detail::Vec3;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert lon (radians), lat (radians) to unit-sphere Vec3.
Vec3 lonlat_to_vec3(double lon, double lat) {
    double cl = std::cos(lat);
    return {cl * std::cos(lon), cl * std::sin(lon), std::sin(lat)};
}

/// Generate a convex spherical quad by perturbing 4 directions around a center.
/// The center is specified by (lon, lat) and the half-width controls quad size.
/// Returns 4 vertices in counter-clockwise order on the sphere.
void make_spherical_quad(Vec3 center, double half_width, Vec3 verts[4]) {
    center = normalize(center);

    // Build a local tangent-plane basis at center
    Vec3 arbitrary = (std::abs(center.z) < 0.9) ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    Vec3 u = normalize(cross(center, arbitrary));
    Vec3 v = cross(center, u);

    // Place 4 vertices at the corners of a square in the tangent plane,
    // then project back to the sphere.
    // Corners: (-hw, -hw), (+hw, -hw), (+hw, +hw), (-hw, +hw) in (u, v) coords
    double hw = half_width;
    double offsets[4][2] = {{-hw, -hw}, {hw, -hw}, {hw, hw}, {-hw, hw}};

    for (int i = 0; i < 4; ++i) {
        double ou = offsets[i][0];
        double ov = offsets[i][1];
        Vec3 pt{center.x + ou * u.x + ov * v.x, center.y + ou * u.y + ov * v.y, center.z + ou * u.z + ov * v.z};
        verts[i] = normalize(pt);
    }
}

/// Generate a random interior point within a quad defined by isoparametric
/// coordinates (xi, eta) ∈ [-1, 1]². Interpolates in the gnomonic projected
/// space (not 3D Cartesian) so that the bilinear weights solver will recover
/// exact (xi, eta). Returns the projected (pu, pv) and the 3D point on sphere.
void bilinear_interpolate_point(const Vec3 &center, const double quad_u[4], const double quad_v[4], double xi, double eta, double &pu, double &pv,
                                Vec3 &point_3d) {
    // Compute bilinear shape functions
    double n0 = (1.0 - xi) * (1.0 - eta) * 0.25;
    double n1 = (1.0 + xi) * (1.0 - eta) * 0.25;
    double n2 = (1.0 + xi) * (1.0 + eta) * 0.25;
    double n3 = (1.0 - xi) * (1.0 + eta) * 0.25;

    // Interpolate in the projected tangent-plane space
    pu = n0 * quad_u[0] + n1 * quad_u[1] + n2 * quad_u[2] + n3 * quad_u[3];
    pv = n0 * quad_v[0] + n1 * quad_v[1] + n2 * quad_v[2] + n3 * quad_v[3];

    // Recover the 3D point via inverse gnomonic projection
    point_3d = GnomonicProjector::inverse(center, pu, pv);
}

// ─── Property 4: True bilinear affine exactness ──────────────────────────────
// For any affine f(x,y,z) = a·x + b·y + c·z and any valid (cell, point) pair,
// the interpolator SHALL reproduce f within 1e-10 relative tolerance.
//
// The gnomonic bilinear interpolator is affine-exact in the projected tangent
// plane. For a sphere point p with gnomonic coordinates (u,v):
//   f(p) / dot(center, p) = f(center) + u·f(east) + v·f(north)
// which is affine in (u,v). Bilinear interpolation preserves this, so:
//   sum_i w_i · [f(v_i) / dot(center, v_i)] = f(target) / dot(center, target)
//
// Strategy: generate a random quad, a random interior target point on the
// sphere, project everything to the tangent plane, compute bilinear weights,
// and verify the gnomonic-scaled affine exactness identity holds within 1e-10.
//
// **Validates: Requirements 2.3**

RC_GTEST_PROP(PropTrueBilinear, AffineExactness, ()) {
    // Generate a random center point on the sphere
    double lon = *rc::gen::map(rc::gen::inRange(-3141, 3142), [](int v) { return v * 0.001; });
    double lat = *rc::gen::map(rc::gen::inRange(-1400, 1401), [](int v) { return v * 0.001; });
    Vec3 center = lonlat_to_vec3(lon, lat);

    // Generate quad half-width: 0.02 to 0.2 radians (~1-11 degrees)
    double half_width = *rc::gen::map(rc::gen::inRange(20, 201), [](int v) { return v * 0.001; });

    // Build the spherical quad
    Vec3 quad_verts[4];
    make_spherical_quad(center, half_width, quad_verts);

    // Generate random isoparametric coordinates for interior point
    // Keep strictly interior: xi, eta ∈ [-0.9, 0.9]
    double xi = *rc::gen::map(rc::gen::inRange(-900, 901), [](int v) { return v * 0.001; });
    double eta = *rc::gen::map(rc::gen::inRange(-900, 901), [](int v) { return v * 0.001; });

    // Project all quad vertices to the tangent plane at center
    double quad_u[4], quad_v[4];
    for (int i = 0; i < 4; ++i) {
        GnomonicProjector::forward(center, quad_verts[i], quad_u[i], quad_v[i]);
    }

    // Generate random affine coefficients: f(x,y,z) = a*x + b*y + c*z
    double a = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });
    double b = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });
    double c = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });

    // Get the interior point by interpolating in projected space, then
    // recovering the 3D point via inverse gnomonic projection.
    double pu, pv;
    Vec3 target_3d;
    bilinear_interpolate_point(center, quad_u, quad_v, xi, eta, pu, pv, target_3d);

    // Compute bilinear weights
    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(quad_u, quad_v, pu, pv, weights);
    RC_PRE(converged);  // Only test when Newton converges

    // Gnomonic-scaled affine exactness:
    // The gnomonic projection gives u_i = dot(east, v_i) / dot(center, v_i).
    // For affine f, the function g(u,v) = f(p) / dot(center, p) is affine
    // in the projected (u,v) coordinates. Bilinear interpolation in (u,v)
    // preserves affine functions, so:
    //   sum_i w_i * g(u_i, v_i) = g(pu, pv)
    // i.e., sum_i w_i * f(v_i)/d_i = f(target)/d_target
    // where d_i = dot(center, v_i).

    // Compute d_i = dot(center, v_i) for each vertex
    double d_verts[4];
    for (int i = 0; i < 4; ++i) {
        d_verts[i] = center.x * quad_verts[i].x + center.y * quad_verts[i].y + center.z * quad_verts[i].z;
    }
    double d_target = center.x * target_3d.x + center.y * target_3d.y + center.z * target_3d.z;

    // Evaluate the gnomonic-scaled affine function at each vertex: g_i = f(v_i)/d_i
    double g_interp = 0.0;
    for (int i = 0; i < 4; ++i) {
        double f_i = a * quad_verts[i].x + b * quad_verts[i].y + c * quad_verts[i].z;
        g_interp += weights[i] * (f_i / d_verts[i]);
    }

    // Expected value: g(target) = f(target) / d_target
    double f_target = a * target_3d.x + b * target_3d.y + c * target_3d.z;
    double g_exact = f_target / d_target;

    // Relative tolerance check
    double denom = std::max(std::abs(g_exact), 1.0e-15);
    double rel_error = std::abs(g_interp - g_exact) / denom;

    RC_ASSERT(rel_error < 1.0e-10);
}

// ─── Property 5: Bilinear weights partition of unity ─────────────────────────
// For any destination point and enclosing source cell, weights SHALL be
// non-negative and sum to 1.0 within 1e-14.
//
// Strategy: generate a random spherical quad, project a random interior point
// onto the tangent plane, compute bilinear weights, and verify all weights are
// non-negative and sum to 1.0.
//
// **Validates: Requirements 2.4**

RC_GTEST_PROP(PropTrueBilinear, WeightsPartitionOfUnity, ()) {
    // Generate a random center point on the sphere (including near-polar regions)
    double lon = *rc::gen::map(rc::gen::inRange(-3141, 3142), [](int v) { return v * 0.001; });
    // Allow latitudes up to ±88° to test near-pole behavior per Requirement 2.4
    double lat = *rc::gen::map(rc::gen::inRange(-1535, 1536), [](int v) { return v * 0.001; });
    Vec3 center = lonlat_to_vec3(lon, lat);

    // Generate quad half-width: 0.01 to 0.15 radians
    double half_width = *rc::gen::map(rc::gen::inRange(10, 151), [](int v) { return v * 0.001; });

    // Build the spherical quad
    Vec3 quad_verts[4];
    make_spherical_quad(center, half_width, quad_verts);

    // Project all quad vertices to the tangent plane at center
    double quad_u[4], quad_v[4];
    bool all_valid = true;
    for (int i = 0; i < 4; ++i) {
        GnomonicProjector::forward(center, quad_verts[i], quad_u[i], quad_v[i]);
        // Ensure projection didn't clamp (point on opposite hemisphere)
        if (quad_u[i] == 0.0 && quad_v[i] == 0.0) {
            all_valid = false;
        }
    }
    RC_PRE(all_valid);

    // Generate random interior point in isoparametric space: xi, eta ∈ [-0.95, 0.95]
    double xi = *rc::gen::map(rc::gen::inRange(-950, 951), [](int v) { return v * 0.001; });
    double eta = *rc::gen::map(rc::gen::inRange(-950, 951), [](int v) { return v * 0.001; });

    // Get the projected target point by interpolating in projected space
    double pu, pv;
    Vec3 target_3d;
    bilinear_interpolate_point(center, quad_u, quad_v, xi, eta, pu, pv, target_3d);

    // Compute bilinear weights
    double weights[4];
    bool converged = GnomonicProjector::bilinear_weights(quad_u, quad_v, pu, pv, weights);
    RC_PRE(converged);  // Only test when Newton converges

    // Check all weights are non-negative
    for (int i = 0; i < 4; ++i) {
        RC_ASSERT(weights[i] >= 0.0);
    }

    // Check weights sum to 1.0 within tolerance 1e-14
    double sum = weights[0] + weights[1] + weights[2] + weights[3];
    double sum_error = std::abs(sum - 1.0);
    RC_ASSERT(sum_error < 1.0e-14);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
