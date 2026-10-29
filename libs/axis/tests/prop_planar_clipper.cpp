// ─── Property-Based Tests: Parallel Planar Clipper Equivalence ───────────────
// Feature: axis-performance-optimizations
//
// Property 1: Parallel Planar Clipper Equivalence
//   For any pair of convex or concave 2D polygons with at most 32 vertices each,
//   the overlap area computed by PlanarClipper::overlap_area() (fixed-capacity,
//   KOKKOS_FUNCTION) SHALL agree with the overlap area computed by the existing
//   sequential compute_polygon_overlap_area() (heap-based std::vector) within
//   relative tolerance 1e-14.
//
// **Validates: Requirements 1.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/planar_clipper.hpp>
#include <cmath>
#include <vector>

namespace {

using axis::detail::PlanarClipper;
using axis::detail::PlanarPolygon;

// ─── Reference Implementation (heap-based std::vector Sutherland-Hodgman) ────
// This is the "existing sequential" reference that the design document refers to
// as compute_polygon_overlap_area(). It uses std::vector for dynamic vertex
// storage — the heap-based approach that PlanarClipper replaces for device
// portability.

struct Vec2 {
    double x, y;
};

/// Shoelace formula for unsigned area of a polygon given as std::vector<Vec2>.
/// Vertices are translated to the first vertex before summation to avoid
/// catastrophic cancellation for polygons far from the origin (keeps this
/// reference consistent with PlanarPolygon::area()).
double shoelace_area(const std::vector<Vec2> &poly) {
    if (poly.size() < 3) return 0.0;
    double a = 0.0;
    int n = static_cast<int>(poly.size());
    const double x0 = poly[0].x;
    const double y0 = poly[0].y;
    for (int i = 1; i < n - 1; ++i) {
        const double dx0 = poly[i].x - x0;
        const double dy0 = poly[i].y - y0;
        const double dx1 = poly[i + 1].x - x0;
        const double dy1 = poly[i + 1].y - y0;
        a += dx0 * dy1 - dx1 * dy0;
    }
    return std::fabs(a) * 0.5;
}

/// Signed distance from point (px, py) to the directed edge from
/// (ex0, ey0) in direction (edx, edy).
double ref_signed_distance(double px, double py, double ex0, double ey0, double edx, double edy) {
    return edx * (py - ey0) - edy * (px - ex0);
}

/// Clip polygon against a single directed edge using std::vector (heap).
std::vector<Vec2> ref_clip_against_edge(const std::vector<Vec2> &input, double ex0, double ey0, double edx, double edy) {
    std::vector<Vec2> output;
    if (input.empty()) return output;

    int n = static_cast<int>(input.size());
    constexpr double eps = -1.0e-14;

    for (int i = 0; i < n; ++i) {
        int prev_idx = (i + n - 1) % n;

        double curr_x = input[i].x;
        double curr_y = input[i].y;
        double prev_x = input[prev_idx].x;
        double prev_y = input[prev_idx].y;

        double d_curr = ref_signed_distance(curr_x, curr_y, ex0, ey0, edx, edy);
        double d_prev = ref_signed_distance(prev_x, prev_y, ex0, ey0, edx, edy);

        bool curr_inside = (d_curr >= eps);
        bool prev_inside = (d_prev >= eps);

        if (curr_inside) {
            if (!prev_inside) {
                double t = d_prev / (d_prev - d_curr);
                double ix = prev_x + t * (curr_x - prev_x);
                double iy = prev_y + t * (curr_y - prev_y);
                output.push_back({ix, iy});
            }
            output.push_back({curr_x, curr_y});
        } else if (prev_inside) {
            double t = d_prev / (d_prev - d_curr);
            double ix = prev_x + t * (curr_x - prev_x);
            double iy = prev_y + t * (curr_y - prev_y);
            output.push_back({ix, iy});
        }
    }
    return output;
}

/// Reference sequential Sutherland-Hodgman polygon overlap area using
/// heap-allocated std::vector<Vec2>. This is the "compute_polygon_overlap_area"
/// that PlanarClipper::overlap_area() must match.
double compute_polygon_overlap_area(const std::vector<Vec2> &subject, const std::vector<Vec2> &clip) {
    if (subject.size() < 3 || clip.size() < 3) return 0.0;

    std::vector<Vec2> current = subject;
    int clip_n = static_cast<int>(clip.size());

    for (int ci = 0; ci < clip_n; ++ci) {
        int cj = (ci + 1) % clip_n;

        double ex0 = clip[ci].x;
        double ey0 = clip[ci].y;
        double edx = clip[cj].x - ex0;
        double edy = clip[cj].y - ey0;

        // Skip degenerate edges
        if (edx * edx + edy * edy < 1.0e-30) continue;

        current = ref_clip_against_edge(current, ex0, ey0, edx, edy);

        if (current.size() < 3) return 0.0;
    }

    return shoelace_area(current);
}

// ─── Polygon Generators ─────────────────────────────────────────────────────

/// Generate a convex polygon as a regular n-gon centered at (cx, cy) with
/// given radius, slightly perturbed.
std::vector<Vec2> make_convex_polygon(double cx, double cy, double radius, int n_verts, const std::vector<double> &perturbations) {
    std::vector<Vec2> poly;
    poly.reserve(n_verts);
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    for (int i = 0; i < n_verts; ++i) {
        double angle = two_pi * static_cast<double>(i) / static_cast<double>(n_verts);
        double r = radius;
        if (i < static_cast<int>(perturbations.size())) {
            r *= (1.0 + perturbations[i] * 0.3);  // up to ±30% radius perturbation
        }
        poly.push_back({cx + r * std::cos(angle), cy + r * std::sin(angle)});
    }
    return poly;
}

/// Generate a concave polygon by creating a star-like shape.
/// Alternates between outer radius and inner radius.
std::vector<Vec2> make_concave_polygon(double cx, double cy, double outer_radius, double inner_radius, int n_points) {
    std::vector<Vec2> poly;
    poly.reserve(n_points);
    constexpr double two_pi = 2.0 * 3.14159265358979323846;
    for (int i = 0; i < n_points; ++i) {
        double angle = two_pi * static_cast<double>(i) / static_cast<double>(n_points);
        double r = (i % 2 == 0) ? outer_radius : inner_radius;
        poly.push_back({cx + r * std::cos(angle), cy + r * std::sin(angle)});
    }
    return poly;
}

/// Convert std::vector<Vec2> to PlanarPolygon<32>.
PlanarPolygon<32> to_planar_polygon(const std::vector<Vec2> &verts) {
    PlanarPolygon<32> poly;
    for (const auto &v : verts) {
        poly.push(v.x, v.y);
    }
    return poly;
}

/// Compare two areas with relative tolerance 1e-14.
/// When both are near zero, use absolute tolerance instead.
bool areas_agree(double a, double b, double rel_tol = 1.0e-14) {
    double max_abs = std::fmax(std::fabs(a), std::fabs(b));
    if (max_abs < 1.0e-20) {
        // Both essentially zero
        return std::fabs(a - b) < 1.0e-20;
    }
    double rel_error = std::fabs(a - b) / max_abs;
    return rel_error <= rel_tol;
}

// ─── Property 1: Parallel Planar Clipper Equivalence (Convex Polygons) ───────
// For random convex polygon pairs, PlanarClipper::overlap_area() agrees with
// the reference heap-based implementation within relative tolerance 1e-14.
//
// **Validates: Requirements 1.4**

RC_GTEST_PROP(PropPlanarClipper, ConvexPolygonEquivalence, ()) {
    // Generate vertex counts for both polygons (3 to 16 vertices)
    int n_verts_a = *rc::gen::inRange(3, 17);
    int n_verts_b = *rc::gen::inRange(3, 17);

    // Generate centers in [-10, 10]
    double cx_a = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });
    double cy_a = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });
    double cx_b = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });
    double cy_b = *rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return v * 0.01; });

    // Generate radii in [0.1, 5.0]
    double radius_a = *rc::gen::map(rc::gen::inRange(10, 500), [](int v) { return v * 0.01; });
    double radius_b = *rc::gen::map(rc::gen::inRange(10, 500), [](int v) { return v * 0.01; });

    // Perturbations for convex polygon vertices (keep within ±30% so polygon stays convex)
    std::vector<double> perturb_a;
    for (int i = 0; i < n_verts_a; ++i) {
        double p = *rc::gen::map(rc::gen::inRange(-100, 101), [](int v) { return v * 0.001; });
        perturb_a.push_back(p);
    }
    std::vector<double> perturb_b;
    for (int i = 0; i < n_verts_b; ++i) {
        double p = *rc::gen::map(rc::gen::inRange(-100, 101), [](int v) { return v * 0.001; });
        perturb_b.push_back(p);
    }

    auto verts_a = make_convex_polygon(cx_a, cy_a, radius_a, n_verts_a, perturb_a);
    auto verts_b = make_convex_polygon(cx_b, cy_b, radius_b, n_verts_b, perturb_b);

    // Compute overlap via reference (heap-based std::vector Sutherland-Hodgman)
    double ref_area = compute_polygon_overlap_area(verts_a, verts_b);

    // Compute overlap via PlanarClipper (fixed-capacity, KOKKOS_FUNCTION)
    auto poly_a = to_planar_polygon(verts_a);
    auto poly_b = to_planar_polygon(verts_b);
    double clipper_area = PlanarClipper::overlap_area<32>(poly_a, poly_b);

    // Verify equivalence within relative tolerance 1e-14
    RC_ASSERT(areas_agree(ref_area, clipper_area, 1.0e-14));
}

// ─── Property 1 (continued): Concave Polygon Equivalence ────────────────────
// For random concave (star-shaped) polygon pairs, PlanarClipper::overlap_area()
// agrees with the reference heap-based implementation within relative tolerance
// 1e-14.
//
// **Validates: Requirements 1.4**

RC_GTEST_PROP(PropPlanarClipper, ConcavePolygonEquivalence, ()) {
    // Generate vertex counts: concave star polygons need even vertex count >= 6
    int half_points_a = *rc::gen::inRange(3, 12);
    int half_points_b = *rc::gen::inRange(3, 12);
    int n_verts_a = half_points_a * 2;  // 6 to 22
    int n_verts_b = half_points_b * 2;

    // Clamp to max 32 vertices
    n_verts_a = std::min(n_verts_a, 32);
    n_verts_b = std::min(n_verts_b, 32);

    // Centers
    double cx_a = *rc::gen::map(rc::gen::inRange(-500, 501), [](int v) { return v * 0.01; });
    double cy_a = *rc::gen::map(rc::gen::inRange(-500, 501), [](int v) { return v * 0.01; });
    double cx_b = *rc::gen::map(rc::gen::inRange(-500, 501), [](int v) { return v * 0.01; });
    double cy_b = *rc::gen::map(rc::gen::inRange(-500, 501), [](int v) { return v * 0.01; });

    // Outer radii in [0.5, 4.0]
    double outer_a = *rc::gen::map(rc::gen::inRange(50, 400), [](int v) { return v * 0.01; });
    double outer_b = *rc::gen::map(rc::gen::inRange(50, 400), [](int v) { return v * 0.01; });

    // Inner radii as fraction of outer: [0.2, 0.8]
    double inner_frac_a = *rc::gen::map(rc::gen::inRange(20, 80), [](int v) { return v * 0.01; });
    double inner_frac_b = *rc::gen::map(rc::gen::inRange(20, 80), [](int v) { return v * 0.01; });
    double inner_a = outer_a * inner_frac_a;
    double inner_b = outer_b * inner_frac_b;

    auto verts_a = make_concave_polygon(cx_a, cy_a, outer_a, inner_a, n_verts_a);
    auto verts_b = make_concave_polygon(cx_b, cy_b, outer_b, inner_b, n_verts_b);

    // Compute overlap via reference
    double ref_area = compute_polygon_overlap_area(verts_a, verts_b);

    // Compute overlap via PlanarClipper
    auto poly_a = to_planar_polygon(verts_a);
    auto poly_b = to_planar_polygon(verts_b);
    double clipper_area = PlanarClipper::overlap_area<32>(poly_a, poly_b);

    // Verify equivalence within relative tolerance 1e-14
    RC_ASSERT(areas_agree(ref_area, clipper_area, 1.0e-14));
}

// ─── Property 1 (continued): Mixed Convex/Concave Equivalence ────────────────
// One convex and one concave polygon — verifies PlanarClipper handles the mixed
// case identically to the reference.
//
// **Validates: Requirements 1.4**

RC_GTEST_PROP(PropPlanarClipper, MixedConvexConcaveEquivalence, ()) {
    // Convex polygon (subject)
    int n_verts_convex = *rc::gen::inRange(3, 12);
    double cx_conv = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double cy_conv = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double r_conv = *rc::gen::map(rc::gen::inRange(20, 300), [](int v) { return v * 0.01; });

    std::vector<double> perturb;
    for (int i = 0; i < n_verts_convex; ++i) {
        double p = *rc::gen::map(rc::gen::inRange(-80, 81), [](int v) { return v * 0.001; });
        perturb.push_back(p);
    }

    // Concave polygon (clip)
    int half_pts = *rc::gen::inRange(3, 10);
    int n_verts_concave = half_pts * 2;
    n_verts_concave = std::min(n_verts_concave, 32);

    double cx_conc = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double cy_conc = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double outer = *rc::gen::map(rc::gen::inRange(30, 300), [](int v) { return v * 0.01; });
    double inner_frac = *rc::gen::map(rc::gen::inRange(25, 75), [](int v) { return v * 0.01; });
    double inner = outer * inner_frac;

    auto verts_a = make_convex_polygon(cx_conv, cy_conv, r_conv, n_verts_convex, perturb);
    auto verts_b = make_concave_polygon(cx_conc, cy_conc, outer, inner, n_verts_concave);

    // Reference
    double ref_area = compute_polygon_overlap_area(verts_a, verts_b);

    // PlanarClipper
    auto poly_a = to_planar_polygon(verts_a);
    auto poly_b = to_planar_polygon(verts_b);
    double clipper_area = PlanarClipper::overlap_area<32>(poly_a, poly_b);

    RC_ASSERT(areas_agree(ref_area, clipper_area, 1.0e-14));
}

// ─── Property 1 (continued): Commutativity ──────────────────────────────────
// overlap_area(A, B) == overlap_area(B, A) for all polygon pairs.
// This is a self-consistency property that both implementations should satisfy.
//
// **Validates: Requirements 1.4**

RC_GTEST_PROP(PropPlanarClipper, OverlapAreaCommutativity, ()) {
    int n_a = *rc::gen::inRange(3, 12);
    int n_b = *rc::gen::inRange(3, 12);

    double cx_a = *rc::gen::map(rc::gen::inRange(-200, 201), [](int v) { return v * 0.01; });
    double cy_a = *rc::gen::map(rc::gen::inRange(-200, 201), [](int v) { return v * 0.01; });
    double cx_b = *rc::gen::map(rc::gen::inRange(-200, 201), [](int v) { return v * 0.01; });
    double cy_b = *rc::gen::map(rc::gen::inRange(-200, 201), [](int v) { return v * 0.01; });
    double r_a = *rc::gen::map(rc::gen::inRange(10, 200), [](int v) { return v * 0.01; });
    double r_b = *rc::gen::map(rc::gen::inRange(10, 200), [](int v) { return v * 0.01; });

    std::vector<double> pert_a, pert_b;
    for (int i = 0; i < n_a; ++i) {
        pert_a.push_back(*rc::gen::map(rc::gen::inRange(-50, 51), [](int v) { return v * 0.001; }));
    }
    for (int i = 0; i < n_b; ++i) {
        pert_b.push_back(*rc::gen::map(rc::gen::inRange(-50, 51), [](int v) { return v * 0.001; }));
    }

    auto verts_a = make_convex_polygon(cx_a, cy_a, r_a, n_a, pert_a);
    auto verts_b = make_convex_polygon(cx_b, cy_b, r_b, n_b, pert_b);

    auto poly_a = to_planar_polygon(verts_a);
    auto poly_b = to_planar_polygon(verts_b);

    double area_ab = PlanarClipper::overlap_area<32>(poly_a, poly_b);
    double area_ba = PlanarClipper::overlap_area<32>(poly_b, poly_a);

    // Commutativity: overlap(A,B) == overlap(B,A)
    // Note: Sutherland-Hodgman is not perfectly commutative due to
    // floating-point differences in the clipping order. Use a relaxed
    // tolerance (1e-10) since intermediate intersection vertices differ
    // depending on which polygon is subject vs clip.
    RC_ASSERT(areas_agree(area_ab, area_ba, 1.0e-10));
}

// ─── Property 1 (continued): Area Bounds ─────────────────────────────────────
// overlap_area(A, B) <= min(area(A), area(B)) — overlap cannot exceed either
// polygon's area.
//
// **Validates: Requirements 1.4**

RC_GTEST_PROP(PropPlanarClipper, OverlapAreaBounds, ()) {
    int n_a = *rc::gen::inRange(3, 14);
    int n_b = *rc::gen::inRange(3, 14);

    double cx_a = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double cy_a = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double cx_b = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double cy_b = *rc::gen::map(rc::gen::inRange(-300, 301), [](int v) { return v * 0.01; });
    double r_a = *rc::gen::map(rc::gen::inRange(10, 300), [](int v) { return v * 0.01; });
    double r_b = *rc::gen::map(rc::gen::inRange(10, 300), [](int v) { return v * 0.01; });

    std::vector<double> pert_a, pert_b;
    for (int i = 0; i < n_a; ++i) {
        pert_a.push_back(*rc::gen::map(rc::gen::inRange(-80, 81), [](int v) { return v * 0.001; }));
    }
    for (int i = 0; i < n_b; ++i) {
        pert_b.push_back(*rc::gen::map(rc::gen::inRange(-80, 81), [](int v) { return v * 0.001; }));
    }

    auto verts_a = make_convex_polygon(cx_a, cy_a, r_a, n_a, pert_a);
    auto verts_b = make_convex_polygon(cx_b, cy_b, r_b, n_b, pert_b);

    auto poly_a = to_planar_polygon(verts_a);
    auto poly_b = to_planar_polygon(verts_b);

    double overlap = PlanarClipper::overlap_area<32>(poly_a, poly_b);
    double area_a = poly_a.area();
    double area_b = poly_b.area();
    double min_area = std::fmin(area_a, area_b);

    // Non-negativity
    RC_ASSERT(overlap >= 0.0);

    // Overlap cannot exceed the smaller polygon's area (with small tolerance
    // for floating-point rounding)
    RC_ASSERT(overlap <= min_area * (1.0 + 1.0e-14));
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
