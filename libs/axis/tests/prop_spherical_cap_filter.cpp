// ─── Property-Based Tests: Spherical Cap Filter ─────────────────────────────
// Feature: axis-performance-optimizations
//
// Property 5: Angular Radius Bound Soundness
//
// For any spherical polygon cell on the unit sphere, the computed angular
// radius (maximum arc-distance from centroid to any vertex) SHALL be greater
// than or equal to the angular distance from the centroid to any point on the
// polygon boundary. This ensures the spherical cap fully contains the cell.
//
// **Validates: Requirements 3.1**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/spherical_cap_filter.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <cmath>
#include <vector>

namespace {

using axis::detail::cross;
using axis::detail::dot;
using axis::detail::length;
using axis::detail::normalize;
using axis::detail::Vec3;

constexpr double pi = 3.14159265358979323846;
constexpr double deg2rad = pi / 180.0;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert lon (degrees), lat (degrees) to unit-sphere Vec3.
Vec3 lonlat_deg_to_vec3(double lon_deg, double lat_deg) {
    double lon = lon_deg * deg2rad;
    double lat = lat_deg * deg2rad;
    double cl = std::cos(lat);
    return Vec3{cl * std::cos(lon), cl * std::sin(lon), std::sin(lat)};
}

/// Spherical linear interpolation (slerp) between two unit vectors on a
/// great-circle arc. Parameter t in [0, 1].
Vec3 slerp(const Vec3 &a, const Vec3 &b, double t) {
    double d = dot(a, b);
    d = std::clamp(d, -1.0, 1.0);
    double omega = std::acos(d);

    // If vectors are nearly identical, linearly interpolate and normalize
    if (omega < 1e-12) {
        return normalize(Vec3{a.x * (1.0 - t) + b.x * t, a.y * (1.0 - t) + b.y * t, a.z * (1.0 - t) + b.z * t});
    }

    double sin_omega = std::sin(omega);
    double coeff_a = std::sin((1.0 - t) * omega) / sin_omega;
    double coeff_b = std::sin(t * omega) / sin_omega;

    return normalize(Vec3{a.x * coeff_a + b.x * coeff_b, a.y * coeff_a + b.y * coeff_b, a.z * coeff_a + b.z * coeff_b});
}

/// Compute angular distance between two unit vectors (radians).
double angular_distance(const Vec3 &a, const Vec3 &b) {
    double d = dot(a, b);
    d = std::clamp(d, -1.0, 1.0);
    return std::acos(d);
}

/// Build a single-cell UnstructuredMesh from vertex coordinates in degrees.
axis::topology::UnstructuredMesh<Kokkos::HostSpace> build_single_cell_mesh(const std::vector<std::pair<double, double>> &vertices) {
    using MemSpace = Kokkos::HostSpace;
    const std::size_t n_nodes = vertices.size();

    Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace> node_coords("node_coords", n_nodes, 2);

    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords(i, 0) = vertices[i].first;   // lon
        node_coords(i, 1) = vertices[i].second;  // lat
    }

    Kokkos::View<axis::index_t *, MemSpace> offsets("offsets", 2);
    offsets(0) = 0;
    offsets(1) = static_cast<axis::index_t>(n_nodes);

    Kokkos::View<axis::index_t *, MemSpace> indices("indices", n_nodes);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        indices(i) = static_cast<axis::index_t>(i);
    }

    return axis::topology::UnstructuredMesh<MemSpace>(std::move(node_coords), std::move(offsets), std::move(indices),
                                                      axis::topology::CoordinateSystem::SphericalDeg);
}

// ─── Property 5: Angular Radius Bound Soundness ─────────────────────────────
//
// For any spherical polygon cell on the unit sphere, the computed angular
// radius (max arc-distance from centroid to any vertex) SHALL be >= the
// angular distance from the centroid to any point on the polygon boundary.
//
// Test strategy:
//   1. Generate random spherical polygon cells at various locations and sizes
//   2. Compute centroid and angular radius via precompute_cap_data()
//   3. Verify angular radius >= distance to each vertex (by definition)
//   4. Sample points along polygon boundary (great-circle arcs between
//      adjacent vertices) and verify each is within angular radius
//
// **Validates: Requirements 3.1**

RC_GTEST_PROP(PropSphericalCapFilter, AngularRadiusBoundSoundness, ()) {
    // Generate a random cell center at various locations:
    //   lon ∈ [-180, 180], lat ∈ [-85, 85] (avoid exact poles for cell shape)
    double center_lon = *rc::gen::map(rc::gen::inRange(-1800, 1801), [](int v) { return v * 0.1; });
    double center_lat = *rc::gen::map(rc::gen::inRange(-850, 851), [](int v) { return v * 0.1; });

    // Number of vertices: 3 to 8 (triangles to octagons)
    int n_verts = *rc::gen::inRange(3, 9);

    // Cell angular size: 0.5° to 30° half-extent
    double half_extent = *rc::gen::map(rc::gen::inRange(5, 301), [](int v) { return v * 0.1; });

    // Generate polygon vertices as points at angular distance `half_extent`
    // from center, equally spaced in azimuth around the center, with random
    // perturbation in radius
    Vec3 center_xyz = lonlat_deg_to_vec3(center_lon, center_lat);

    // Build a local coordinate frame (tangent plane at center)
    Vec3 arbitrary = (std::abs(center_xyz.z) < 0.9) ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    Vec3 u = normalize(cross(center_xyz, arbitrary));
    Vec3 v_dir = cross(center_xyz, u);

    // Generate vertices
    std::vector<std::pair<double, double>> vertices;
    std::vector<Vec3> vertex_xyz;
    vertices.reserve(n_verts);
    vertex_xyz.reserve(n_verts);

    double radius_rad = half_extent * deg2rad;

    for (int i = 0; i < n_verts; ++i) {
        double azimuth = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n_verts);

        // Random radial perturbation: 50% to 100% of half_extent
        double frac = *rc::gen::map(rc::gen::inRange(50, 101), [](int v) { return v * 0.01; });
        double r = radius_rad * frac;

        double cos_r = std::cos(r);
        double sin_r = std::sin(r);
        double ca = std::cos(azimuth);
        double sa = std::sin(azimuth);

        Vec3 pt = normalize(Vec3{center_xyz.x * cos_r + u.x * sin_r * ca + v_dir.x * sin_r * sa,
                                 center_xyz.y * cos_r + u.y * sin_r * ca + v_dir.y * sin_r * sa,
                                 center_xyz.z * cos_r + u.z * sin_r * ca + v_dir.z * sin_r * sa});

        // Convert back to lon/lat degrees
        double lat = std::asin(std::clamp(pt.z, -1.0, 1.0)) / deg2rad;
        double lon = std::atan2(pt.y, pt.x) / deg2rad;

        vertices.push_back({lon, lat});
        vertex_xyz.push_back(pt);
    }

    // Build mesh and compute cap data
    auto mesh = build_single_cell_mesh(vertices);
    auto cap_data = axis::detail::precompute_cap_data<Kokkos::HostSpace>(mesh);

    Vec3 centroid = cap_data.centroids(0);
    double angular_radius = cap_data.angular_radii(0);

    // Sanity: centroid should be unit length
    double clen = std::sqrt(dot(centroid, centroid));
    RC_ASSERT(std::abs(clen - 1.0) < 1e-12);

    // Angular radius should be positive
    RC_ASSERT(angular_radius > 0.0);

    // ── Check 1: Angular radius >= distance to each vertex ──────────────────
    for (int i = 0; i < n_verts; ++i) {
        Vec3 vert = lonlat_deg_to_vec3(vertices[i].first, vertices[i].second);
        double dist = angular_distance(centroid, vert);
        RC_ASSERT(angular_radius >= dist - 1e-12);
    }

    // ── Check 2: Angular radius >= distance to boundary sample points ───────
    // Sample points along great-circle arcs between consecutive vertices.
    // Use 10 sample points per edge.
    constexpr int n_samples_per_edge = 10;

    for (int i = 0; i < n_verts; ++i) {
        int j = (i + 1) % n_verts;

        Vec3 v_start = lonlat_deg_to_vec3(vertices[i].first, vertices[i].second);
        Vec3 v_end = lonlat_deg_to_vec3(vertices[j].first, vertices[j].second);

        for (int s = 1; s < n_samples_per_edge; ++s) {
            double t = static_cast<double>(s) / static_cast<double>(n_samples_per_edge);

            Vec3 boundary_pt = slerp(v_start, v_end, t);
            double dist = angular_distance(centroid, boundary_pt);

            // The angular radius must bound all boundary points.
            // Allow a tiny numerical tolerance for floating-point slerp.
            RC_ASSERT(angular_radius >= dist - 1e-10);
        }
    }
}

// ─── Property 5 (variant): Polar and large-cell stress test ──────────────────
// Additional coverage for cells near poles and cells spanning large angular
// extents, where the centroid / radius computation can be numerically tricky.
//
// **Validates: Requirements 3.1**

RC_GTEST_PROP(PropSphericalCapFilter, AngularRadiusBoundSoundnessPolarCells, ()) {
    // Generate cells near the poles: lat ∈ [70, 89] or [-89, -70]
    bool north = *rc::gen::arbitrary<bool>();
    double center_lat = north ? *rc::gen::map(rc::gen::inRange(700, 890), [](int v) { return v * 0.1; })
                              : *rc::gen::map(rc::gen::inRange(-890, -700), [](int v) { return v * 0.1; });
    double center_lon = *rc::gen::map(rc::gen::inRange(-1800, 1801), [](int v) { return v * 0.1; });

    // Smaller cells near poles: 1° to 15° half-extent
    double half_extent = *rc::gen::map(rc::gen::inRange(10, 151), [](int v) { return v * 0.1; });

    int n_verts = *rc::gen::inRange(3, 7);

    Vec3 center_xyz = lonlat_deg_to_vec3(center_lon, center_lat);

    Vec3 arbitrary = (std::abs(center_xyz.z) < 0.9) ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    Vec3 u = normalize(cross(center_xyz, arbitrary));
    Vec3 v_dir = cross(center_xyz, u);

    std::vector<std::pair<double, double>> vertices;
    double radius_rad = half_extent * deg2rad;

    for (int i = 0; i < n_verts; ++i) {
        double azimuth = 2.0 * pi * static_cast<double>(i) / static_cast<double>(n_verts);

        double frac = *rc::gen::map(rc::gen::inRange(60, 101), [](int v) { return v * 0.01; });
        double r = radius_rad * frac;

        double cos_r = std::cos(r);
        double sin_r = std::sin(r);
        double ca = std::cos(azimuth);
        double sa = std::sin(azimuth);

        Vec3 pt = normalize(Vec3{center_xyz.x * cos_r + u.x * sin_r * ca + v_dir.x * sin_r * sa,
                                 center_xyz.y * cos_r + u.y * sin_r * ca + v_dir.y * sin_r * sa,
                                 center_xyz.z * cos_r + u.z * sin_r * ca + v_dir.z * sin_r * sa});

        double lat = std::asin(std::clamp(pt.z, -1.0, 1.0)) / deg2rad;
        double lon = std::atan2(pt.y, pt.x) / deg2rad;

        vertices.push_back({lon, lat});
    }

    auto mesh = build_single_cell_mesh(vertices);
    auto cap_data = axis::detail::precompute_cap_data<Kokkos::HostSpace>(mesh);

    Vec3 centroid = cap_data.centroids(0);
    double angular_radius = cap_data.angular_radii(0);

    RC_ASSERT(angular_radius > 0.0);

    // Verify all vertices are within angular radius
    for (int i = 0; i < n_verts; ++i) {
        Vec3 vert = lonlat_deg_to_vec3(vertices[i].first, vertices[i].second);
        double dist = angular_distance(centroid, vert);
        RC_ASSERT(angular_radius >= dist - 1e-12);
    }

    // Verify boundary sample points are within angular radius
    constexpr int n_samples = 10;
    for (int i = 0; i < n_verts; ++i) {
        int j = (i + 1) % n_verts;
        Vec3 v_start = lonlat_deg_to_vec3(vertices[i].first, vertices[i].second);
        Vec3 v_end = lonlat_deg_to_vec3(vertices[j].first, vertices[j].second);

        for (int s = 1; s < n_samples; ++s) {
            double t = static_cast<double>(s) / static_cast<double>(n_samples);
            Vec3 boundary_pt = slerp(v_start, v_end, t);
            double dist = angular_distance(centroid, boundary_pt);
            RC_ASSERT(angular_radius >= dist - 1e-10);
        }
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
