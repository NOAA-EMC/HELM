// AXIS unit test: detail/spherical_geometry.hpp
// Tests the numerically robust spherical geometric primitives derived from
// Chen et al. (2026) "Accurate and Robust Geometric Algorithms for Regridding
// on the Sphere".

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/spherical_geometry.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <utility>
#include <vector>

namespace {

class KokkosEnv : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnv);

using namespace axis::detail::spherical;
constexpr double deg2rad = pi / 180.0;
constexpr double EARTH_R2 = 1.0;  // unit sphere

// ─────────────────────────────────────────────────────────────────────────────
// EFT (Error-Free Transformations) basic sanity
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, TwoSumExact) {
    double a = 1.0;
    double b = 1e-16;  // below single ulp of 1.0 in double
    double s, t;
    two_sum(a, b, s, t);
    // s + t must reproduce a + b exactly.
    EXPECT_DOUBLE_EQ(s + t, a + b);
}

TEST(SphericalGeometry, TwoProductExact) {
    double a = 1.0 + 1e-10;
    double b = 1.0 - 1e-10;
    double p, e;
    two_product(a, b, p, e);
    // p + e should give a*b with extended precision.
    // The true result is 1 - 1e-20.
    double true_val = a * b;
    EXPECT_DOUBLE_EQ(p + e, true_val);
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate conversions
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, LonLatToXyzRoundtrip) {
    double lon = 45.0 * deg2rad;
    double lat = 30.0 * deg2rad;

    Vec3 p = lonlat_to_xyz(lon, lat);

    // Should be on unit sphere.
    EXPECT_NEAR(norm(p), 1.0, 1e-15);

    // Roundtrip.
    double lon2, lat2;
    xyz_to_lonlat(p, lon2, lat2);
    EXPECT_NEAR(lon2, lon, 1e-14);
    EXPECT_NEAR(lat2, lat, 1e-14);
}

TEST(SphericalGeometry, PolesToXyz) {
    Vec3 north = lonlat_to_xyz(0.0, pi / 2.0);
    EXPECT_NEAR(north.x, 0.0, 1e-15);
    EXPECT_NEAR(north.y, 0.0, 1e-15);
    EXPECT_NEAR(north.z, 1.0, 1e-15);

    Vec3 south = lonlat_to_xyz(0.0, -pi / 2.0);
    EXPECT_NEAR(south.z, -1.0, 1e-15);
}

// ─────────────────────────────────────────────────────────────────────────────
// Robust orientation predicate
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, OrientSphereBasic) {
    // Equatorial triangle: (0°,0°), (90°,0°), (0°,90°)
    Vec3 a = lonlat_to_xyz(0.0, 0.0);
    Vec3 b = lonlat_to_xyz(pi / 2.0, 0.0);
    Vec3 c = lonlat_to_xyz(0.0, pi / 2.0);

    double orient = robust_orient_sphere(a, b, c);
    // C is to the left of A->B (CCW ordering), so orient > 0.
    EXPECT_GT(orient, 0.0);

    // Reversed: C on the right.
    double orient_rev = robust_orient_sphere(b, a, c);
    EXPECT_LT(orient_rev, 0.0);
}

TEST(SphericalGeometry, OrientSphereColinear) {
    // Three points on the equator: (0,0), (45,0), (90,0)
    Vec3 a = lonlat_to_xyz(0.0, 0.0);
    Vec3 b = lonlat_to_xyz(pi / 4.0, 0.0);
    Vec3 c = lonlat_to_xyz(pi / 2.0, 0.0);

    double orient = robust_orient_sphere(a, c, b);
    // b is on the great circle through a and c, so orient ≈ 0.
    EXPECT_NEAR(orient, 0.0, 1e-14);
}

// ─────────────────────────────────────────────────────────────────────────────
// Great-circle arc intersection
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, GcArcIntersectionCrossing) {
    // Arc 1: equator from (0°,0°) to (90°,0°)
    // Arc 2: meridian from (45°,-45°) to (45°,45°)
    // They should intersect at (45°, 0°).
    Vec3 a1 = lonlat_to_xyz(0.0, 0.0);
    Vec3 a2 = lonlat_to_xyz(pi / 2.0, 0.0);
    Vec3 b1 = lonlat_to_xyz(pi / 4.0, -pi / 4.0);
    Vec3 b2 = lonlat_to_xyz(pi / 4.0, pi / 4.0);

    Vec3 p{};
    bool found = great_circle_arc_intersection(a1, a2, b1, b2, p);
    EXPECT_TRUE(found);

    // Check intersection is near (45°, 0°).
    Vec3 expected = lonlat_to_xyz(pi / 4.0, 0.0);
    EXPECT_NEAR(p.x, expected.x, 1e-10);
    EXPECT_NEAR(p.y, expected.y, 1e-10);
    EXPECT_NEAR(p.z, expected.z, 1e-10);
}

TEST(SphericalGeometry, GcArcIntersectionNoIntersection) {
    // Two arcs that don't cross.
    Vec3 a1 = lonlat_to_xyz(0.0, 0.0);
    Vec3 a2 = lonlat_to_xyz(pi / 4.0, 0.0);
    Vec3 b1 = lonlat_to_xyz(pi / 2.0, pi / 4.0);
    Vec3 b2 = lonlat_to_xyz(3.0 * pi / 4.0, pi / 4.0);

    Vec3 p{};
    bool found = great_circle_arc_intersection(a1, a2, b1, b2, p);
    EXPECT_FALSE(found);
}

// ─────────────────────────────────────────────────────────────────────────────
// Constant-latitude intersection
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, ConstLatIntersectionMeridianCrossingEquator) {
    // Meridian arc from (0°,-45°) to (0°,45°) crossing equator (lat=0).
    Vec3 a1 = lonlat_to_xyz(0.0, -pi / 4.0);
    Vec3 a2 = lonlat_to_xyz(0.0, pi / 4.0);

    std::array<Vec3, 2> pts{};
    int count = great_circle_const_lat_intersection(a1, a2, 0.0, pts);
    EXPECT_GE(count, 1);

    // The intersection should be at (0°, 0°) = (1, 0, 0).
    bool found_origin = false;
    for (int i = 0; i < count; ++i) {
        if (std::abs(pts[i].x - 1.0) < 1e-10 && std::abs(pts[i].y) < 1e-10 && std::abs(pts[i].z) < 1e-10) {
            found_origin = true;
        }
    }
    EXPECT_TRUE(found_origin);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical polygon area
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, SphericalTriangleAreaOctant) {
    // Spherical triangle covering one octant of the sphere:
    // Vertices: (0°,0°), (90°,0°), (0°,90°)
    // Area should be π/2 steradians (one eighth of 4π).
    std::vector<Vec3> tri = {lonlat_to_xyz(0.0, 0.0), lonlat_to_xyz(pi / 2.0, 0.0), lonlat_to_xyz(0.0, pi / 2.0)};

    double area = spherical_polygon_area(tri);
    EXPECT_NEAR(area, pi / 2.0, 1e-10);
}

TEST(SphericalGeometry, SphericalTriangleAreaSmall) {
    // A small triangle near (10°, 50°) with ~1° sides.
    // For small triangles, spherical area ≈ planar area * cos(lat).
    double lon0 = 10.0 * deg2rad;
    double lat0 = 50.0 * deg2rad;
    double dlat = 1.0 * deg2rad;
    double dlon = 1.0 * deg2rad;

    std::vector<Vec3> tri = {lonlat_to_xyz(lon0, lat0), lonlat_to_xyz(lon0 + dlon, lat0), lonlat_to_xyz(lon0, lat0 + dlat)};

    double area = spherical_polygon_area(tri);

    // Expected area ≈ 0.5 * dlon * dlat * cos(lat0) (on unit sphere)
    double expected = 0.5 * dlon * dlat * std::cos(lat0);
    EXPECT_NEAR(area, expected, expected * 0.01);  // Within 1% for 1° triangle
}

TEST(SphericalGeometry, SphericalQuadAreaHemisphere) {
    // A quad covering a quarter of the sphere (one face of a cubed sphere-like partition).
    // Equatorial band from 0° to 90° lon, 0° to 90° lat.
    // Exact area = π steradians.
    std::vector<Vec3> quad = {lonlat_to_xyz(0.0, 0.0), lonlat_to_xyz(pi / 2.0, 0.0), lonlat_to_xyz(pi / 2.0, pi / 2.0), lonlat_to_xyz(0.0, pi / 2.0)};

    double area = spherical_polygon_area(quad);
    // A spherical quadrilateral from (0,0) to (90°,90°) covers 1/8 of the sphere.
    // Full sphere = 4π, so expected area = 4π/8 = π/2 steradians.
    EXPECT_NEAR(area, pi / 2.0, 1e-8);
}

TEST(SphericalGeometry, LargeGreatCircleTriangleUsesAtan2ForNegativeDenominator) {
    using MemSpace = Kokkos::HostSpace;
    Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace> coords("coords", 3, 2);
    const double lon[] = {0.0, 120.0, 250.0};
    const double lat[] = {0.0, 10.0, 20.0};
    for (std::size_t i = 0; i < 3; ++i) {
        coords(i, 0) = lon[i];
        coords(i, 1) = lat[i];
    }
    Kokkos::View<axis::index_t *, MemSpace> offsets("offsets", 2);
    Kokkos::View<axis::index_t *, MemSpace> indices("indices", 3);
    offsets(0) = 0;
    offsets(1) = 3;
    indices(0) = 0;
    indices(1) = 1;
    indices(2) = 2;
    axis::topology::UnstructuredMesh<MemSpace> mesh(std::move(coords), std::move(offsets), std::move(indices),
                                                    axis::topology::CoordinateSystem::SphericalDeg);
    mesh.compute_areas();

    // This valid triangle has a negative denominator in the robust spherical
    // excess expression. Its smaller spherical excess is about 4.472 steradians.
    EXPECT_NEAR(mesh.cell_areas()(0), 4.472373859838174, 1e-12);
    EXPECT_NE(mesh.cell_areas()(0), axis::detail::spherical::pi);
}

// ─────────────────────────────────────────────────────────────────────────────
// Spherical polygon clipping
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, ClipIdenticalPolygons) {
    // Clipping a polygon against itself should return (approximately) itself.
    std::vector<Vec3> poly = {lonlat_to_xyz(0.0, 0.0), lonlat_to_xyz(pi / 4.0, 0.0), lonlat_to_xyz(pi / 4.0, pi / 4.0), lonlat_to_xyz(0.0, pi / 4.0)};

    auto clipped = spherical_clip_polygon(poly, poly);
    EXPECT_GE(clipped.size(), 3u);

    // The area of the clipped polygon should match the original.
    double area_orig = spherical_polygon_area(poly);
    double area_clip = spherical_polygon_area(clipped);
    EXPECT_NEAR(area_clip, area_orig, area_orig * 1e-8);
}

TEST(SphericalGeometry, ClipNonOverlapping) {
    // Two polygons on opposite sides of the sphere.
    std::vector<Vec3> poly_a = {lonlat_to_xyz(0.0, 0.0), lonlat_to_xyz(10.0 * deg2rad, 0.0), lonlat_to_xyz(10.0 * deg2rad, 10.0 * deg2rad),
                                lonlat_to_xyz(0.0, 10.0 * deg2rad)};

    std::vector<Vec3> poly_b = {lonlat_to_xyz(pi, 0.0), lonlat_to_xyz(pi + 10.0 * deg2rad, 0.0), lonlat_to_xyz(pi + 10.0 * deg2rad, 10.0 * deg2rad),
                                lonlat_to_xyz(pi, 10.0 * deg2rad)};

    auto clipped = spherical_clip_polygon(poly_a, poly_b);
    // Should be empty or have zero area.
    if (clipped.size() >= 3) {
        double area = spherical_polygon_area(clipped);
        EXPECT_NEAR(area, 0.0, 1e-12);
    } else {
        EXPECT_LT(clipped.size(), 3u);
    }
}

TEST(SphericalGeometry, ClipPartialOverlap) {
    // Two quads sharing a 5° strip:
    // Poly A: [0°,10°] lon × [0°,10°] lat
    // Poly B: [5°,15°] lon × [0°,10°] lat
    // Overlap: [5°,10°] lon × [0°,10°] lat (half of A).
    std::vector<Vec3> poly_a = {lonlat_to_xyz(0.0, 0.0), lonlat_to_xyz(10.0 * deg2rad, 0.0), lonlat_to_xyz(10.0 * deg2rad, 10.0 * deg2rad),
                                lonlat_to_xyz(0.0, 10.0 * deg2rad)};

    std::vector<Vec3> poly_b = {lonlat_to_xyz(5.0 * deg2rad, 0.0), lonlat_to_xyz(15.0 * deg2rad, 0.0), lonlat_to_xyz(15.0 * deg2rad, 10.0 * deg2rad),
                                lonlat_to_xyz(5.0 * deg2rad, 10.0 * deg2rad)};

    double overlap_area = spherical_polygon_overlap_area(poly_a, poly_b);
    double area_a = spherical_polygon_area(poly_a);

    // The overlap should be approximately half of poly_a.
    EXPECT_NEAR(overlap_area, area_a * 0.5, area_a * 0.05);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience conversion helpers
// ─────────────────────────────────────────────────────────────────────────────

TEST(SphericalGeometry, PolygonLonLatDegToXyz) {
    std::vector<double> lons = {0.0, 90.0, 90.0, 0.0};
    std::vector<double> lats = {0.0, 0.0, 45.0, 45.0};

    auto verts = polygon_lonlat_deg_to_xyz(lons, lats);
    EXPECT_EQ(verts.size(), 4u);

    // All should be on unit sphere.
    for (const auto &v : verts) {
        EXPECT_NEAR(norm(v), 1.0, 1e-15);
    }
}

}  // namespace
