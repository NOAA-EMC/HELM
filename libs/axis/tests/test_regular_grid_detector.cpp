// AXIS unit test: RegularGridInfo and detect_regular_grid()
// Verifies that the detector correctly identifies uniform regular lat-lon grids
// and rejects non-uniform or non-quad meshes.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

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
static auto *const kenv = ::testing::AddGlobalTestEnvironment(new KokkosEnv);
}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Helper: build a uniform regular lat-lon grid as an UnstructuredMesh.
// Grid covers [lon0, lon0 + ni*dlon] x [lat0, lat0 + nj*dlat]
static topology::UnstructuredMesh<MemSpace> make_regular_grid(std::size_t ni, std::size_t nj, double lon0, double dlon, double lat0, double dlat) {
    // Center coordinates
    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon0 + (static_cast<double>(i) + 0.5) * dlon;
            cy(i + j * ni) = lat0 + (static_cast<double>(j) + 0.5) * dlat;
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);

    // Set corners (uniform grid boundaries)
    const std::size_t nc_i = ni + 1;
    const std::size_t nc_j = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_i * nc_j);
    Kokkos::View<double *, MemSpace> cry("cry", nc_i * nc_j);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_i) = lon0 + static_cast<double>(i) * dlon;
            cry(i + j * nc_i) = lat0 + static_cast<double>(j) * dlat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// Helper: build a non-uniform grid (varying delta_lon)
static topology::UnstructuredMesh<MemSpace> make_nonuniform_grid(std::size_t ni, std::size_t nj) {
    // Non-uniform longitude spacing: exponentially increasing
    std::vector<double> lon_bounds(ni + 1);
    lon_bounds[0] = 0.0;
    for (std::size_t i = 1; i <= ni; ++i) {
        lon_bounds[i] = lon_bounds[i - 1] + static_cast<double>(i) * 0.5;
    }

    double dlat = 1.0;

    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = 0.5 * (lon_bounds[i] + lon_bounds[i + 1]);
            cy(i + j * ni) = (static_cast<double>(j) + 0.5) * dlat;
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);

    const std::size_t nc_i = ni + 1;
    const std::size_t nc_j = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_i * nc_j);
    Kokkos::View<double *, MemSpace> cry("cry", nc_i * nc_j);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_i) = lon_bounds[i];
            cry(i + j * nc_i) = static_cast<double>(j) * dlat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Regular grid is detected correctly
// ─────────────────────────────────────────────────────────────────────────────

TEST(RegularGridDetector, DetectsUniformGrid) {
    auto mesh = make_regular_grid(10, 5, 0.0, 1.0, -2.5, 1.0);
    auto info = detail::detect_regular_grid(mesh);

    EXPECT_TRUE(info.is_regular);
    EXPECT_EQ(info.ni, 10u);
    EXPECT_EQ(info.nj, 5u);
    EXPECT_NEAR(info.delta_lon, 1.0, 1e-12);
    EXPECT_NEAR(info.delta_lat, 1.0, 1e-12);
    EXPECT_NEAR(info.lon_min, 0.0, 1e-12);
    EXPECT_NEAR(info.lon_max, 10.0, 1e-12);
    EXPECT_NEAR(info.lat_min, -2.5, 1e-12);
    EXPECT_NEAR(info.lat_max, 2.5, 1e-12);
}

TEST(RegularGridDetector, DetectsTypicalLatLonGrid) {
    // Typical 360x180 degree global grid with 0.25 degree spacing
    auto mesh = make_regular_grid(4, 3, 0.0, 0.25, -0.375, 0.25);
    auto info = detail::detect_regular_grid(mesh);

    EXPECT_TRUE(info.is_regular);
    EXPECT_EQ(info.ni, 4u);
    EXPECT_EQ(info.nj, 3u);
    EXPECT_NEAR(info.delta_lon, 0.25, 1e-12);
    EXPECT_NEAR(info.delta_lat, 0.25, 1e-12);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Non-uniform grid is rejected
// ─────────────────────────────────────────────────────────────────────────────

TEST(RegularGridDetector, RejectsNonUniformGrid) {
    auto mesh = make_nonuniform_grid(5, 3);
    auto info = detail::detect_regular_grid(mesh);

    EXPECT_FALSE(info.is_regular);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Empty mesh returns is_regular=false
// ─────────────────────────────────────────────────────────────────────────────

TEST(RegularGridDetector, RejectsEmptyMesh) {
    topology::UnstructuredMesh<MemSpace> empty_mesh;
    auto info = detail::detect_regular_grid(empty_mesh);

    EXPECT_FALSE(info.is_regular);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: 1x1 grid is detected (minimal case)
// ─────────────────────────────────────────────────────────────────────────────

TEST(RegularGridDetector, Detects1x1Grid) {
    auto mesh = make_regular_grid(1, 1, 0.0, 1.0, 0.0, 1.0);
    auto info = detail::detect_regular_grid(mesh);

    // A single cell has only 1 delta per axis — trivially uniform
    EXPECT_TRUE(info.is_regular);
    EXPECT_EQ(info.ni, 1u);
    EXPECT_EQ(info.nj, 1u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: RegularGridInfo struct defaults
// ─────────────────────────────────────────────────────────────────────────────

TEST(RegularGridDetector, DefaultStructIsNotRegular) {
    detail::RegularGridInfo info;
    EXPECT_FALSE(info.is_regular);
    EXPECT_EQ(info.ni, 0u);
    EXPECT_EQ(info.nj, 0u);
    EXPECT_DOUBLE_EQ(info.delta_lon, 0.0);
    EXPECT_DOUBLE_EQ(info.delta_lat, 0.0);
}

}  // namespace axis::test

namespace axis::test {
TEST(RegularGridDetector, PermutedCellConnectivityRejectsArithmeticIndexing) {
    auto mesh = make_regular_grid(4, 3, 0, 1, 0, 1);
    auto reordered = Kokkos::View<index_t *, MemSpace>("permuted_connectivity", mesh.conn_indices_view().extent(0));
    auto original = mesh.conn_indices_view();
    for (std::size_t c = 0; c < mesh.n_cells(); ++c)
        for (std::size_t k = 0; k < 4; ++k) reordered(c * 4 + k) = original((mesh.n_cells() - 1 - c) * 4 + k);
    topology::UnstructuredMesh<MemSpace> permuted(mesh.node_coords_view(), mesh.conn_offsets_view(), reordered, mesh.coord_system());
    EXPECT_TRUE(detail::detect_regular_grid(mesh).is_regular);
    EXPECT_TRUE(detail::detect_rectilinear_grid(mesh).is_rectilinear);
    EXPECT_FALSE(detail::detect_regular_grid(permuted).is_regular);
    EXPECT_FALSE(detail::detect_rectilinear_grid(permuted).is_rectilinear);
}

TEST(RegularGridDetector, DuplicateCellsRejectCoordinateSetOnlyDetection) {
    auto mesh = make_regular_grid(4, 3, 0, 1, 0, 1);
    auto broken = Kokkos::View<index_t *, MemSpace>("duplicated_connectivity", mesh.conn_indices_view().extent(0));
    Kokkos::deep_copy(broken, mesh.conn_indices_view());
    for (int k = 0; k < 4; ++k) broken(4 + k) = broken(k);
    topology::UnstructuredMesh<MemSpace> duplicate(mesh.node_coords_view(), mesh.conn_offsets_view(), broken, mesh.coord_system());
    EXPECT_FALSE(detail::detect_regular_grid(duplicate).is_regular);
    EXPECT_FALSE(detail::detect_rectilinear_grid(duplicate).is_rectilinear);
}
}  // namespace axis::test
