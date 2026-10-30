// AXIS unit test: RectilinearGridInfo and detect_rectilinear_grid()
// Verifies that the detector correctly identifies non-uniform rectilinear grids.

#include <gtest/gtest.h>

#include <axis/detail/regular_grid_detector.hpp>
#include <axis/topology/structured_grid.hpp>

#include "common/test_kokkos_env.hpp"

namespace {}  // namespace

namespace axis::test {

TEST(RectilinearGridDetector, DetectsNonUniformRectilinearGrid) {
    // Generate simple non-uniform rectilinear coordinate spacing
    std::vector<double> lons = {0.0, 1.5, 4.0, 5.5, 9.0};  // non-uniform spacing
    std::vector<double> lats = {0.0, 2.0, 3.5, 6.0};

    const std::size_t ni = lons.size() - 1;
    const std::size_t nj = lats.size() - 1;

    Kokkos::View<double *, Kokkos::HostSpace> center_lons("center_lons", ni * nj);
    Kokkos::View<double *, Kokkos::HostSpace> center_lats("center_lats", ni * nj);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lons("corner_lons", (ni + 1) * (nj + 1));
    Kokkos::View<double *, Kokkos::HostSpace> corner_lats("corner_lats", (ni + 1) * (nj + 1));

    // Fill corner nodes
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            corner_lons(i + j * (ni + 1)) = lons[i];
            corner_lats(i + j * (ni + 1)) = lats[j];
        }
    }

    // Fill cell centers as average of corners
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            center_lons(i + j * ni) = 0.5 * (lons[i] + lons[i + 1]);
            center_lats(i + j * ni) = 0.5 * (lats[j] + lats[j + 1]);
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lons, center_lats, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lons, corner_lats);

    auto mesh = grid.to_unstructured();
    auto info = axis::detail::detect_rectilinear_grid(mesh);

    EXPECT_TRUE(info.is_rectilinear);
    EXPECT_EQ(info.ni, ni);
    EXPECT_EQ(info.nj, nj);
    ASSERT_EQ(info.unique_lons.extent(0), lons.size());
    ASSERT_EQ(info.unique_lats.extent(0), lats.size());
    EXPECT_DOUBLE_EQ(info.unique_lons(0), 0.0);
    EXPECT_DOUBLE_EQ(info.unique_lons(4), 9.0);
    EXPECT_DOUBLE_EQ(info.unique_lats(0), 0.0);
    EXPECT_DOUBLE_EQ(info.unique_lats(3), 6.0);
}

}  // namespace axis::test
