// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/reduced_gaussian_grid.hpp>
#include <cmath>
#include <stdexcept>
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

using namespace axis::topology;

ReducedGaussianGrid asymmetric_grid() {
    std::vector<ReducedGaussianRow> rows{{{10.0, 190.0, 370.0}}, {{-110.0, 10.0, 130.0, 250.0}}};
    return ReducedGaussianGrid({-60.0, 0.0, 60.0}, std::move(rows), CoordinateSystem::SphericalDeg, LongitudePeriodicity{true, 360.0, true});
}

TEST(ReducedGaussianGridTest, PreservesRaggedRowOffsetsAndCellCount) {
    auto grid = asymmetric_grid();
    ASSERT_EQ(grid.row_offsets().size(), 3u);
    EXPECT_EQ(grid.row_offsets()[0], 0);
    EXPECT_EQ(grid.row_offsets()[1], 2);
    EXPECT_EQ(grid.row_offsets()[2], 5);
    auto mesh = grid.to_unstructured();
    EXPECT_EQ(mesh.n_cells(), 5u);
    EXPECT_EQ(mesh.conn_offsets().extent(0), 6u);
    EXPECT_EQ(mesh.conn_offsets()(0), 0);
    EXPECT_EQ(mesh.conn_offsets()(5), mesh.conn_indices().extent(0));
}

TEST(ReducedGaussianGridTest, ConformingModeSplitsHangingLongitudeEdges) {
    auto grid = asymmetric_grid();
    auto mesh = grid.to_unstructured(true);
    auto offsets = mesh.conn_offsets();
    EXPECT_EQ(offsets(1) - offsets(0), 5);
    EXPECT_EQ(offsets(2) - offsets(1), 5);
    EXPECT_EQ(offsets(3) - offsets(2), 4);
    EXPECT_EQ(offsets(4) - offsets(3), 4);
    EXPECT_EQ(offsets(5) - offsets(4), 5);

    // The outer rows retain their own subdivisions; the shared latitude
    // boundary contains the deterministic four-point union.
    EXPECT_EQ(mesh.n_nodes(), 9u);
}

TEST(ReducedGaussianGridTest, NonconformingModeKeepsSourceCellQuads) {
    auto mesh = asymmetric_grid().to_unstructured(false);
    for (std::size_t c = 0; c < mesh.n_cells(); ++c) {
        EXPECT_EQ(mesh.conn_offsets()(c + 1) - mesh.conn_offsets()(c), 4);
    }
}

TEST(ReducedGaussianGridTest, ComputesConstantLatitudeAreaFromDeclaredBounds) {
    auto mesh = asymmetric_grid().to_unstructured();
    const auto areas = mesh.cell_areas();
    double total = 0.0;
    for (std::size_t c = 0; c < areas.extent(0); ++c) {
        EXPECT_GT(areas(c), 0.0);
        total += areas(c);
    }
    EXPECT_NEAR(total, 4.0 * std::acos(-1.0) * std::sin(std::acos(-1.0) / 3.0), 1e-12);
}

TEST(ReducedGaussianGridTest, RejectsInvalidRaggedRowsAndLatitudeBounds) {
    EXPECT_THROW(ReducedGaussianGrid({-90.0, 90.0}, {{{0.0, 180.0, 360.0}}, {{0.0, 360.0}}}, CoordinateSystem::SphericalDeg,
                                     LongitudePeriodicity{true, 360.0, true}),
                 std::invalid_argument);
    EXPECT_THROW(ReducedGaussianGrid({-90.0, -90.0}, {{{0.0, 360.0}}}, CoordinateSystem::SphericalDeg, LongitudePeriodicity{true, 360.0, true}),
                 std::invalid_argument);
}

}  // namespace
