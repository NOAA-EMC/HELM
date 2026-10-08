// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/multi_face_grid.hpp>
#include <axis/types.hpp>
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

using Space = Kokkos::HostSpace;
using namespace axis::topology;

StructuredFace<Space> face(double x0, double y0, double x1, double y1) {
    StructuredFace<Space> result;
    result.ni = 1;
    result.nj = 1;
    result.node_coords = Kokkos::View<double **, Kokkos::LayoutLeft, Space>("face", 4, 2);
    result.node_coords(0, 0) = x0;
    result.node_coords(0, 1) = y0;
    result.node_coords(1, 0) = x1;
    result.node_coords(1, 1) = y0;
    result.node_coords(2, 0) = x0;
    result.node_coords(2, 1) = y1;
    result.node_coords(3, 0) = x1;
    result.node_coords(3, 1) = y1;
    return result;
}

TEST(MultiFaceGridTest, JoinsOrientedEdgesWithDeterministicGlobalIds) {
    std::vector<StructuredFace<Space>> faces{face(0, 0, 1, 1), face(1, 0, 2, 1)};
    MultiFaceGrid<Space> grid(faces, {{0, FaceEdge::East, 1, FaceEdge::West, false}}, CoordinateSystem::SphericalDeg);
    const auto a = grid.to_unstructured();
    const auto b = grid.to_unstructured();
    EXPECT_EQ(a.n_nodes(), 6u);
    EXPECT_EQ(a.n_cells(), 2u);
    for (std::size_t k = 0; k < a.conn_indices().extent(0); ++k) EXPECT_EQ(a.conn_indices()(k), b.conn_indices()(k));
    EXPECT_EQ(a.conn_indices()(1), a.conn_indices()(4));
    EXPECT_EQ(a.conn_indices()(2), a.conn_indices()(7));
}

TEST(MultiFaceGridTest, SupportsRotatedAndReversedFaceEdgeContacts) {
    auto a = face(0, 0, 1, 1);
    auto rotated = face(1, 0, 2, 1);
    // Rotate the second face's local axes: its south edge matches face A east.
    rotated.node_coords(0, 0) = 1;
    rotated.node_coords(0, 1) = 0;
    rotated.node_coords(1, 0) = 1;
    rotated.node_coords(1, 1) = 1;
    rotated.node_coords(2, 0) = 2;
    rotated.node_coords(2, 1) = 0;
    rotated.node_coords(3, 0) = 2;
    rotated.node_coords(3, 1) = 1;
    MultiFaceGrid<Space> rotated_grid({a, rotated}, {{0, FaceEdge::East, 1, FaceEdge::South, false}}, CoordinateSystem::SphericalDeg);
    EXPECT_EQ(rotated_grid.to_unstructured().n_nodes(), 6u);

    auto reversed = face(1, 0, 2, 1);
    reversed.node_coords(0, 1) = 1;
    reversed.node_coords(2, 1) = 0;
    reversed.reverse_cell_orientation = true;
    MultiFaceGrid<Space> reversed_grid({a, reversed}, {{0, FaceEdge::East, 1, FaceEdge::West, true}}, CoordinateSystem::SphericalDeg);
    EXPECT_EQ(reversed_grid.to_unstructured().n_nodes(), 6u);
}

TEST(MultiFaceGridTest, AllowsUnequalFaceDimensionsWhenContactVertexCountsMatch) {
    auto a = face(0, 0, 1, 1);
    StructuredFace<Space> b;
    b.ni = 2;
    b.nj = 1;
    b.node_coords = Kokkos::View<double **, Kokkos::LayoutLeft, Space>("wide_face", 6, 2);
    const double x[] = {1, 2, 3};
    for (std::size_t i = 0; i <= b.ni; ++i) {
        b.node_coords(i, 0) = x[i];
        b.node_coords(i, 1) = 0;
        b.node_coords(i + (b.ni + 1), 0) = x[i];
        b.node_coords(i + (b.ni + 1), 1) = 1;
    }
    MultiFaceGrid<Space> grid({a, b}, {{0, FaceEdge::East, 1, FaceEdge::West, false}}, CoordinateSystem::SphericalDeg);
    const auto mesh = grid.to_unstructured();
    EXPECT_EQ(mesh.n_cells(), 3u);
    EXPECT_EQ(mesh.n_nodes(), 8u);
}

TEST(MultiFaceGridTest, ExplicitEquivalenceSupportsMultiFaceJunctions) {
    std::vector<StructuredFace<Space>> faces{face(0, 0, 1, 1), face(1, 1, 2, 2), face(0, 1, 1, 2)};
    const std::vector<std::vector<FaceVertexRef>> equivalences{{{0, 1, 1}, {1, 0, 0}, {2, 1, 0}}};
    MultiFaceGrid<Space> grid(std::move(faces), {}, CoordinateSystem::SphericalDeg, equivalences);
    EXPECT_EQ(grid.to_unstructured().n_nodes(), 10u);
}

TEST(MultiFaceGridTest, RejectsInconsistentEdgeDimensionsAndCoordinates) {
    auto small = face(0, 0, 1, 1);
    auto tall = face(1, 0, 2, 2);
    tall.nj = 2;
    tall.node_coords = Kokkos::View<double **, Kokkos::LayoutLeft, Space>("tall", 6, 2);
    EXPECT_THROW((MultiFaceGrid<Space>({small, tall}, {{0, FaceEdge::East, 1, FaceEdge::West, false}}, CoordinateSystem::SphericalDeg)),
                 std::invalid_argument);

    auto inconsistent = face(5, 0, 6, 1);
    MultiFaceGrid<Space> mismatch({small, inconsistent}, {{0, FaceEdge::East, 1, FaceEdge::West, false}}, CoordinateSystem::SphericalDeg);
    EXPECT_THROW(mismatch.to_unstructured(), std::invalid_argument);
}
}  // namespace
