// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

class MeshBuilderTest : public ::testing::Test {
   protected:
    static void SetUpTestSuite() {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }

    static void TearDownTestSuite() {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};

using MemorySpace = Kokkos::HostSpace;
using Mesh = axis::topology::UnstructuredMesh<MemorySpace>;

TEST_F(MeshBuilderTest, AdoptsMixedArityConnectivityAndAttributes) {
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> coords("coords", 7, 2);
    Kokkos::View<axis::index_t *, MemorySpace> offsets("offsets", 4);
    Kokkos::View<axis::index_t *, MemorySpace> indices("indices", 12);
    Kokkos::View<double *, MemorySpace> areas("areas", 3);
    Kokkos::View<int *, MemorySpace> mask("mask", 3);

    const axis::index_t host_offsets[] = {0, 3, 7, 12};
    const axis::index_t host_indices[] = {0, 1, 2, 0, 1, 2, 3, 1, 4, 5, 6, 2};
    for (std::size_t i = 0; i < 4; ++i) offsets(i) = host_offsets[i];
    for (std::size_t i = 0; i < 12; ++i) indices(i) = host_indices[i];
    for (std::size_t i = 0; i < 3; ++i) {
        areas(i) = static_cast<double>(i + 1);
        mask(i) = i == 1 ? 0 : 1;
    }

    const axis::topology::GeometryMetadata geometry{axis::topology::GeometryProvenance::SourceAuthoritative,
                                                    axis::topology::BoundaryModel::GreatCircle, axis::topology::AreaModel::SourceSupplied};
    auto mesh = axis::topology::make_unstructured(std::move(coords), std::move(offsets), std::move(indices),
                                                  axis::topology::CoordinateSystem::SphericalDeg, std::move(areas), std::move(mask), geometry);

    EXPECT_EQ(mesh.n_nodes(), 7u);
    EXPECT_EQ(mesh.n_cells(), 3u);
    EXPECT_EQ(mesh.conn_offsets()(1), 3);
    EXPECT_EQ(mesh.conn_offsets()(2), 7);
    EXPECT_EQ(mesh.conn_offsets()(3), 12);
    EXPECT_EQ(mesh.conn_indices()(11), 2);
    EXPECT_DOUBLE_EQ(mesh.cell_areas()(2), 3.0);
    EXPECT_EQ(mesh.cell_mask()(1), 0);
    EXPECT_EQ(mesh.geometry_metadata().provenance, axis::topology::GeometryProvenance::SourceAuthoritative);
    EXPECT_EQ(mesh.geometry_metadata().boundary_model, axis::topology::BoundaryModel::GreatCircle);
    EXPECT_EQ(mesh.geometry_metadata().area_model, axis::topology::AreaModel::SourceSupplied);
}

TEST_F(MeshBuilderTest, RejectsMalformedOffsets) {
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> coords("coords", 3, 2);
    Kokkos::View<axis::index_t *, MemorySpace> offsets("offsets", 2);
    Kokkos::View<axis::index_t *, MemorySpace> indices("indices", 3);
    offsets(0) = 0;
    offsets(1) = 2;
    indices(0) = 0;
    indices(1) = 1;
    indices(2) = 2;

    EXPECT_THROW(
        axis::topology::make_unstructured(std::move(coords), std::move(offsets), std::move(indices), axis::topology::CoordinateSystem::SphericalDeg),
        std::invalid_argument);
}

TEST_F(MeshBuilderTest, RejectsTerminalOffsetAtIndexLimitWhenStorageIsSmaller) {
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> coords("coords", 3, 2);
    Kokkos::View<axis::index_t *, MemorySpace> offsets("offsets", 2);
    Kokkos::View<axis::index_t *, MemorySpace> indices("indices", 3);
    offsets(0) = 0;
    offsets(1) = std::numeric_limits<axis::index_t>::max();
    indices(0) = 0;
    indices(1) = 1;
    indices(2) = 2;

    EXPECT_THROW(
        axis::topology::make_unstructured(std::move(coords), std::move(offsets), std::move(indices), axis::topology::CoordinateSystem::SphericalDeg),
        std::invalid_argument);
}

TEST_F(MeshBuilderTest, RejectsOutOfRangeNodeIndex) {
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> coords("coords", 3, 2);
    Kokkos::View<axis::index_t *, MemorySpace> offsets("offsets", 2);
    Kokkos::View<axis::index_t *, MemorySpace> indices("indices", 3);
    offsets(0) = 0;
    offsets(1) = 3;
    indices(0) = 0;
    indices(1) = 1;
    indices(2) = 3;

    EXPECT_THROW(
        axis::topology::make_unstructured(std::move(coords), std::move(offsets), std::move(indices), axis::topology::CoordinateSystem::SphericalDeg),
        std::invalid_argument);
}

TEST_F(MeshBuilderTest, RejectsMismatchedOptionalAttributeExtent) {
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> coords("coords", 3, 2);
    Kokkos::View<axis::index_t *, MemorySpace> offsets("offsets", 2);
    Kokkos::View<axis::index_t *, MemorySpace> indices("indices", 3);
    Kokkos::View<int *, MemorySpace> mask("mask", 2);
    offsets(0) = 0;
    offsets(1) = 3;
    indices(0) = 0;
    indices(1) = 1;
    indices(2) = 2;

    EXPECT_THROW(axis::topology::make_unstructured(std::move(coords), std::move(offsets), std::move(indices),
                                                   axis::topology::CoordinateSystem::SphericalDeg, {}, std::move(mask)),
                 std::invalid_argument);
}

}  // namespace
