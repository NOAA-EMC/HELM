// AXIS unit test: AMIO-style vs Python-style descriptor produces identical output
// Verifies that two independently-constructed GridDescriptors with the same
// data yield meshes with identical geometry (producer-agnostic contract).

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace {

}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Test: two CF descriptors with identical center_x, center_y, ni, nj data
// but constructed independently ("AMIO-style" and "Python-style") produce
// meshes with identical node counts, cell counts, and coordinates.
TEST(ProducerEquivalence, TwoCfDescriptorsSameMesh) {
    // Shared raw data: a 2x2 CF structured grid
    const std::size_t ni = 2, nj = 2;
    std::vector<double> lons = {0.5, 1.5, 0.5, 1.5};  // [ni*nj]
    std::vector<double> lats = {0.5, 0.5, 1.5, 1.5};  // [ni*nj]

    // "AMIO-style" descriptor (e.g., decoded from NetCDF by AMIO)
    ingest::GridDescriptor amio_desc;
    amio_desc.kind = ingest::ConventionKind::CF;
    amio_desc.coord_system = ingest::CoordinateSystem::Cartesian3D;
    amio_desc.cf.grid_mapping_name = "latitude_longitude";
    amio_desc.buffers.center_x = field_view<const double, 1>(lons.data(), ni * nj);
    amio_desc.buffers.center_y = field_view<const double, 1>(lats.data(), ni * nj);
    amio_desc.buffers.ni = ni;
    amio_desc.buffers.nj = nj;

    // "Python-style" descriptor (e.g., built from numpy arrays via pybind11)
    ingest::GridDescriptor python_desc;
    python_desc.kind = ingest::ConventionKind::CF;
    python_desc.coord_system = ingest::CoordinateSystem::Cartesian3D;
    python_desc.cf.grid_mapping_name = "latitude_longitude";
    python_desc.buffers.center_x = field_view<const double, 1>(lons.data(), ni * nj);
    python_desc.buffers.center_y = field_view<const double, 1>(lats.data(), ni * nj);
    python_desc.buffers.ni = ni;
    python_desc.buffers.nj = nj;

    // Both go through the same from_descriptor funnel
    auto mesh_amio = topology::MeshFactory::from_descriptor<MemSpace>(amio_desc);
    auto mesh_python = topology::MeshFactory::from_descriptor<MemSpace>(python_desc);

    // Verify identical structure
    EXPECT_EQ(mesh_amio.n_nodes(), mesh_python.n_nodes());
    EXPECT_EQ(mesh_amio.n_cells(), mesh_python.n_cells());

    // Verify identical node coordinates
    auto coords_a = mesh_amio.node_coords();
    auto coords_p = mesh_python.node_coords();
    ASSERT_EQ(coords_a.extent(0), coords_p.extent(0));
    for (std::size_t i = 0; i < coords_a.extent(0); ++i) {
        for (std::size_t d = 0; d < coords_a.extent(1); ++d) {
            EXPECT_EQ(coords_a(i, d), coords_p(i, d)) << "Node " << i << " dim " << d << " differs";
        }
    }

    // Verify identical connectivity
    auto off_a = mesh_amio.conn_offsets();
    auto off_p = mesh_python.conn_offsets();
    ASSERT_EQ(off_a.extent(0), off_p.extent(0));
    for (std::size_t i = 0; i < off_a.extent(0); ++i) {
        EXPECT_EQ(off_a[i], off_p[i]);
    }
}

// Test: UGRID descriptors from two "producers" produce the same mesh
TEST(ProducerEquivalence, TwoUgridDescriptorsSameMesh) {
    // A simple 1-cell triangle mesh: 3 nodes, 1 triangular cell
    std::vector<double> coords_raw = {0.0, 1.0, 0.5, 0.0, 0.0, 1.0};
    // node_coords shape: [3, 2]
    std::vector<index_t> offsets = {0, 3};  // 1 cell with 3 nodes
    std::vector<index_t> indices = {0, 1, 2};

    field_view<const double, 2> nc(coords_raw.data(), 3, 2);
    field_view<const index_t, 1> co(offsets.data(), 2);
    field_view<const index_t, 1> ci(indices.data(), 3);

    ingest::GridDescriptor desc_a;
    desc_a.kind = ingest::ConventionKind::UGRID;
    desc_a.coord_system = ingest::CoordinateSystem::Cartesian3D;
    desc_a.ugrid.topology_dimension = 2;
    desc_a.ugrid.start_index = 0;
    desc_a.buffers.node_coords = nc;
    desc_a.buffers.conn_offsets = co;
    desc_a.buffers.conn_indices = ci;

    ingest::GridDescriptor desc_b;
    desc_b.kind = ingest::ConventionKind::UGRID;
    desc_b.coord_system = ingest::CoordinateSystem::Cartesian3D;
    desc_b.ugrid.topology_dimension = 2;
    desc_b.ugrid.start_index = 0;
    desc_b.buffers.node_coords = nc;
    desc_b.buffers.conn_offsets = co;
    desc_b.buffers.conn_indices = ci;

    auto mesh_a = topology::MeshFactory::from_descriptor<MemSpace>(desc_a);
    auto mesh_b = topology::MeshFactory::from_descriptor<MemSpace>(desc_b);

    EXPECT_EQ(mesh_a.n_nodes(), mesh_b.n_nodes());
    EXPECT_EQ(mesh_a.n_cells(), mesh_b.n_cells());
    EXPECT_EQ(mesh_a.n_nodes(), std::size_t(3));
    EXPECT_EQ(mesh_a.n_cells(), std::size_t(1));
}

}  // namespace axis::test
