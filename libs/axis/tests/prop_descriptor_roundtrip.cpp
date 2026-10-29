// ─── Property-Based Tests: Descriptor Round-Trip ─────────────────────────────
// Feature: helm-axis-microlibrary, Property 23: Descriptor Round-Trip
//          (Mesh → Descriptor → Mesh)
//
// Export mesh via mesh_egress, build UGRID GridDescriptor from those views,
// feed to MeshFactory::from_descriptor, verify identical node/cell/connectivity.
//
// **Validates: Requirements 21.1, 15.2, 15.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <vector>

namespace {

/// Build a simple ni x nj regular-grid UnstructuredMesh on HostSpace via
/// StructuredGrid::to_unstructured(). This gives us a mesh with known geometry
/// and well-formed CSR connectivity suitable for the round-trip test.
axis::topology::UnstructuredMesh<Kokkos::HostSpace> build_regular_mesh(std::size_t ni, std::size_t nj, double lon_start, double lat_start,
                                                                       double dlon, double dlat) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("clon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("clat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("crlon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("crlat", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_lon(idx) = lon_start + (static_cast<double>(i) + 0.5) * dlon;
            center_lat(idx) = lat_start + (static_cast<double>(j) + 0.5) * dlat;
        }
    }

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_lon(idx) = lon_start + static_cast<double>(i) * dlon;
            corner_lat(idx) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    return grid.to_unstructured();
}

// ─── Property 23: Descriptor Round-Trip (Mesh → Descriptor → Mesh) ──────────
// Generate an UnstructuredMesh (via StructuredGrid), export via mesh_egress,
// construct a UGRID GridDescriptor from those views, feed to from_descriptor,
// and verify the reconstructed mesh is identical to the original.
//
// **Validates: Requirements 21.1, 15.2, 15.4**

RC_GTEST_PROP(PropDescriptorRoundtrip, MeshEgressToFromDescriptorIsIdentity, ()) {
    // Generate random grid dimensions (small to keep tests fast)
    const auto ni = *rc::gen::inRange<std::size_t>(2, 8);
    const auto nj = *rc::gen::inRange<std::size_t>(2, 8);

    // Random starting coordinate and cell size
    const double lon_start = *rc::gen::map(rc::gen::inRange(0, 180), [](int v) { return static_cast<double>(v); });
    const double lat_start = *rc::gen::map(rc::gen::inRange(-80, 60), [](int v) { return static_cast<double>(v); });
    const double dlon = *rc::gen::map(rc::gen::inRange(1, 10), [](int v) { return static_cast<double>(v) * 0.5; });
    const double dlat = *rc::gen::map(rc::gen::inRange(1, 10), [](int v) { return static_cast<double>(v) * 0.5; });

    // 1. Build the original mesh
    auto original = build_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat);

    // 2. Export the mesh's views (simulating mesh_egress by using the actual
    //    UnstructuredMesh accessors directly — these are the same non-owning
    //    field_view members that MeshEgress would hold).
    auto orig_node_coords = original.node_coords();
    auto orig_conn_offsets = original.conn_offsets();
    auto orig_conn_indices = original.conn_indices();
    auto orig_cell_areas = original.cell_areas();
    auto orig_cell_mask = original.cell_mask();

    // 3. Build a UGRID GridDescriptor from the mesh's views
    axis::ingest::GridDescriptor descriptor;
    descriptor.kind = axis::ingest::ConventionKind::UGRID;
    descriptor.coord_system = axis::ingest::CoordinateSystem::SphericalDeg;
    descriptor.ugrid.topology_dimension = 2;
    descriptor.ugrid.start_index = 0;

    // Assign the non-owning views into the descriptor buffers
    descriptor.buffers.node_coords = orig_node_coords;
    descriptor.buffers.conn_offsets = orig_conn_offsets;
    descriptor.buffers.conn_indices = orig_conn_indices;
    descriptor.buffers.cell_areas = orig_cell_areas;
    descriptor.buffers.cell_mask = orig_cell_mask;

    // 4. Reconstruct the mesh via from_descriptor
    auto reconstructed = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(descriptor);

    // 5. Verify identical n_nodes and n_cells
    RC_ASSERT(reconstructed.n_nodes() == original.n_nodes());
    RC_ASSERT(reconstructed.n_cells() == original.n_cells());

    // 6. Verify identical node coordinates (bitwise)
    auto recon_coords = reconstructed.node_coords();
    RC_ASSERT(orig_node_coords.extent(0) == recon_coords.extent(0));
    RC_ASSERT(orig_node_coords.extent(1) == recon_coords.extent(1));

    for (std::size_t i = 0; i < orig_node_coords.extent(0); ++i) {
        for (std::size_t d = 0; d < orig_node_coords.extent(1); ++d) {
            RC_ASSERT(orig_node_coords(i, d) == recon_coords(i, d));
        }
    }

    // 7. Verify identical CSR connectivity offsets
    auto recon_offsets = reconstructed.conn_offsets();
    RC_ASSERT(orig_conn_offsets.extent(0) == recon_offsets.extent(0));

    for (std::size_t i = 0; i < orig_conn_offsets.extent(0); ++i) {
        RC_ASSERT(orig_conn_offsets(i) == recon_offsets(i));
    }

    // 8. Verify identical CSR connectivity indices
    auto recon_indices = reconstructed.conn_indices();
    RC_ASSERT(orig_conn_indices.extent(0) == recon_indices.extent(0));

    for (std::size_t i = 0; i < orig_conn_indices.extent(0); ++i) {
        RC_ASSERT(orig_conn_indices(i) == recon_indices(i));
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
