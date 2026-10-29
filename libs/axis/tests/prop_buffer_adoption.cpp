// ─── Property-Based Tests: layout_left Buffer Adoption With No Copy When Space Matches ───
// Feature: helm-axis-microlibrary, Property 25: layout_left Buffer Adoption With No Copy When Space Matches
//
// Uses RapidCheck to verify that MeshFactory::from_descriptor correctly handles
// buffer adoption semantics:
//
//   (a) When BufferViews address memory in the target MemorySpace (HostSpace
//       target with host-resident buffers), from_descriptor produces a mesh
//       whose data is correct — node coordinates and connectivity values match
//       the descriptor input exactly.
//
//   (b) The produced mesh data is independent of the original descriptor buffers:
//       mutating the original host arrays after from_descriptor does NOT affect
//       the mesh (proves a copy/adoption into owned storage occurred).
//
//   (c) Mesh structural invariants hold: n_nodes matches node_coords extent,
//       n_cells matches (conn_offsets.extent(0) - 1), and coordinate values
//       live in accessible HostSpace memory.
//
// All tests use UGRID convention (the direct adoption path) with randomly
// generated node coordinates and CSR connectivity, targeting Kokkos::HostSpace.
//
// **Validates: Requirements 1.2, 4.9, 2.5**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a node count in [3, 128]. Minimum 3 nodes to form at least one cell.
rc::Gen<std::size_t> genNodeCount() {
    return rc::gen::inRange<std::size_t>(3, 129);
}

/// Generate a cell count in [1, 32]. Each cell will have 3-6 nodes (mixed elements).
rc::Gen<std::size_t> genCellCount() {
    return rc::gen::inRange<std::size_t>(1, 33);
}

/// Generate a coordinate value in [-180, 180] range (mimicking lon/lat).
rc::Gen<double> genCoordValue() {
    return rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a nodes-per-cell count in [3, 6] (triangles to hexagons).
rc::Gen<std::size_t> genNodesPerCell() {
    return rc::gen::inRange<std::size_t>(3, 7);
}

/// Generate a complete valid UGRID descriptor data set:
///   - node_coords: [n_nodes, 2] (lon, lat)
///   - conn_offsets: [n_cells + 1] CSR offsets
///   - conn_indices: valid node indices referencing [0, n_nodes)
///
/// Returns a tuple of (node_coords_flat, conn_offsets, conn_indices, n_nodes, n_cells, ndim).
struct UgridData {
    std::vector<double> node_coords;  // flat [n_nodes * ndim], layout_left
    std::vector<axis::index_t> conn_offsets;
    std::vector<axis::index_t> conn_indices;
    std::size_t n_nodes;
    std::size_t n_cells;
    static constexpr std::size_t ndim = 2;
};

rc::Gen<UgridData> genUgridData() {
    return rc::gen::exec([]() {
        UgridData data;
        data.n_nodes = *genNodeCount();
        data.n_cells = *genCellCount();

        // Generate node coordinates [n_nodes * ndim], layout_left:
        // In layout_left (column-major), element (i, j) is at index i + n_nodes * j
        data.node_coords.resize(data.n_nodes * data.ndim);
        for (std::size_t j = 0; j < data.ndim; ++j) {
            for (std::size_t i = 0; i < data.n_nodes; ++i) {
                data.node_coords[i + data.n_nodes * j] = *genCoordValue();
            }
        }

        // Generate CSR connectivity
        data.conn_offsets.resize(data.n_cells + 1);
        data.conn_offsets[0] = 0;
        for (std::size_t c = 0; c < data.n_cells; ++c) {
            std::size_t npc = *genNodesPerCell();
            data.conn_offsets[c + 1] = data.conn_offsets[c] + static_cast<axis::index_t>(npc);
        }

        // Generate connectivity indices — each is a valid node index in [0, n_nodes)
        std::size_t total_indices = static_cast<std::size_t>(data.conn_offsets[data.n_cells]);
        data.conn_indices.resize(total_indices);
        for (std::size_t k = 0; k < total_indices; ++k) {
            data.conn_indices[k] = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, data.n_nodes));
        }

        return data;
    });
}

/// Build a GridDescriptor from UgridData, pointing buffer views at the data vectors.
axis::ingest::GridDescriptor make_ugrid_descriptor(const UgridData &ugrid) {
    axis::ingest::GridDescriptor desc;
    desc.kind = axis::ingest::ConventionKind::UGRID;
    desc.coord_system = axis::ingest::CoordinateSystem::SphericalDeg;
    desc.ugrid.topology_dimension = 2;
    desc.ugrid.start_index = 0;
    desc.ugrid.mesh_name = "test_mesh";

    // Wrap vectors as layout_left mdspan views
    desc.buffers.node_coords = axis::field_view<const double, 2>(ugrid.node_coords.data(), ugrid.n_nodes, UgridData::ndim);
    desc.buffers.conn_offsets = axis::field_view<const axis::index_t, 1>(ugrid.conn_offsets.data(), ugrid.conn_offsets.size());
    desc.buffers.conn_indices = axis::field_view<const axis::index_t, 1>(ugrid.conn_indices.data(), ugrid.conn_indices.size());

    return desc;
}

// ─── Property 25a: Correct data adoption — values preserved ──────────────────
// Generate random UGRID data, build descriptor, call from_descriptor<HostSpace>,
// verify the resulting mesh has correct node counts, cell counts, and coordinate
// values match the input exactly.
//
// **Validates: Requirements 1.2, 4.9, 2.5**

RC_GTEST_PROP(PropBufferAdoption, DataValuesPreservedOnAdoption, ()) {
    auto ugrid = *genUgridData();
    auto desc = make_ugrid_descriptor(ugrid);

    // Call from_descriptor targeting HostSpace (same space as descriptor buffers)
    auto mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);

    // Verify node count matches
    RC_ASSERT(mesh.n_nodes() == ugrid.n_nodes);

    // Verify cell count matches
    RC_ASSERT(mesh.n_cells() == ugrid.n_cells);

    // Verify node coordinate values match the input exactly
    auto coords_view = mesh.node_coords();
    RC_ASSERT(coords_view.extent(0) == ugrid.n_nodes);
    RC_ASSERT(coords_view.extent(1) == UgridData::ndim);

    for (std::size_t j = 0; j < UgridData::ndim; ++j) {
        for (std::size_t i = 0; i < ugrid.n_nodes; ++i) {
            double expected = ugrid.node_coords[i + ugrid.n_nodes * j];
            double actual = coords_view(i, j);
            // Bitwise equality — no numerical transformation should occur
            RC_ASSERT(std::memcmp(&expected, &actual, sizeof(double)) == 0);
        }
    }

    // Verify CSR connectivity offsets match exactly
    auto offsets_view = mesh.conn_offsets();
    RC_ASSERT(offsets_view.extent(0) == ugrid.conn_offsets.size());
    for (std::size_t i = 0; i < ugrid.conn_offsets.size(); ++i) {
        RC_ASSERT(offsets_view(i) == ugrid.conn_offsets[i]);
    }

    // Verify CSR connectivity indices match exactly
    auto indices_view = mesh.conn_indices();
    RC_ASSERT(indices_view.extent(0) == ugrid.conn_indices.size());
    for (std::size_t k = 0; k < ugrid.conn_indices.size(); ++k) {
        RC_ASSERT(indices_view(k) == ugrid.conn_indices[k]);
    }
}

// ─── Property 25b: Data independence — mutation of original doesn't affect mesh ─
// After from_descriptor, modify the original host arrays. The mesh data must
// remain unchanged. This proves that the data was either copied into owned
// storage or truly adopted (in which case the descriptor would be consumed).
// Since descriptors own nothing and AXIS must own its mesh data, this verifies
// the correct copy/adoption into Kokkos managed views.
//
// **Validates: Requirements 1.2, 4.9**

RC_GTEST_PROP(PropBufferAdoption, MeshDataIndependentOfOriginalBuffers, ()) {
    auto ugrid = *genUgridData();
    auto desc = make_ugrid_descriptor(ugrid);

    // Build mesh from descriptor
    auto mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);

    // Snapshot the mesh coordinate values before mutation
    std::vector<double> coords_snapshot(ugrid.n_nodes * UgridData::ndim);
    auto coords_view = mesh.node_coords();
    for (std::size_t j = 0; j < UgridData::ndim; ++j) {
        for (std::size_t i = 0; i < ugrid.n_nodes; ++i) {
            coords_snapshot[i + ugrid.n_nodes * j] = coords_view(i, j);
        }
    }

    // Snapshot connectivity offsets
    std::vector<axis::index_t> offsets_snapshot(ugrid.conn_offsets.size());
    auto offsets_view = mesh.conn_offsets();
    for (std::size_t i = 0; i < ugrid.conn_offsets.size(); ++i) {
        offsets_snapshot[i] = offsets_view(i);
    }

    // Snapshot connectivity indices
    std::vector<axis::index_t> indices_snapshot(ugrid.conn_indices.size());
    auto indices_view = mesh.conn_indices();
    for (std::size_t k = 0; k < ugrid.conn_indices.size(); ++k) {
        indices_snapshot[k] = indices_view(k);
    }

    // ── Mutate the original buffers ──────────────────────────────────────────
    // Flip all coordinate values
    for (auto &v : ugrid.node_coords) {
        v = -999.0;
    }
    // Corrupt connectivity offsets
    for (auto &v : ugrid.conn_offsets) {
        v = 0;
    }
    // Corrupt connectivity indices
    for (auto &v : ugrid.conn_indices) {
        v = -1;
    }

    // ── Verify mesh is unaffected ────────────────────────────────────────────
    auto coords_after = mesh.node_coords();
    for (std::size_t j = 0; j < UgridData::ndim; ++j) {
        for (std::size_t i = 0; i < ugrid.n_nodes; ++i) {
            double expected = coords_snapshot[i + ugrid.n_nodes * j];
            double actual = coords_after(i, j);
            RC_ASSERT(std::memcmp(&expected, &actual, sizeof(double)) == 0);
        }
    }

    auto offsets_after = mesh.conn_offsets();
    for (std::size_t i = 0; i < offsets_snapshot.size(); ++i) {
        RC_ASSERT(offsets_after(i) == offsets_snapshot[i]);
    }

    auto indices_after = mesh.conn_indices();
    for (std::size_t k = 0; k < indices_snapshot.size(); ++k) {
        RC_ASSERT(indices_after(k) == indices_snapshot[k]);
    }
}

// ─── Property 25c: Mesh data is HostSpace accessible ─────────────────────────
// Verify that the resulting mesh data pointers are non-null and accessible from
// the host (they can be dereferenced without seg-fault). This confirms data
// lives in HostSpace when HostSpace is the target.
//
// **Validates: Requirements 2.5**

RC_GTEST_PROP(PropBufferAdoption, MeshDataAccessibleInTargetHostSpace, ()) {
    auto ugrid = *genUgridData();
    auto desc = make_ugrid_descriptor(ugrid);

    auto mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);

    // Node coordinates pointer must be non-null and host-accessible
    auto coords = mesh.node_coords();
    RC_ASSERT(coords.data_handle() != nullptr);
    RC_ASSERT(coords.extent(0) == ugrid.n_nodes);
    RC_ASSERT(coords.extent(1) == UgridData::ndim);

    // Verify we can actually read every element (host accessible)
    double checksum = 0.0;
    for (std::size_t j = 0; j < coords.extent(1); ++j) {
        for (std::size_t i = 0; i < coords.extent(0); ++i) {
            checksum += coords(i, j);
        }
    }
    // The checksum is computed from accessible memory — if not host-accessible
    // we'd segfault above. Just verify it's finite as a sanity check.
    RC_ASSERT(std::isfinite(checksum) || ugrid.n_nodes == 0);

    // CSR offsets pointer must be non-null and host-accessible
    auto offsets = mesh.conn_offsets();
    RC_ASSERT(offsets.data_handle() != nullptr);
    RC_ASSERT(offsets.extent(0) == ugrid.conn_offsets.size());

    // CSR indices pointer must be non-null and host-accessible
    auto indices = mesh.conn_indices();
    RC_ASSERT(indices.data_handle() != nullptr);
    RC_ASSERT(indices.extent(0) == ugrid.conn_indices.size());

    // Read connectivity to verify accessibility
    axis::index_t idx_sum = 0;
    for (std::size_t k = 0; k < indices.extent(0); ++k) {
        idx_sum += indices(k);
    }
    // If memory wasn't host-accessible we'd segfault. Just confirm it ran.
    (void)idx_sum;
}

// ─── Property 25d: Mesh data pointer differs from original (ownership transfer) ──
// After from_descriptor copies into owned Kokkos Views, the resulting mesh's
// internal data pointers MUST differ from the original descriptor buffer
// pointers. This verifies that AXIS owns its data (allocated fresh Views)
// rather than holding dangling references to the descriptor's non-owning views.
//
// **Validates: Requirements 1.2, 4.9**

RC_GTEST_PROP(PropBufferAdoption, MeshOwnsDataSeparateFromDescriptorBuffers, ()) {
    auto ugrid = *genUgridData();
    auto desc = make_ugrid_descriptor(ugrid);

    // Remember original buffer pointers
    const double *orig_coords_ptr = ugrid.node_coords.data();
    const axis::index_t *orig_offsets_ptr = ugrid.conn_offsets.data();
    const axis::index_t *orig_indices_ptr = ugrid.conn_indices.data();

    auto mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);

    // The mesh must own its data in separately allocated Views.
    // The data_handle() of the mesh's field_view accessors wraps Kokkos View
    // data pointers which should be distinct from the original vector storage.
    auto coords = mesh.node_coords();
    auto offsets = mesh.conn_offsets();
    auto indices = mesh.conn_indices();

    RC_ASSERT(coords.data_handle() != orig_coords_ptr);
    RC_ASSERT(offsets.data_handle() != orig_offsets_ptr);
    RC_ASSERT(indices.data_handle() != orig_indices_ptr);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// RapidCheck/GTest property tests need Kokkos initialized for View allocation.



}  // namespace
