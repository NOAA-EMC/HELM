// ─── Property-Based Tests: CSR Connectivity Validity ─────────────────────────
// Feature: helm-axis-microlibrary, Property 3: CSR Connectivity Validity
//
// Uses RapidCheck to verify that for meshes produced by any source, the CSR
// connectivity arrays satisfy the fundamental invariants:
//   1. offsets[0] == 0
//   2. offsets is non-decreasing: offsets[c] <= offsets[c+1] for all c
//   3. offsets[n_cells] == conn_indices.extent(0) (total connectivity entries)
//   4. All values in conn_indices are in range [0, n_nodes)
//
// Meshes are generated via:
//   (a) StructuredGrid::to_unstructured() with random dimensions
//   (b) Directly constructed UnstructuredMesh with varying sizes
//
// All tests execute on Kokkos::HostSpace (host-space access for validation).
//
// **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <limits>
#include <vector>

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a grid dimension in [1, 32]. Small enough for fast property
/// iteration but large enough to exercise multi-cell connectivity.
rc::Gen<std::size_t> genGridDim() {
    return rc::gen::inRange<std::size_t>(1, 33);
}

/// Generate a coordinate value in a realistic range for lon/lat.
rc::Gen<double> genCoord() {
    return rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a vector of coordinate values of a given size.
rc::Gen<std::vector<double>> genCoordVector(std::size_t n) {
    return rc::gen::container<std::vector<double>>(n, genCoord());
}

// ─── Helper: Validate CSR invariants on a mesh ───────────────────────────────

/// Check all four CSR invariants on a given UnstructuredMesh. Returns true
/// if all invariants hold; uses RC_ASSERT internally so failures report
/// the specific violated invariant.
void validateCsrInvariants(const axis::topology::UnstructuredMesh<Kokkos::HostSpace> &mesh) {
    const std::size_t n_cells = mesh.n_cells();
    const std::size_t n_nodes = mesh.n_nodes();

    auto offsets = mesh.conn_offsets();
    auto indices = mesh.conn_indices();

    // The offsets array must have exactly n_cells + 1 entries.
    RC_ASSERT(offsets.extent(0) == n_cells + 1);

    // ── Invariant 1: offsets[0] == 0 ─────────────────────────────────────────
    // Validates: Requirement 19.2
    RC_ASSERT(offsets[0] == 0);

    // ── Invariant 2: offsets is non-decreasing ───────────────────────────────
    // Validates: Requirement 19.1
    for (std::size_t c = 0; c < n_cells; ++c) {
        RC_ASSERT(offsets[c] <= offsets[c + 1]);
    }

    // ── Invariant 3: offsets[n_cells] == indices.extent(0) ───────────────────
    // Validates: Requirement 19.3
    RC_ASSERT(static_cast<std::size_t>(offsets[n_cells]) == indices.extent(0));

    // ── Invariant 4: All indices in [0, n_nodes) ─────────────────────────────
    // Validates: Requirement 19.4
    const std::size_t nnz = indices.extent(0);
    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(indices[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(indices[k]) < n_nodes);
    }
}

// ─── Property 3a: CSR validity on meshes from StructuredGrid::to_unstructured()
//
// Generate a random StructuredGrid with dimensions ni × nj and random center
// coordinates, convert to UnstructuredMesh, and verify all CSR invariants hold.
//
// **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(CsrConnectivityProperty3, FromStructuredGrid, ()) {
    // Generate random grid dimensions
    const std::size_t ni = *genGridDim();
    const std::size_t nj = *genGridDim();
    const std::size_t n_centers = ni * nj;

    // Generate a valid rectilinear center layout. Arbitrary independent lon/
    // lat samples are not a structured grid and are correctly rejected by the
    // rectilinear corner policy.
    std::vector<double> lon_data(n_centers), lat_data(n_centers);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            const std::size_t cell = i + j * ni;
            lon_data[cell] = -80.0 + 10.0 * static_cast<double>(i);
            lat_data[cell] = -60.0 + 10.0 * static_cast<double>(j);
        }
    }

    // Create Kokkos views from generated data
    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", n_centers);

    for (std::size_t i = 0; i < n_centers; ++i) {
        center_lon(i) = lon_data[i];
        center_lat(i) = lat_data[i];
    }

    // Construct StructuredGrid and convert to UnstructuredMesh
    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);

    auto mesh = grid.to_unstructured();

    // Validate all CSR invariants
    validateCsrInvariants(mesh);
}

// ─── Property 3b: CSR validity on directly-constructed UnstructuredMesh
//
// Construct an UnstructuredMesh directly with randomized but well-formed CSR
// arrays (triangle or quad cells of varying count), and verify the CSR
// invariants hold on the resulting mesh.
//
// **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(CsrConnectivityProperty3, DirectlyConstructedMesh, ()) {
    // Generate random mesh parameters
    const std::size_t n_cells = *rc::gen::inRange<std::size_t>(1, 65);
    // Nodes per cell: 3 (triangle) or 4 (quad)
    const std::size_t verts_per_cell = *rc::gen::element(std::size_t{3}, std::size_t{4});
    const std::size_t nnz = n_cells * verts_per_cell;
    // Ensure enough nodes for all indices to be valid
    const std::size_t n_nodes = *rc::gen::inRange<std::size_t>(verts_per_cell, std::max(verts_per_cell, n_cells * verts_per_cell) + 1);

    // Build CSR offsets (uniform cells: each cell has verts_per_cell nodes)
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> conn_offsets("conn_offsets", n_cells + 1);
    for (std::size_t c = 0; c <= n_cells; ++c) {
        conn_offsets(c) = static_cast<axis::index_t>(c * verts_per_cell);
    }

    // Build CSR indices with random valid node indices in [0, n_nodes)
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> conn_indices("conn_indices", nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
        const auto idx = *rc::gen::inRange<std::size_t>(0, n_nodes);
        conn_indices(k) = static_cast<axis::index_t>(idx);
    }

    // Build node coordinates [n_nodes, 2]
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> node_coords("node_coords", n_nodes, std::size_t{2});
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords(i, 0) = static_cast<double>(i);        // lon
        node_coords(i, 1) = static_cast<double>(i) * 0.5;  // lat
    }

    // Construct UnstructuredMesh directly
    axis::topology::UnstructuredMesh<Kokkos::HostSpace> mesh(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices),
                                                             axis::topology::CoordinateSystem::SphericalDeg);

    // Validate all CSR invariants
    validateCsrInvariants(mesh);
}

// ─── Property 3c: CSR validity with mixed-element meshes (varying verts/cell)
//
// Construct meshes where different cells have different numbers of vertices
// (simulating mixed triangle/quad/polygon meshes), and verify CSR invariants.
//
// **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(CsrConnectivityProperty3, MixedElementMesh, ()) {
    // Generate a random number of cells
    const std::size_t n_cells = *rc::gen::inRange<std::size_t>(1, 33);

    // Generate random vertices-per-cell for each cell (3 to 6)
    std::vector<std::size_t> verts_per_cell(n_cells);
    std::size_t total_verts = 0;
    for (std::size_t c = 0; c < n_cells; ++c) {
        verts_per_cell[c] = *rc::gen::inRange<std::size_t>(3, 7);
        total_verts += verts_per_cell[c];
    }

    // Need enough nodes to satisfy all indices
    const std::size_t n_nodes =
        *rc::gen::inRange<std::size_t>(std::max(std::size_t{3}, *std::max_element(verts_per_cell.begin(), verts_per_cell.end())), total_verts + 10);

    // Build CSR offsets (non-uniform cells)
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> conn_offsets("conn_offsets", n_cells + 1);
    conn_offsets(0) = 0;
    for (std::size_t c = 0; c < n_cells; ++c) {
        conn_offsets(c + 1) = conn_offsets(c) + static_cast<axis::index_t>(verts_per_cell[c]);
    }

    // Build CSR indices with random valid node indices
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> conn_indices("conn_indices", total_verts);
    for (std::size_t k = 0; k < total_verts; ++k) {
        const auto idx = *rc::gen::inRange<std::size_t>(0, n_nodes);
        conn_indices(k) = static_cast<axis::index_t>(idx);
    }

    // Build node coordinates [n_nodes, 2]
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> node_coords("node_coords", n_nodes, std::size_t{2});
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords(i, 0) = static_cast<double>(i) * 1.5;
        node_coords(i, 1) = static_cast<double>(i) * 0.7;
    }

    // Construct UnstructuredMesh directly
    axis::topology::UnstructuredMesh<Kokkos::HostSpace> mesh(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices),
                                                             axis::topology::CoordinateSystem::SphericalDeg);

    // Validate all CSR invariants
    validateCsrInvariants(mesh);
}

RC_GTEST_PROP(CsrConnectivityProperty3, RejectsIndexLimitTerminalOffset, ()) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> coords("coords", 3, 2);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> offsets("offsets", 2);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> indices("indices", 3);
    offsets(0) = 0;
    offsets(1) = std::numeric_limits<axis::index_t>::max();
    indices(0) = 0;
    indices(1) = 1;
    indices(2) = 2;

    bool rejected = false;
    try {
        (void)axis::topology::make_unstructured(std::move(coords), std::move(offsets), std::move(indices),
                                                axis::topology::CoordinateSystem::SphericalDeg);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    RC_ASSERT(rejected);
}

// ─── Property 3d: CSR validity preserved for single-cell meshes (edge case)
//
// Generate meshes with exactly 1 cell to test the minimal CSR case where
// offsets has exactly 2 entries.
//
// **Validates: Requirements 19.1, 19.2, 19.3, 19.4**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(CsrConnectivityProperty3, SingleCellMesh, ()) {
    // 1x1 StructuredGrid produces exactly 1 quad cell
    auto lon_data = *genCoordVector(1);
    auto lat_data = *genCoordVector(1);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", 1);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", 1);
    center_lon(0) = lon_data[0];
    center_lat(0) = lat_data[0];

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(1, 1, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);

    auto mesh = grid.to_unstructured();

    // Should have exactly 1 cell, 4 nodes (corners of the single quad)
    RC_ASSERT(mesh.n_cells() == 1);
    RC_ASSERT(mesh.n_nodes() == 4);

    // Validate CSR invariants
    validateCsrInvariants(mesh);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// RapidCheck/GTest property tests need Kokkos initialized for View allocation
// and parallel kernel execution in to_unstructured().

class KokkosEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

// Register the Kokkos environment with GTest
static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

}  // namespace
