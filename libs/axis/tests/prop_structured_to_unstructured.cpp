// ─── Property-Based Tests: Structured-to-Unstructured Conversion ─────────────
// Feature: helm-axis-microlibrary, Property 2: Structured-to-Unstructured
//          Conversion Preserves Geometry
//
// Uses RapidCheck to verify that for any valid StructuredGrid with random
// dimensions and corner coordinates, calling to_unstructured() produces an
// UnstructuredMesh with:
//   1. Exactly ni * nj cells
//   2. Each cell has exactly 4 nodes (quadrilateral)
//   3. The 4 corner node coordinates of each cell match the expected values
//      from the original corner arrays
//
// All tests execute on Kokkos::HostSpace (host-only property test).
//
// **Validates: Requirements 18.1, 18.2**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate grid dimension in [2, 20]. Small enough for fast tests but large
/// enough to exercise non-trivial connectivity.
rc::Gen<std::size_t> genDim() {
    return rc::gen::inRange<std::size_t>(2, 21);
}

/// Generate a vector of random doubles in [-180, 180] with given size.
/// Range chosen to mimic geographic coordinate values.
rc::Gen<std::vector<double>> genCoordVector(std::size_t n) {
    return rc::gen::container<std::vector<double>>(
        n, rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return static_cast<double>(v) / 100.0; }));
}

// ─── Property 2a: Cell count is exactly ni * nj ─────────────────────────────
// Generate random grid dimensions and coordinates, convert to unstructured,
// verify the resulting mesh has exactly ni*nj cells.
//
// **Validates: Requirements 18.1**

RC_GTEST_PROP(PropStructuredToUnstructured, CellCountEqualsNiTimesNj, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    // Generate center coordinates [ni*nj]
    auto center_lon_vec = *genCoordVector(n_centers);
    auto center_lat_vec = *genCoordVector(n_centers);

    // Generate corner coordinates [(ni+1)*(nj+1)]
    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    // Build Kokkos views from generated data
    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("corner_lon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("corner_lat", n_corners);

    for (std::size_t k = 0; k < n_centers; ++k) {
        center_lon(k) = center_lon_vec[k];
        center_lat(k) = center_lat_vec[k];
    }
    for (std::size_t k = 0; k < n_corners; ++k) {
        corner_lon(k) = corner_lon_vec[k];
        corner_lat(k) = corner_lat_vec[k];
    }

    // Construct StructuredGrid and set corners
    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    // Convert to unstructured
    auto mesh = grid.to_unstructured();

    // Verify: exactly ni*nj cells
    RC_ASSERT(mesh.n_cells() == ni * nj);
}

// ─── Property 2b: Each cell has exactly 4 nodes (quadrilateral) ──────────────
// For the converted mesh, verify every cell has exactly 4 connectivity entries
// (all cells are quads).
//
// **Validates: Requirements 18.1**

RC_GTEST_PROP(PropStructuredToUnstructured, AllCellsAreQuads, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto center_lon_vec = *genCoordVector(n_centers);
    auto center_lat_vec = *genCoordVector(n_centers);
    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("corner_lon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("corner_lat", n_corners);

    for (std::size_t k = 0; k < n_centers; ++k) {
        center_lon(k) = center_lon_vec[k];
        center_lat(k) = center_lat_vec[k];
    }
    for (std::size_t k = 0; k < n_corners; ++k) {
        corner_lon(k) = corner_lon_vec[k];
        corner_lat(k) = corner_lat_vec[k];
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    auto mesh = grid.to_unstructured();

    // Access CSR offsets
    auto offsets = mesh.conn_offsets();
    const std::size_t n_cells = mesh.n_cells();

    // Every cell must have exactly 4 nodes
    for (std::size_t c = 0; c < n_cells; ++c) {
        auto nodes_in_cell = offsets[c + 1] - offsets[c];
        RC_ASSERT(nodes_in_cell == 4);
    }
}

// ─── Property 2c: Corner coordinates of each cell match original corners ─────
// For each cell (i,j), the 4 node coordinates in the unstructured mesh must
// equal the corresponding corner values from the original corner arrays.
//
// Corner node mapping (CCW winding as implemented in structured_grid.cpp):
//   Node 0 (bottom-left):  corner index = i     + j     * (ni+1)
//   Node 1 (bottom-right): corner index = (i+1) + j     * (ni+1)
//   Node 2 (top-right):    corner index = (i+1) + (j+1) * (ni+1)
//   Node 3 (top-left):     corner index = i     + (j+1) * (ni+1)
//
// **Validates: Requirements 18.2**

RC_GTEST_PROP(PropStructuredToUnstructured, CornerCoordinatesPreserved, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto center_lon_vec = *genCoordVector(n_centers);
    auto center_lat_vec = *genCoordVector(n_centers);
    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("corner_lon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("corner_lat", n_corners);

    for (std::size_t k = 0; k < n_centers; ++k) {
        center_lon(k) = center_lon_vec[k];
        center_lat(k) = center_lat_vec[k];
    }
    for (std::size_t k = 0; k < n_corners; ++k) {
        corner_lon(k) = corner_lon_vec[k];
        corner_lat(k) = corner_lat_vec[k];
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    auto mesh = grid.to_unstructured();

    // Access mesh arrays
    auto node_coords = mesh.node_coords();  // [n_nodes, 2]
    auto offsets = mesh.conn_offsets();     // [n_cells + 1]
    auto indices = mesh.conn_indices();     // [nnz]

    const std::size_t nip1 = ni + 1;

    // For each cell (i, j), verify the 4 corner node coordinates match
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            const std::size_t cell_idx = i + j * ni;

            // CSR slice for this cell
            auto start = static_cast<std::size_t>(offsets[cell_idx]);
            auto end = static_cast<std::size_t>(offsets[cell_idx + 1]);
            RC_ASSERT(end - start == 4);

            // Get the 4 node indices
            auto n0 = static_cast<std::size_t>(indices[start + 0]);  // bottom-left
            auto n1 = static_cast<std::size_t>(indices[start + 1]);  // bottom-right
            auto n2 = static_cast<std::size_t>(indices[start + 2]);  // top-right
            auto n3 = static_cast<std::size_t>(indices[start + 3]);  // top-left

            // Expected corner indices in the original corner arrays
            std::size_t bl_idx = i + j * nip1;              // bottom-left
            std::size_t br_idx = (i + 1) + j * nip1;        // bottom-right
            std::size_t tr_idx = (i + 1) + (j + 1) * nip1;  // top-right
            std::size_t tl_idx = i + (j + 1) * nip1;        // top-left

            // Access node coordinates via data_handle with layout_left indexing.
            // For [n_nodes, 2] layout_left: element (row, col) is at
            // offset row + n_nodes * col.
            const auto *coords_ptr = node_coords.data_handle();
            const std::size_t n_nodes_total = mesh.n_nodes();

            // Bottom-left node: lon and lat must match
            RC_ASSERT(coords_ptr[n0 + n_nodes_total * 0] == corner_lon_vec[bl_idx]);
            RC_ASSERT(coords_ptr[n0 + n_nodes_total * 1] == corner_lat_vec[bl_idx]);

            // Bottom-right node
            RC_ASSERT(coords_ptr[n1 + n_nodes_total * 0] == corner_lon_vec[br_idx]);
            RC_ASSERT(coords_ptr[n1 + n_nodes_total * 1] == corner_lat_vec[br_idx]);

            // Top-right node
            RC_ASSERT(coords_ptr[n2 + n_nodes_total * 0] == corner_lon_vec[tr_idx]);
            RC_ASSERT(coords_ptr[n2 + n_nodes_total * 1] == corner_lat_vec[tr_idx]);

            // Top-left node
            RC_ASSERT(coords_ptr[n3 + n_nodes_total * 0] == corner_lon_vec[tl_idx]);
            RC_ASSERT(coords_ptr[n3 + n_nodes_total * 1] == corner_lat_vec[tl_idx]);
        }
    }
}

TEST(PropStructuredToUnstructured, NonSquareAffineCentersExtrapolateEveryPerimeter) {
    constexpr std::size_t ni = 3;
    constexpr std::size_t nj = 2;
    Kokkos::View<double *, Kokkos::HostSpace> lon("lon", ni * nj);
    Kokkos::View<double *, Kokkos::HostSpace> lat("lat", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            lon(i + j * ni) = 4.0 + 2.0 * static_cast<double>(i);
            lat(i + j * ni) = -3.0 + 6.0 * static_cast<double>(j);
        }
    }
    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, std::move(lon), std::move(lat), axis::topology::CoordinateSystem::SphericalDeg);
    const auto mesh =
        grid.to_unstructured(axis::topology::CornerPolicy::RectilinearMidpoint, axis::topology::LongitudePeriodicity{false, 360.0, true});
    const auto coords = mesh.node_coords();
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            const std::size_t node = i + j * (ni + 1);
            const double expected_lon = 3.0 + 2.0 * static_cast<double>(i);
            const double expected_lat = -6.0 + 6.0 * static_cast<double>(j);
            EXPECT_NEAR(coords(node, 0), expected_lon, 1e-12);
            EXPECT_NEAR(coords(node, 1), expected_lat, 1e-12);
        }
    }
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();
    EXPECT_EQ(offsets(0), 0);
    EXPECT_EQ(indices(0), 0);
    EXPECT_EQ(indices(1), 1);
    EXPECT_EQ(indices(2), 5);
    EXPECT_EQ(indices(3), 4);
}

// ─── Property 2d: Node count is exactly (ni+1)*(nj+1) ───────────────────────
// The conversion produces one node per corner vertex in the structured grid.
//
// **Validates: Requirements 18.1**

RC_GTEST_PROP(PropStructuredToUnstructured, NodeCountEqualsCornerGrid, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto center_lon_vec = *genCoordVector(n_centers);
    auto center_lat_vec = *genCoordVector(n_centers);
    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("corner_lon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("corner_lat", n_corners);

    for (std::size_t k = 0; k < n_centers; ++k) {
        center_lon(k) = center_lon_vec[k];
        center_lat(k) = center_lat_vec[k];
    }
    for (std::size_t k = 0; k < n_corners; ++k) {
        corner_lon(k) = corner_lon_vec[k];
        corner_lat(k) = corner_lat_vec[k];
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    auto mesh = grid.to_unstructured();

    // Verify: exactly (ni+1)*(nj+1) nodes
    RC_ASSERT(mesh.n_nodes() == (ni + 1) * (nj + 1));
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// Property tests need Kokkos initialized for StructuredGrid and the conversion
// kernel.

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

static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

}  // namespace
