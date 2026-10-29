// ─── Property-Based Tests: Gmsh Export Round-Trip ────────────────────────────
// Feature: helm-axis-microlibrary, Property 7: Gmsh Export Round-Trip
//
// Uses RapidCheck to verify that for any UnstructuredMesh (produced from a
// random StructuredGrid), writing it with GmshWriter::write and reading the
// resulting .msh file back reconstructs a mesh with:
//   - Identical node count
//   - Identical cell count
//   - Node coordinates equal within round-off (%.17g preserves ~16 sig. digits)
//   - Identical connectivity (accounting for 0-based vs 1-based index shift)
//
// The custom .msh parser handles only the subset GmshWriter emits:
//   Gmsh v2.2 ASCII, triangles (type 2) and quadrilaterals (type 3).
//
// All tests execute on Kokkos::HostSpace (host-only property test).
//
// **Validates: Requirements 16.1, 16.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/topology/gmsh_writer.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ─── Minimal Gmsh .msh v2.2 ASCII Parser (test-only) ────────────────────────
// Parses just the $Nodes and $Elements sections written by GmshWriter.
// Only handles element types 2 (triangle) and 3 (quad).

struct ParsedNode {
    std::size_t id;  // 1-based in file, stored as-is
    double x, y, z;
};

struct ParsedElement {
    std::size_t id;                  // 1-based element ID
    int type;                        // 2=tri, 3=quad
    std::vector<std::size_t> nodes;  // 1-based node IDs as stored in file
};

struct ParsedMsh {
    std::vector<ParsedNode> nodes;
    std::vector<ParsedElement> elements;
};

/// Parse a .msh v2.2 ASCII file written by GmshWriter.
/// Throws std::runtime_error on parse failure.
ParsedMsh parse_msh_file(const std::string &filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open .msh file: " + filepath);
    }

    ParsedMsh result;
    std::string line;

    while (std::getline(ifs, line)) {
        // ── $Nodes section ──────────────────────────────────────────────
        if (line == "$Nodes") {
            std::size_t n_nodes = 0;
            ifs >> n_nodes;
            result.nodes.reserve(n_nodes);

            for (std::size_t i = 0; i < n_nodes; ++i) {
                ParsedNode nd{};
                ifs >> nd.id >> nd.x >> nd.y >> nd.z;
                result.nodes.push_back(nd);
            }
            // Consume up to $EndNodes
            std::getline(ifs, line);  // consume newline after last node
            std::getline(ifs, line);  // "$EndNodes"
            continue;
        }

        // ── $Elements section ───────────────────────────────────────────
        if (line == "$Elements") {
            std::size_t n_elements = 0;
            ifs >> n_elements;
            result.elements.reserve(n_elements);

            for (std::size_t i = 0; i < n_elements; ++i) {
                ParsedElement elem{};
                int n_tags = 0;
                ifs >> elem.id >> elem.type >> n_tags;

                // Skip tags
                for (int t = 0; t < n_tags; ++t) {
                    int tag_val = 0;
                    ifs >> tag_val;
                }

                // Read node IDs based on element type
                int n_elem_nodes = 0;
                if (elem.type == 2) {
                    n_elem_nodes = 3;  // triangle
                } else if (elem.type == 3) {
                    n_elem_nodes = 4;  // quad
                } else {
                    // Skip unknown element types (shouldn't happen with our writer)
                    std::getline(ifs, line);
                    continue;
                }

                elem.nodes.resize(static_cast<std::size_t>(n_elem_nodes));
                for (int n = 0; n < n_elem_nodes; ++n) {
                    ifs >> elem.nodes[static_cast<std::size_t>(n)];
                }
                result.elements.push_back(std::move(elem));
            }
            // Consume up to $EndElements
            std::getline(ifs, line);  // consume newline
            std::getline(ifs, line);  // "$EndElements"
            continue;
        }
    }

    return result;
}

// ─── Temp file utility ───────────────────────────────────────────────────────

/// Generate a unique temporary .msh file path and return it.
/// Caller is responsible for cleanup.
std::string make_temp_msh_path() {
    // Use random suffix to avoid collisions in parallel test runs
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(100000, 999999);
    std::string path = "/tmp/axis_prop_gmsh_" + std::to_string(dist(rng)) + ".msh";
    return path;
}

/// RAII guard for temp file cleanup.
struct TempFileGuard {
    std::string path;
    ~TempFileGuard() {
        if (!path.empty()) {
            std::remove(path.c_str());
        }
    }
};

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate grid dimension in [2, 8]. Small for fast tests but exercises
/// non-trivial connectivity.
rc::Gen<std::size_t> genDim() {
    return rc::gen::inRange<std::size_t>(2, 9);
}

/// Generate a coordinate value in a reasonable range [-180, 180] with
/// fractional precision. Uses integer generation mapped to double for
/// reproducibility.
rc::Gen<double> genCoord() {
    return rc::gen::map(rc::gen::inRange(-18000000, 18000001), [](int v) { return static_cast<double>(v) / 100000.0; });
}

/// Generate a vector of random coordinate doubles with given size.
rc::Gen<std::vector<double>> genCoordVector(std::size_t n) {
    return rc::gen::container<std::vector<double>>(n, genCoord());
}

// ─── Helper: Build UnstructuredMesh from random StructuredGrid ───────────────

/// Create an UnstructuredMesh<HostSpace> from a random StructuredGrid with
/// given dimensions and corner coordinates.
axis::topology::UnstructuredMesh<Kokkos::HostSpace> build_random_mesh(std::size_t ni, std::size_t nj, const std::vector<double> &corner_lon_vec,
                                                                      const std::vector<double> &corner_lat_vec) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    // Center coords (not used after conversion, just needed for construction)
    Kokkos::View<double *, Kokkos::HostSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("center_lat", n_centers);

    // Fill centers with averages (doesn't matter for the test, we just need a
    // valid StructuredGrid)
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            const std::size_t idx = i + j * ni;
            center_lon(idx) = 0.0;
            center_lat(idx) = 0.0;
        }
    }

    // Corner coords
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("corner_lon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("corner_lat", n_corners);

    for (std::size_t k = 0; k < n_corners; ++k) {
        corner_lon(k) = corner_lon_vec[k];
        corner_lat(k) = corner_lat_vec[k];
    }

    // Build StructuredGrid and convert to UnstructuredMesh
    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::Cartesian3D);
    grid.set_corners(corner_lon, corner_lat);

    return grid.to_unstructured();
}

// ─── Property 7a: Node Count Preserved After Round-Trip ──────────────────────
// Write mesh via GmshWriter, read back, verify same number of nodes.
//
// **Validates: Requirements 16.1, 16.3**

RC_GTEST_PROP(PropGmshRoundtrip, NodeCountPreserved, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    auto mesh = build_random_mesh(ni, nj, corner_lon_vec, corner_lat_vec);

    // Write to temp file
    std::string tmp_path = make_temp_msh_path();
    TempFileGuard guard{tmp_path};

    axis::topology::GmshWriter::write(tmp_path, mesh);

    // Parse back
    auto parsed = parse_msh_file(tmp_path);

    // Verify node count matches
    RC_ASSERT(parsed.nodes.size() == mesh.n_nodes());
}

// ─── Property 7b: Cell Count Preserved After Round-Trip ──────────────────────
// Write mesh via GmshWriter, read back, verify same number of elements (cells).
//
// **Validates: Requirements 16.1, 16.3**

RC_GTEST_PROP(PropGmshRoundtrip, CellCountPreserved, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    auto mesh = build_random_mesh(ni, nj, corner_lon_vec, corner_lat_vec);

    // Write to temp file
    std::string tmp_path = make_temp_msh_path();
    TempFileGuard guard{tmp_path};

    axis::topology::GmshWriter::write(tmp_path, mesh);

    // Parse back
    auto parsed = parse_msh_file(tmp_path);

    // All cells from StructuredGrid conversion are quads (4 nodes each),
    // so all are writable in MSH v2.2 (type 3). Cell count must match.
    RC_ASSERT(parsed.elements.size() == mesh.n_cells());
}

// ─── Property 7c: Coordinates Preserved Within Round-Off ─────────────────────
// After round-trip, each node's x/y/z coordinates must match the original
// within the precision of %.17g formatting (~1e-15 relative tolerance).
//
// **Validates: Requirements 16.1, 16.3**

RC_GTEST_PROP(PropGmshRoundtrip, CoordinatesWithinRoundoff, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    auto mesh = build_random_mesh(ni, nj, corner_lon_vec, corner_lat_vec);

    // Write to temp file
    std::string tmp_path = make_temp_msh_path();
    TempFileGuard guard{tmp_path};

    axis::topology::GmshWriter::write(tmp_path, mesh);

    // Parse back
    auto parsed = parse_msh_file(tmp_path);

    // Access original node coordinates
    auto coords = mesh.node_coords();  // [n_nodes, ndim]
    const std::size_t n_nodes = mesh.n_nodes();
    const auto *coords_ptr = coords.data_handle();
    const std::size_t ndim = coords.extent(1);

    // Nodes in the parsed file are 1-based; in AXIS they are 0-based.
    // The writer outputs them sequentially: node id = i+1 for axis index i.
    for (std::size_t i = 0; i < n_nodes; ++i) {
        const auto &pn = parsed.nodes[i];

        // Verify node ID is sequential 1-based
        RC_ASSERT(pn.id == i + 1);

        // Original coordinates (layout_left: [n_nodes, ndim])
        const double orig_x = coords_ptr[i + n_nodes * 0];
        const double orig_y = (ndim > 1) ? coords_ptr[i + n_nodes * 1] : 0.0;
        const double orig_z = (ndim > 2) ? coords_ptr[i + n_nodes * 2] : 0.0;

        // %.17g provides ~16 significant digits. For values in [-180, 180]
        // the round-trip error should be < 1e-13. Use a generous tolerance
        // of 1e-12 to account for parsing variability.
        const double tol = 1e-12;

        RC_ASSERT(std::abs(pn.x - orig_x) <= tol * (1.0 + std::abs(orig_x)));
        RC_ASSERT(std::abs(pn.y - orig_y) <= tol * (1.0 + std::abs(orig_y)));
        RC_ASSERT(std::abs(pn.z - orig_z) <= tol * (1.0 + std::abs(orig_z)));
    }
}

// ─── Property 7d: Connectivity Preserved After Round-Trip ────────────────────
// After round-trip, each element's node connectivity (adjusted from 1-based
// back to 0-based) must match the original CSR connectivity exactly.
//
// **Validates: Requirements 16.1, 16.3**

RC_GTEST_PROP(PropGmshRoundtrip, ConnectivityPreserved, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    auto corner_lon_vec = *genCoordVector(n_corners);
    auto corner_lat_vec = *genCoordVector(n_corners);

    auto mesh = build_random_mesh(ni, nj, corner_lon_vec, corner_lat_vec);

    // Write to temp file
    std::string tmp_path = make_temp_msh_path();
    TempFileGuard guard{tmp_path};

    axis::topology::GmshWriter::write(tmp_path, mesh);

    // Parse back
    auto parsed = parse_msh_file(tmp_path);

    // Access original CSR connectivity
    auto offsets = mesh.conn_offsets();
    auto indices = mesh.conn_indices();
    const std::size_t n_cells = mesh.n_cells();

    RC_ASSERT(parsed.elements.size() == n_cells);

    // For each cell, compare parsed connectivity (1-based) to original (0-based)
    for (std::size_t c = 0; c < n_cells; ++c) {
        const auto &elem = parsed.elements[c];

        const auto start = static_cast<std::size_t>(offsets[c]);
        const auto end = static_cast<std::size_t>(offsets[c + 1]);
        const std::size_t n_cell_nodes = end - start;

        // Element node count must match
        RC_ASSERT(elem.nodes.size() == n_cell_nodes);

        // All cells from StructuredGrid are quads
        RC_ASSERT(elem.type == 3);  // Gmsh type 3 = quad

        // Verify connectivity: parsed nodes are 1-based, original are 0-based
        for (std::size_t n = 0; n < n_cell_nodes; ++n) {
            const auto parsed_node_0based = elem.nodes[n] - 1;  // convert to 0-based
            const auto orig_node = static_cast<std::size_t>(indices[start + n]);
            RC_ASSERT(parsed_node_0based == orig_node);
        }
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// Property tests need Kokkos initialized for StructuredGrid conversion kernel
// and GmshWriter (which may do Kokkos::deep_copy for device meshes).



}  // namespace
