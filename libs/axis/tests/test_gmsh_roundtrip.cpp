// AXIS unit test: Gmsh export → reimport round-trip
// Writes a 2x2 mesh to a temp .msh file, reads it back by parsing the ASCII
// format manually, and verifies node count, cell count, and coordinates.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/gmsh_writer.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static std::string make_temp_msh_path() {
    namespace fs = std::filesystem;
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<unsigned long long> dist;
    const fs::path dir = fs::temp_directory_path();

    for (int attempt = 0; attempt < 128; ++attempt) {
        const fs::path candidate = dir / ("axis_gmsh_roundtrip_" + std::to_string(dist(rng)) + ".msh");
        if (!fs::exists(candidate)) {
            return candidate.string();
        }
    }

    throw std::runtime_error("Failed to generate unique temp .msh path");
}

// Build a 2x2 mesh covering [0,2] x [0,2]
static topology::UnstructuredMesh<MemSpace> make_2x2_mesh() {
    const std::size_t n = 2;
    const double dx = 1.0;
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }
    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = static_cast<double>(i) * dx;
            cry(i + j * nc) = static_cast<double>(j) * dx;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));
    return grid.to_unstructured();
}

// Minimal Gmsh v2.2 ASCII parser for reading back nodes and elements.
struct GmshReadResult {
    std::size_t n_nodes{0};
    std::size_t n_elements{0};
    std::vector<std::array<double, 3>> node_coords;
    std::vector<std::vector<int>> elements;  // node indices (1-based from file)
};

static GmshReadResult read_gmsh_file(const std::string &path) {
    GmshReadResult result;
    std::ifstream in(path);
    std::string line;

    while (std::getline(in, line)) {
        if (line == "$Nodes") {
            std::getline(in, line);
            result.n_nodes = std::stoull(line);
            result.node_coords.resize(result.n_nodes);
            for (std::size_t i = 0; i < result.n_nodes; ++i) {
                std::getline(in, line);
                std::istringstream ss(line);
                int id;
                double x, y, z;
                ss >> id >> x >> y >> z;
                result.node_coords[i] = {x, y, z};
            }
        } else if (line == "$Elements") {
            std::getline(in, line);
            result.n_elements = std::stoull(line);
            result.elements.resize(result.n_elements);
            for (std::size_t i = 0; i < result.n_elements; ++i) {
                std::getline(in, line);
                std::istringstream ss(line);
                int id, elem_type, n_tags;
                ss >> id >> elem_type >> n_tags;
                // skip tags
                for (int t = 0; t < n_tags; ++t) {
                    int tag;
                    ss >> tag;
                }
                // read remaining node indices
                int node_id;
                while (ss >> node_id) {
                    result.elements[i].push_back(node_id);
                }
            }
        }
    }
    return result;
}

// Test: Write 2x2 mesh to .msh, read back, verify node/element counts match.
TEST(GmshRoundtrip, WriteThenRead) {
    auto mesh = make_2x2_mesh();

    // Use a temp file path
    std::string tmp_path = make_temp_msh_path();

    // Write
    topology::GmshWriter::write(tmp_path, mesh);

    // Read back
    auto gmsh = read_gmsh_file(tmp_path);

    // Verify counts
    EXPECT_EQ(gmsh.n_nodes, mesh.n_nodes());
    EXPECT_EQ(gmsh.n_elements, mesh.n_cells());

    // Verify node coordinates match within round-off (Gmsh ASCII has limited precision)
    auto orig_coords = mesh.node_coords();
    for (std::size_t i = 0; i < gmsh.n_nodes; ++i) {
        EXPECT_NEAR(gmsh.node_coords[i][0], orig_coords(i, 0), 1e-6) << "Node " << i << " x mismatch";
        EXPECT_NEAR(gmsh.node_coords[i][1], orig_coords(i, 1), 1e-6) << "Node " << i << " y mismatch";
    }

    // Cleanup
    std::remove(tmp_path.c_str());
}

// Test: Verify that each element has the expected number of nodes (quads have 4)
TEST(GmshRoundtrip, QuadrilateralElements) {
    auto mesh = make_2x2_mesh();
    std::string tmp_path = make_temp_msh_path();

    topology::GmshWriter::write(tmp_path, mesh);
    auto gmsh = read_gmsh_file(tmp_path);

    // All cells in a structured-to-unstructured 2x2 grid are quads (4 nodes)
    for (std::size_t i = 0; i < gmsh.n_elements; ++i) {
        EXPECT_EQ(gmsh.elements[i].size(), std::size_t(4)) << "Element " << i << " is not a quadrilateral";
    }

    std::remove(tmp_path.c_str());
}

}  // namespace axis::test
