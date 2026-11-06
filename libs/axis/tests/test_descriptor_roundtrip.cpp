// AXIS unit test: mesh → UGRID descriptor → mesh round-trip
// Builds a mesh, exports via mesh_egress, constructs a UGRID GridDescriptor
// from those views, feeds it to from_descriptor, and verifies identity.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
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
static auto *const kenv = ::testing::AddGlobalTestEnvironment(new KokkosEnv);
}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Build a 2x2 mesh for a simple roundtrip
static topology::UnstructuredMesh<MemSpace> make_small_mesh() {
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

// Test: mesh → UGRID descriptor → from_descriptor → mesh has same geometry
TEST(DescriptorRoundtrip, MeshToUgridAndBack) {
    auto original = make_small_mesh();

    // Build a UGRID GridDescriptor directly from mesh accessors
    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::UGRID;
    desc.coord_system = ingest::CoordinateSystem::Cartesian3D;
    desc.ugrid.topology_dimension = 2;
    desc.ugrid.start_index = 0;
    desc.buffers.node_coords = original.node_coords();
    desc.buffers.conn_offsets = original.conn_offsets();
    desc.buffers.conn_indices = original.conn_indices();

    // Feed back through from_descriptor
    auto rebuilt = topology::MeshFactory::from_descriptor<MemSpace>(desc);

    // Verify identity: same node count, cell count
    EXPECT_EQ(rebuilt.n_nodes(), original.n_nodes());
    EXPECT_EQ(rebuilt.n_cells(), original.n_cells());

    // Verify node coordinates match
    auto orig_coords = original.node_coords();
    auto new_coords = rebuilt.node_coords();
    ASSERT_EQ(orig_coords.extent(0), new_coords.extent(0));
    ASSERT_EQ(orig_coords.extent(1), new_coords.extent(1));
    for (std::size_t i = 0; i < orig_coords.extent(0); ++i) {
        for (std::size_t d = 0; d < orig_coords.extent(1); ++d) {
            EXPECT_NEAR(orig_coords(i, d), new_coords(i, d), 1e-14) << "Node " << i << " dim " << d;
        }
    }

    // Verify connectivity matches
    auto orig_offsets = original.conn_offsets();
    auto new_offsets = rebuilt.conn_offsets();
    ASSERT_EQ(orig_offsets.extent(0), new_offsets.extent(0));
    for (std::size_t i = 0; i < orig_offsets.extent(0); ++i) {
        EXPECT_EQ(orig_offsets[i], new_offsets[i]);
    }

    auto orig_indices = original.conn_indices();
    auto new_indices = rebuilt.conn_indices();
    ASSERT_EQ(orig_indices.extent(0), new_indices.extent(0));
    for (std::size_t i = 0; i < orig_indices.extent(0); ++i) {
        EXPECT_EQ(orig_indices[i], new_indices[i]);
    }
}

// Test: coordinate system is preserved through the roundtrip
TEST(DescriptorRoundtrip, PreservesCoordinateSystem) {
    auto original = make_small_mesh();

    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::UGRID;
    desc.coord_system = ingest::CoordinateSystem::Cartesian3D;
    desc.ugrid.topology_dimension = 2;
    desc.ugrid.start_index = 0;
    desc.buffers.node_coords = original.node_coords();
    desc.buffers.conn_offsets = original.conn_offsets();
    desc.buffers.conn_indices = original.conn_indices();

    auto rebuilt = topology::MeshFactory::from_descriptor<MemSpace>(desc);
    EXPECT_EQ(rebuilt.coord_system(), topology::CoordinateSystem::Cartesian3D);
}

TEST(DescriptorRoundtrip, ReducedGaussianRaggedRowsRouteThroughMeshFactory) {
    std::vector<double> latitude_bounds{-90.0, 0.0, 90.0};
    std::vector<index_t> row_offsets{0, 2, 5};
    std::vector<double> longitude_bounds{0.0, 180.0, 360.0, 0.0, 120.0, 240.0, 360.0};
    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::ReducedGaussian;
    desc.coord_system = ingest::CoordinateSystem::SphericalDeg;
    desc.reduced_gaussian.n_rows = 2;
    desc.longitude_periodic = true;
    desc.longitude_period = 360.0;
    desc.buffers.latitude_bounds = field_view<const double, 1>(latitude_bounds.data(), latitude_bounds.size());
    desc.buffers.row_cell_offsets = field_view<const index_t, 1>(row_offsets.data(), row_offsets.size());
    desc.buffers.row_longitude_boundaries = field_view<const double, 1>(longitude_bounds.data(), longitude_bounds.size());

    const auto mesh = topology::MeshFactory::from_descriptor<MemSpace>(desc);
    EXPECT_EQ(mesh.n_cells(), 5u);
    EXPECT_GT(mesh.n_nodes(), 0u);
    EXPECT_EQ(mesh.geometry_metadata().boundary_model, topology::BoundaryModel::ConstantLatitude);
    double area = 0.0;
    for (std::size_t cell = 0; cell < mesh.n_cells(); ++cell) area += mesh.cell_areas()(cell);
    EXPECT_NEAR(area, 4.0 * 3.14159265358979323846, 1e-12);
}

}  // namespace axis::test
