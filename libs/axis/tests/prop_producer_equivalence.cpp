// ─── Property-Based Tests: Producer-Equivalence ─────────────────────────────
// Feature: helm-axis-microlibrary, Property 9: Producer-Equivalence
//          (Descriptor Is Producer-Agnostic)
//
// Uses RapidCheck to verify that constructing paired "AMIO-style" and
// "Python-style" GridDescriptors with IDENTICAL fields (same ni, nj, same
// coordinate data stored in SEPARATE buffers) and feeding both through
// MeshFactory::from_descriptor produces meshes with:
//   - Identical n_nodes
//   - Identical n_cells
//   - Identical CSR connectivity (offsets and indices match element-by-element)
//   - Identical node coordinates
//
// This proves from_descriptor has NO producer-specific branch — identical
// descriptor fields produce identical meshes regardless of who populated them.
//
// All tests execute on Kokkos::HostSpace (host-only property test).
//
// **Validates: Requirements 20.1, 20.2, 3.8**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate grid dimension in [2, 15]. Small enough for fast tests but large
/// enough to exercise non-trivial connectivity.
rc::Gen<std::size_t> genDim() {
    return rc::gen::inRange<std::size_t>(2, 16);
}

/// Generate a vector of random coordinate doubles in [-180, 180] with given
/// size. Uses integer scaling to avoid floating-point generation overhead.
rc::Gen<std::vector<double>> genCoordVector(std::size_t n) {
    return rc::gen::container<std::vector<double>>(
        n, rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return static_cast<double>(v) / 100.0; }));
}

std::pair<std::vector<double>, std::vector<double>> make_rectilinear_centers(std::size_t ni, std::size_t nj) {
    std::vector<double> longitude(ni * nj), latitude(ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            const std::size_t cell = i + j * ni;
            longitude[cell] = -120.0 + 5.0 * static_cast<double>(i);
            latitude[cell] = -60.0 + 5.0 * static_cast<double>(j);
        }
    }
    return {std::move(longitude), std::move(latitude)};
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Build a CF GridDescriptor from raw vectors, wrapping them as mdspan
// buffer views. Each call wraps the given vectors — different calls with the
// same DATA but different VECTOR INSTANCES simulate different producers
// (AMIO vs Python) writing identical field values into separate memory.
// ─────────────────────────────────────────────────────────────────────────────

axis::ingest::GridDescriptor make_cf_descriptor(std::size_t ni, std::size_t nj, const std::vector<double> &center_x,
                                                const std::vector<double> &center_y) {
    axis::ingest::GridDescriptor desc{};
    desc.kind = axis::ingest::ConventionKind::CF;
    desc.coord_system = axis::ingest::CoordinateSystem::SphericalDeg;

    // BufferViews wrapping the given vectors as non-owning mdspan views.
    desc.buffers.ni = ni;
    desc.buffers.nj = nj;
    desc.buffers.center_x = axis::field_view<const double, 1>{center_x.data(), center_x.size()};
    desc.buffers.center_y = axis::field_view<const double, 1>{center_y.data(), center_y.size()};

    return desc;
}

// ─── Property 9a: Identical CF descriptors from separate buffers produce ─────
//     meshes with identical n_nodes and n_cells.
//
// **Validates: Requirements 20.1, 20.2, 3.8**

RC_GTEST_PROP(PropProducerEquivalence, IdenticalNodeAndCellCounts, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();

    // Generate coordinate data ONCE (the "truth").
    auto [coords_x, coords_y] = make_rectilinear_centers(ni, nj);

    // ── Descriptor A: "AMIO-style" (copies the data into its own buffer) ────
    std::vector<double> buffer_a_x(coords_x.begin(), coords_x.end());
    std::vector<double> buffer_a_y(coords_y.begin(), coords_y.end());
    auto desc_a = make_cf_descriptor(ni, nj, buffer_a_x, buffer_a_y);

    // ── Descriptor B: "Python-style" (copies the same data into a SEPARATE
    //    buffer — simulating numpy producing it independently) ────────────────
    std::vector<double> buffer_b_x(coords_x.begin(), coords_x.end());
    std::vector<double> buffer_b_y(coords_y.begin(), coords_y.end());
    auto desc_b = make_cf_descriptor(ni, nj, buffer_b_x, buffer_b_y);

    // Sanity: the two buffers are at DIFFERENT addresses.
    RC_ASSERT(buffer_a_x.data() != buffer_b_x.data());
    RC_ASSERT(buffer_a_y.data() != buffer_b_y.data());

    // ── Call from_descriptor on both ─────────────────────────────────────────
    auto mesh_a = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_a);
    auto mesh_b = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_b);

    // ── Verify identical counts ──────────────────────────────────────────────
    RC_ASSERT(mesh_a.n_nodes() == mesh_b.n_nodes());
    RC_ASSERT(mesh_a.n_cells() == mesh_b.n_cells());
}

// ─── Property 9b: Identical CF descriptors from separate buffers produce ─────
//     meshes with identical CSR connectivity (offsets and indices).
//
// **Validates: Requirements 20.1, 20.2, 3.8**

RC_GTEST_PROP(PropProducerEquivalence, IdenticalCSRConnectivity, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();

    auto [coords_x, coords_y] = make_rectilinear_centers(ni, nj);

    // Descriptor A: AMIO-style (separate buffer copy)
    std::vector<double> buffer_a_x(coords_x.begin(), coords_x.end());
    std::vector<double> buffer_a_y(coords_y.begin(), coords_y.end());
    auto desc_a = make_cf_descriptor(ni, nj, buffer_a_x, buffer_a_y);

    // Descriptor B: Python-style (separate buffer copy, same values)
    std::vector<double> buffer_b_x(coords_x.begin(), coords_x.end());
    std::vector<double> buffer_b_y(coords_y.begin(), coords_y.end());
    auto desc_b = make_cf_descriptor(ni, nj, buffer_b_x, buffer_b_y);

    auto mesh_a = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_a);
    auto mesh_b = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_b);

    // ── Verify identical CSR offsets ─────────────────────────────────────────
    auto offsets_a = mesh_a.conn_offsets();
    auto offsets_b = mesh_b.conn_offsets();
    RC_ASSERT(offsets_a.extent(0) == offsets_b.extent(0));

    for (std::size_t i = 0; i < offsets_a.extent(0); ++i) {
        RC_ASSERT(offsets_a[i] == offsets_b[i]);
    }

    // ── Verify identical CSR indices ─────────────────────────────────────────
    auto indices_a = mesh_a.conn_indices();
    auto indices_b = mesh_b.conn_indices();
    RC_ASSERT(indices_a.extent(0) == indices_b.extent(0));

    for (std::size_t i = 0; i < indices_a.extent(0); ++i) {
        RC_ASSERT(indices_a[i] == indices_b[i]);
    }
}

// ─── Property 9c: Identical CF descriptors from separate buffers produce ─────
//     meshes with identical node coordinates (bitwise).
//
// **Validates: Requirements 20.1, 20.2, 3.8**

RC_GTEST_PROP(PropProducerEquivalence, IdenticalNodeCoordinates, ()) {
    const std::size_t ni = *genDim();
    const std::size_t nj = *genDim();

    auto [coords_x, coords_y] = make_rectilinear_centers(ni, nj);

    // Descriptor A: AMIO-style
    std::vector<double> buffer_a_x(coords_x.begin(), coords_x.end());
    std::vector<double> buffer_a_y(coords_y.begin(), coords_y.end());
    auto desc_a = make_cf_descriptor(ni, nj, buffer_a_x, buffer_a_y);

    // Descriptor B: Python-style
    std::vector<double> buffer_b_x(coords_x.begin(), coords_x.end());
    std::vector<double> buffer_b_y(coords_y.begin(), coords_y.end());
    auto desc_b = make_cf_descriptor(ni, nj, buffer_b_x, buffer_b_y);

    auto mesh_a = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_a);
    auto mesh_b = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_b);

    // ── Verify identical node coordinates [n_nodes, ndim] ────────────────────
    auto coords_a = mesh_a.node_coords();
    auto coords_b = mesh_b.node_coords();

    RC_ASSERT(coords_a.extent(0) == coords_b.extent(0));
    RC_ASSERT(coords_a.extent(1) == coords_b.extent(1));

    const std::size_t n_nodes = coords_a.extent(0);
    const std::size_t ndim = coords_a.extent(1);

    // layout_left: element (row, col) at offset row + n_nodes * col
    const double *ptr_a = coords_a.data_handle();
    const double *ptr_b = coords_b.data_handle();

    for (std::size_t d = 0; d < ndim; ++d) {
        for (std::size_t n = 0; n < n_nodes; ++n) {
            RC_ASSERT(ptr_a[n + n_nodes * d] == ptr_b[n + n_nodes * d]);
        }
    }
}

// ─── Property 9d: Producer-equivalence holds for UGRID convention too ────────
// Construct paired UGRID descriptors from separate buffers, verify identical
// mesh output. This covers the unstructured path through from_descriptor.
//
// **Validates: Requirements 20.1, 20.2, 3.8**

RC_GTEST_PROP(PropProducerEquivalence, UgridProducerEquivalence, ()) {
    // Generate a small unstructured mesh: random triangle mesh.
    const std::size_t n_nodes = *rc::gen::inRange<std::size_t>(4, 20);
    const std::size_t n_cells = *rc::gen::inRange<std::size_t>(2, 10);
    const std::size_t ndim = 2;

    // Random node coordinates [n_nodes, 2]
    auto node_x = *genCoordVector(n_nodes);
    auto node_y = *genCoordVector(n_nodes);

    // Build a flat node_coords buffer in layout_left [n_nodes, 2]:
    // col 0 = x coords (indices 0..n_nodes-1), col 1 = y coords (n_nodes..2*n_nodes-1)
    std::vector<double> node_coords_data(n_nodes * ndim);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        node_coords_data[i] = node_x[i];            // col 0
        node_coords_data[i + n_nodes] = node_y[i];  // col 1
    }

    // Build valid CSR connectivity (each cell is a triangle: 3 node indices).
    std::vector<axis::index_t> offsets_data(n_cells + 1);
    offsets_data[0] = 0;
    for (std::size_t c = 0; c < n_cells; ++c) {
        offsets_data[c + 1] = offsets_data[c] + 3;
    }
    const std::size_t nnz = static_cast<std::size_t>(offsets_data.back());

    // Random node indices in [0, n_nodes)
    std::vector<axis::index_t> indices_data(nnz);
    for (std::size_t k = 0; k < nnz; ++k) {
        indices_data[k] = *rc::gen::inRange<axis::index_t>(0, static_cast<axis::index_t>(n_nodes));
    }

    // ── Build Descriptor A (AMIO-style): separate buffer copies ──────────────
    std::vector<double> a_coords(node_coords_data.begin(), node_coords_data.end());
    std::vector<axis::index_t> a_offsets(offsets_data.begin(), offsets_data.end());
    std::vector<axis::index_t> a_indices(indices_data.begin(), indices_data.end());

    axis::ingest::GridDescriptor desc_a{};
    desc_a.kind = axis::ingest::ConventionKind::UGRID;
    desc_a.coord_system = axis::ingest::CoordinateSystem::SphericalDeg;
    desc_a.buffers.node_coords = axis::field_view<const double, 2>{a_coords.data(), n_nodes, ndim};
    desc_a.buffers.conn_offsets = axis::field_view<const axis::index_t, 1>{a_offsets.data(), a_offsets.size()};
    desc_a.buffers.conn_indices = axis::field_view<const axis::index_t, 1>{a_indices.data(), a_indices.size()};

    // ── Build Descriptor B (Python-style): separate buffer copies of same data
    std::vector<double> b_coords(node_coords_data.begin(), node_coords_data.end());
    std::vector<axis::index_t> b_offsets(offsets_data.begin(), offsets_data.end());
    std::vector<axis::index_t> b_indices(indices_data.begin(), indices_data.end());

    axis::ingest::GridDescriptor desc_b{};
    desc_b.kind = axis::ingest::ConventionKind::UGRID;
    desc_b.coord_system = axis::ingest::CoordinateSystem::SphericalDeg;
    desc_b.buffers.node_coords = axis::field_view<const double, 2>{b_coords.data(), n_nodes, ndim};
    desc_b.buffers.conn_offsets = axis::field_view<const axis::index_t, 1>{b_offsets.data(), b_offsets.size()};
    desc_b.buffers.conn_indices = axis::field_view<const axis::index_t, 1>{b_indices.data(), b_indices.size()};

    // Sanity: buffers are at different addresses.
    RC_ASSERT(a_coords.data() != b_coords.data());

    // ── Call from_descriptor on both ─────────────────────────────────────────
    auto mesh_a = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_a);
    auto mesh_b = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc_b);

    // ── Verify identical counts ──────────────────────────────────────────────
    RC_ASSERT(mesh_a.n_nodes() == mesh_b.n_nodes());
    RC_ASSERT(mesh_a.n_cells() == mesh_b.n_cells());

    // ── Verify identical CSR connectivity ────────────────────────────────────
    auto off_a = mesh_a.conn_offsets();
    auto off_b = mesh_b.conn_offsets();
    RC_ASSERT(off_a.extent(0) == off_b.extent(0));
    for (std::size_t i = 0; i < off_a.extent(0); ++i) {
        RC_ASSERT(off_a[i] == off_b[i]);
    }

    auto idx_a = mesh_a.conn_indices();
    auto idx_b = mesh_b.conn_indices();
    RC_ASSERT(idx_a.extent(0) == idx_b.extent(0));
    for (std::size_t i = 0; i < idx_a.extent(0); ++i) {
        RC_ASSERT(idx_a[i] == idx_b[i]);
    }

    // ── Verify identical node coordinates ────────────────────────────────────
    auto nc_a = mesh_a.node_coords();
    auto nc_b = mesh_b.node_coords();
    RC_ASSERT(nc_a.extent(0) == nc_b.extent(0));
    RC_ASSERT(nc_a.extent(1) == nc_b.extent(1));

    const double *pa = nc_a.data_handle();
    const double *pb = nc_b.data_handle();
    const std::size_t total = nc_a.extent(0) * nc_a.extent(1);
    for (std::size_t i = 0; i < total; ++i) {
        RC_ASSERT(pa[i] == pb[i]);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// Property tests need Kokkos initialized for MeshFactory (uses StructuredGrid
// conversion kernel and Kokkos::deep_copy).

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
