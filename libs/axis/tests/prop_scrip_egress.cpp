// ─── Property-Based Tests: SCRIP Egress 1-Based Indexing ─────────────────────
// Feature: axis-v2-improvements, Property 16
//
// Property 16: SCRIP egress 1-based indexing
//   For any InterpolationMatrix, ScripEgress row/col values SHALL equal
//   original indices + 1, with minimum value 1.
//
// **Validates: Requirements 10.2**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/scrip_egress.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Helper: Build a simple UnstructuredMesh with n_cells quad cells ─────────
// Creates a planar grid of quads in SphericalDeg coordinates for testing.
// Each cell is a unit square at (i, 0)-(i+1, 0)-(i+1, 1)-(i, 1).

axis::topology::UnstructuredMesh<MemSpace> buildSimpleMesh(std::size_t n_cells) {
    // Each cell has 4 unique nodes (no sharing for simplicity)
    const std::size_t n_nodes = n_cells * 4;
    const std::size_t ndim = 2;

    Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace> node_coords("node_coords", n_nodes, ndim);

    Kokkos::View<axis::index_t *, MemSpace> conn_offsets("conn_offsets", n_cells + 1);

    Kokkos::View<axis::index_t *, MemSpace> conn_indices("conn_indices", n_cells * 4);

    for (std::size_t c = 0; c < n_cells; ++c) {
        const std::size_t base_node = c * 4;
        const double lon_start = static_cast<double>(c) * 10.0;

        // 4 nodes of a quad: (lon, lat) pairs
        node_coords(base_node + 0, 0) = lon_start;
        node_coords(base_node + 0, 1) = 0.0;
        node_coords(base_node + 1, 0) = lon_start + 10.0;
        node_coords(base_node + 1, 1) = 0.0;
        node_coords(base_node + 2, 0) = lon_start + 10.0;
        node_coords(base_node + 2, 1) = 10.0;
        node_coords(base_node + 3, 0) = lon_start;
        node_coords(base_node + 3, 1) = 10.0;

        // CSR connectivity: 4 nodes per cell
        conn_offsets(c) = static_cast<axis::index_t>(c * 4);
        conn_indices(c * 4 + 0) = static_cast<axis::index_t>(base_node + 0);
        conn_indices(c * 4 + 1) = static_cast<axis::index_t>(base_node + 1);
        conn_indices(c * 4 + 2) = static_cast<axis::index_t>(base_node + 2);
        conn_indices(c * 4 + 3) = static_cast<axis::index_t>(base_node + 3);
    }
    conn_offsets(n_cells) = static_cast<axis::index_t>(n_cells * 4);

    return axis::topology::UnstructuredMesh<MemSpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices),
                                                      axis::topology::CoordinateSystem::SphericalDeg);
}

// ─── Helper: Build a random InterpolationMatrix with valid indices ────────────

struct MatrixData {
    axis::solver::InterpolationMatrix<MemSpace> matrix;
    std::size_t n_src;
    std::size_t n_dst;
    std::size_t nnz;
};

MatrixData generateRandomMatrix() {
    const auto n_src = *rc::gen::inRange<std::size_t>(2, 20);
    const auto n_dst = *rc::gen::inRange<std::size_t>(2, 20);
    const auto max_nnz = std::min(n_src * n_dst, static_cast<std::size_t>(50));
    const auto nnz = *rc::gen::inRange<std::size_t>(1, max_nnz + 1);

    Kokkos::View<double *, MemSpace> fl("fl", nnz);
    Kokkos::View<axis::index_t *, MemSpace> fr("fr", nnz);
    Kokkos::View<axis::index_t *, MemSpace> fc("fc", nnz);
    Kokkos::View<double *, MemSpace> fa("fa", n_src);
    Kokkos::View<double *, MemSpace> fb("fb", n_dst);
    Kokkos::View<double *, MemSpace> aa("aa", n_src);
    Kokkos::View<double *, MemSpace> ab("ab", n_dst);

    for (std::size_t k = 0; k < nnz; ++k) {
        fl(k) = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 1000.0; });
        fr(k) = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_dst));
        fc(k) = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_src));
    }

    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    axis::solver::InterpolationMatrix<MemSpace> matrix(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb), std::move(aa),
                                                       std::move(ab), n_src, n_dst);

    return MatrixData{std::move(matrix), n_src, n_dst, nnz};
}

// ─── Property 16: SCRIP egress 1-based indexing ──────────────────────────────
//
// For any InterpolationMatrix, scrip_egress() SHALL produce row/col arrays
// where each value equals the original 0-based index + 1, and all values >= 1.
//
// **Validates: Requirements 10.2**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropScripEgress, ColIs1BasedOffset, ()) {
    auto data = generateRandomMatrix();

    // Build simple meshes with n_src and n_dst cells
    auto src_mesh = buildSimpleMesh(data.n_src);
    auto dst_mesh = buildSimpleMesh(data.n_dst);

    axis::solver::RegridConfig config{};
    config.method = axis::solver::InterpolationMethod::Bilinear;

    auto result = axis::ingest::scrip_egress(data.matrix, src_mesh, dst_mesh, config);
    const auto &egress = result.get();

    // Verify n_s matches
    RC_ASSERT(egress.n_s == data.nnz);

    // Get original 0-based column indices from the matrix
    const auto orig_col = data.matrix.factor_col();

    // For each entry k: egress.col(k) == matrix.factor_col()(k) + 1
    for (std::size_t k = 0; k < data.nnz; ++k) {
        const auto egress_col_val = egress.col[k];
        const auto orig_col_val = orig_col[k];
        RC_ASSERT(egress_col_val == orig_col_val + 1);
    }
}

RC_GTEST_PROP(PropScripEgress, RowIs1BasedOffset, ()) {
    auto data = generateRandomMatrix();

    auto src_mesh = buildSimpleMesh(data.n_src);
    auto dst_mesh = buildSimpleMesh(data.n_dst);

    axis::solver::RegridConfig config{};
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;

    auto result = axis::ingest::scrip_egress(data.matrix, src_mesh, dst_mesh, config);
    const auto &egress = result.get();

    RC_ASSERT(egress.n_s == data.nnz);

    // Get original 0-based row indices from the matrix
    const auto orig_row = data.matrix.factor_row();

    // For each entry k: egress.row(k) == matrix.factor_row()(k) + 1
    for (std::size_t k = 0; k < data.nnz; ++k) {
        const auto egress_row_val = egress.row[k];
        const auto orig_row_val = orig_row[k];
        RC_ASSERT(egress_row_val == orig_row_val + 1);
    }
}

RC_GTEST_PROP(PropScripEgress, AllIndicesAtLeastOne, ()) {
    auto data = generateRandomMatrix();

    auto src_mesh = buildSimpleMesh(data.n_src);
    auto dst_mesh = buildSimpleMesh(data.n_dst);

    axis::solver::RegridConfig config{};
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;

    auto result = axis::ingest::scrip_egress(data.matrix, src_mesh, dst_mesh, config);
    const auto &egress = result.get();

    RC_ASSERT(egress.n_s == data.nnz);

    // Verify all indices >= 1 (no zero indices in SCRIP convention)
    for (std::size_t k = 0; k < data.nnz; ++k) {
        RC_ASSERT(egress.col[k] >= 1);
        RC_ASSERT(egress.row[k] >= 1);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
