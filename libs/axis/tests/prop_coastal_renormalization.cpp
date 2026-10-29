// ─── Property-Based Tests: Coastal Renormalization & Extrapolation ───────────
// Feature: helm-axis-microlibrary, Property: Coastal Renormalization & Extrapolation
//
// Uses RapidCheck to verify that under NearestWet extrapolation, even with randomly
// masked source meshes, destination cells are resolved/extrapolated to active
// (wet) source cells, and partition of unity (weight row sums) is preserved.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cmath>

namespace {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

/// Build a structured geographic grid to unstructured mesh with optional mask.
axis::topology::UnstructuredMesh<MemSpace> build_grid_mesh(std::size_t ni, std::size_t nj, double x_start, double y_start, double dx, double dy,
                                                           Kokkos::View<int *, MemSpace> mask = {}) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, MemSpace> center_x("cx", n_centers);
    Kokkos::View<double *, MemSpace> center_y("cy", n_centers);
    Kokkos::View<double *, MemSpace> corner_x("crx", n_corners);
    Kokkos::View<double *, MemSpace> corner_y("cry", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_x(idx) = x_start + (static_cast<double>(i) + 0.5) * dx;
            center_y(idx) = y_start + (static_cast<double>(j) + 0.5) * dy;
        }
    }

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_x(idx) = x_start + static_cast<double>(i) * dx;
            corner_y(idx) = y_start + static_cast<double>(j) * dy;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, center_x, center_y, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_x, corner_y);

    auto mesh = grid.to_unstructured();

    if (mask.extent(0) > 0) {
        return axis::topology::UnstructuredMesh<MemSpace>(mesh.node_coords_view(), mesh.conn_offsets_view(), mesh.conn_indices_view(),
                                                          mesh.coord_system(), mesh.cell_areas_view(), std::move(mask));
    }

    return mesh;
}

RC_GTEST_PROP(PropCoastalRenormalization, RenormalizeToWetCells, ()) {
    // Generate small random dimensions to keep tests fast
    const std::size_t src_ni = *rc::gen::inRange<std::size_t>(2, 6);
    const std::size_t src_nj = *rc::gen::inRange<std::size_t>(2, 6);
    const std::size_t n_src_cells = src_ni * src_nj;

    const std::size_t dst_ni = *rc::gen::inRange<std::size_t>(2, 6);
    const std::size_t dst_nj = *rc::gen::inRange<std::size_t>(2, 6);
    const std::size_t n_dst_cells = dst_ni * dst_nj;

    // Generate random mask: each cell 0 (dry) or 1 (wet)
    auto mask_vec = *rc::gen::container<std::vector<int>>(n_src_cells, rc::gen::inRange(0, 2));

    // Force at least one wet cell so extrapolation/re-normalization is physically defined
    const auto wet_idx = *rc::gen::inRange<std::size_t>(0, n_src_cells);
    mask_vec[wet_idx] = 1;

    // Build mask View
    Kokkos::View<int *, MemSpace> src_mask("src_mask", n_src_cells);
    for (std::size_t i = 0; i < n_src_cells; ++i) {
        src_mask(i) = mask_vec[i];
    }

    const double domain_size = 4.0;
    const double src_dx = domain_size / static_cast<double>(src_ni);
    const double src_dy = domain_size / static_cast<double>(src_nj);
    const double dst_dx = domain_size / static_cast<double>(dst_ni);
    const double dst_dy = domain_size / static_cast<double>(dst_nj);

    // Build meshes on identical domain
    auto src_mesh = build_grid_mesh(src_ni, src_nj, 0.0, 0.0, src_dx, src_dy, src_mask);
    auto dst_mesh = build_grid_mesh(dst_ni, dst_nj, 0.0, 0.0, dst_dx, dst_dy);

    // Generate nearest-neighbor weights with NearestWet extrapolation enabled
    RegridConfig config;
    config.method = InterpolationMethod::NearestNeighbor;
    config.unmapped = UnmappedAction::Ignore;
    config.extrap_method = ExtrapolationAction::NearestWet;

    auto W = WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // Verify properties of the generated matrix:
    // 1. Every row of the matrix corresponds to a destination cell.
    // 2. Since NearestWet is enabled and there is at least one active/wet source cell,
    //    every destination cell MUST map to at least one active source cell. Thus, row sum of weights
    //    for each destination cell should be exactly 1.0 (with high precision)!
    // 3. No weight entry in W can reference a dry/masked source cell (col_idx must have mask == 1).

    std::vector<double> row_sums(n_dst_cells, 0.0);

    const auto nnz = W.nnz();
    auto factor_row = W.factor_row();
    auto factor_col = W.factor_col();
    auto factor_list = W.factor_list();

    for (std::size_t k = 0; k < nnz; ++k) {
        const auto r = static_cast<std::size_t>(factor_row[k]);
        const auto c = static_cast<std::size_t>(factor_col[k]);
        const auto val = factor_list[k];

        // Ensure the source cell is wet
        RC_ASSERT(mask_vec[c] == 1);

        row_sums[r] += val;
    }

    for (std::size_t r = 0; r < n_dst_cells; ++r) {
        // Assert partition of unity (sum is exactly 1.0)
        RC_ASSERT(std::abs(row_sums[r] - 1.0) < 1e-11);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
