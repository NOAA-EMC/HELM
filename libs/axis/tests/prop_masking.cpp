// ─── Property-Based Tests: Masking Excludes Masked Cells ─────────────────────
// Feature: axis-v2-improvements, Property 14: Masking excludes masked cells
//
// For any mesh with masked source cells, the resulting InterpolationMatrix
// SHALL contain zero entries referencing those cells.
//
// **Validates: Requirements 8.1**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

/// Build a simple ni x nj Cartesian regular-grid UnstructuredMesh on HostSpace.
/// Uses Cartesian3D coordinate system (matches LineType::Cartesian requirement).
axis::topology::UnstructuredMesh<Kokkos::HostSpace> build_cartesian_mesh(std::size_t ni, std::size_t nj, double x_start, double y_start, double dx,
                                                                         double dy, Kokkos::View<int *, Kokkos::HostSpace> mask = {}) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, Kokkos::HostSpace> center_x("cx", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_y("cy", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_x("crx", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_y("cry", n_corners);

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

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_x, center_y, axis::topology::CoordinateSystem::Cartesian3D);
    grid.set_corners(corner_x, corner_y);

    auto mesh = grid.to_unstructured();

    // If a mask was provided, rebuild with the mask
    if (mask.extent(0) > 0) {
        return axis::topology::UnstructuredMesh<Kokkos::HostSpace>(mesh.node_coords_view(), mesh.conn_offsets_view(), mesh.conn_indices_view(),
                                                                   mesh.coord_system(), mesh.cell_areas_view(), std::move(mask));
    }

    return mesh;
}

// ─── Property 14: Masking excludes masked cells ──────────────────────────────
// For any mesh with masked source cells, the resulting InterpolationMatrix
// SHALL contain zero entries referencing those cells.
//
// Strategy:
// 1. Generate random grid dimensions for source and destination
// 2. Generate a random source mask (at least one cell masked, at least one active)
// 3. Generate weights using Conservative1stOrder with Cartesian line_type
// 4. Verify: for all entries k, factor_col(k) never references a masked cell
//
// **Validates: Requirements 8.1**

RC_GTEST_PROP(PropMasking, MaskedSourceCellsExcludedFromWeights, ()) {
    // Random grid dimensions (small to keep test fast)
    const auto src_ni = *rc::gen::inRange<std::size_t>(2, 6);
    const auto src_nj = *rc::gen::inRange<std::size_t>(2, 6);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 6);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 6);

    const std::size_t n_src_cells = src_ni * src_nj;

    // Generate a random mask: each cell independently 0 or 1
    // Ensure at least one cell is masked and at least one is active
    auto mask_vec = *rc::gen::container<std::vector<int>>(n_src_cells, rc::gen::inRange(0, 2));

    // Force at least one masked cell
    const auto mask_idx = *rc::gen::inRange<std::size_t>(0, n_src_cells);
    mask_vec[mask_idx] = 0;

    // Force at least one active cell (pick a different cell if possible)
    const auto active_idx = *rc::gen::inRange<std::size_t>(0, n_src_cells);
    if (active_idx != mask_idx) {
        mask_vec[active_idx] = 1;
    } else {
        // If same index, force the next cell active
        mask_vec[(mask_idx + 1) % n_src_cells] = 1;
    }

    // Build the source mask Kokkos view
    Kokkos::View<int *, Kokkos::HostSpace> src_mask("src_mask", n_src_cells);
    for (std::size_t i = 0; i < n_src_cells; ++i) {
        src_mask(i) = mask_vec[i];
    }

    // Collect masked cell indices for assertion
    std::vector<std::size_t> masked_cells;
    for (std::size_t i = 0; i < n_src_cells; ++i) {
        if (mask_vec[i] == 0) {
            masked_cells.push_back(i);
        }
    }

    // Build overlapping meshes (same domain) with Cartesian coordinate system
    const double domain_size = 4.0;
    const double src_dx = domain_size / static_cast<double>(src_ni);
    const double src_dy = domain_size / static_cast<double>(src_nj);
    const double dst_dx = domain_size / static_cast<double>(dst_ni);
    const double dst_dy = domain_size / static_cast<double>(dst_nj);

    auto src_mesh = build_cartesian_mesh(src_ni, src_nj, 0.0, 0.0, src_dx, src_dy, src_mask);
    auto dst_mesh = build_cartesian_mesh(dst_ni, dst_nj, 0.0, 0.0, dst_dx, dst_dy);

    // Configure conservative weight generation with Cartesian line type
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    // Verify: no weight entry references any masked source cell
    const auto nnz = matrix.nnz();
    auto factor_col = matrix.factor_col();

    for (std::size_t k = 0; k < nnz; ++k) {
        const auto col_idx = static_cast<std::size_t>(factor_col[k]);
        for (const auto &masked_idx : masked_cells) {
            RC_ASSERT(col_idx != masked_idx);
        }
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
