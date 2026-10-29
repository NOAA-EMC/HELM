// ─── Property-Based Tests: Bilinear Regular-Grid Fast-Path ───────────────────
// Feature: bilinear-regular-grid-fastpath, Property 3: Sparsity Structure Invariant
//
// For any regular source-destination grid pair processed by the fast-path,
// each interior destination cell (whose containing source cell is not at a
// grid boundary in non-periodic mode) SHALL have exactly 4 non-zero entries
// in the InterpolationMatrix, and ALL column indices SHALL be in the valid
// range [0, n_src_cells) and ALL row indices in [0, n_dst_cells).
//
// **Validates: Requirements 5.2, 5.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Kokkos Initialization ───────────────────────────────────────────────────



// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate source grid dimension in [2, 100].
rc::Gen<std::size_t> genSrcDim() {
    return rc::gen::inRange<std::size_t>(2, 101);
}

/// Generate destination grid dimension in [2, 100].
rc::Gen<std::size_t> genDstDim() {
    return rc::gen::inRange<std::size_t>(2, 101);
}

/// Generate a positive delta for grid spacing in (0.1, 5.0].
rc::Gen<double> genPositiveDelta() {
    return rc::gen::map(rc::gen::inRange(1, 501), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a longitude starting value in [-180, 80).
/// Capped to avoid overflow when adding ni * delta_lon.
rc::Gen<double> genLonMin() {
    return rc::gen::map(rc::gen::inRange(-18000, 8000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a latitude starting value in [-80, 80).
rc::Gen<double> genLatMin() {
    return rc::gen::map(rc::gen::inRange(-8000, 8000), [](int v) { return static_cast<double>(v) / 100.0; });
}

// ─── Helper: Build a uniform regular-grid UnstructuredMesh ───────────────────

/// Constructs a regular lat-lon grid as an UnstructuredMesh.
/// Grid covers [lon0, lon0 + ni*dlon] × [lat0, lat0 + nj*dlat].
axis::topology::UnstructuredMesh<MemSpace> make_regular_grid(std::size_t ni, std::size_t nj, double lon0, double dlon, double lat0, double dlat) {
    const std::size_t n_centers = ni * nj;
    const std::size_t nc_i = ni + 1;
    const std::size_t nc_j = nj + 1;
    const std::size_t n_corners = nc_i * nc_j;

    Kokkos::View<double *, MemSpace> cx("cx", n_centers);
    Kokkos::View<double *, MemSpace> cy("cy", n_centers);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon0 + (static_cast<double>(i) + 0.5) * dlon;
            cy(i + j * ni) = lat0 + (static_cast<double>(j) + 0.5) * dlat;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> crx("crx", n_corners);
    Kokkos::View<double *, MemSpace> cry("cry", n_corners);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_i) = lon0 + static_cast<double>(i) * dlon;
            cry(i + j * nc_i) = lat0 + static_cast<double>(j) * dlat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Helper: Determine if a destination cell is interior ─────────────────────

/// A destination cell is "interior" if its centroid maps to a source cell index
/// (i, j) such that i ∈ [1, ni-2] and j ∈ [1, nj-2], i.e. not at any boundary.
/// For such cells, all 4 neighbors (i, j), (i+1, j), (i, j+1), (i+1, j+1) are
/// valid without clamping.
bool is_interior_dst_cell(double dst_lon, double dst_lat, double src_lon_min, double src_lat_min, double src_dlon, double src_dlat,
                          std::size_t src_ni, std::size_t src_nj) {
    // Compute source cell indices for the destination centroid
    double fi = (dst_lon - src_lon_min) / src_dlon - 0.5;
    double fj = (dst_lat - src_lat_min) / src_dlat - 0.5;

    int i = static_cast<int>(std::floor(fi));
    int j = static_cast<int>(std::floor(fj));

    // Interior means i in [0, ni-2] and j in [0, nj-2]
    // (so that i+1 <= ni-1 and j+1 <= nj-1 are valid)
    return (i >= 0 && i <= static_cast<int>(src_ni) - 2 && j >= 0 && j <= static_cast<int>(src_nj) - 2);
}

// ─── Property 3a: All index bounds are valid ─────────────────────────────────
// For any regular source-destination grid pair, verify that ALL column indices
// in the InterpolationMatrix are in [0, n_src_cells) and ALL row indices are
// in [0, n_dst_cells).
//
// **Validates: Requirements 5.2, 5.4**

RC_GTEST_PROP(PropBilinearRectSparsity, AllIndicesInValidRange, ()) {
    const auto src_ni = *genSrcDim();
    const auto src_nj = *genSrcDim();
    const auto src_dlon = *genPositiveDelta();
    const auto src_dlat = *genPositiveDelta();
    const auto src_lon0 = *genLonMin();
    const auto src_lat0 = *genLatMin();

    // Generate a destination grid that lies within the source domain
    // to ensure all points are mapped (avoid unmapped complexity).
    const auto dst_ni = *genDstDim();
    const auto dst_nj = *genDstDim();

    // Source grid extent
    const double src_lon_max = src_lon0 + static_cast<double>(src_ni) * src_dlon;
    const double src_lat_max = src_lat0 + static_cast<double>(src_nj) * src_dlat;

    // Place destination grid strictly inside source domain with some margin
    const double margin_lon = src_dlon * 0.5;
    const double margin_lat = src_dlat * 0.5;
    const double dst_lon0 = src_lon0 + margin_lon;
    const double dst_lat0 = src_lat0 + margin_lat;
    const double dst_lon_extent = (src_lon_max - src_lon0) - 2.0 * margin_lon;
    const double dst_lat_extent = (src_lat_max - src_lat0) - 2.0 * margin_lat;

    // Skip degenerate cases where source grid is too small for margin
    RC_PRE(dst_lon_extent > 0.0);
    RC_PRE(dst_lat_extent > 0.0);

    const double dst_dlon = dst_lon_extent / static_cast<double>(dst_ni);
    const double dst_dlat = dst_lat_extent / static_cast<double>(dst_nj);

    auto src_mesh = make_regular_grid(src_ni, src_nj, src_lon0, src_dlon, src_lat0, src_dlat);
    auto dst_mesh = make_regular_grid(dst_ni, dst_nj, dst_lon0, dst_dlon, dst_lat0, dst_dlat);

    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    const auto nnz = matrix.nnz();

    RC_ASSERT(n_src == src_ni * src_nj);
    RC_ASSERT(n_dst == dst_ni * dst_nj);

    auto factor_col = matrix.factor_col();
    auto factor_row = matrix.factor_row();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_col[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_col[k]) < n_src);
        RC_ASSERT(factor_row[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_row[k]) < n_dst);
    }
}

// ─── Property 3b: Interior destination cells have exactly 4 entries ──────────
// For any regular source-destination grid pair, each interior destination cell
// (whose containing source cell is not at a grid boundary) SHALL have exactly
// 4 non-zero entries in the InterpolationMatrix.
//
// **Validates: Requirements 5.2, 5.4**

RC_GTEST_PROP(PropBilinearRectSparsity, InteriorCellsHaveExactly4Entries, ()) {
    const auto src_ni = *genSrcDim();
    const auto src_nj = *genSrcDim();
    const auto src_dlon = *genPositiveDelta();
    const auto src_dlat = *genPositiveDelta();
    const auto src_lon0 = *genLonMin();
    const auto src_lat0 = *genLatMin();

    // Generate a destination grid strictly inside the source domain interior.
    // We offset by 1.5 * src_delta on each side so all destination cell
    // centroids map to interior source cells (i in [1, ni-2], j in [1, nj-2]).
    RC_PRE(src_ni >= 4);  // need at least 4 cells for interior to exist
    RC_PRE(src_nj >= 4);

    const double src_lon_max = src_lon0 + static_cast<double>(src_ni) * src_dlon;
    const double src_lat_max = src_lat0 + static_cast<double>(src_nj) * src_dlat;

    // Offset so all dst centroids land in interior source cells
    const double dst_lon0 = src_lon0 + 1.5 * src_dlon;
    const double dst_lat0 = src_lat0 + 1.5 * src_dlat;
    const double dst_lon_max = src_lon_max - 1.5 * src_dlon;
    const double dst_lat_max = src_lat_max - 1.5 * src_dlat;
    const double dst_lon_extent = dst_lon_max - dst_lon0;
    const double dst_lat_extent = dst_lat_max - dst_lat0;

    RC_PRE(dst_lon_extent > 0.0);
    RC_PRE(dst_lat_extent > 0.0);

    // Use a small-ish destination grid to keep test fast
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 50);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 50);

    const double dst_dlon = dst_lon_extent / static_cast<double>(dst_ni);
    const double dst_dlat = dst_lat_extent / static_cast<double>(dst_nj);

    auto src_mesh = make_regular_grid(src_ni, src_nj, src_lon0, src_dlon, src_lat0, src_dlat);
    auto dst_mesh = make_regular_grid(dst_ni, dst_nj, dst_lon0, dst_dlon, dst_lat0, dst_dlat);

    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    const auto nnz = matrix.nnz();

    RC_ASSERT(n_dst == dst_ni * dst_nj);

    auto factor_row = matrix.factor_row();
    auto factor_col = matrix.factor_col();

    // Count entries per destination row
    std::vector<std::size_t> row_counts(n_dst, 0);
    for (std::size_t k = 0; k < nnz; ++k) {
        auto r = static_cast<std::size_t>(factor_row[k]);
        RC_ASSERT(r < n_dst);
        row_counts[r]++;
    }

    // Every destination cell should have exactly 4 entries since all are
    // interior (their centroids are well within the source grid interior)
    for (std::size_t r = 0; r < n_dst; ++r) {
        RC_ASSERT(row_counts[r] == 4u);
    }

    // Additionally verify all column indices are in valid range
    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_col[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_col[k]) < n_src);
    }
}

}  // namespace
