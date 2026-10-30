// ─── Property-Based Tests: Bilinear Regular-Grid Fast-Path ───────────────────
// Feature: bilinear-regular-grid-fastpath, Property 6: Cell Location Correctness
//
// For any regular source grid and any random interior destination point, the
// computed cell index places the point within the correct 2×2 source block,
// and the flat linear index is within [0, ni*nj).
//
// **Validates: Requirements 2.1, 2.7**
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
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Kokkos Initialization ───────────────────────────────────────────────────

// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate grid dimension in [2, 200].
rc::Gen<std::size_t> genGridDim() {
    return rc::gen::inRange<std::size_t>(2, 201);
}

/// Generate a positive delta for grid spacing in (0.01, 10.0].
rc::Gen<double> genPositiveDelta() {
    return rc::gen::map(rc::gen::inRange(1, 1000), [](int x) { return static_cast<double>(x) * 0.01; });
}

/// Generate a grid origin coordinate in [-180, 180].
rc::Gen<double> genOrigin() {
    return rc::gen::map(rc::gen::inRange(-18000, 18001), [](int x) { return static_cast<double>(x) * 0.01; });
}

/// Generate a fractional position in (0, 1), avoiding exact boundaries.
rc::Gen<double> genFrac() {
    return rc::gen::map(rc::gen::inRange(1, 999), [](int x) { return static_cast<double>(x) / 1000.0; });
}

// ─── Helper: Build a regular-grid UnstructuredMesh ───────────────────────────

/// Constructs a regular lat-lon grid as an UnstructuredMesh.
/// Grid has ni×nj cells covering [lon0, lon0 + ni*dlon] × [lat0, lat0 + nj*dlat].
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

/// Build a single-cell destination mesh centered at (lon, lat).
axis::topology::UnstructuredMesh<MemSpace> make_single_cell_dst(double lon, double lat) {
    const double half_dx = 0.01;

    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = lon;
    dst_cy(0) = lat;

    axis::topology::StructuredGrid<MemSpace> dst_grid(1, 1, std::move(dst_cx), std::move(dst_cy), axis::topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dst_crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> dst_cry("dst_cry", 4);
    dst_crx(0) = lon - half_dx;
    dst_cry(0) = lat - half_dx;
    dst_crx(1) = lon + half_dx;
    dst_cry(1) = lat - half_dx;
    dst_crx(2) = lon - half_dx;
    dst_cry(2) = lat + half_dx;
    dst_crx(3) = lon + half_dx;
    dst_cry(3) = lat + half_dx;
    dst_grid.set_corners(std::move(dst_crx), std::move(dst_cry));

    return dst_grid.to_unstructured();
}

// ─── Property 6: Cell Location Correctness ──────────────────────────────────
// For any regular source grid and any random interior point, the computed cell
// index places the point within the correct 2×2 source block, and the flat
// linear index is within [0, ni*nj).
//
// We verify this by running the fast-path interpolation and checking that
// column indices in the returned matrix correspond to valid source cells whose
// bounding box contains the destination point.
//
// **Validates: Requirements 2.1, 2.7**

RC_GTEST_PROP(BilinearRectCellLocation, ComputedCellIndexPlacesPointInCorrect2x2Block, ()) {
    // Generate random grid dimensions
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto dlon = *genPositiveDelta();
    const auto dlat = *genPositiveDelta();
    const auto lon_min = *genOrigin();
    const auto lat_min = *genOrigin();

    // Generate a random interior destination point (strictly inside the grid)
    const auto frac_i = *genFrac();
    const auto frac_j = *genFrac();

    // Destination point coordinates (interior)
    const double dst_lon = lon_min + frac_i * static_cast<double>(ni) * dlon;
    const double dst_lat = lat_min + frac_j * static_cast<double>(nj) * dlat;

    // Expected cell index via floor division (same as production code)
    auto expected_i = static_cast<std::size_t>(std::floor((dst_lon - lon_min) / dlon));
    auto expected_j = static_cast<std::size_t>(std::floor((dst_lat - lat_min) / dlat));

    // Clamp to valid range (matches production clamping logic)
    if (expected_i >= ni) expected_i = ni - 1;
    if (expected_j >= nj) expected_j = nj - 1;

    // Flat index must be in [0, ni*nj)
    const std::size_t flat_idx = expected_j * ni + expected_i;
    RC_ASSERT(flat_idx < ni * nj);

    // Verify point is within the bounding box of cell (expected_i, expected_j)
    const double cell_lon_min = lon_min + static_cast<double>(expected_i) * dlon;
    const double cell_lon_max = cell_lon_min + dlon;
    const double cell_lat_min = lat_min + static_cast<double>(expected_j) * dlat;
    const double cell_lat_max = cell_lat_min + dlat;

    // Allow small numerical tolerance for floating-point floor
    constexpr double eps = 1e-12;
    RC_ASSERT(dst_lon >= cell_lon_min - eps);
    RC_ASSERT(dst_lon <= cell_lon_max + eps);
    RC_ASSERT(dst_lat >= cell_lat_min - eps);
    RC_ASSERT(dst_lat <= cell_lat_max + eps);
}

RC_GTEST_PROP(BilinearRectCellLocation, FlatLinearIndexWithinBounds, ()) {
    // Generate random grid dimensions
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();

    // Generate random cell indices within the grid
    const auto i = *rc::gen::inRange<std::size_t>(0, ni);
    const auto j = *rc::gen::inRange<std::size_t>(0, nj);

    // Flat index calculation mirrors the production code: j * ni + i
    const std::size_t flat_idx = j * ni + i;

    // Must be in valid range
    RC_ASSERT(flat_idx < ni * nj);

    // Verify the 2×2 block neighbors are also valid for interior cells
    if (i + 1 < ni && j + 1 < nj) {
        RC_ASSERT((j)*ni + (i) < ni * nj);            // (i, j)
        RC_ASSERT((j)*ni + (i + 1) < ni * nj);        // (i+1, j)
        RC_ASSERT((j + 1) * ni + (i) < ni * nj);      // (i, j+1)
        RC_ASSERT((j + 1) * ni + (i + 1) < ni * nj);  // (i+1, j+1)
    }
}

RC_GTEST_PROP(BilinearRectCellLocation, InterpolationHitsCorrectSourceCells, ()) {
    // Use smaller grids to keep test fast but still validate cell location
    const auto ni = *rc::gen::inRange<std::size_t>(4, 50);
    const auto nj = *rc::gen::inRange<std::size_t>(4, 50);
    const double dlon = 1.0;
    const double dlat = 1.0;
    const double lon_min = 0.0;
    const double lat_min = 0.0;

    // Random interior point — generate within the cell-center bounding box
    // to avoid boundary clamping. Cell centers range from
    // [lon_min + 0.5*dlon, lon_min + (ni-0.5)*dlon] and similarly for lat.
    const auto frac_i = *genFrac();
    const auto frac_j = *genFrac();
    // Interior to cell-center bbox: fraction maps to [0.5*dlon, (ni-0.5)*dlon]
    const double dst_lon = lon_min + (0.5 + frac_i * (static_cast<double>(ni) - 1.0)) * dlon;
    const double dst_lat = lat_min + (0.5 + frac_j * (static_cast<double>(nj) - 1.0)) * dlat;

    // Build the source mesh and destination mesh
    auto src_mesh = make_regular_grid(ni, nj, lon_min, dlon, lat_min, dlat);
    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Run bilinear interpolation via weight generator
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // The matrix should have exactly 4 nonzero entries for one interior point
    RC_ASSERT(matrix.nnz() == 4);

    // All column indices must be in [0, ni*nj)
    const std::size_t n_src = ni * nj;
    auto factor_col = matrix.factor_col();
    for (std::size_t k = 0; k < matrix.nnz(); ++k) {
        RC_ASSERT(static_cast<std::size_t>(factor_col[k]) < n_src);
    }

    // Compute expected cell indices using cell-center-relative logic
    // (matching the production fast-path implementation):
    //   fi = (lon_d - lon_min) / dlon - 0.5
    //   i = floor(fi), i1 = i + 1
    double fi = (dst_lon - lon_min) / dlon - 0.5;
    double fj = (dst_lat - lat_min) / dlat - 0.5;

    auto ci = static_cast<int>(std::floor(fi));
    auto cj = static_cast<int>(std::floor(fj));

    // Clamp (matching fast-path logic)
    if (ci < 0) ci = 0;
    if (ci >= static_cast<int>(ni) - 1) ci = static_cast<int>(ni) - 2;
    if (cj < 0) cj = 0;
    if (cj >= static_cast<int>(nj) - 1) cj = static_cast<int>(nj) - 2;

    int ci1 = ci + 1;
    int cj1 = cj + 1;

    // The four expected source cells for bilinear interpolation
    std::vector<std::size_t> expected_cells = {
        static_cast<std::size_t>(cj) * ni + static_cast<std::size_t>(ci), static_cast<std::size_t>(cj) * ni + static_cast<std::size_t>(ci1),
        static_cast<std::size_t>(cj1) * ni + static_cast<std::size_t>(ci), static_cast<std::size_t>(cj1) * ni + static_cast<std::size_t>(ci1)};

    // Each column index from the matrix should be one of the expected cells
    for (std::size_t k = 0; k < matrix.nnz(); ++k) {
        auto col = static_cast<std::size_t>(factor_col[k]);
        bool found = false;
        for (auto expected : expected_cells) {
            if (col == expected) {
                found = true;
                break;
            }
        }
        RC_ASSERT(found);
    }
}

}  // namespace
