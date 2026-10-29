// ─── Property-Based Tests: Bilinear Regular-Grid Fast-Path ───────────────────
// Feature: bilinear-regular-grid-fastpath, Property 1: Partition of unity
//
// For any regular source grid and for any destination cell centroid (whether
// interior, clamped, or in the wraparound cell), the four bilinear weights
// produced by the fast-path SHALL sum to 1.0 within relative tolerance 1e-15,
// and each individual weight SHALL be in the range [0.0, 1.0].
//
// **Validates: Requirements 3.3, 3.4**
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

/// Generate grid dimension in [2, 200].
rc::Gen<std::size_t> genGridDim() {
    return rc::gen::inRange<std::size_t>(2, 201);
}

/// Generate a positive delta for grid spacing in (0.01, 10.0].
rc::Gen<double> genPositiveDelta() {
    return rc::gen::map(rc::gen::inRange(1, 1001), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a longitude starting value in [-180, 180).
rc::Gen<double> genLonMin() {
    return rc::gen::map(rc::gen::inRange(-18000, 18000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a latitude starting value in [-90, 60).
/// Upper bound chosen so that lat_min + nj*delta_lat stays reasonable.
rc::Gen<double> genLatMin() {
    return rc::gen::map(rc::gen::inRange(-9000, 6000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a fractional offset in [0.0, 1.0] for positioning destination
/// points relative to the grid.
rc::Gen<double> genFrac() {
    return rc::gen::map(rc::gen::inRange(0, 10001), [](int v) { return static_cast<double>(v) / 10000.0; });
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

/// Build a single-cell destination mesh centered at (lon, lat) with a tiny
/// extent (±half_dx in each direction).
axis::topology::UnstructuredMesh<MemSpace> make_single_cell_dst(double lon, double lat) {
    const double half_dx = 0.01;  // tiny cell to ensure centroid is at (lon, lat)

    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = lon;
    dst_cy(0) = lat;

    axis::topology::StructuredGrid<MemSpace> dst_grid(1, 1, std::move(dst_cx), std::move(dst_cy), axis::topology::CoordinateSystem::SphericalDeg);

    // Corners: 2×2 nodes around the single cell
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

// ─── Property 1: Partition of Unity ─────────────────────────────────────────
// For any regular source grid and for any destination cell centroid (interior,
// clamped, or wraparound), the four bilinear weights sum to 1.0 within
// relative tolerance 1e-15, and each individual weight is in [0.0, 1.0].
//
// **Validates: Requirements 3.3, 3.4**

RC_GTEST_PROP(PropBilinearRectUnity, InteriorPointWeightsSumToOne, ()) {
    // Generate random source grid parameters
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto delta_lon = *genPositiveDelta();
    const auto delta_lat = *genPositiveDelta();
    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();

    // Generate a destination point strictly inside the grid
    const auto frac_lon = *genFrac();
    const auto frac_lat = *genFrac();

    const double lon_max = lon_min + static_cast<double>(ni) * delta_lon;
    const double lat_max = lat_min + static_cast<double>(nj) * delta_lat;

    // Place destination inside the grid domain using fractional offsets
    const double dst_lon = lon_min + frac_lon * (lon_max - lon_min);
    const double dst_lat = lat_min + frac_lat * (lat_max - lat_min);

    // Build source mesh
    auto src_mesh = make_regular_grid(ni, nj, lon_min, delta_lon, lat_min, delta_lat);

    // Build single-cell destination mesh
    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Configure bilinear interpolation with Ignore unmapped policy
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    // Generate weights
    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // The destination point is inside the grid, so it must be mapped
    const auto nnz = matrix.nnz();
    RC_ASSERT(nnz > 0u);

    // Collect weights for row 0 (our single destination cell)
    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();

    double weight_sum = 0.0;
    std::size_t entry_count = 0;

    for (std::size_t k = 0; k < nnz; ++k) {
        if (rows[k] == 0) {
            double w = vals[k];

            // Each weight must be in [0.0, 1.0]
            RC_ASSERT(w >= 0.0);
            RC_ASSERT(w <= 1.0);

            weight_sum += w;
            ++entry_count;
        }
    }

    // Must have entries for this destination cell
    RC_ASSERT(entry_count > 0u);

    // Partition of unity: weights sum to 1.0 within 1e-15 relative tolerance
    RC_ASSERT(std::abs(weight_sum - 1.0) < 1e-15);
}

RC_GTEST_PROP(PropBilinearRectUnity, BoundaryClampedPointWeightsSumToOne, ()) {
    // Generate random source grid parameters
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto delta_lon = *genPositiveDelta();
    const auto delta_lat = *genPositiveDelta();
    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();

    const double lon_max = lon_min + static_cast<double>(ni) * delta_lon;
    const double lat_max = lat_min + static_cast<double>(nj) * delta_lat;

    // Generate destination point at or near grid boundary (clamped region)
    // Use a small epsilon inside the boundary to ensure it's within bounds
    // but exercises the boundary clamping logic.
    const auto edge_choice = *rc::gen::inRange(0, 4);
    double dst_lon, dst_lat;

    const double eps = delta_lon * 0.01;
    const double eps_lat = delta_lat * 0.01;
    const auto frac = *genFrac();

    switch (edge_choice) {
        case 0:  // Near left boundary
            dst_lon = lon_min + eps;
            dst_lat = lat_min + frac * (lat_max - lat_min);
            break;
        case 1:  // Near right boundary
            dst_lon = lon_max - eps;
            dst_lat = lat_min + frac * (lat_max - lat_min);
            break;
        case 2:  // Near bottom boundary
            dst_lon = lon_min + frac * (lon_max - lon_min);
            dst_lat = lat_min + eps_lat;
            break;
        case 3:  // Near top boundary
            dst_lon = lon_min + frac * (lon_max - lon_min);
            dst_lat = lat_max - eps_lat;
            break;
        default:
            dst_lon = lon_min + 0.5 * (lon_max - lon_min);
            dst_lat = lat_min + 0.5 * (lat_max - lat_min);
            break;
    }

    // Build source mesh
    auto src_mesh = make_regular_grid(ni, nj, lon_min, delta_lon, lat_min, delta_lat);

    // Build single-cell destination mesh
    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Configure bilinear interpolation with Ignore unmapped policy
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    // Generate weights
    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto nnz = matrix.nnz();
    // If the point is inside, it must be mapped
    RC_ASSERT(nnz > 0u);

    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();

    double weight_sum = 0.0;
    std::size_t entry_count = 0;

    for (std::size_t k = 0; k < nnz; ++k) {
        if (rows[k] == 0) {
            double w = vals[k];

            // Each weight must be in [0.0, 1.0]
            RC_ASSERT(w >= 0.0);
            RC_ASSERT(w <= 1.0);

            weight_sum += w;
            ++entry_count;
        }
    }

    RC_ASSERT(entry_count > 0u);

    // Partition of unity: weights sum to 1.0 within 1e-15 relative tolerance
    RC_ASSERT(std::abs(weight_sum - 1.0) < 1e-15);
}

RC_GTEST_PROP(PropBilinearRectUnity, PeriodicWraparoundWeightsSumToOne, ()) {
    // Generate random periodic source grid (spanning exactly 360° longitude)
    const auto ni = *rc::gen::inRange<std::size_t>(4, 201);
    const auto nj = *rc::gen::inRange<std::size_t>(2, 201);
    const double delta_lon = 360.0 / static_cast<double>(ni);
    const auto delta_lat = *genPositiveDelta();
    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();

    const double lat_max = lat_min + static_cast<double>(nj) * delta_lat;

    // Generate destination point in the wraparound region
    // (near the seam between last and first longitude cell)
    const auto frac_lat = *genFrac();
    const auto wrap_frac = *genFrac();  // position within the wraparound cell

    // Place dst in the wraparound cell: between the last cell center and the
    // first cell center (across the seam)
    const double last_cell_left = lon_min + static_cast<double>(ni - 1) * delta_lon;
    const double dst_lon = last_cell_left + wrap_frac * delta_lon;
    const double dst_lat = lat_min + frac_lat * (lat_max - lat_min);

    // Build source mesh (periodic: spans 360° in longitude)
    auto src_mesh = make_regular_grid(ni, nj, lon_min, delta_lon, lat_min, delta_lat);

    // Build single-cell destination mesh
    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Configure bilinear interpolation with Ignore unmapped policy
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    // Generate weights
    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto nnz = matrix.nnz();
    // Periodic grid: no point should be unmapped in longitude
    RC_ASSERT(nnz > 0u);

    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();

    double weight_sum = 0.0;
    std::size_t entry_count = 0;

    for (std::size_t k = 0; k < nnz; ++k) {
        if (rows[k] == 0) {
            double w = vals[k];

            // Each weight must be in [0.0, 1.0]
            RC_ASSERT(w >= 0.0);
            RC_ASSERT(w <= 1.0);

            weight_sum += w;
            ++entry_count;
        }
    }

    RC_ASSERT(entry_count > 0u);

    // Partition of unity: weights sum to 1.0 within 1e-15 relative tolerance
    RC_ASSERT(std::abs(weight_sum - 1.0) < 1e-15);
}

}  // namespace
