// ─── Property-Based Tests: Regular-Grid Detection Correctness ────────────────
// Feature: axis-performance-optimizations, Property 2: Regular-Grid Detection
//          Correctness
//
// For any mesh whose cell vertex coordinates form a logically rectangular grid
// where max(delta_lon) - min(delta_lon) < 1e-10 * mean(delta_lon) AND
// max(delta_lat) - min(delta_lat) < 1e-10 * mean(delta_lat), the
// detect_regular_grid() function SHALL return is_regular = true.
// For any mesh where either axis exceeds the threshold, it SHALL return
// is_regular = false.
//
// **Validates: Requirements 2.1, 2.6**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate grid dimension in [2, 30]. Large enough to exercise uniformity
/// checks across multiple deltas, small enough for fast property iterations.
rc::Gen<std::size_t> genGridDim() {
    return rc::gen::inRange<std::size_t>(2, 31);
}

/// Generate a positive delta for grid spacing in (0.01, 10.0].
/// Avoids degenerate zero-width cells.
rc::Gen<double> genPositiveDelta() {
    return rc::gen::map(rc::gen::inRange(1, 1001), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a longitude starting value in [-180, 180).
rc::Gen<double> genLonMin() {
    return rc::gen::map(rc::gen::inRange(-18000, 18000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a latitude starting value in [-90, 90).
rc::Gen<double> genLatMin() {
    return rc::gen::map(rc::gen::inRange(-9000, 9000), [](int v) { return static_cast<double>(v) / 100.0; });
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

// ─── Helper: Build a non-uniform grid with perturbed spacing ─────────────────

/// Constructs a grid where spacing is perturbed beyond the uniformity
/// threshold. The perturbation is applied to at least one axis to guarantee
/// that max(delta) - min(delta) >= 1e-10 * mean(delta).
///
/// @param perturb_lon If true, perturb longitude spacing
/// @param perturb_lat If true, perturb latitude spacing
/// @param perturbation_factor How much to perturb (relative to delta)
axis::topology::UnstructuredMesh<MemSpace> make_nonuniform_grid(std::size_t ni, std::size_t nj, double lon0, double base_dlon, double lat0,
                                                                double base_dlat, bool perturb_lon, bool perturb_lat, double perturbation_factor) {
    const std::size_t nc_i = ni + 1;
    const std::size_t nc_j = nj + 1;
    const std::size_t n_corners = nc_i * nc_j;

    // Build longitude boundary coordinates with optional perturbation.
    // For non-uniform: alternate between (1 - pf) * dlon and (1 + pf) * dlon
    std::vector<double> lon_bounds(nc_i);
    lon_bounds[0] = lon0;
    for (std::size_t i = 1; i <= ni; ++i) {
        double factor = 1.0;
        if (perturb_lon) {
            // Alternate wider/narrower cells
            factor = (i % 2 == 0) ? (1.0 + perturbation_factor) : (1.0 - perturbation_factor);
        }
        lon_bounds[i] = lon_bounds[i - 1] + base_dlon * factor;
    }

    // Build latitude boundary coordinates with optional perturbation
    std::vector<double> lat_bounds(nc_j);
    lat_bounds[0] = lat0;
    for (std::size_t j = 1; j <= nj; ++j) {
        double factor = 1.0;
        if (perturb_lat) {
            factor = (j % 2 == 0) ? (1.0 + perturbation_factor) : (1.0 - perturbation_factor);
        }
        lat_bounds[j] = lat_bounds[j - 1] + base_dlat * factor;
    }

    // Build center coordinates from boundaries
    const std::size_t n_centers = ni * nj;
    Kokkos::View<double *, MemSpace> cx("cx", n_centers);
    Kokkos::View<double *, MemSpace> cy("cy", n_centers);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = 0.5 * (lon_bounds[i] + lon_bounds[i + 1]);
            cy(i + j * ni) = 0.5 * (lat_bounds[j] + lat_bounds[j + 1]);
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> crx("crx", n_corners);
    Kokkos::View<double *, MemSpace> cry("cry", n_corners);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_i) = lon_bounds[i];
            cry(i + j * nc_i) = lat_bounds[j];
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Property 2a: Uniform grids are detected as regular ─────────────────────
// Generate random regular grids with varying ni, nj, lon_min, delta_lon,
// lat_min, delta_lat. Verify detect_regular_grid() returns is_regular = true
// and the detected parameters match the input.
//
// **Validates: Requirements 2.1, 2.6**

RC_GTEST_PROP(PropRegularGridDetector, UniformGridDetectedAsRegular, ()) {
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto dlon = *genPositiveDelta();
    const auto dlat = *genPositiveDelta();
    const auto lon0 = *genLonMin();
    const auto lat0 = *genLatMin();

    auto mesh = make_regular_grid(ni, nj, lon0, dlon, lat0, dlat);
    auto info = axis::detail::detect_regular_grid(mesh);

    // Must detect as regular
    RC_ASSERT(info.is_regular);

    // Detected dimensions must match
    RC_ASSERT(info.ni == ni);
    RC_ASSERT(info.nj == nj);

    // Detected spacing must match within floating-point tolerance
    RC_ASSERT(std::abs(info.delta_lon - dlon) < 1e-12 * dlon);
    RC_ASSERT(std::abs(info.delta_lat - dlat) < 1e-12 * dlat);

    // Detected bounds must match
    RC_ASSERT(std::abs(info.lon_min - lon0) < 1e-12);
    RC_ASSERT(std::abs(info.lon_max - (lon0 + ni * dlon)) < 1e-12 * (ni * dlon));
    RC_ASSERT(std::abs(info.lat_min - lat0) < 1e-12);
    RC_ASSERT(std::abs(info.lat_max - (lat0 + nj * dlat)) < 1e-12 * (nj * dlat));
}

// ─── Property 2b: Non-uniform grids are rejected ────────────────────────────
// Generate grids where the spacing on at least one axis is perturbed beyond
// the 1e-10 relative threshold. Verify detect_regular_grid() returns
// is_regular = false.
//
// The perturbation factor is chosen large enough (>= 1e-6) that
// max(delta) - min(delta) clearly exceeds 1e-10 * mean(delta).
//
// **Validates: Requirements 2.1, 2.6**

RC_GTEST_PROP(PropRegularGridDetector, NonUniformGridRejected, ()) {
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto base_dlon = *genPositiveDelta();
    const auto base_dlat = *genPositiveDelta();
    const auto lon0 = *genLonMin();
    const auto lat0 = *genLatMin();

    // Generate a perturbation factor well above the threshold.
    // Range [1e-6, 0.5] ensures max-min > 1e-10 * mean for grids with ni,nj >= 2.
    const auto pf_int = *rc::gen::inRange(1, 500001);
    const double perturbation_factor = static_cast<double>(pf_int) / 1000000.0;

    // Choose which axis to perturb (at least one must be perturbed)
    const auto perturb_mode = *rc::gen::inRange(0, 3);
    bool perturb_lon = (perturb_mode == 0 || perturb_mode == 2);
    bool perturb_lat = (perturb_mode == 1 || perturb_mode == 2);

    auto mesh = make_nonuniform_grid(ni, nj, lon0, base_dlon, lat0, base_dlat, perturb_lon, perturb_lat, perturbation_factor);
    auto info = axis::detail::detect_regular_grid(mesh);

    // Verify: the alternating spacing pattern with perturbation_factor >= 1e-6
    // and ni,nj >= 2 produces max(delta)-min(delta) = 2*pf*base_delta which
    // must exceed 1e-10 * mean(delta) = 1e-10 * base_delta.
    // Since 2*pf >= 2e-6 >> 1e-10, the grid must be rejected.
    RC_ASSERT(!info.is_regular);
}

// ─── Property 2c: Borderline uniform grid (within threshold) is accepted ────
// Generate a grid with extremely tiny perturbation that stays well below the
// 1e-10 * mean(delta) threshold. Verify it's still detected as regular.
//
// **Validates: Requirements 2.1, 2.6**

RC_GTEST_PROP(PropRegularGridDetector, TinyPerturbationStillDetectedAsRegular, ()) {
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto base_dlon = *genPositiveDelta();
    const auto base_dlat = *genPositiveDelta();
    const auto lon0 = *genLonMin();
    const auto lat0 = *genLatMin();

    // Use a perturbation factor well below the threshold (1e-13).
    // max(delta) - min(delta) = 2 * 1e-13 * base_delta
    // threshold = 1e-10 * mean(delta) ≈ 1e-10 * base_delta
    // 2e-13 << 1e-10, so the grid should still be detected as regular.
    constexpr double tiny_pf = 1.0e-13;

    auto mesh = make_nonuniform_grid(ni, nj, lon0, base_dlon, lat0, base_dlat, true, true, tiny_pf);
    auto info = axis::detail::detect_regular_grid(mesh);

    RC_ASSERT(info.is_regular);
}

// ─── Property 2d: Detected parameters are self-consistent ───────────────────
// For any grid detected as regular, verify that the reported parameters are
// internally consistent: lon_max ≈ lon_min + ni * delta_lon, and similarly
// for latitude.
//
// **Validates: Requirements 2.1, 2.6**

RC_GTEST_PROP(PropRegularGridDetector, DetectedParamsAreConsistent, ()) {
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();
    const auto dlon = *genPositiveDelta();
    const auto dlat = *genPositiveDelta();
    const auto lon0 = *genLonMin();
    const auto lat0 = *genLatMin();

    auto mesh = make_regular_grid(ni, nj, lon0, dlon, lat0, dlat);
    auto info = axis::detail::detect_regular_grid(mesh);

    RC_PRE(info.is_regular);

    // lon_max should equal lon_min + ni * delta_lon
    double expected_lon_max = info.lon_min + static_cast<double>(info.ni) * info.delta_lon;
    RC_ASSERT(std::abs(info.lon_max - expected_lon_max) < 1e-10 * std::abs(expected_lon_max) + 1e-12);

    // lat_max should equal lat_min + nj * delta_lat
    double expected_lat_max = info.lat_min + static_cast<double>(info.nj) * info.delta_lat;
    RC_ASSERT(std::abs(info.lat_max - expected_lat_max) < 1e-10 * std::abs(expected_lat_max) + 1e-12);

    // ni * nj must equal the original cell count
    RC_ASSERT(info.ni * info.nj == mesh.n_cells());
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
