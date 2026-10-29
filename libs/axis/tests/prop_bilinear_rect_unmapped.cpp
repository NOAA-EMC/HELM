// ─── Property-Based Tests: Bilinear Regular-Grid Fast-Path ───────────────────
// Feature: bilinear-regular-grid-fastpath, Property 5: Unmapped Point Policy
//          Enforcement
//
// For any non-periodic source grid and for any destination cell centroid that
// lies strictly outside the source grid bounding box: when
// RegridConfig::unmapped is Error, the fast-path SHALL throw
// std::runtime_error; when unmapped is Ignore, the corresponding destination
// row SHALL have zero entries in the InterpolationMatrix.
//
// **Validates: Requirements 7.1, 7.2**
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
#include <stdexcept>
#include <vector>

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Kokkos Initialization ───────────────────────────────────────────────────



// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate grid dimension in [2, 50].
rc::Gen<std::size_t> genGridDim() {
    return rc::gen::inRange<std::size_t>(2, 51);
}

/// Direction of out-of-bounds placement relative to the source bounding box.
enum class OutsideDirection { Left, Right, Below, Above };

/// Generate one of the four outside directions.
rc::Gen<OutsideDirection> genOutsideDirection() {
    return rc::gen::map(rc::gen::inRange(0, 4), [](int v) { return static_cast<OutsideDirection>(v); });
}

/// Generate an offset in [1.0, 10.0] degrees to place the point outside bbox.
rc::Gen<double> genOutsideOffset() {
    return rc::gen::map(rc::gen::inRange(100, 1001), [](int v) { return static_cast<double>(v) / 100.0; });
}

// ─── Helper: Build a non-periodic regular grid as UnstructuredMesh ────────────
//
// Creates a regular grid with ni×nj cells covering [lon_min, lon_max] ×
// [lat_min, lat_max] where lon_range < 360° to ensure non-periodicity.
// ─────────────────────────────────────────────────────────────────────────────

axis::topology::UnstructuredMesh<MemSpace> make_nonperiodic_regular_grid(std::size_t ni, std::size_t nj, double lon_min, double lon_max,
                                                                         double lat_min, double lat_max) {
    const double delta_lon = (lon_max - lon_min) / static_cast<double>(ni);
    const double delta_lat = (lat_max - lat_min) / static_cast<double>(nj);

    const std::size_t n_centers = ni * nj;
    Kokkos::View<double *, MemSpace> cx("cx", n_centers);
    Kokkos::View<double *, MemSpace> cy("cy", n_centers);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_min + (static_cast<double>(i) + 0.5) * delta_lon;
            cy(i + j * ni) = lat_min + (static_cast<double>(j) + 0.5) * delta_lat;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    const std::size_t nc_lon = ni + 1;
    const std::size_t nc_lat = nj + 1;
    const std::size_t n_corners = nc_lon * nc_lat;
    Kokkos::View<double *, MemSpace> crx("crx", n_corners);
    Kokkos::View<double *, MemSpace> cry("cry", n_corners);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_lon) = lon_min + static_cast<double>(i) * delta_lon;
            cry(i + j * nc_lon) = lat_min + static_cast<double>(j) * delta_lat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Helper: Build a single-cell destination mesh at a given point ────────────

axis::topology::UnstructuredMesh<MemSpace> make_single_cell_dst(double lon, double lat) {
    const double half_dx = 0.25;  // small cell around the point

    Kokkos::View<double *, MemSpace> cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> cy("dst_cy", 1);
    cx(0) = lon;
    cy(0) = lat;

    axis::topology::StructuredGrid<MemSpace> grid(1, 1, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> cry("dst_cry", 4);
    crx(0) = lon - half_dx;
    cry(0) = lat - half_dx;
    crx(1) = lon + half_dx;
    cry(1) = lat - half_dx;
    crx(2) = lon - half_dx;
    cry(2) = lat + half_dx;
    crx(3) = lon + half_dx;
    cry(3) = lat + half_dx;
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Helper: Compute outside destination point given source bounds ────────────

/// Given the source grid bounding box and a direction, produce a destination
/// point that lies at least `offset` degrees outside the boundary.
std::pair<double, double> compute_outside_point(double lon_min, double lon_max, double lat_min, double lat_max, OutsideDirection dir, double offset) {
    const double lon_center = (lon_min + lon_max) * 0.5;
    const double lat_center = (lat_min + lat_max) * 0.5;

    switch (dir) {
        case OutsideDirection::Left:
            return {lon_min - offset, lat_center};
        case OutsideDirection::Right:
            return {lon_max + offset, lat_center};
        case OutsideDirection::Below:
            return {lon_center, lat_min - offset};
        case OutsideDirection::Above:
            return {lon_center, lat_max + offset};
    }
    // unreachable
    return {lon_min - offset, lat_center};
}

// ─── Helper: Count entries in a specific row of the interpolation matrix ─────

std::size_t count_row_entries(const axis::solver::InterpolationMatrix<MemSpace> &matrix, axis::index_t row_idx) {
    std::size_t count = 0;
    const auto nnz = matrix.nnz();
    auto rows = matrix.factor_row();
    for (std::size_t k = 0; k < nnz; ++k) {
        if (rows[k] == row_idx) {
            ++count;
        }
    }
    return count;
}

// ─── Property 5a: Unmapped Error policy throws std::runtime_error ────────────
//
// For any non-periodic source grid and for any destination cell centroid that
// lies strictly outside the source grid bounding box, when
// RegridConfig::unmapped is Error, the fast-path SHALL throw
// std::runtime_error.
//
// **Validates: Requirements 7.1, 7.2**

RC_GTEST_PROP(PropBilinearRectUnmapped, ErrorPolicyThrows, ()) {
    // Generate random non-periodic source grid dimensions
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();

    // Fixed non-periodic source grid bounds: lon ∈ [10, 90], lat ∈ [10, 60]
    // This ensures lon_range = 80° < 360° → non-periodic.
    const double src_lon_min = 10.0;
    const double src_lon_max = 90.0;
    const double src_lat_min = 10.0;
    const double src_lat_max = 60.0;

    auto src_mesh = make_nonperiodic_regular_grid(ni, nj, src_lon_min, src_lon_max, src_lat_min, src_lat_max);

    // Generate a destination point strictly outside the source bbox
    const auto dir = *genOutsideDirection();
    const auto offset = *genOutsideOffset();

    auto [dst_lon, dst_lat] = compute_outside_point(src_lon_min, src_lon_max, src_lat_min, src_lat_max, dir, offset);

    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Configure for Error unmapped policy
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Error;

    // The fast-path should throw std::runtime_error for the unmapped point
    bool threw_runtime_error = false;
    try {
        axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);
    } catch (const std::runtime_error &) {
        threw_runtime_error = true;
    }

    RC_ASSERT(threw_runtime_error);
}

// ─── Property 5b: Unmapped Ignore policy produces zero-row ───────────────────
//
// For any non-periodic source grid and for any destination cell centroid that
// lies strictly outside the source grid bounding box, when
// RegridConfig::unmapped is Ignore, the corresponding destination row SHALL
// have zero entries in the InterpolationMatrix.
//
// **Validates: Requirements 7.1, 7.2**

RC_GTEST_PROP(PropBilinearRectUnmapped, IgnorePolicyProducesZeroRow, ()) {
    // Generate random non-periodic source grid dimensions
    const auto ni = *genGridDim();
    const auto nj = *genGridDim();

    // Fixed non-periodic source grid bounds: lon ∈ [10, 90], lat ∈ [10, 60]
    const double src_lon_min = 10.0;
    const double src_lon_max = 90.0;
    const double src_lat_min = 10.0;
    const double src_lat_max = 60.0;

    auto src_mesh = make_nonperiodic_regular_grid(ni, nj, src_lon_min, src_lon_max, src_lat_min, src_lat_max);

    // Generate a destination point strictly outside the source bbox
    const auto dir = *genOutsideDirection();
    const auto offset = *genOutsideOffset();

    auto [dst_lon, dst_lat] = compute_outside_point(src_lon_min, src_lon_max, src_lat_min, src_lat_max, dir, offset);

    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Configure for Ignore unmapped policy
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // The destination row for the unmapped cell should have zero entries
    RC_ASSERT(matrix.n_dst() == 1u);
    RC_ASSERT(count_row_entries(matrix, 0) == 0u);
}

}  // namespace
