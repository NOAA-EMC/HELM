// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file tests/test_bilinear_rect.cpp
/// @brief GTest unit tests for the bilinear regular-grid fast-path.
///
/// Tests verify correct weight computation via analytic index arithmetic,
/// longitude wraparound for global grids, exact weight values at cell centers
/// and grid corners, transparent BVH fallback for non-regular grids, and
/// unmapped point policy enforcement.

#include <gtest/gtest.h>

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

#include "common/test_kokkos_env.hpp"

namespace {}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Build a regular lat-lon grid as UnstructuredMesh
//
// Creates a regular grid with ni×nj cells covering [lon_min, lon_max] ×
// [lat_min, lat_max]. Cell centers are at:
//   lon = lon_min + (i + 0.5) * delta_lon
//   lat = lat_min + (j + 0.5) * delta_lat
// Node/corner coordinates are at:
//   lon = lon_min + i * delta_lon   for i in [0, ni]
//   lat = lat_min + j * delta_lat   for j in [0, nj]
// ─────────────────────────────────────────────────────────────────────────────
static topology::UnstructuredMesh<MemSpace> make_regular_grid(std::size_t ni, std::size_t nj, double lon_min, double lon_max, double lat_min,
                                                              double lat_max) {
    const double delta_lon = (lon_max - lon_min) / static_cast<double>(ni);
    const double delta_lat = (lat_max - lat_min) / static_cast<double>(nj);

    // Cell centers
    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_min + (static_cast<double>(i) + 0.5) * delta_lon;
            cy(i + j * ni) = lat_min + (static_cast<double>(j) + 0.5) * delta_lat;
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);

    // Corner coordinates: (ni+1)×(nj+1) nodes
    const std::size_t nc_lon = ni + 1;
    const std::size_t nc_lat = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_lon * nc_lat);
    Kokkos::View<double *, MemSpace> cry("cry", nc_lon * nc_lat);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_lon) = lon_min + static_cast<double>(i) * delta_lon;
            cry(i + j * nc_lon) = lat_min + static_cast<double>(j) * delta_lat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));
    return grid.to_unstructured();
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: Collect weights for a specific destination row from the matrix
// ─────────────────────────────────────────────────────────────────────────────
struct WeightEntry {
    index_t col;
    double weight;
};

static std::vector<WeightEntry> get_row_entries(const solver::InterpolationMatrix<MemSpace> &matrix, index_t row_idx) {
    std::vector<WeightEntry> entries;
    const auto nnz = matrix.nnz();
    auto rows = matrix.factor_row();
    auto cols = matrix.factor_col();
    auto vals = matrix.factor_list();
    for (std::size_t k = 0; k < nnz; ++k) {
        if (rows[k] == row_idx) {
            entries.push_back({cols[k], vals[k]});
        }
    }
    return entries;
}

// =============================================================================
// Test 1: 4×2 → 2×1 grid with known analytic weights (hand-computed)
//
// Source: 4×2 grid covering [0, 40] × [0, 20] in degrees.
//   delta_lon = 10, delta_lat = 10
//   Cell centers: (5,5), (15,5), (25,5), (35,5), (5,15), (15,15), (25,15), (35,15)
//
// Destination: 2×1 grid covering [0, 40] × [0, 20].
//   delta_lon = 20, delta_lat = 20
//   Cell center: (10, 10) for dst cell 0, (30, 10) for dst cell 1
//
// For dst cell 0 at (10, 10):
//   fi = (10 - 0) / 10 - 0.5 = 0.5
//   fj = (10 - 0) / 10 - 0.5 = 0.5
//   i = floor(0.5) = 0, j = floor(0.5) = 0
//   tx = 0.5 - 0 = 0.5, ty = 0.5 - 0 = 0.5
//   Weights: (1-0.5)*(1-0.5)=0.25, 0.5*(1-0.5)=0.25, (1-0.5)*0.5=0.25, 0.5*0.5=0.25
//   Source cells: (0,0)=0, (1,0)=1, (0,1)=4, (1,1)=5
//
// For dst cell 1 at (30, 10):
//   fi = (30 - 0) / 10 - 0.5 = 2.5
//   fj = (10 - 0) / 10 - 0.5 = 0.5
//   i = floor(2.5) = 2, j = floor(0.5) = 0
//   tx = 2.5 - 2 = 0.5, ty = 0.5 - 0 = 0.5
//   Weights: 0.25, 0.25, 0.25, 0.25
//   Source cells: (2,0)=2, (3,0)=3, (2,1)=6, (3,1)=7
// =============================================================================
TEST(BilinearRect, KnownAnalyticWeights4x2To2x1) {
    auto src_mesh = make_regular_grid(4, 2, 0.0, 40.0, 0.0, 20.0);
    auto dst_mesh = make_regular_grid(2, 1, 0.0, 40.0, 0.0, 20.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    ASSERT_EQ(matrix.n_dst(), 2u);
    ASSERT_EQ(matrix.n_src(), 8u);

    // Dst cell 0: all 4 weights should be 0.25
    auto entries0 = get_row_entries(matrix, 0);
    ASSERT_EQ(entries0.size(), 4u);
    double sum0 = 0.0;
    for (const auto &e : entries0) {
        EXPECT_NEAR(e.weight, 0.25, 1e-12) << "Dst cell 0: expected weight 0.25 at col " << e.col;
        sum0 += e.weight;
    }
    EXPECT_NEAR(sum0, 1.0, 1e-15);

    // Dst cell 1: all 4 weights should be 0.25
    auto entries1 = get_row_entries(matrix, 1);
    ASSERT_EQ(entries1.size(), 4u);
    double sum1 = 0.0;
    for (const auto &e : entries1) {
        EXPECT_NEAR(e.weight, 0.25, 1e-12) << "Dst cell 1: expected weight 0.25 at col " << e.col;
        sum1 += e.weight;
    }
    EXPECT_NEAR(sum1, 1.0, 1e-15);
}

// =============================================================================
// Test 2: Global 360° grid, destination point at 359.5° — verify wraparound
//
// Source: 36×18 global grid [0, 360] × [-90, 90].
//   delta_lon = 10, cell centers at 5, 15, ..., 355.
//
// Destination: single cell whose center is at (359.5, 0.0).
//   fi = (359.5 - 0) / 10 - 0.5 = 35.45
//   i = floor(35.45) = 35 (the last cell), j = related to lat
//   i1 wraps around: since ni=36, i1 = (35+1) % 36 = 0
//
// This verifies that the wraparound uses cells (ni-1) = 35 and 0.
// =============================================================================
TEST(BilinearRect, GlobalGridWraparoundAt359_5) {
    // Source: 36×18 global grid
    auto src_mesh = make_regular_grid(36, 18, 0.0, 360.0, -90.0, 90.0);

    // Destination: single cell centered at (359.5, 0.0)
    // Create a tiny 1×1 grid centered there
    // Cell center at (359.5, 0.0), corners offset by ±0.5° in each direction
    const double dst_lon_center = 359.5;
    const double dst_lat_center = 0.0;
    const double half_dx = 0.5;

    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = dst_lon_center;
    dst_cy(0) = dst_lat_center;

    topology::StructuredGrid<MemSpace> dst_grid(1, 1, std::move(dst_cx), std::move(dst_cy), topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dst_crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> dst_cry("dst_cry", 4);
    dst_crx(0) = dst_lon_center - half_dx;
    dst_cry(0) = dst_lat_center - half_dx;
    dst_crx(1) = dst_lon_center + half_dx;
    dst_cry(1) = dst_lat_center - half_dx;
    dst_crx(2) = dst_lon_center - half_dx;
    dst_cry(2) = dst_lat_center + half_dx;
    dst_crx(3) = dst_lon_center + half_dx;
    dst_cry(3) = dst_lat_center + half_dx;
    dst_grid.set_corners(std::move(dst_crx), std::move(dst_cry));

    auto dst_mesh = dst_grid.to_unstructured();

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Error;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    ASSERT_EQ(matrix.n_dst(), 1u);
    ASSERT_GT(matrix.nnz(), 0u);

    // Get the entries for the single destination cell
    auto entries = get_row_entries(matrix, 0);
    ASSERT_EQ(entries.size(), 4u);

    // Verify that the source cells include both the last column (i=35) and
    // the first column (i=0) — wraparound
    // Source cells are indexed as j*36+i.
    // For the j index: lat_center=0, fi_lat = (0 - (-90))/10 - 0.5 = 8.5
    //   j = floor(8.5) = 8, j1 = 9
    // Expected columns: 8*36+35=323, 8*36+0=288, 9*36+35=359, 9*36+0=324
    bool has_last_col = false;
    bool has_first_col = false;
    for (const auto &e : entries) {
        std::size_t col = static_cast<std::size_t>(e.col);
        std::size_t i_col = col % 36;
        if (i_col == 35) has_last_col = true;
        if (i_col == 0) has_first_col = true;
    }
    EXPECT_TRUE(has_last_col) << "Wraparound should use the last longitude column (i=35)";
    EXPECT_TRUE(has_first_col) << "Wraparound should use the first longitude column (i=0)";

    // Verify partition of unity
    double wsum = 0.0;
    for (const auto &e : entries) wsum += e.weight;
    EXPECT_NEAR(wsum, 1.0, 1e-15);
}

// =============================================================================
// Test 3: Destination exactly at source cell center — weight = 1.0 on that cell
//
// Source: 4×4 grid [0, 40] × [0, 40], delta_lon=10, delta_lat=10.
//   Cell (1,1) center = (15, 15).
//
// Destination: single cell centered at (15, 15).
//   fi = (15 - 0) / 10 - 0.5 = 1.0 → i = floor(1.0) = 1, tx = 1.0 - 1 = 0.0
//   fj = (15 - 0) / 10 - 0.5 = 1.0 → j = floor(1.0) = 1, ty = 1.0 - 1 = 0.0
//   Weights: (1-0)*(1-0)=1.0, 0*anything=0, etc.
//   Only cell (1,1) = flat index 1 + 1*4 = 5 should have weight 1.0.
// =============================================================================
TEST(BilinearRect, DestAtSourceCellCenter) {
    auto src_mesh = make_regular_grid(4, 4, 0.0, 40.0, 0.0, 40.0);

    // Destination: single cell centered at (15, 15) = center of src cell (1,1)
    const double dst_lon = 15.0;
    const double dst_lat = 15.0;
    const double half_dx = 0.5;

    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = dst_lon;
    dst_cy(0) = dst_lat;

    topology::StructuredGrid<MemSpace> dst_grid(1, 1, std::move(dst_cx), std::move(dst_cy), topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dst_crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> dst_cry("dst_cry", 4);
    dst_crx(0) = dst_lon - half_dx;
    dst_cry(0) = dst_lat - half_dx;
    dst_crx(1) = dst_lon + half_dx;
    dst_cry(1) = dst_lat - half_dx;
    dst_crx(2) = dst_lon - half_dx;
    dst_cry(2) = dst_lat + half_dx;
    dst_crx(3) = dst_lon + half_dx;
    dst_cry(3) = dst_lat + half_dx;
    dst_grid.set_corners(std::move(dst_crx), std::move(dst_cry));

    auto dst_mesh = dst_grid.to_unstructured();

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    ASSERT_EQ(matrix.n_dst(), 1u);

    auto entries = get_row_entries(matrix, 0);
    ASSERT_EQ(entries.size(), 4u);

    // Source cell (1,1) has flat index = 1 + 1*4 = 5
    const index_t target_col = 5;

    for (const auto &e : entries) {
        if (e.col == target_col) {
            EXPECT_NEAR(e.weight, 1.0, 1e-12) << "Cell center weight should be 1.0";
        } else {
            EXPECT_NEAR(e.weight, 0.0, 1e-12) << "Non-center cell weight should be 0.0 at col " << e.col;
        }
    }
}

// =============================================================================
// Test 4: Destination exactly at grid corner (intersection of 4 cells) — all
// four weights = 0.25
//
// Source: 4×4 grid [0, 40] × [0, 40], delta_lon=10, delta_lat=10.
//   Corner between cells (1,1), (2,1), (1,2), (2,2) is at node (2,2) = (20, 20).
//   This point is exactly between cell centers (15,15), (25,15), (15,25), (25,25).
//
// Destination: single cell centered at (20, 20).
//   fi = (20 - 0) / 10 - 0.5 = 1.5 → i = floor(1.5) = 1, tx = 0.5
//   fj = (20 - 0) / 10 - 0.5 = 1.5 → j = floor(1.5) = 1, ty = 0.5
//   Weights: 0.25, 0.25, 0.25, 0.25
// =============================================================================
TEST(BilinearRect, DestAtGridCornerAllWeightsEqual) {
    auto src_mesh = make_regular_grid(4, 4, 0.0, 40.0, 0.0, 40.0);

    // Destination: single cell centered at (20, 20) = grid corner
    const double dst_lon = 20.0;
    const double dst_lat = 20.0;
    const double half_dx = 0.5;

    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = dst_lon;
    dst_cy(0) = dst_lat;

    topology::StructuredGrid<MemSpace> dst_grid(1, 1, std::move(dst_cx), std::move(dst_cy), topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dst_crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> dst_cry("dst_cry", 4);
    dst_crx(0) = dst_lon - half_dx;
    dst_cry(0) = dst_lat - half_dx;
    dst_crx(1) = dst_lon + half_dx;
    dst_cry(1) = dst_lat - half_dx;
    dst_crx(2) = dst_lon - half_dx;
    dst_cry(2) = dst_lat + half_dx;
    dst_crx(3) = dst_lon + half_dx;
    dst_cry(3) = dst_lat + half_dx;
    dst_grid.set_corners(std::move(dst_crx), std::move(dst_cry));

    auto dst_mesh = dst_grid.to_unstructured();

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    ASSERT_EQ(matrix.n_dst(), 1u);

    auto entries = get_row_entries(matrix, 0);
    ASSERT_EQ(entries.size(), 4u);

    double wsum = 0.0;
    for (const auto &e : entries) {
        EXPECT_NEAR(e.weight, 0.25, 1e-12) << "All four weights at grid corner should be 0.25, got " << e.weight << " at col " << e.col;
        wsum += e.weight;
    }
    EXPECT_NEAR(wsum, 1.0, 1e-15);
}

// =============================================================================
// Test 5: Non-regular grid pair fed to generate_bilinear() — verify BVH path
// is used transparently
//
// Feed non-uniform grids (varying cell spacing) and ensure correct results.
// The detector should return is_regular=false, triggering the BVH fallback.
// We just verify that the result is still valid (non-empty, correct dimensions).
// =============================================================================
TEST(BilinearRect, NonRegularGridUsesBVHFallback) {
    // Create a non-uniform source grid: 4×4 but with varying spacing
    const std::size_t ni = 4;
    const std::size_t nj = 4;

    // Non-uniform longitude positions: 0, 5, 20, 30, 40 (unequal spacing)
    std::vector<double> lon_bounds = {0.0, 5.0, 20.0, 30.0, 40.0};
    std::vector<double> lat_bounds = {0.0, 5.0, 20.0, 30.0, 40.0};

    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = (lon_bounds[i] + lon_bounds[i + 1]) * 0.5;
            cy(i + j * ni) = (lat_bounds[j] + lat_bounds[j + 1]) * 0.5;
        }
    }

    topology::StructuredGrid<MemSpace> src_grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);

    // Set corners for the non-uniform grid
    const std::size_t nc_lon = ni + 1;
    const std::size_t nc_lat = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_lon * nc_lat);
    Kokkos::View<double *, MemSpace> cry("cry", nc_lon * nc_lat);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_lon) = lon_bounds[i];
            cry(i + j * nc_lon) = lat_bounds[j];
        }
    }
    src_grid.set_corners(std::move(crx), std::move(cry));
    auto src_mesh = src_grid.to_unstructured();

    // Destination: a regular 2×2 grid that lies within the source domain
    auto dst_mesh = make_regular_grid(2, 2, 5.0, 30.0, 5.0, 30.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // BVH path should still produce valid results
    EXPECT_EQ(matrix.n_src(), ni * nj);
    EXPECT_EQ(matrix.n_dst(), 4u);
    EXPECT_GT(matrix.nnz(), 0u);

    // Verify partition of unity for each destination cell
    for (index_t r = 0; r < 4; ++r) {
        auto entries = get_row_entries(matrix, r);
        ASSERT_GT(entries.size(), 0u) << "Dst cell " << r << " has no entries";
        double wsum = 0.0;
        for (const auto &e : entries) {
            EXPECT_GE(e.weight, 0.0);
            EXPECT_LE(e.weight, 1.0);
            wsum += e.weight;
        }
        EXPECT_NEAR(wsum, 1.0, 1e-10) << "BVH fallback should still satisfy partition of unity at row " << r;
    }
}

// =============================================================================
// Test 6: Unmapped Error policy with out-of-bounds point on non-periodic grid
// — verify exception message contains cell index
//
// Source: 4×4 grid [10, 50] × [10, 50] (non-periodic, not 360°).
// Destination: single cell centered at (5, 5) — outside source bounds.
// =============================================================================
TEST(BilinearRect, UnmappedErrorPolicyThrowsWithCellIndex) {
    // Non-periodic source grid covering [10, 50] × [10, 50]
    auto src_mesh = make_regular_grid(4, 4, 10.0, 50.0, 10.0, 50.0);

    // Destination: single cell centered at (5, 5) — outside source domain
    const double dst_lon = 5.0;
    const double dst_lat = 5.0;
    const double half_dx = 1.0;

    Kokkos::View<double *, MemSpace> dst_cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> dst_cy("dst_cy", 1);
    dst_cx(0) = dst_lon;
    dst_cy(0) = dst_lat;

    topology::StructuredGrid<MemSpace> dst_grid(1, 1, std::move(dst_cx), std::move(dst_cy), topology::CoordinateSystem::SphericalDeg);

    Kokkos::View<double *, MemSpace> dst_crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> dst_cry("dst_cry", 4);
    dst_crx(0) = dst_lon - half_dx;
    dst_cry(0) = dst_lat - half_dx;
    dst_crx(1) = dst_lon + half_dx;
    dst_cry(1) = dst_lat - half_dx;
    dst_crx(2) = dst_lon - half_dx;
    dst_cry(2) = dst_lat + half_dx;
    dst_crx(3) = dst_lon + half_dx;
    dst_cry(3) = dst_lat + half_dx;
    dst_grid.set_corners(std::move(dst_crx), std::move(dst_cry));

    auto dst_mesh = dst_grid.to_unstructured();

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Error;

    // Should throw std::runtime_error with "0" (the unmapped cell index) in msg
    try {
        solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);
        FAIL() << "Expected std::runtime_error for unmapped out-of-bounds point";
    } catch (const std::runtime_error &e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("0"), std::string::npos) << "Error message should contain the cell index '0', got: " << msg;
    }
}

// =============================================================================
// Test 7: Unmapped Ignore policy with out-of-bounds point — verify zero-row
// in output matrix (no entries for that destination cell)
//
// Source: 4×4 grid [10, 50] × [10, 50] (non-periodic).
// Destination: 2×1 regular grid where one cell is inside and one is outside.
// The outside cell should have no entries in the matrix.
// =============================================================================
TEST(BilinearRect, UnmappedIgnorePolicyProducesZeroRow) {
    // Non-periodic source grid covering [10, 50] × [10, 50]
    auto src_mesh = make_regular_grid(4, 4, 10.0, 50.0, 10.0, 50.0);

    // Destination: 2×1 REGULAR grid covering [0, 20] × [25, 35]
    // This gives cell centers at (5, 30) and (15, 30).
    // Cell 0 at lon=5 is outside source [10, 50], cell 1 at lon=15 is inside.
    auto dst_mesh = make_regular_grid(2, 1, 0.0, 20.0, 25.0, 35.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    ASSERT_EQ(matrix.n_dst(), 2u);

    // Cell 0 (center at lon=5, outside source lon_min=10) should have no entries
    auto entries0 = get_row_entries(matrix, 0);
    EXPECT_EQ(entries0.size(), 0u) << "Out-of-bounds destination cell should have zero entries with Ignore policy";

    // Cell 1 (center at lon=15, inside) should have entries with valid weights
    auto entries1 = get_row_entries(matrix, 1);
    EXPECT_GT(entries1.size(), 0u) << "In-bounds destination cell should have weight entries";

    double wsum = 0.0;
    for (const auto &e : entries1) {
        EXPECT_GE(e.weight, 0.0);
        EXPECT_LE(e.weight, 1.0);
        wsum += e.weight;
    }
    EXPECT_NEAR(wsum, 1.0, 1e-12);
}

}  // namespace axis::test
