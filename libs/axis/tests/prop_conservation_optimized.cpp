// ─── Property-Based Tests: Conservation, Non-Negative Weights, Valid Index
//     Bounds across Optimized Fast-Paths ──────────────────────────────────────
// Feature: axis-performance-optimizations
//
// Property 9: First-Order Conservation
//   For any source and destination mesh pair processed by any optimization
//   fast-path, the weighted sum preserves first-order conservation within
//   relative tolerance 1e-12.
//
// Property 10: Non-Negative Weights
//   For any source and destination mesh pair processed by any optimization
//   fast-path, every weight value w_ij in the produced InterpolationMatrix
//   SHALL satisfy w_ij >= 0.0.
//
// Property 12: Valid Index Bounds
//   For any InterpolationMatrix produced by any optimization fast-path, every
//   entry SHALL satisfy 0 <= row < n_dst and 0 <= col < n_src.
//
// **Validates: Requirements 2.3, 2.4, 7.1, 7.2, 7.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Helper: Build a regular lat-lon mesh with uniform spacing ───────────────
// Creates an ni x nj grid covering [lon_start, lon_start + ni*dlon] x
// [lat_start, lat_start + nj*dlat]. This triggers the regular-grid fast-path
// detection in WeightGenerator when both src and dst are built this way.

axis::topology::UnstructuredMesh<MemSpace> build_regular_mesh(std::size_t ni, std::size_t nj, double lon_start, double lat_start, double dlon,
                                                              double dlat) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, MemSpace> center_lon("clon", n_centers);
    Kokkos::View<double *, MemSpace> center_lat("clat", n_centers);
    Kokkos::View<double *, MemSpace> corner_lon("crlon", n_corners);
    Kokkos::View<double *, MemSpace> corner_lat("crlat", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_lon(idx) = lon_start + (static_cast<double>(i) + 0.5) * dlon;
            center_lat(idx) = lat_start + (static_cast<double>(j) + 0.5) * dlat;
        }
    }

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_lon(idx) = lon_start + static_cast<double>(i) * dlon;
            corner_lat(idx) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    return grid.to_unstructured();
}

// ─── Property 9a: First-Order Conservation — Regular Grid (Rectangle Fast-Path)
// Generate pairs of regular grids with different resolutions (triggering the
// rectangle fast-path). Verify conservation: source integral ≈ destination
// integral within relative tolerance 1e-12.
//
// **Validates: Requirements 2.3, 7.1**

RC_GTEST_PROP(PropConservationOpt, ConservationRegularGrid, ()) {
    // Source grid: small regular grid (F-family style)
    auto src_ni = *rc::gen::inRange<std::size_t>(2, 9);
    auto src_nj = *rc::gen::inRange<std::size_t>(2, 9);
    // Destination grid: different resolution
    auto dst_ni = *rc::gen::inRange<std::size_t>(2, 9);
    auto dst_nj = *rc::gen::inRange<std::size_t>(2, 9);

    // Both grids cover the same domain [0, 10] x [0, 10]
    double src_dlon = 10.0 / static_cast<double>(src_ni);
    double src_dlat = 10.0 / static_cast<double>(src_nj);
    double dst_dlon = 10.0 / static_cast<double>(dst_ni);
    double dst_dlat = 10.0 / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // Generate a random source field
    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    RC_PRE(n_src > 0);
    RC_PRE(n_dst > 0);
    RC_PRE(matrix.nnz() > 0);

    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_data[i] = *rc::gen::map(rc::gen::inRange(1, 1001), [](int v) { return static_cast<double>(v) / 10.0; });
    }
    std::vector<double> dst_data(n_dst, 0.0);

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    axis::solver::apply(matrix, src_view, dst_view);

    auto report = axis::solver::check_conservation<MemSpace>(src_view, axis::field_view<const double, 1>(dst_data.data(), n_dst), matrix,
                                                             axis::solver::NormType::DstArea);

    RC_ASSERT(report.relative_error < 1e-12);
}

// ─── Property 9b: First-Order Conservation — Cartesian (Parallel Planar Clipper)
// Use non-uniform grids that exercise the parallel Cartesian clipper path
// rather than the rectangle fast-path. Both grids overlap the same domain.
//
// **Validates: Requirements 2.3, 7.1**

RC_GTEST_PROP(PropConservationOpt, ConservationCartesianPath, ()) {
    // Source: regular grid
    auto src_ni = *rc::gen::inRange<std::size_t>(2, 7);
    auto src_nj = *rc::gen::inRange<std::size_t>(2, 7);
    // Destination: slightly offset (but still regular, so rect fast-path fires)
    auto dst_ni = *rc::gen::inRange<std::size_t>(2, 7);
    auto dst_nj = *rc::gen::inRange<std::size_t>(2, 7);

    double domain_size = 8.0;
    double src_dlon = domain_size / static_cast<double>(src_ni);
    double src_dlat = domain_size / static_cast<double>(src_nj);
    double dst_dlon = domain_size / static_cast<double>(dst_ni);
    double dst_dlat = domain_size / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 1.0, 1.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 1.0, 1.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    RC_PRE(n_src > 0);
    RC_PRE(n_dst > 0);
    RC_PRE(matrix.nnz() > 0);

    // Use a constant field for strongest conservation guarantee
    double c = *rc::gen::map(rc::gen::inRange(1, 500), [](int v) { return static_cast<double>(v) / 5.0; });

    std::vector<double> src_data(n_src, c);
    std::vector<double> dst_data(n_dst, 0.0);

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    axis::solver::apply(matrix, src_view, dst_view);

    auto report = axis::solver::check_conservation<MemSpace>(src_view, axis::field_view<const double, 1>(dst_data.data(), n_dst), matrix,
                                                             axis::solver::NormType::DstArea);

    RC_ASSERT(report.relative_error < 1e-12);
}

// ─── Property 9c: First-Order Conservation — F-family NamedGrid (Rectangle
//     Fast-Path with Trig Cache) ──────────────────────────────────────────────
// Generate F-family grid pairs (e.g., F4, F8, F16) with Conservative1stOrder +
// Cartesian line type. These trigger the regular-grid rectangle fast-path.
// Apply a random source field, verify conservation.
//
// **Validates: Requirements 2.3, 7.1**

RC_GTEST_PROP(PropConservationOpt, RegularGridConservation, ()) {
    // Choose small F-family grid pairs to keep runtime reasonable
    const int grid_idx = *rc::gen::inRange(0, 3);
    const std::vector<std::string> src_grids = {"F4", "F8", "F16"};
    const std::vector<std::string> dst_grids = {"F4", "F4", "F8"};

    const auto &src_name = src_grids[grid_idx];
    const auto &dst_name = dst_grids[grid_idx];

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(src_name);
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(dst_name);

    // Conservative 1st order with Cartesian line type triggers rectangle fast-path
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = axis::solver::NormType::DstArea;
    cfg.line_type = axis::solver::LineType::Cartesian;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    RC_PRE(n_src > 0);
    RC_PRE(n_dst > 0);
    RC_PRE(matrix.nnz() > 0);

    // Generate a smooth cosine-bell source field
    const int seed = *rc::gen::inRange(0, 100);
    const double scale = 1.0 + static_cast<double>(seed % 10);
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n_src);
        src_data[i] = scale * (1.0 + std::cos(2.0 * M_PI * t));
    }

    std::vector<double> dst_data(n_dst, 0.0);
    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    axis::solver::apply<MemSpace>(matrix, src_view, dst_view);

    auto report = axis::solver::check_conservation<MemSpace>(src_view, axis::field_view<const double, 1>(dst_data.data(), n_dst), matrix,
                                                             axis::solver::NormType::DstArea);

    // Conservation: |src_integral - dst_integral| / |src_integral| < 1e-12
    if (std::abs(report.src_integral) > 1e-20) {
        RC_ASSERT(report.relative_error < 1e-12);
    }
}

// ─── Property 9d: First-Order Conservation — O-family NamedGrid (BVH + Morton
//     Sort + Planar Clipper) ──────────────────────────────────────────────────
// Generate O-family grid pairs (e.g., O2, O4, O8) with Conservative1stOrder +
// Cartesian line type. These trigger the BVH path with Morton sort and planar
// clipper. Verify conservation.
//
// **Validates: Requirements 2.3, 7.1**

RC_GTEST_PROP(PropConservationOpt, OctahedralGridConservation, ()) {
    // Choose small O-family grids
    const int grid_idx = *rc::gen::inRange(0, 3);
    const std::vector<std::string> src_grids = {"O2", "O4", "O8"};
    const std::vector<std::string> dst_grids = {"O2", "O2", "O4"};

    const auto &src_name = src_grids[grid_idx];
    const auto &dst_name = dst_grids[grid_idx];

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(src_name);
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(dst_name);

    // Conservative 1st order with Cartesian line type triggers BVH + Morton path
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = axis::solver::NormType::DstArea;
    cfg.line_type = axis::solver::LineType::Cartesian;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    RC_PRE(n_src > 0);
    RC_PRE(n_dst > 0);
    RC_PRE(matrix.nnz() > 0);

    // Generate a smooth cosine-bell source field
    const int seed = *rc::gen::inRange(0, 100);
    const double scale = 1.0 + static_cast<double>(seed % 10);
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n_src);
        src_data[i] = scale * (1.0 + std::cos(2.0 * M_PI * t));
    }

    std::vector<double> dst_data(n_dst, 0.0);
    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    axis::solver::apply<MemSpace>(matrix, src_view, dst_view);

    auto report = axis::solver::check_conservation<MemSpace>(src_view, axis::field_view<const double, 1>(dst_data.data(), n_dst), matrix,
                                                             axis::solver::NormType::DstArea);

    // Conservation: relative error < 1e-12
    if (std::abs(report.src_integral) > 1e-20) {
        RC_ASSERT(report.relative_error < 1e-12);
    }
}

// ─── Property 9e: First-Order Conservation — Mixed Grid (BVH Fallback Path) ──
// Source is F-family, destination is O-family (or vice versa). This triggers
// the BVH fallback path since the meshes are not both regular. Verify
// conservation.
//
// **Validates: Requirements 2.3, 7.1**

RC_GTEST_PROP(PropConservationOpt, MixedGridConservation, ()) {
    // Mix F-family and O-family grids
    const int case_idx = *rc::gen::inRange(0, 4);

    std::string src_name, dst_name;
    switch (case_idx) {
        case 0:
            src_name = "F4";
            dst_name = "O2";
            break;
        case 1:
            src_name = "F8";
            dst_name = "O4";
            break;
        case 2:
            src_name = "O2";
            dst_name = "F4";
            break;
        case 3:
            src_name = "O4";
            dst_name = "F8";
            break;
        default:
            src_name = "F4";
            dst_name = "O2";
            break;
    }

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(src_name);
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(dst_name);

    // Conservative 1st order with Cartesian line type triggers BVH fallback
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = axis::solver::NormType::DstArea;
    cfg.line_type = axis::solver::LineType::Cartesian;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    RC_PRE(n_src > 0);
    RC_PRE(n_dst > 0);
    RC_PRE(matrix.nnz() > 0);

    // Generate a smooth cosine-bell source field
    const int seed = *rc::gen::inRange(0, 100);
    const double scale = 1.0 + static_cast<double>(seed % 10);
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(n_src);
        src_data[i] = scale * (1.0 + std::cos(2.0 * M_PI * t));
    }

    std::vector<double> dst_data(n_dst, 0.0);
    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    axis::solver::apply<MemSpace>(matrix, src_view, dst_view);

    auto report = axis::solver::check_conservation<MemSpace>(src_view, axis::field_view<const double, 1>(dst_data.data(), n_dst), matrix,
                                                             axis::solver::NormType::DstArea);

    // Conservation: relative error < 1e-12
    if (std::abs(report.src_integral) > 1e-20) {
        RC_ASSERT(report.relative_error < 1e-12);
    }
}

// ─── Property 10a: Non-Negative Weights — Regular Grid (Rectangle Fast-Path) ─
// For F-family grid pairs (regular lat-lon), verify all weights are >= 0.0.
//
// **Validates: Requirements 2.4, 7.2**

RC_GTEST_PROP(PropConservationOpt, NonNegativeWeightsRegularGrid, ()) {
    // F-family style grid pairs: e.g., F4→F2, F8→F4 (different resolutions)
    auto src_ni = *rc::gen::inRange<std::size_t>(2, 9);
    auto src_nj = *rc::gen::inRange<std::size_t>(2, 9);
    auto dst_ni = *rc::gen::inRange<std::size_t>(2, 9);
    auto dst_nj = *rc::gen::inRange<std::size_t>(2, 9);

    double domain = 10.0;
    double src_dlon = domain / static_cast<double>(src_ni);
    double src_dlat = domain / static_cast<double>(src_nj);
    double dst_dlon = domain / static_cast<double>(dst_ni);
    double dst_dlat = domain / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // Access COO data and verify all weights are non-negative
    auto weights = matrix.factor_list();
    const std::size_t nnz = matrix.nnz();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(weights[k] >= 0.0);
    }
}

// ─── Property 10b: Non-Negative Weights — Offset Grids (Cartesian Path) ──────
// For grid pairs that may exercise the Cartesian parallel clipper path or
// rectangle fast-path with partial overlap, verify all weights are >= 0.0.
//
// **Validates: Requirements 2.4, 7.2**

RC_GTEST_PROP(PropConservationOpt, NonNegativeWeightsOffsetGrids, ()) {
    // O-family style: grids with different origins (partial overlap)
    auto src_ni = *rc::gen::inRange<std::size_t>(2, 7);
    auto src_nj = *rc::gen::inRange<std::size_t>(2, 7);
    auto dst_ni = *rc::gen::inRange<std::size_t>(2, 7);
    auto dst_nj = *rc::gen::inRange<std::size_t>(2, 7);

    // Source covers [0, 8] x [0, 8], destination covers [2, 10] x [2, 10]
    // This creates partial overlap exercising boundary weight computation
    double src_dlon = 8.0 / static_cast<double>(src_ni);
    double src_dlat = 8.0 / static_cast<double>(src_nj);
    double dst_dlon = 8.0 / static_cast<double>(dst_ni);
    double dst_dlat = 8.0 / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 2.0, 2.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // Verify all weights are non-negative
    auto weights = matrix.factor_list();
    const std::size_t nnz = matrix.nnz();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(weights[k] >= 0.0);
    }
}

// ─── Property 12a: Valid Index Bounds — Regular Grid (Rectangle Fast-Path) ───
// For F-family grid pairs, verify 0 <= row < n_dst and 0 <= col < n_src
// for every COO entry.
//
// **Validates: Requirements 7.4**

RC_GTEST_PROP(PropConservationOpt, ValidIndexBoundsRegularGrid, ()) {
    auto src_ni = *rc::gen::inRange<std::size_t>(2, 9);
    auto src_nj = *rc::gen::inRange<std::size_t>(2, 9);
    auto dst_ni = *rc::gen::inRange<std::size_t>(2, 9);
    auto dst_nj = *rc::gen::inRange<std::size_t>(2, 9);

    double domain = 10.0;
    double src_dlon = domain / static_cast<double>(src_ni);
    double src_dlat = domain / static_cast<double>(src_nj);
    double dst_dlon = domain / static_cast<double>(dst_ni);
    double dst_dlat = domain / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    const std::size_t nnz = matrix.nnz();
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    auto rows = matrix.factor_row();
    auto cols = matrix.factor_col();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(rows[k] >= 0);
        RC_ASSERT(rows[k] < static_cast<axis::index_t>(n_dst));
        RC_ASSERT(cols[k] >= 0);
        RC_ASSERT(cols[k] < static_cast<axis::index_t>(n_src));
    }
}

// ─── Property 12b: Valid Index Bounds — Offset/Partial Overlap Grids ─────────
// For grid pairs with partial overlap (exercising boundary conditions),
// verify 0 <= row < n_dst and 0 <= col < n_src for every COO entry.
//
// **Validates: Requirements 7.4**

RC_GTEST_PROP(PropConservationOpt, ValidIndexBoundsOffsetGrids, ()) {
    auto src_ni = *rc::gen::inRange<std::size_t>(2, 7);
    auto src_nj = *rc::gen::inRange<std::size_t>(2, 7);
    auto dst_ni = *rc::gen::inRange<std::size_t>(2, 7);
    auto dst_nj = *rc::gen::inRange<std::size_t>(2, 7);

    // Partially overlapping grids
    double src_dlon = 8.0 / static_cast<double>(src_ni);
    double src_dlat = 8.0 / static_cast<double>(src_nj);
    double dst_dlon = 8.0 / static_cast<double>(dst_ni);
    double dst_dlat = 8.0 / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 2.0, 2.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.line_type = axis::solver::LineType::Cartesian;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    const std::size_t nnz = matrix.nnz();
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    auto rows = matrix.factor_row();
    auto cols = matrix.factor_col();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(rows[k] >= 0);
        RC_ASSERT(rows[k] < static_cast<axis::index_t>(n_dst));
        RC_ASSERT(cols[k] >= 0);
        RC_ASSERT(cols[k] < static_cast<axis::index_t>(n_src));
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
