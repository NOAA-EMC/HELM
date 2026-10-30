// ─── Property-Based Tests: Sparse Matrix Index Bounds ────────────────────────
// Feature: helm-axis-microlibrary, Property 10: Sparse Matrix Index Bounds
//
// For any generated InterpolationMatrix, verify all factor_col[k] in [0, n_src)
// and factor_row[k] in [0, n_dst). Tests Bilinear and NearestNeighbor methods
// on small randomly-sized mesh pairs.
//
// **Validates: Requirements 8.4**
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

/// Build a simple ni x nj regular-grid UnstructuredMesh on HostSpace.
/// The grid covers [lon_start, lon_start + ni*dlon] x [lat_start, lat_start + nj*dlat].
axis::topology::UnstructuredMesh<Kokkos::HostSpace> build_regular_mesh(std::size_t ni, std::size_t nj, double lon_start, double lat_start,
                                                                       double dlon, double dlat) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("clon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("clat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("crlon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("crlat", n_corners);

    // Fill centers
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_lon(idx) = lon_start + (static_cast<double>(i) + 0.5) * dlon;
            center_lat(idx) = lat_start + (static_cast<double>(j) + 0.5) * dlat;
        }
    }

    // Fill corners
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_lon(idx) = lon_start + static_cast<double>(i) * dlon;
            corner_lat(idx) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    return grid.to_unstructured();
}

// ─── Property 10a: Bilinear — all indices within bounds ──────────────────────

RC_GTEST_PROP(PropSparseMatrixBounds, BilinearIndicesInBounds, ()) {
    // Generate small mesh dimensions [2, 8]
    const auto src_ni = *rc::gen::inRange<std::size_t>(2, 9);
    const auto src_nj = *rc::gen::inRange<std::size_t>(2, 9);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 9);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 9);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, 1.0, 1.0);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, 1.0, 1.0);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Bilinear;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    const auto nnz = matrix.nnz();

    auto factor_col = matrix.factor_col();
    auto factor_row = matrix.factor_row();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_col[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_col[k]) < n_src);
        RC_ASSERT(factor_row[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_row[k]) < n_dst);
    }
}

// ─── Property 10b: NearestNeighbor — all indices within bounds ───────────────

RC_GTEST_PROP(PropSparseMatrixBounds, NearestNeighborIndicesInBounds, ()) {
    const auto src_ni = *rc::gen::inRange<std::size_t>(2, 9);
    const auto src_nj = *rc::gen::inRange<std::size_t>(2, 9);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 9);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 9);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, 1.0, 1.0);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, 1.0, 1.0);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    const auto nnz = matrix.nnz();

    auto factor_col = matrix.factor_col();
    auto factor_row = matrix.factor_row();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_col[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_col[k]) < n_src);
        RC_ASSERT(factor_row[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_row[k]) < n_dst);
    }
}

// ─── Property 10c: Conservative — all indices within bounds ──────────────────

RC_GTEST_PROP(PropSparseMatrixBounds, ConservativeIndicesInBounds, ()) {
    const auto src_ni = *rc::gen::inRange<std::size_t>(2, 7);
    const auto src_nj = *rc::gen::inRange<std::size_t>(2, 7);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 7);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 7);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, 2.0, 2.0);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, 2.0, 2.0);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();
    const auto nnz = matrix.nnz();

    auto factor_col = matrix.factor_col();
    auto factor_row = matrix.factor_row();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_col[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_col[k]) < n_src);
        RC_ASSERT(factor_row[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(factor_row[k]) < n_dst);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
