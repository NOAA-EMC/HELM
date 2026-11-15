// ─── Property-Based Tests: Conservative Weight Non-Negativity ────────────────
// Feature: helm-axis-microlibrary, Property 11: Conservative Weight Non-Negativity
//
// For any matrix generated with Conservative1stOrder, verify all factor_list
// values >= 0. Conservative weights represent area overlap fractions and must
// never be negative.
//
// **Validates: Requirements 8.2**
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

#include "common/test_kokkos_env.hpp"

namespace {

/// Build a simple ni x nj regular-grid UnstructuredMesh on HostSpace.
axis::topology::UnstructuredMesh<Kokkos::HostSpace> build_regular_mesh(std::size_t ni, std::size_t nj, double lon_start, double lat_start,
                                                                       double dlon, double dlat) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, Kokkos::HostSpace> center_lon("clon", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> center_lat("clat", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lon("crlon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> corner_lat("crlat", n_corners);

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

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, center_lon, center_lat, axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);

    return grid.to_unstructured();
}

// ─── Property 11a: Conservative DstArea weights are non-negative ─────────────

RC_GTEST_PROP(PropConservativeNonneg, DstAreaWeightsNonNegative, ()) {
    const auto src_ni = *rc::gen::inRange<std::size_t>(2, 8);
    const auto src_nj = *rc::gen::inRange<std::size_t>(2, 8);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 8);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 8);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, 2.0, 2.0);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, 2.0, 2.0);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::DstArea;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    const auto nnz = matrix.nnz();
    auto factor_list = matrix.factor_list();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_list[k] >= 0.0);
    }
}

// ─── Property 11b: Conservative FracArea weights are non-negative ────────────

RC_GTEST_PROP(PropConservativeNonneg, FracAreaWeightsNonNegative, ()) {
    const auto src_ni = *rc::gen::inRange<std::size_t>(2, 8);
    const auto src_nj = *rc::gen::inRange<std::size_t>(2, 8);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(2, 8);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(2, 8);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, 2.0, 2.0);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, 2.0, 2.0);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::FracArea;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    const auto nnz = matrix.nnz();
    auto factor_list = matrix.factor_list();

    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(factor_list[k] >= 0.0);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
