// ─── Property-Based Tests: FracArea Normalization ────────────────────────────
// Feature: helm-axis-microlibrary, Property 16: FracArea Normalization Bakes
//          Fraction In
//
// For FracArea matrices, verify that apply produces the true interpolated
// value directly without needing division by frac_b. When the source field is
// constant c, apply should yield c at all covered cells.
//
// **Validates: Requirements 10.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
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

// ─── Property 16: FracArea apply yields true value directly ──────────────────
// For a constant field c, FracArea normalization bakes the fraction into the
// weights, so apply directly yields c at covered cells without division.
//
// **Validates: Requirements 10.3**

RC_GTEST_PROP(PropFracareaNorm, ApplyYieldsTrueValueDirectly, ()) {
    const auto src_ni = *rc::gen::inRange<std::size_t>(3, 7);
    const auto src_nj = *rc::gen::inRange<std::size_t>(3, 7);
    const auto dst_ni = *rc::gen::inRange<std::size_t>(3, 7);
    const auto dst_nj = *rc::gen::inRange<std::size_t>(3, 7);

    // Both grids cover [0, 10] x [0, 10]
    double src_dlon = 10.0 / static_cast<double>(src_ni);
    double src_dlat = 10.0 / static_cast<double>(src_nj);
    double dst_dlon = 10.0 / static_cast<double>(dst_ni);
    double dst_dlat = 10.0 / static_cast<double>(dst_nj);

    auto src_mesh = build_regular_mesh(src_ni, src_nj, 0.0, 0.0, src_dlon, src_dlat);
    auto dst_mesh = build_regular_mesh(dst_ni, dst_nj, 0.0, 0.0, dst_dlon, dst_dlat);

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    config.norm_type = axis::solver::NormType::FracArea;
    config.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

    // Use a constant source field
    const double c = *rc::gen::map(rc::gen::inRange(1, 1001), [](int v) { return static_cast<double>(v) / 10.0; });

    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();

    std::vector<double> src_data(n_src, c);
    std::vector<double> dst_data(n_dst, 0.0);

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    // Apply FracArea weights — should produce true value directly
    axis::solver::apply(matrix, src_view, dst_view);

    // Verify: at covered cells, dst ≈ c without any division by frac_b
    auto frac_b = matrix.frac_b();
    const double tol = 1e-10 * c;
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (frac_b[j] > 0.99) {
            // Fully covered cells should have the true value directly
            RC_ASSERT(std::abs(dst_data[j] - c) < tol + 1e-12);
        }
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
