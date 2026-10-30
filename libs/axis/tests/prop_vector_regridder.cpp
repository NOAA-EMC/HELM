// ─── Property-Based Tests: Vector Regridder Coupled Rotation ──────────────────
// Feature: helm-axis-microlibrary, Property: Vector Regridder Coupled Rotation
//
// Uses RapidCheck to verify that the VectorWeightGenerator correctly computes
// coupled weights that perform spatial interpolation and local coordinate frame
// rotation simultaneously.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cmath>

#include "common/test_kokkos_env.hpp"

namespace {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

/// Generate valid grid dimensions for structured grids
rc::Gen<std::size_t> genGridDim() {
    return rc::gen::inRange<std::size_t>(2, 6);
}

/// Generate a valid rotation angle in radians [-pi, pi] by scaling an integer
rc::Gen<double> genAngle() {
    return rc::gen::map(rc::gen::inRange(-3141, 3142), [](int v) { return static_cast<double>(v) / 1000.0; });
}

/// Build a simple structured grid
axis::topology::UnstructuredMesh<MemSpace> build_mesh(std::size_t ni, std::size_t nj, double x_start, double y_start, double dx, double dy) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, MemSpace> center_x("cx", n_centers);
    Kokkos::View<double *, MemSpace> center_y("cy", n_centers);
    Kokkos::View<double *, MemSpace> corner_x("crx", n_corners);
    Kokkos::View<double *, MemSpace> corner_y("cry", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_x(idx) = x_start + (static_cast<double>(i) + 0.5) * dx;
            center_y(idx) = y_start + (static_cast<double>(j) + 0.5) * dy;
        }
    }

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_x(idx) = x_start + static_cast<double>(i) * dx;
            corner_y(idx) = y_start + static_cast<double>(j) * dy;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, center_x, center_y, axis::topology::CoordinateSystem::Cartesian3D);
    grid.set_corners(corner_x, corner_y);

    return grid.to_unstructured();
}

RC_GTEST_PROP(PropVectorRegridder, CoupledRotationAndInterpolation, ()) {
    const std::size_t ni = *genGridDim();
    const std::size_t nj = *genGridDim();
    const std::size_t n_points = ni * nj;

    // Use a fixed domain size
    const double domain_size = 4.0;
    const double dx = domain_size / static_cast<double>(ni);
    const double dy = domain_size / static_cast<double>(nj);

    auto src_mesh = build_mesh(ni, nj, 0.0, 0.0, dx, dy);

    // Generate randomized rotation angles for source and destination
    auto src_angles = *rc::gen::container<std::vector<double>>(n_points, genAngle());
    auto dst_angles = *rc::gen::container<std::vector<double>>(n_points, genAngle());

    Kokkos::View<double *, MemSpace> src_alpha("src_alpha", n_points);
    Kokkos::View<double *, MemSpace> dst_alpha("dst_alpha", n_points);
    for (std::size_t i = 0; i < n_points; ++i) {
        src_alpha(i) = src_angles[i];
        dst_alpha(i) = dst_angles[i];
    }

    GridRotation<MemSpace> src_rot{src_alpha};
    GridRotation<MemSpace> dst_rot{dst_alpha};

    RegridConfig config;
    config.method = InterpolationMethod::Bilinear;
    config.unmapped = UnmappedAction::Ignore;

    // Generate coupled vector weights (regridding the same mesh onto itself)
    auto [W_u, W_v] = VectorWeightGenerator<MemSpace>::generate(src_mesh, src_mesh, src_rot, dst_rot, config);

    // Verify matrix dimensions
    RC_ASSERT(W_u.n_src() == n_points * 2);
    RC_ASSERT(W_u.n_dst() == n_points);
    RC_ASSERT(W_v.n_src() == n_points * 2);
    RC_ASSERT(W_v.n_dst() == n_points);

    // Set source field to a randomized unit vector (u_src, v_src) per cell
    // We want to verify that for a given cell, the applied weights correctly rotate the vector.
    // For identity spatial mapping, W_ji is non-zero only when j == i, and its value is 1.0.
    // Therefore, the applied vector should be rotated by (dst_alpha[j] - src_alpha[j]).
    Kokkos::View<double *, MemSpace> src_uv("src_uv", n_points * 2);
    auto vec_angles = *rc::gen::container<std::vector<double>>(n_points, genAngle());
    for (std::size_t i = 0; i < n_points; ++i) {
        src_uv(i) = std::cos(vec_angles[i]);             // u_src
        src_uv(i + n_points) = std::sin(vec_angles[i]);  // v_src
    }

    Kokkos::View<double *, MemSpace> dst_u("dst_u", n_points);
    Kokkos::View<double *, MemSpace> dst_v("dst_v", n_points);

    axis::field_view<const double, 1> src_uv_view(src_uv.data(), src_uv.extent(0));
    axis::field_view<double, 1> dst_u_view(dst_u.data(), dst_u.extent(0));
    axis::field_view<double, 1> dst_v_view(dst_v.data(), dst_v.extent(0));

    apply(W_u, src_uv_view, dst_u_view);
    apply(W_v, src_uv_view, dst_v_view);

    for (std::size_t idx = 0; idx < n_points; ++idx) {
        double u_val = src_uv(idx);
        double v_val = src_uv(idx + n_points);

        double diff_alpha = dst_alpha(idx) - src_alpha(idx);
        double cos_d = std::cos(diff_alpha);
        double sin_d = std::sin(diff_alpha);

        double expected_u = u_val * cos_d + v_val * sin_d;
        double expected_v = -u_val * sin_d + v_val * cos_d;

        RC_ASSERT(std::abs(dst_u(idx) - expected_u) < 1e-11);
        RC_ASSERT(std::abs(dst_v(idx) - expected_v) < 1e-11);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
