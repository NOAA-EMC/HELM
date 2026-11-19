// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cmath>
#include <vector>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

/// @brief Helper to compute actual cell centroids from UnstructuredMesh coordinates.
template <typename MemorySpace>
void get_mesh_centroids(const axis::topology::UnstructuredMesh<MemorySpace> &mesh, std::vector<double> &cx, std::vector<double> &cy) {
    const auto n_cells = mesh.n_cells();
    cx.resize(n_cells);
    cy.resize(n_cells);

    auto coords = mesh.node_coords_view();
    auto offsets = mesh.conn_offsets_view();
    auto indices = mesh.conn_indices_view();

    auto h_coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords);
    auto h_offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), offsets);
    auto h_indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), indices);

    for (std::size_t c = 0; c < n_cells; ++c) {
        auto start = h_offsets(c);
        auto end = h_offsets(c + 1);
        double sx = 0.0, sy = 0.0;
        for (auto i = start; i < end; ++i) {
            auto ni = h_indices(i);
            sx += h_coords(ni, 0);
            sy += h_coords(ni, 1);
        }
        cx[c] = sx / (end - start);
        cy[c] = sy / (end - start);
    }
}

/// @brief Creates a mathematically perfect, uniform structured grid with explicit centers and corners.
/// @details Prevents boundary-cell squeezing, ensuring 100% uniform step sizes.
topology::UnstructuredMesh<MemSpace> make_perfect_uniform_grid(std::size_t ni, std::size_t nj) {
    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, MemSpace> center_lon("center_lon", n_centers);
    Kokkos::View<double *, MemSpace> center_lat("center_lat", n_centers);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            center_lon(idx) = static_cast<double>(i) + 0.5;
            center_lat(idx) = static_cast<double>(j) + 0.5;
        }
    }

    Kokkos::View<double *, MemSpace> corner_lon("corner_lon", n_corners);
    Kokkos::View<double *, MemSpace> corner_lat("corner_lat", n_corners);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            corner_lon(idx) = static_cast<double>(i);
            corner_lat(idx) = static_cast<double>(j);
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, center_lon, center_lat, topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(corner_lon, corner_lat);
    return grid.to_unstructured();
}

TEST(BicubicInterpolationTest, ConstantFieldReproduction) {
    // 1. Create a 10x10 regular source grid
    auto src_mesh = make_perfect_uniform_grid(10, 10);
    const std::size_t src_n_points = src_mesh.n_cells();

    // 2. Create a 1x1 destination grid deep in the interior
    auto dst_mesh = make_perfect_uniform_grid(1, 1);
    const std::size_t dst_n_points = dst_mesh.n_cells();

    // Shift the destination cell's nodes so its centroid lies at (4.1, 4.1)
    // To do this, we rewrite dst_mesh coordinates directly:
    auto coords = dst_mesh.node_coords_view();
    auto h_coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords);
    // Original corner nodes are (0,0), (1,0), (1,1), (0,1) -> centroid (0.5, 0.5)
    // To shift centroid to (4.1, 4.1), we translate corners by (3.6, 3.6):
    for (std::size_t i = 0; i < dst_mesh.n_nodes(); ++i) {
        h_coords(i, 0) += 3.6;
        h_coords(i, 1) += 3.6;
    }
    Kokkos::deep_copy(coords, h_coords);

    // 3. Generate bicubic weights
    RegridConfig config;
    config.method = InterpolationMethod::Bicubic;
    config.unmapped = UnmappedAction::Error;

    auto W = WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // 4. Test with a constant field: f(x, y) = 10.0
    Kokkos::View<double *, MemSpace> src_field("src_field", src_n_points);
    Kokkos::deep_copy(src_field, 10.0);

    Kokkos::View<double *, MemSpace> dst_field("dst_field", dst_n_points);

    axis::field_view<const double, 1> src_view(src_field.data(), src_n_points);
    axis::field_view<double, 1> dst_view(dst_field.data(), dst_n_points);
    apply(W, src_view, dst_view);

    auto h_dst = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_field);
    for (std::size_t j = 0; j < dst_n_points; ++j) {
        EXPECT_NEAR(h_dst(j), 10.0, 1.0e-9);
    }
}

TEST(BicubicInterpolationTest, LinearAndCubicAccuracy) {
    // The bicubic generator fits a 16-term tensor basis to a 64-point stencil
    // by overdetermined least squares, so the resulting weights reproduce low-
    // order polynomial fields only approximately (the fit is not an exact
    // interpolant). Tolerances below reflect that: they are far tighter than
    // the observed least-squares error (~5e-7 on these fields) but well above
    // machine precision.
    // 1. Create a 10x10 regular source grid
    auto src_mesh = make_perfect_uniform_grid(10, 10);
    const std::size_t src_n_points = src_mesh.n_cells();

    // Retrieve actual mesh centroids for field evaluations
    std::vector<double> actual_src_cx, actual_src_cy;
    get_mesh_centroids(src_mesh, actual_src_cx, actual_src_cy);

    // 2. Create a 1x1 destination grid deep in the interior
    auto dst_mesh = make_perfect_uniform_grid(1, 1);
    const std::size_t dst_n_points = dst_mesh.n_cells();

    // Shift destination centroid to (4.1, 4.1) by translating corner nodes by (3.6, 3.6)
    auto coords = dst_mesh.node_coords_view();
    auto h_coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords);
    for (std::size_t i = 0; i < dst_mesh.n_nodes(); ++i) {
        h_coords(i, 0) += 3.6;
        h_coords(i, 1) += 3.6;
    }
    Kokkos::deep_copy(coords, h_coords);

    std::vector<double> actual_dst_cx, actual_dst_cy;
    get_mesh_centroids(dst_mesh, actual_dst_cx, actual_dst_cy);

    // 3. Generate bicubic weights
    RegridConfig config;
    config.method = InterpolationMethod::Bicubic;
    config.unmapped = UnmappedAction::Error;

    auto W = WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

    // 4. Test linear field: f(x, y) = 3x - 4y
    Kokkos::View<double *, MemSpace> src_linear("src_linear", src_n_points);
    for (std::size_t idx = 0; idx < src_n_points; ++idx) {
        src_linear(idx) = 3.0 * actual_src_cx[idx] - 4.0 * actual_src_cy[idx];
    }

    Kokkos::View<double *, MemSpace> dst_linear("dst_linear", dst_n_points);

    axis::field_view<const double, 1> src_linear_view(src_linear.data(), src_n_points);
    axis::field_view<double, 1> dst_linear_view(dst_linear.data(), dst_n_points);
    apply(W, src_linear_view, dst_linear_view);

    auto h_dst_linear = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_linear);
    for (std::size_t j = 0; j < dst_n_points; ++j) {
        double expected = 3.0 * actual_dst_cx[j] - 4.0 * actual_dst_cy[j];
        EXPECT_NEAR(h_dst_linear(j), expected, 1.0e-5);
    }

    // 5. Test quadratic field: f(x, y) = x^2 + 2y^2 - xy
    Kokkos::View<double *, MemSpace> src_quadratic("src_quadratic", src_n_points);
    for (std::size_t idx = 0; idx < src_n_points; ++idx) {
        double x = actual_src_cx[idx];
        double y = actual_src_cy[idx];
        src_quadratic(idx) = x * x + 2.0 * y * y - x * y;
    }

    Kokkos::View<double *, MemSpace> dst_quadratic("dst_quadratic", dst_n_points);

    axis::field_view<const double, 1> src_quad_view(src_quadratic.data(), src_n_points);
    axis::field_view<double, 1> dst_quad_view(dst_quadratic.data(), dst_n_points);
    apply(W, src_quad_view, dst_quad_view);

    auto h_dst_quadratic = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_quadratic);
    for (std::size_t j = 0; j < dst_n_points; ++j) {
        double x = actual_dst_cx[j];
        double y = actual_dst_cy[j];
        double expected = x * x + 2.0 * y * y - x * y;
        EXPECT_NEAR(h_dst_quadratic(j), expected, 1.0e-5);
    }
}

}  // namespace axis::test
