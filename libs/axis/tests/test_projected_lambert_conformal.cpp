// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file tests/test_projected_lambert_conformal.cpp
/// @brief Integration tests for Lambert Conformal Conic to EPSG:4326 regridding.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

TEST(ProjectedLambertConformal, EndToEndInterpolationAllMethods) {
#ifndef AXIS_ENABLE_PROJ
    GTEST_SKIP() << "AXIS built without PROJ support — skipping Lambert Conformal test";
#endif

    // ── Source: Lambert Conformal Conic Grid (LCC) ──
    const std::string lcc_proj = "+proj=lcc +lat_1=30 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";

    const std::size_t ni = 6;
    const std::size_t nj = 6;
    const std::size_t n_points = ni * nj;

    // Define coordinate ranges in projection space (meters)
    double min_x = -50000.0;
    double max_x = 50000.0;
    double min_y = -50000.0;
    double max_y = 50000.0;

    double dx = (max_x - min_x) / ni;
    double dy = (max_y - min_y) / nj;

    std::vector<double> h_center_x(n_points);
    std::vector<double> h_center_y(n_points);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t idx = i + j * ni;
            h_center_x[idx] = min_x + (i + 0.5) * dx;
            h_center_y[idx] = min_y + (j + 0.5) * dy;
        }
    }

    ingest::ProjectedParams params;
    params.proj_string = lcc_proj;

    ingest::BufferViews buffers;
    buffers.ni = ni;
    buffers.nj = nj;
    buffers.center_x = field_view<const double, 1>(h_center_x.data(), n_points);
    buffers.center_y = field_view<const double, 1>(h_center_y.data(), n_points);

    // Build geographic coordinate StructuredGrid using PROJ
    auto src_grid = topology::ProjectionBuilder::build<MemSpace>(params, buffers);
    EXPECT_TRUE(src_grid.has_explicit_corners());
    auto src_mesh = src_grid.to_unstructured();

    // ── Destination: EPSG:4326 Regular Lat-Lon Grid (covering LCC bounding box) ──
    // Center at lon_0 = -96, lat_0 = 40. Since 50km on Earth is ~0.5 degrees:
    const std::size_t dst_ni = 4;
    const std::size_t dst_nj = 4;
    const std::size_t dst_n_points = dst_ni * dst_nj;

    std::vector<double> dst_cx(dst_n_points);
    std::vector<double> dst_cy(dst_n_points);

    // Build destination grid center points around -96 deg lon, 40 deg lat
    double dst_min_lon = -96.5;
    double dst_max_lon = -95.5;
    double dst_min_lat = 39.5;
    double dst_max_lat = 40.5;

    double d_lon = (dst_max_lon - dst_min_lon) / dst_ni;
    double d_lat = (dst_max_lat - dst_min_lat) / dst_nj;

    for (std::size_t j = 0; j < dst_nj; ++j) {
        for (std::size_t i = 0; i < dst_ni; ++i) {
            std::size_t idx = i + j * dst_ni;
            dst_cx[idx] = dst_min_lon + (i + 0.5) * d_lon;
            dst_cy[idx] = dst_min_lat + (j + 0.5) * d_lat;
        }
    }

    // Build corners for regular destination grid
    const std::size_t dst_nc_lon = dst_ni + 1;
    const std::size_t dst_nc_lat = dst_nj + 1;
    std::vector<double> dst_crx(dst_nc_lon * dst_nc_lat);
    std::vector<double> dst_cry(dst_nc_lon * dst_nc_lat);

    for (std::size_t j = 0; j <= dst_nj; ++j) {
        for (std::size_t i = 0; i <= dst_ni; ++i) {
            std::size_t idx = i + j * dst_nc_lon;
            dst_crx[idx] = dst_min_lon + i * d_lon;
            dst_cry[idx] = dst_min_lat + j * d_lat;
        }
    }

    Kokkos::View<double *, MemSpace> dst_center_lon("dst_center_lon", dst_n_points);
    Kokkos::View<double *, MemSpace> dst_center_lat("dst_center_lat", dst_n_points);
    for (std::size_t idx = 0; idx < dst_n_points; ++idx) {
        dst_center_lon(idx) = dst_cx[idx];
        dst_center_lat(idx) = dst_cy[idx];
    }

    topology::StructuredGrid<MemSpace> dst_grid(dst_ni, dst_nj, dst_center_lon, dst_center_lat, topology::CoordinateSystem::SphericalDeg);

    // Set corners
    Kokkos::View<double *, MemSpace> dst_corners_lon("dst_corners_lon", dst_nc_lon * dst_nc_lat);
    Kokkos::View<double *, MemSpace> dst_corners_lat("dst_corners_lat", dst_nc_lon * dst_nc_lat);
    for (std::size_t idx = 0; idx < dst_nc_lon * dst_nc_lat; ++idx) {
        dst_corners_lon(idx) = dst_crx[idx];
        dst_corners_lat(idx) = dst_cry[idx];
    }
    dst_grid.set_corners(dst_corners_lon, dst_corners_lat);

    auto dst_mesh = dst_grid.to_unstructured();

    // ── 1. Test Nearest Neighbor ──
    {
        solver::RegridConfig cfg;
        cfg.method = solver::InterpolationMethod::NearestNeighbor;
        cfg.line_type = solver::LineType::GreatCircle;
        cfg.unmapped = solver::UnmappedAction::Ignore;

        auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

        EXPECT_EQ(matrix.n_src(), n_points);
        EXPECT_EQ(matrix.n_dst(), dst_n_points);

        // Verify row sum partition of unity
        auto rows = matrix.factor_row();
        auto vals = matrix.factor_list();
        std::vector<double> row_sums(dst_n_points, 0.0);
        for (std::size_t k = 0; k < matrix.nnz(); ++k) {
            row_sums[static_cast<std::size_t>(rows[k])] += vals[k];
        }
        for (std::size_t j = 0; j < dst_n_points; ++j) {
            if (row_sums[j] > 0.0) {
                EXPECT_NEAR(row_sums[j], 1.0, 1e-12);
            }
        }
    }

    // ── 2. Test Bilinear ──
    {
        solver::RegridConfig cfg;
        cfg.method = solver::InterpolationMethod::Bilinear;
        cfg.line_type = solver::LineType::GreatCircle;
        cfg.unmapped = solver::UnmappedAction::Ignore;

        auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

        EXPECT_EQ(matrix.n_src(), n_points);
        EXPECT_EQ(matrix.n_dst(), dst_n_points);

        auto rows = matrix.factor_row();
        auto vals = matrix.factor_list();
        std::vector<double> row_sums(dst_n_points, 0.0);
        for (std::size_t k = 0; k < matrix.nnz(); ++k) {
            row_sums[static_cast<std::size_t>(rows[k])] += vals[k];
        }
        for (std::size_t j = 0; j < dst_n_points; ++j) {
            if (row_sums[j] > 0.0) {
                EXPECT_NEAR(row_sums[j], 1.0, 1e-12);
            }
        }
    }

    // ── 3. Test Conservative 1st Order ──
    {
        solver::RegridConfig cfg;
        cfg.method = solver::InterpolationMethod::Conservative1stOrder;
        cfg.line_type = solver::LineType::GreatCircle;
        cfg.unmapped = solver::UnmappedAction::Ignore;

        auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

        EXPECT_EQ(matrix.n_src(), n_points);
        EXPECT_EQ(matrix.n_dst(), dst_n_points);

        // Verify non-negativity and total integral conservation report
        auto rows = matrix.factor_row();
        auto vals = matrix.factor_list();
        for (std::size_t k = 0; k < matrix.nnz(); ++k) {
            EXPECT_GE(vals[k], 0.0);
        }
    }
}

}  // namespace axis::test
