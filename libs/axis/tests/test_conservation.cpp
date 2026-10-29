// AXIS unit test: first-order conservation (Σ src·area_a·frac_a ≈ Σ dst·area_b)
// Tests that conservative regridding preserves global integrals.

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace {
namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Helper: build a simple NxN structured grid covering [0, size] x [0, size]
// with uniform cells, then convert to UnstructuredMesh.
static topology::UnstructuredMesh<MemSpace> make_uniform_mesh(std::size_t n, double size) {
    const std::size_t n_cells = n;
    const double dx = size / static_cast<double>(n);

    // Center coordinates for an n x n grid
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }

    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    // Set corners for conservative regridding
    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = static_cast<double>(i) * dx;
            cry(i + j * nc) = static_cast<double>(j) * dx;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// Test: Conservative regridding of a known field on identical 3x3 meshes
// that tile the same domain preserves the global integral.
TEST(Conservation, IdenticalMeshesPreserveIntegral) {
    auto src_mesh = make_uniform_mesh(3, 3.0);
    auto dst_mesh = make_uniform_mesh(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // Apply a constant field value of 5.0
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();
    std::vector<double> src_data(n_src, 5.0);
    std::vector<double> dst_data(n_dst, 0.0);

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // Check conservation: source_integral ≈ destination_integral
    auto report =
        solver::check_conservation<MemSpace>(src_view, field_view<const double, 1>(dst_data.data(), n_dst), matrix, solver::NormType::DstArea);

    EXPECT_NEAR(report.src_integral, report.dst_integral, 1e-12 * std::abs(report.src_integral));
    EXPECT_LT(report.relative_error, 1e-12);
}

// Test: Conservation with a spatially-varying field
TEST(Conservation, VaryingFieldPreservesIntegral) {
    auto src_mesh = make_uniform_mesh(3, 3.0);
    auto dst_mesh = make_uniform_mesh(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    // Field: value = cell index + 1
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) src_data[i] = static_cast<double>(i + 1);
    std::vector<double> dst_data(n_dst, 0.0);

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    auto report =
        solver::check_conservation<MemSpace>(src_view, field_view<const double, 1>(dst_data.data(), n_dst), matrix, solver::NormType::DstArea);

    EXPECT_NEAR(report.src_integral, report.dst_integral, 1e-12 * std::abs(report.src_integral));
}

static topology::UnstructuredMesh<MemSpace> make_regular_spherical_grid(std::size_t ni, std::size_t nj, double lon_min, double lon_max,
                                                                        double lat_min, double lat_max) {
    const double delta_lon = (lon_max - lon_min) / static_cast<double>(ni);
    const double delta_lat = (lat_max - lat_min) / static_cast<double>(nj);

    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_min + (static_cast<double>(i) + 0.5) * delta_lon;
            cy(i + j * ni) = lat_min + (static_cast<double>(j) + 0.5) * delta_lat;
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);

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

TEST(Conservation, GlobalPeriodicLongitudeMapping0To360ToMinus180To180) {
    // 0->360 source mesh (global)
    auto src_mesh = make_regular_spherical_grid(36, 18, 0.0, 360.0, -90.0, 90.0);
    // -180->180 destination mesh
    auto dst_mesh = make_regular_spherical_grid(36, 18, -180.0, 180.0, -90.0, 90.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_dst = matrix.n_dst();
    EXPECT_EQ(n_dst, 36 * 18);

    // Compute row sums
    std::vector<double> row_sums(n_dst, 0.0);
    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();
    for (std::size_t k = 0; k < matrix.nnz(); ++k) {
        row_sums[rows[k]] += vals[k];
    }

    // Every destination cell should be fully covered and have row sum of approximately 1.0.
    // Specifically, check cells in the western hemisphere (indices where longitude is negative: 0 to 17)
    for (std::size_t j = 0; j < 18; ++j) {
        for (std::size_t i = 0; i < 36; ++i) {
            std::size_t idx = j * 36 + i;
            // Longitudes corresponding to indices i < 18 are western hemisphere (-180.0 to 0.0)
            if (i < 18) {
                EXPECT_NEAR(row_sums[idx], 1.0, 1e-12) << "Western hemisphere cell at i=" << i << ", j=" << j << " (index " << idx << ") is empty!";
            } else {
                EXPECT_NEAR(row_sums[idx], 1.0, 1e-12) << "Eastern hemisphere cell at i=" << i << ", j=" << j << " (index " << idx << ") is empty!";
            }
        }
    }
}

static topology::UnstructuredMesh<MemSpace> make_regular_spherical_grid_synthesized(std::size_t ni, std::size_t nj, double lon_min, double lon_max,
                                                                                    double lat_min, double lat_max) {
    const double delta_lon = (lon_max - lon_min) / static_cast<double>(ni);
    const double delta_lat = (lat_max - lat_min) / static_cast<double>(nj);

    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_min + (static_cast<double>(i) + 0.5) * delta_lon;
            cy(i + j * ni) = lat_min + (static_cast<double>(j) + 0.5) * delta_lat;
        }
    }

    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);
    // Do NOT call set_corners, let them synthesize.
    return grid.to_unstructured();
}

TEST(Conservation, GlobalPeriodicLongitudeMappingWithSynthesizedCorners) {
    // 0->360 source mesh (global) with synthesized corners
    auto src_mesh = make_regular_spherical_grid_synthesized(72, 46, 0.0, 360.0, -90.0, 90.0);
    // -180->180 destination mesh with synthesized corners
    auto dst_mesh = make_regular_spherical_grid_synthesized(72, 46, -180.0, 180.0, -90.0, 90.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::GreatCircle;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_dst = matrix.n_dst();
    EXPECT_EQ(n_dst, 72 * 46);

    // Compute row sums
    std::vector<double> row_sums(n_dst, 0.0);
    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();
    for (std::size_t k = 0; k < matrix.nnz(); ++k) {
        row_sums[rows[k]] += vals[k];
    }

    // Corner synthesis detects the global 360-degree longitude span and wraps
    // periodically across the dateline, so every destination cell (including the
    // boundary columns i=35 and i=36) is fully covered with a row sum of ~1.0.
    for (std::size_t j = 0; j < 46; ++j) {
        for (std::size_t i = 0; i < 72; ++i) {
            std::size_t idx = j * 72 + i;
            if (i < 36) {
                EXPECT_NEAR(row_sums[idx], 1.0, 1e-12)
                    << "Western hemisphere cell at i=" << i << ", j=" << j << " (index " << idx << ") has incorrect coverage!";
            } else {
                EXPECT_NEAR(row_sums[idx], 1.0, 1e-12)
                    << "Eastern hemisphere cell at i=" << i << ", j=" << j << " (index " << idx << ") has incorrect coverage!";
            }
        }
    }
}

}  // namespace axis::test
