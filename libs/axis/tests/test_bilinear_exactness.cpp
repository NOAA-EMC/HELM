// AXIS unit test: bilinear exactness on affine fields f(x,y) = a·x + b·y + c
// Verifies that bilinear interpolation exactly reproduces linear fields.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

#include "common/test_kokkos_env.hpp"

namespace {}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Build a uniform NxN grid in Cartesian space covering [0, size]^2
static topology::UnstructuredMesh<MemSpace> make_bilinear_mesh(std::size_t n, double size) {
    const double dx = size / static_cast<double>(n);
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }
    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

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

// Test: f(x,y) = x + y on two identical 4x4 grids should be exactly reproduced
TEST(BilinearExactness, AffineFieldReproduction) {
    const std::size_t n = 4;
    const double size = 4.0;
    const double dx = size / static_cast<double>(n);

    auto src_mesh = make_bilinear_mesh(n, size);
    auto dst_mesh = make_bilinear_mesh(n, size);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    // Set source field: f(x,y) = x + y at cell centers
    std::vector<double> src_data(n_src);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            double x = (static_cast<double>(i) + 0.5) * dx;
            double y = (static_cast<double>(j) + 0.5) * dx;
            src_data[i + j * n] = x + y;
        }
    }

    std::vector<double> dst_data(n_dst, 0.0);
    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // Verify exact reproduction (identical grids → identity map for bilinear)
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            double x = (static_cast<double>(i) + 0.5) * dx;
            double y = (static_cast<double>(j) + 0.5) * dx;
            double expected = x + y;
            EXPECT_NEAR(dst_data[i + j * n], expected, 1e-10) << "Bilinear exactness failed at cell (" << i << "," << j << ")";
        }
    }
}

// Test: f(x,y) = 2x - 3y + 1 with different coefficients
TEST(BilinearExactness, GeneralAffineCoefficients) {
    const std::size_t n = 4;
    const double size = 4.0;
    const double dx = size / static_cast<double>(n);
    const double a = 2.0, b = -3.0, c_coeff = 1.0;

    auto src_mesh = make_bilinear_mesh(n, size);
    auto dst_mesh = make_bilinear_mesh(n, size);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Bilinear;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    std::vector<double> src_data(n_src);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            double x = (static_cast<double>(i) + 0.5) * dx;
            double y = (static_cast<double>(j) + 0.5) * dx;
            src_data[i + j * n] = a * x + b * y + c_coeff;
        }
    }

    std::vector<double> dst_data(n_dst, 0.0);
    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            double x = (static_cast<double>(i) + 0.5) * dx;
            double y = (static_cast<double>(j) + 0.5) * dx;
            double expected = a * x + b * y + c_coeff;
            EXPECT_NEAR(dst_data[i + j * n], expected, 1e-10) << "General affine failed at cell (" << i << "," << j << ")";
        }
    }
}

}  // namespace axis::test
