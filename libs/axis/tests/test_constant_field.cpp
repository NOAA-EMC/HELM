// AXIS unit test: partition-of-unity constant-field preservation
// Verifies that conservative regridding of a constant field yields that same
// constant at all fully-covered destination cells after adjust_by_fraction.

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

}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static topology::UnstructuredMesh<MemSpace> make_grid(std::size_t n, double size) {
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

// Test: constant source field c=7.0, conservative DstArea, after
// adjust_by_fraction every fully-covered dst cell equals c.
TEST(ConstantField, PartitionOfUnityDstArea) {
    constexpr double c = 7.0;

    auto src_mesh = make_grid(3, 3.0);
    auto dst_mesh = make_grid(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();
    std::vector<double> src_data(n_src, c);
    std::vector<double> dst_data(n_dst, 0.0);

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // adjust_by_fraction to recover true value from DstArea normalization
    solver::adjust_by_fraction<MemSpace>(dst_view, matrix.frac_b());

    // All fully-covered cells (frac_b ≈ 1.0) should have value == c
    auto frac_b = matrix.frac_b();
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (std::abs(frac_b[j] - 1.0) < 1e-10) {
            EXPECT_NEAR(dst_data[j], c, 1e-10) << "Cell " << j << " deviates from constant " << c;
        }
    }
}

// Test: FracArea normalization should directly yield the constant.
TEST(ConstantField, FracAreaDirectlyYieldsConstant) {
    constexpr double c = 3.14;

    auto src_mesh = make_grid(3, 3.0);
    auto dst_mesh = make_grid(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::FracArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();
    std::vector<double> src_data(n_src, c);
    std::vector<double> dst_data(n_dst, 0.0);

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // FracArea: no adjustment needed, result should be c at fully-covered cells
    auto frac_b = matrix.frac_b();
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (std::abs(frac_b[j] - 1.0) < 1e-10) {
            EXPECT_NEAR(dst_data[j], c, 1e-10) << "FracArea cell " << j << " deviates from constant " << c;
        }
    }
}

}  // namespace axis::test
