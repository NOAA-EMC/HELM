// AXIS unit test: DstArea vs FracArea normalization semantics
// Verifies that DstArea produces dst_raw = frac_b * dst_true, and that
// FracArea bakes the fraction in so no division is needed.

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
#include <cmath>

namespace {

}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static topology::UnstructuredMesh<MemSpace> make_mesh(std::size_t n, double size) {
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

// Test: DstArea normalization — for a constant field, dst_raw = frac_b * c
// and after adjust_by_fraction, result == c.
TEST(ConservationNorm, DstAreaRawEqualsFracTimesTrue) {
    constexpr double c = 5.0;
    auto src_mesh = make_mesh(3, 3.0);
    auto dst_mesh = make_mesh(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);
    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();

    std::vector<double> src_data(n_src, c);
    std::vector<double> dst_data(n_dst, 0.0);

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // Before adjustment: dst_raw(j) should == frac_b(j) * c
    auto frac_b = matrix.frac_b();
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (frac_b[j] > 0.0) {
            EXPECT_NEAR(dst_data[j], frac_b[j] * c, 1e-10) << "DstArea raw at cell " << j;
        }
    }

    // After adjustment: should recover c
    solver::adjust_by_fraction<MemSpace>(dst_view, frac_b);
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (frac_b[j] > 0.0) {
            EXPECT_NEAR(dst_data[j], c, 1e-10) << "DstArea adjusted at cell " << j;
        }
    }
}

// Test: FracArea normalization — dst directly equals c (no adjustment needed)
TEST(ConservationNorm, FracAreaDirectlyYieldsTrue) {
    constexpr double c = 5.0;
    auto src_mesh = make_mesh(3, 3.0);
    auto dst_mesh = make_mesh(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::FracArea;
    cfg.line_type = solver::LineType::Cartesian;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);
    const auto n_src = matrix.n_src();
    const auto n_dst = matrix.n_dst();

    std::vector<double> src_data(n_src, c);
    std::vector<double> dst_data(n_dst, 0.0);

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // FracArea: dst directly equals true value
    auto frac_b = matrix.frac_b();
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (std::abs(frac_b[j] - 1.0) < 1e-10) {
            EXPECT_NEAR(dst_data[j], c, 1e-10) << "FracArea at cell " << j;
        }
    }
}

// Test: Conservation holds under both normalization types
TEST(ConservationNorm, BothNormsConserve) {
    constexpr double c = 3.0;
    auto src_mesh = make_mesh(3, 3.0);
    auto dst_mesh = make_mesh(3, 3.0);

    for (auto norm : {solver::NormType::DstArea, solver::NormType::FracArea}) {
        solver::RegridConfig cfg;
        cfg.method = solver::InterpolationMethod::Conservative1stOrder;
        cfg.norm_type = norm;
        cfg.line_type = solver::LineType::Cartesian;

        auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);
        const auto n_src = matrix.n_src();
        const auto n_dst = matrix.n_dst();

        std::vector<double> src_data(n_src, c);
        std::vector<double> dst_data(n_dst, 0.0);

        field_view<const double, 1> src_view(src_data.data(), n_src);
        field_view<double, 1> dst_view(dst_data.data(), n_dst);

        solver::apply<MemSpace>(matrix, src_view, dst_view);

        auto report = solver::check_conservation<MemSpace>(src_view, field_view<const double, 1>(dst_data.data(), n_dst), matrix, norm);

        EXPECT_LT(report.relative_error, 1e-12) << "Conservation violated for norm type " << static_cast<int>(norm);
    }
}

}  // namespace axis::test
