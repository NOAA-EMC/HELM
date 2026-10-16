// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <axis/detail/linear_geometry.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
using Space = Kokkos::HostSpace;
using Mesh = axis::topology::UnstructuredMesh<Space>;
using namespace axis::detail::linear;
class LinearKokkosEnv : public testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
auto *const env = testing::AddGlobalTestEnvironment(new LinearKokkosEnv);

// Shared-node triangles. Spherical grids are constructed in a gnomonic chart,
// so all grid lines are great circles and both resolutions tile the same domain.
Mesh grid(int n, bool sphere = false, int axis = 0, double size = 1, bool mask_right = false, bool reverse = false) {
    int nodes = (n + 1) * (n + 1), cells = 2 * n * n;
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> coords("nodes", nodes, sphere ? 2 : 3);
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            double x = size * i / n, y = size * j / n;
            int k = i + (n + 1) * j;
            if (sphere) {
                double u = 0.18 * (x - 0.5), v = 0.18 * (y - 0.5);
                Point p = axis == 2 ? unit(Point{u, v, 1}) : axis == 1 ? unit(Point{-1, u, v}) : unit(Point{1, u, v});
                coords(k, 0) = std::atan2(p[1], p[0]) * 180 / 3.14159265358979323846;
                coords(k, 1) = std::asin(p[2]) * 180 / 3.14159265358979323846;
            } else {
                coords(k, 0) = x;
                coords(k, 1) = y;
                coords(k, 2) = 0;
            }
        }
    Kokkos::View<axis::index_t *, Space> offsets("offsets", cells + 1), indices("indices", 3 * cells);
    Kokkos::View<int *, Space> mask("mask", mask_right ? cells : 0);
    int c = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int a = i + (n + 1) * j, b = a + 1, d = a + n + 1, e = d + 1;
            for (auto triangle : {std::array<int, 3>{a, b, e}, std::array<int, 3>{a, e, d}}) {
                offsets(c) = 3 * c;
                for (int k = 0; k < 3; ++k) indices(3 * c + k) = triangle[(reverse != (sphere && axis == 1)) ? 2 - k : k];
                if (mask_right) mask(c) = i < n / 2 ? 1 : 0;
                ++c;
            }
        }
    offsets(c) = 3 * c;
    return Mesh(coords, offsets, indices, sphere ? axis::topology::CoordinateSystem::SphericalDeg : axis::topology::CoordinateSystem::Cartesian3D, {},
                mask);
}
std::vector<Point> polygon(const Mesh &mesh, std::size_t i) {
    std::vector<Point> p;
    auto co = mesh.node_coords();
    auto off = mesh.conn_offsets();
    auto idx = mesh.conn_indices();
    bool sphere = mesh.coord_system() == axis::topology::CoordinateSystem::SphericalDeg;
    for (axis::index_t k = off[i]; k < off[i + 1]; ++k) {
        double x = co(idx[k], 0), y = co(idx[k], 1);
        if (sphere) {
            x *= 3.14159265358979323846 / 180;
            y *= 3.14159265358979323846 / 180;
            p.push_back({std::cos(y) * std::cos(x), std::cos(y) * std::sin(x), std::sin(y)});
        } else
            p.push_back({x, y, 0});
    }
    return p;
}
std::vector<double> linear_values(const Mesh &mesh) {
    std::vector<double> q(mesh.n_cells());
    bool sphere = mesh.coord_system() == axis::topology::CoordinateSystem::SphericalDeg;
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto p = polygon(mesh, i);
        auto m = sphere ? spherical_moments(p) : planar_moments(p);
        q[i] = 2 + 0.7 * m.mean[0] - 0.4 * m.mean[1] + 0.3 * m.mean[2];
    }
    return q;
}
// Independent analytic triangle average of 1+x^2+y^2.
std::vector<double> quadratic_values(const Mesh &mesh) {
    std::vector<double> q(mesh.n_cells(), 1);
    for (std::size_t i = 0; i < q.size(); ++i) {
        auto p = polygon(mesh, i);
        for (int d = 0; d < 2; ++d)
            q[i] += (p[0][d] * p[0][d] + p[1][d] * p[1][d] + p[2][d] * p[2][d] + p[0][d] * p[1][d] + p[1][d] * p[2][d] + p[2][d] * p[0][d]) / 6;
    }
    return q;
}
auto generate(const Mesh &a, const Mesh &b, bool limited = false, axis::solver::NormType norm = axis::solver::NormType::DstArea, bool first = false) {
    axis::solver::RegridConfig c;
    c.method = first ? axis::solver::InterpolationMethod::Conservative1stOrder : axis::solver::InterpolationMethod::Conservative2ndOrder;
    c.line_type =
        a.coord_system() == axis::topology::CoordinateSystem::SphericalDeg ? axis::solver::LineType::GreatCircle : axis::solver::LineType::Cartesian;
    c.norm_type = norm;
    c.use_limiter = limited;
    c.extrap_method = axis::solver::ExtrapolationAction::None;
    return axis::solver::WeightGenerator::generate(a, b, c);
}
std::vector<double> run(const auto &m, const std::vector<double> &q) {
    std::vector<double> dst(m.n_dst());
    axis::solver::apply(m, axis::field_view<const double, 1>{q.data(), q.size()}, axis::field_view<double, 1>{dst.data(), dst.size()});
    return dst;
}
double rms(const std::vector<double> &a, const std::vector<double> &b) {
    double e = 0;
    for (std::size_t i = 0; i < a.size(); ++i) e += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(e / a.size());
}
double integral(const std::vector<double> &q, auto areas, auto fractions, bool frac) {
    double result = 0;
    for (std::size_t i = 0; i < q.size(); ++i) result += q[i] * areas[i] * (frac ? fractions[i] : 1);
    return result;
}
TEST(LinearConservative, PlanarLinearExactnessAndCsrPermutation) {
    auto a = grid(4), b = grid(7);
    auto m = generate(a, b);
    auto q = linear_values(a), expected = linear_values(b);
    EXPECT_LT(rms(run(m, q), expected), 2e-13);
    m.to_csr();
    EXPECT_LT(rms(run(m, q), expected), 2e-13);
    for (auto &v : q) v = 5 - 2 * v;
    for (auto &v : expected) v = 5 - 2 * v;
    EXPECT_LT(rms(run(m, q), expected), 4e-13);  // same geometry, different field
}
TEST(LinearConservative, ConstantsAndConservationBothModesAndNormalizations) {
    for (bool limited : {false, true})
        for (auto norm : {axis::solver::NormType::DstArea, axis::solver::NormType::FracArea}) {
            auto a = grid(4), b = grid(7);
            auto m = generate(a, b, limited, norm);
            m.to_csr();
            std::vector<double> constant(a.n_cells(), 3.25);
            auto result = run(m, constant);
            for (double v : result) EXPECT_NEAR(v, 3.25, 3e-13);
            auto q = quadratic_values(a);
            result = run(m, q);
            EXPECT_NEAR(integral(result, m.area_b(), m.frac_b(), norm == axis::solver::NormType::FracArea),
                        integral(q, m.area_a(), m.frac_a(), false), 3e-13);
        }
}
TEST(LinearConservative, LimitedSharpFrontHasNoNewExtrema) {
    auto a = grid(6), b = grid(11);
    std::vector<double> q(a.n_cells());
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = planar_moments(polygon(a, i)).mean[0] < 0.5 ? 0 : 1;
    auto unlimited = generate(a, b), limited = generate(a, b, true);
    auto raw = run(unlimited, q), bounded = run(limited, q);
    double outside = 0;
    for (double v : raw) outside = std::max(outside, std::max(-v, v - 1));
    EXPECT_GT(outside, 1e-4);
    for (double v : bounded) {
        EXPECT_GE(v, -2e-13);
        EXPECT_LE(v, 1 + 2e-13);
    }
    EXPECT_NEAR(integral(bounded, limited.area_b(), limited.frac_b(), false), integral(q, limited.area_a(), limited.frac_a(), false), 3e-13);
}
TEST(LinearConservative, SmoothFieldImprovementAndRefinement) {
    double previous = 0;
    for (int n : {4, 8, 16}) {
        auto a = grid(n), b = grid(2 * n + 1);
        auto q = quadratic_values(a), expected = quadratic_values(b);
        auto first = generate(a, b, false, axis::solver::NormType::DstArea, true), second = generate(a, b);
        double e1 = rms(run(first, q), expected), e2 = rms(run(second, q), expected);
        std::cout << "planar n=" << n << " first_rms=" << e1 << " linear_rms=" << e2 << "\n";
        EXPECT_LT(e2, e1 * 0.6);
        if (previous > 0) EXPECT_LT(e2, previous * 0.4);
        previous = e2;
    }
}
TEST(LinearConservative, SphericalConservationAndAccuracyAcrossSeamAndPole) {
    for (int chart : {0, 1, 2})
        for (bool limited : {false, true}) {
            auto a = grid(4, true, chart), b = grid(7, true, chart);
            auto m = generate(a, b, limited, axis::solver::NormType::FracArea);
            auto q = linear_values(a), expected = linear_values(b), out = run(m, q);
            EXPECT_NEAR(integral(out, m.area_b(), m.frac_b(), true), integral(q, m.area_a(), m.frac_a(), false), 2e-12);
            auto first = generate(a, b, false, axis::solver::NormType::FracArea, true);
            double e1 = rms(run(first, q), expected), e2 = rms(out, expected);
            std::cout << "sphere chart=" << chart << " limited=" << limited << " first_rms=" << e1 << " linear_rms=" << e2 << "\n";
            EXPECT_LT(e2, e1);
            m.to_csr();
            EXPECT_LT(rms(run(m, q), out), 3e-13);
        }
}
TEST(LinearConservative, ExactSphericalVectorMomentOctant) {
    auto m = spherical_moments({Point{1, 0, 0}, Point{0, 1, 0}, Point{0, 0, 1}});
    EXPECT_NEAR(m.area, 3.14159265358979323846 / 2, 1e-15);
    for (double v : m.mean) EXPECT_NEAR(v, 0.5, 1e-15);
}
TEST(LinearConservative, DryValuesDoNotContaminateGradientsOrBounds) {
    auto a = grid(4, false, 0, 1, true), b = grid(7);
    auto m = generate(a, b, true, axis::solver::NormType::FracArea);
    std::vector<double> q(a.n_cells(), 2);
    auto mask = a.cell_mask();
    for (std::size_t i = 0; i < q.size(); ++i)
        if (mask[i] == 0) q[i] = 1e20;
    auto out = run(m, q);
    for (std::size_t j = 0; j < out.size(); ++j) EXPECT_NEAR(out[j], m.frac_b()[j] > 0 ? 2 : 0, 3e-13);
}
TEST(LinearConservative, PartialDestinationCoverageConservesWholeSource) {
    auto a = grid(4), b = grid(7, false, 0, 1.2);
    auto q = quadratic_values(a);
    for (auto norm : {axis::solver::NormType::DstArea, axis::solver::NormType::FracArea}) {
        auto m = generate(a, b, true, norm);
        auto out = run(m, q);
        EXPECT_NEAR(integral(out, m.area_b(), m.frac_b(), norm == axis::solver::NormType::FracArea), integral(q, m.area_a(), m.frac_a(), false),
                    3e-13);
    }
}
TEST(LinearConservative, PartialSourceCoverageIntegratesReconstructedField) {
    const double length = 0.65;
    auto a = grid(4), b = grid(7, false, 0, length);
    auto q = linear_values(a);
    auto m = generate(a, b);
    auto out = run(m, q);
    // Integral of 2+0.7*x-0.4*y over [0,length]^2, independently analytic.
    const double exact = 2 * length * length + 0.15 * length * length * length;
    EXPECT_NEAR(integral(out, m.area_b(), m.frac_b(), false), exact, 3e-13);
    // Area fractions alone describe a piecewise constant field, not this integral.
    EXPECT_GT(std::abs(integral(q, m.area_a(), m.frac_a(), true) - exact), 1e-5);
}
TEST(LinearConservative, RejectsUnsupportedConstantLatitudeGeometry) {
    auto raw = grid(4, true), b = grid(7, true);
    axis::topology::GeometryMetadata geometry;
    geometry.boundary_model = axis::topology::BoundaryModel::ConstantLatitude;
    Mesh a(raw.node_coords_view(), raw.conn_offsets_view(), raw.conn_indices_view(), raw.coord_system(), {}, {}, geometry);
    EXPECT_THROW(generate(a, b), std::invalid_argument);
}
TEST(LinearConservative, BatchUsesEachFieldsOwnGradients) {
    auto a = grid(4), b = grid(7);
    auto m = generate(a, b);
    m.to_csr();
    auto q = linear_values(a), expected = linear_values(b);
    Kokkos::View<double **, Kokkos::LayoutRight, Space> src("src", a.n_cells(), 2), dst("dst", b.n_cells(), 2);
    for (std::size_t i = 0; i < q.size(); ++i) {
        src(i, 0) = q[i];
        src(i, 1) = 7 - 2 * q[i];
    }
    using In = Kokkos::View<const double **, Kokkos::LayoutRight, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Out = Kokkos::View<double **, Kokkos::LayoutRight, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    axis::solver::batch_apply(m, In(src.data(), a.n_cells(), 2), Out(dst.data(), b.n_cells(), 2));
    for (std::size_t j = 0; j < expected.size(); ++j) {
        EXPECT_NEAR(dst(j, 0), expected[j], 3e-13);
        EXPECT_NEAR(dst(j, 1), 7 - 2 * expected[j], 5e-13);
    }
}
TEST(LinearConservative, MaskedApplyPreservesInactiveOutputs) {
    auto a = grid(4), b = grid(7);
    auto m = generate(a, b);
    auto q = linear_values(a), expected = linear_values(b);
    std::vector<int> mask(b.n_cells(), 1);
    std::vector<double> out(b.n_cells(), -99);
    mask[0] = 0;
    axis::solver::apply(m, axis::field_view<const double, 1>{q.data(), q.size()}, axis::field_view<double, 1>{out.data(), out.size()},
                        axis::field_view<const int, 1>{mask.data(), mask.size()});
    EXPECT_EQ(out[0], -99);
    for (std::size_t j = 1; j < out.size(); ++j) EXPECT_NEAR(out[j], expected[j], 3e-13);
}
TEST(LinearConservative, RankDeficientStencilFallsBackWithoutNaNs) {
    auto a = grid(1), b = grid(3);
    auto m = generate(a, b);
    std::vector<double> q(a.n_cells(), 4);
    for (double v : run(m, q)) EXPECT_NEAR(v, 4, 3e-13);
}
TEST(LinearConservative, ClockwiseRingsAndLargeProjectedCoordinates) {
    auto a = grid(4, false, 0, 1, false, true), b = grid(7, false, 0, 1, false, true);
    auto ca = a.node_coords_view(), cb = b.node_coords_view();
    for (std::size_t i = 0; i < a.n_nodes(); ++i) {
        ca(i, 0) += 1e6;
        ca(i, 1) += 2e6;
    }
    for (std::size_t i = 0; i < b.n_nodes(); ++i) {
        cb(i, 0) += 1e6;
        cb(i, 1) += 2e6;
    }
    auto m = generate(a, b);
    auto q = linear_values(a);
    EXPECT_LT(rms(run(m, q), linear_values(b)), 1e-8);
}
TEST(LinearConservative, ExtentValidationDoesNotWriteDestination) {
    auto m = generate(grid(4), grid(7));
    std::vector<double> q(m.n_src() - 1, 1), dst(m.n_dst(), -99);
    EXPECT_THROW(axis::solver::apply(m, axis::field_view<const double, 1>{q.data(), q.size()}, axis::field_view<double, 1>{dst.data(), dst.size()}),
                 std::invalid_argument);
    for (double v : dst) EXPECT_EQ(v, -99);
}
TEST(LinearConservative, EmptyOverlapCsrAndUnmappedPolicy) {
    auto a = grid(4), b = grid(7);
    auto coords = b.node_coords_view();
    for (std::size_t i = 0; i < b.n_nodes(); ++i) coords(i, 0) += 2;
    auto m = generate(a, b, true);
    EXPECT_EQ(m.nnz(), 0u);
    m.to_csr();
    for (double value : run(m, linear_values(a))) EXPECT_EQ(value, 0);
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
    config.line_type = axis::solver::LineType::Cartesian;
    config.extrap_method = axis::solver::ExtrapolationAction::None;
    config.unmapped = axis::solver::UnmappedAction::Error;
    EXPECT_THROW(axis::solver::WeightGenerator::generate(a, b, config), std::runtime_error);
}
TEST(LinearConservative, RejectsNonconservativeExtrapolation) {
    auto a = grid(4), b = grid(7);
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
    config.line_type = axis::solver::LineType::Cartesian;
    EXPECT_THROW(axis::solver::WeightGenerator::generate(a, b, config), std::invalid_argument);
}
TEST(LinearConservative, RejectsConcaveGeometry) {
    std::vector<Point> p{{0, 0, 0}, {1, 0, 0}, {0.3, 0.3, 0}, {0, 1, 0}};
    EXPECT_THROW(orient_convex(p, false), std::invalid_argument);
}
}  // namespace
