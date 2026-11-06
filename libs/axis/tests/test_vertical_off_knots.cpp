// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <array>
#include <axis/detail/tspack.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
namespace {
using Space = Kokkos::HostSpace;
class VerticalEnvironment : public testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
auto *const env = testing::AddGlobalTestEnvironment(new VerticalEnvironment);
TEST(VerticalOffKnots, AffineUniformAndVaryingDescending) {
    for (double sigma : {0., 1e-8, .05, 2., 50., 1000.})
        for (bool descending : {false, true}) {
            Kokkos::View<double **, Space> f("f", 3, 5), out("out", 3, 6), xs("xs", 3, 5), xd("xd", 3, 6);
            const double levels[] = {0, .1, .4, 1.1, 2.};
            for (int c = 0; c < 3; ++c) {
                for (int i = 0; i < 5; ++i) {
                    double z = levels[descending ? 4 - i : i] * (1 + .1 * c) + .2 * c;
                    xs(c, i) = z;
                    f(c, i) = (c - 1.5) * z + 3;
                }
                for (int j = 0; j < 6; ++j) {
                    double z = (.03 + .35 * (descending ? 5 - j : j)) * (1 + .1 * c) + .2 * c;
                    xd(c, j) = z;
                }
            }
            axis::solver::VerticalRegridder<Space>::interpolate(f, out, xs, xd, sigma);
            Kokkos::fence();
            for (int c = 0; c < 3; ++c)
                for (int j = 0; j < 6; ++j) EXPECT_NEAR(out(c, j), (c - 1.5) * xd(c, j) + 3, 2e-12) << sigma;
        }
}
TEST(VerticalOffKnots, IndependentNaturalSplineReferences) {
    const double levels[] = {0, .2, .65, 1.2, 2.}, values[] = {1, -.3, .8, 2, -.2}, targets[] = {.05, .3, .5, .9, 1.5, 1.8};
    // Cubic: SciPy CubicSpline(bc_type="natural"). Positive tension: independent
    // dense solution in the basis {1,t,cosh(2t),sinh(2t)}, matching values,
    // first/second derivatives, and zero endpoint second derivatives.
    const double cubic[] = {.6060863926840361, -.45480026992328254, .0942155415171455, 1.6720457462257141, 1.5527526174055524, .5824631491726474};
    const double tension[] = {.6180073623577887, -.43282300696950127, .14051435859803263, 1.613414645649971, 1.5327098182523669, .5574695237993192};
    Kokkos::View<double **, Space> f("f", 1, 5), out("out", 1, 6);
    Kokkos::View<double *, Space> xs("xs", 5), xd("xd", 6);
    for (int i = 0; i < 5; ++i) {
        xs(i) = levels[i];
        f(0, i) = values[i];
    }
    for (int j = 0; j < 6; ++j) xd(j) = targets[j];
    for (double sigma : {0., 2.}) {
        axis::solver::VerticalRegridder<Space>::interpolate(f, out, xs, xd, sigma);
        Kokkos::fence();
        for (int j = 0; j < 6; ++j) EXPECT_NEAR(out(0, j), sigma == 0 ? cubic[j] : tension[j], 3e-12);
    }
}
TEST(VerticalOffKnots, HermiteEndpointValuesAndDerivatives) {
    using axis::detail::tspack::evaluate_spline;
    for (double sigma : {0., 1e-8, .05, .1, 2., 50.}) {
        auto f = [&](double z) { return evaluate_spline(z, 0, 1, 1, 4, -.2, 2, sigma); };
        EXPECT_DOUBLE_EQ(f(0), 1);
        EXPECT_DOUBLE_EQ(f(1), 4);
        double h = 1e-6;
        EXPECT_NEAR((-3 * f(0) + 4 * f(h) - f(2 * h)) / (2 * h), -.2, 2e-7) << sigma;
        EXPECT_NEAR((3 * f(1) - 4 * f(1 - h) + f(1 - 2 * h)) / (2 * h), 2., 2e-7) << sigma;
    }
}
TEST(VerticalOffKnots, TinyTensionLimitAndLargeTensionFinite) {
    using namespace axis::detail::tspack;
    const double x[] = {0, .2, .65, 1.2, 2}, y[] = {1, -.3, .8, 2, -.2};
    double d[5], scratch[5];
    for (double sigma : {1e-12, .099999, .100001, 1000., 1e6, 1e308}) {
        solve_column_spline<256>(x, y, 5, sigma, d, scratch);
        for (int i = 0; i < 4; ++i)
            for (double t : {.001, .25, .5, .999}) {
                double v = evaluate_spline(x[i] + t * (x[i + 1] - x[i]), x[i], x[i + 1], y[i], y[i + 1], d[i], d[i + 1], sigma);
                EXPECT_TRUE(std::isfinite(v)) << sigma;
                if (sigma < 1e-10) {
                    double dc[5], sc[5];
                    solve_column_spline<256>(x, y, 5, 0, dc, sc);
                    EXPECT_NEAR(v, evaluate_spline(x[i] + t * (x[i + 1] - x[i]), x[i], x[i + 1], y[i], y[i + 1], dc[i], dc[i + 1], 0), 3e-12);
                }
                if (sigma >= 1e6) EXPECT_NEAR(v, (1 - t) * y[i] + t * y[i + 1], 2e-5);
            }
    }
}
TEST(VerticalOffKnots, CubicSmoothRefinement) {
    double previous = 0.0;
    for (int n : {8, 16, 32}) {
        Kokkos::View<double **, Space> f("f", 1, n + 1), out("out", 1, n);
        Kokkos::View<double *, Space> xs("xs", n + 1), xd("xd", n);
        auto truth = [](double z) { return std::sin(std::numbers::pi * z) + z; };
        for (int i = 0; i <= n; ++i) {
            xs(i) = double(i) / n;
            f(0, i) = truth(xs(i));
        }
        for (int j = 0; j < n; ++j) xd(j) = (j + .5) / n;
        axis::solver::VerticalRegridder<Space>::interpolate(f, out, xs, xd, 0.0);
        Kokkos::fence();
        double rms = 0;
        for (int j = 0; j < n; ++j) {
            double e = out(0, j) - truth(xd(j));
            rms += e * e;
        }
        rms = std::sqrt(rms / n);
        std::cout << "vertical_cubic_refinement," << n << ',' << rms << '\n';
        if (previous > 0) EXPECT_GT(previous / rms, 14.0);
        previous = rms;
    }
}
TEST(VerticalOffKnots, TwoLevelsAreLinearForEveryTension) {
    Kokkos::View<double **, Space> f("f", 1, 2), out("out", 1, 3);
    Kokkos::View<double *, Space> xs("xs", 2), xd("xd", 3);
    xs(0) = 3;
    xs(1) = 1;
    f(0, 0) = 7;
    f(0, 1) = 3;
    xd(0) = 2.9;
    xd(1) = 2.1;
    xd(2) = 1.3;
    for (double sigma : {0., 1e-12, 2., 1000.}) {
        axis::solver::VerticalRegridder<Space>::interpolate(f, out, xs, xd, sigma);
        Kokkos::fence();
        for (int j = 0; j < 3; ++j) EXPECT_NEAR(out(0, j), 2 * xd(j) + 1, 2e-12);
    }
}
TEST(VerticalOffKnots, RejectInsufficientLevelsAndNonfiniteTension) {
    Kokkos::View<double **, Space> f("f", 1, 1), out("out", 1, 1);
    Kokkos::View<double *, Space> x("x", 1);
    EXPECT_THROW(axis::solver::VerticalRegridder<Space>::interpolate(f, out, x, x), std::invalid_argument);
    Kokkos::View<double **, Space> f2("f2", 1, 2);
    Kokkos::View<double *, Space> x2("x2", 2);
    for (double sigma : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        EXPECT_THROW(axis::solver::VerticalRegridder<Space>::interpolate(f2, out, x2, x, sigma), std::invalid_argument);
}
}  // namespace
