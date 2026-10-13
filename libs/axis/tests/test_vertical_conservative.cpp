// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <axis/solver/vertical_conservative.hpp>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
namespace {
using Space = Kokkos::HostSpace;
using Op = axis::solver::VerticalConservativeRegridder<Space>;
using Order = axis::solver::VerticalConservativeOrder;
using Norm = axis::solver::VerticalConservativeNorm;
class Env : public testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
auto *const env = testing::AddGlobalTestEnvironment(new Env);
Op::Field edges(const std::vector<double> &v, int rows = 1) {
    Op::Field x("edges", rows, v.size());
    for (int c = 0; c < rows; ++c)
        for (size_t i = 0; i < v.size(); ++i) x(c, i) = v[i];
    return x;
}
double integral(Op::Field v, Op::ConstField e, int c = 0) {
    double sum = 0;
    for (size_t i = 0; i < v.extent(1); ++i) sum += v(c, i) * std::abs(e(e.extent(0) == 1 ? 0 : c, i + 1) - e(e.extent(0) == 1 ? 0 : c, i));
    return sum;
}
TEST(VerticalConservative, KnownOverlapAndBothNormalizations) {
    auto s = edges({0, 1, 3}), d = edges({-.5, .5, 2, 4});
    Op op(s, d, 1);
    Op::Field q("q", 1, 2), v("v", 1, 3);
    q(0, 0) = 2;
    q(0, 1) = 4;
    op.apply(q, v);
    EXPECT_DOUBLE_EQ(v(0, 0), 1);
    EXPECT_NEAR(v(0, 1), 10. / 3, 1e-14);
    EXPECT_DOUBLE_EQ(v(0, 2), 2);
    EXPECT_DOUBLE_EQ(op.destination_coverage()(0, 0), .5);
    EXPECT_DOUBLE_EQ(op.destination_coverage()(0, 2), .5);
    EXPECT_NEAR(integral(v, d), 10, 1e-14);
    op.apply(q, v, Order::First, Norm::CoveredMeasure);
    EXPECT_DOUBLE_EQ(v(0, 0), 2);
    EXPECT_DOUBLE_EQ(v(0, 2), 4);
    double sum = 0;
    for (int j = 0; j < 3; ++j) sum += v(0, j) * op.destination_covered_measure()(0, j);
    EXPECT_NEAR(sum, 10, 1e-14);
}
TEST(VerticalConservative, LimitedPartialIntegralIncludesSlope) {
    auto s = edges({0, 1, 2, 3}), d = edges({1, 1.5});
    Op op(s, d, 1);
    Op::Field q("q", 1, 3), v("v", 1, 1);
    q(0, 0) = 1;
    q(0, 1) = 2;
    q(0, 2) = 3;
    op.apply(q, v, Order::LimitedLinear);
    EXPECT_NEAR(v(0, 0), 1.75, 1e-14);
    auto total = op.covered_source_integral(q, Order::LimitedLinear);
    EXPECT_NEAR(total(0), .875, 1e-14);
    EXPECT_NEAR(total(0), integral(v, d), 1e-14);
    EXPECT_NE(total(0), q(0, 1) * op.source_covered_measure()(0, 1));
}
TEST(VerticalConservative, WorkspaceReuseRecomputesSlopesForEachTracer) {
    auto s = edges({0, 1, 2, 3}), d = edges({0, .5, 1.5, 2.5, 3});
    Op op(s, d, 2);
    Op::Field q("q", 2, 3), expected("expected", 2, 4), result("result", 2, 4), work("work", 2, 3);
    for (int trial = 0; trial < 3; ++trial) {
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 3; ++i) q(c, i) = trial == 1 ? 7. : (trial == 0 ? i + c : 3 - i + c);
        op.apply(q, expected, Order::LimitedLinear);
        op.apply(q, result, Order::LimitedLinear, Norm::DestinationMeasure, work);
        auto stock = op.covered_source_integral(q, Order::LimitedLinear, work);
        for (int c = 0; c < 2; ++c) {
            for (int j = 0; j < 4; ++j) EXPECT_DOUBLE_EQ(result(c, j), expected(c, j));
            EXPECT_NEAR(stock(c), integral(q, s, c), 1e-14);
        }
    }
}
TEST(VerticalConservative, DescendingIrregularVaryingConservation) {
    auto s = edges({0, .2, .7, 1.3, 2}, 3), d = edges({0, .1, .6, 1.5, 1.8, 2}, 3);
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < 5; ++i) s(c, i) = c + .8 * (c + 1) * s(c, i);
        for (int j = 0; j < 6; ++j) d(c, j) = c + .8 * (c + 1) * d(c, j);
    }
    for (int i = 0; i < 2; ++i) std::swap(s(1, i), s(1, 4 - i));
    for (int j = 0; j < 3; ++j) std::swap(d(2, j), d(2, 5 - j));
    Op op(s, d, 3);
    Op::Field q("q", 3, 4), v("v", 3, 5);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 4; ++i) q(c, i) = 1 + c + .1 * i * i;
    for (auto order : {Order::First, Order::LimitedLinear}) {
        op.apply(q, v, order);
        auto total = op.covered_source_integral(q, order);
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(integral(v, d, c), integral(q, s, c), 2e-14);
            EXPECT_NEAR(total(c), integral(v, d, c), 2e-14);
            for (int j = 0; j < 5; ++j) EXPECT_NEAR(op.destination_coverage()(c, j), 1, 1e-14);
        }
    }
}
TEST(VerticalConservative, MasksDryLayersAndNoLeakage) {
    auto s = edges({0, 1, 1, 2, 3}), d = edges({0, .5, 1, 2, 3, 4});
    Kokkos::View<int **, Space> sm("sm", 1, 4), dm("dm", 1, 5);
    Kokkos::deep_copy(sm, 1);
    Kokkos::deep_copy(dm, 1);
    sm(0, 2) = 0;
    dm(0, 0) = 0;
    Op op(s, d, 1, sm, dm);
    Op::Field q("q", 1, 4), v("v", 1, 5);
    q(0, 0) = 2;
    q(0, 1) = NAN;
    q(0, 2) = NAN;
    q(0, 3) = 4;
    // The operator owns masks and geometry: changing the caller's views is safe.
    Kokkos::deep_copy(sm, 0);
    Kokkos::deep_copy(dm, 0);
    Kokkos::deep_copy(s, 999.);
    for (auto order : {Order::First, Order::LimitedLinear}) {
        op.apply(q, v, order);
        EXPECT_DOUBLE_EQ(v(0, 0), 0);
        EXPECT_DOUBLE_EQ(v(0, 1), 2);
        EXPECT_DOUBLE_EQ(v(0, 2), 0);
        EXPECT_DOUBLE_EQ(v(0, 3), 4);
        EXPECT_DOUBLE_EQ(v(0, 4), 0);
        EXPECT_NEAR(op.covered_source_integral(q, order)(0), 5, 1e-14);
    }
}
TEST(VerticalConservative, PositivityAndBoundsAtSharpInterfaces) {
    auto s = edges({0, .02, .1, .5, .7, 1}), d = edges({0, .01, .03, .08, .2, .4, .6, .8, .9, 1});
    Op op(s, d, 1);
    Op::Field q("q", 1, 5), v("v", 1, 9);
    double values[] = {0, 0, 1, 1, 0};
    for (int i = 0; i < 5; ++i) q(0, i) = values[i];
    for (auto order : {Order::First, Order::LimitedLinear}) {
        op.apply(q, v, order);
        for (int j = 0; j < 9; ++j) {
            EXPECT_GE(v(0, j), -1e-15);
            EXPECT_LE(v(0, j), 1 + 1e-15);
        }
        EXPECT_NEAR(integral(v, d), integral(q, s), 1e-14);
    }
}
TEST(VerticalConservative, ConstantSharedGeometryAndMaskedPartialCoverage) {
    auto s = edges({0, 1, 2, 3}), d = edges({0, .5, 1.5, 2.5, 3});
    Kokkos::View<int **, Space> mask("mask", 2, 3);
    Kokkos::deep_copy(mask, 1);
    mask(1, 1) = 0;
    Op op(s, d, 2, mask);
    Op::Field q("q", 2, 3), v("v", 2, 4);
    Kokkos::deep_copy(q, 7.);
    for (auto order : {Order::First, Order::LimitedLinear}) {
        op.apply(q, v, order);
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < 4; ++j) EXPECT_NEAR(v(c, j), 7 * op.destination_coverage()(c, j), 1e-14);
        op.apply(q, v, order, Norm::CoveredMeasure);
        for (int c = 0; c < 2; ++c)
            for (int j = 0; j < 4; ++j) EXPECT_NEAR(v(c, j), 7, 1e-14);
    }
}
TEST(VerticalConservative, RepeatedRemapsPreserveInventory) {
    auto a = edges({0, .1, .3, .6, .8, 1}), b = edges({0, .05, .15, .4, .7, .9, 1});
    Op ab(a, b, 1), ba(b, a, 1);
    Op::Field q("q", 1, 5), v("v", 1, 6);
    for (auto order : {Order::First, Order::LimitedLinear}) {
        for (int i = 0; i < 5; ++i) q(0, i) = i < 2 ? 0 : 1;
        double initial = integral(q, a);
        for (int step = 0; step < 1000; ++step) {
            ab.apply(q, v, order);
            ba.apply(v, q, order);
        }
        double drift = std::abs(integral(q, a) - initial);
        std::cout << "vertical_repeated_inventory_drift," << (order == Order::First ? "first" : "limited") << ',' << drift << '\n';
        EXPECT_LT(drift, 1e-12);
    }
}
TEST(VerticalConservative, SmoothRefinementLimitedImprovesFirst) {
    for (int n : {8, 16, 32}) {
        std::vector<double> se(n + 1), de(2 * n + 2);
        for (int i = 0; i <= n; ++i) se[i] = double(i) / n;
        for (int j = 0; j <= 2 * n + 1; ++j) de[j] = double(j) / (2 * n + 1);
        auto s = edges(se), d = edges(de);
        Op op(s, d, 1);
        Op::Field q("q", 1, n), v("v", 1, 2 * n + 1);
        auto mean = [](double a, double b) { return 1 + (a * a + a * b + b * b) / 3; };
        for (int i = 0; i < n; ++i) q(0, i) = mean(se[i], se[i + 1]);
        double errors[2];
        int k = 0;
        for (auto order : {Order::First, Order::LimitedLinear}) {
            op.apply(q, v, order);
            double sum = 0;
            for (int j = 0; j < 2 * n + 1; ++j) {
                double e = v(0, j) - mean(de[j], de[j + 1]);
                sum += e * e / (2 * n + 1);
            }
            errors[k++] = std::sqrt(sum);
            EXPECT_NEAR(integral(v, d), 4. / 3, 2e-14);
        }
        std::cout << "vertical_accuracy," << n << ',' << errors[0] << ',' << errors[1] << '\n';
        EXPECT_LT(errors[1], errors[0]);
    }
}
TEST(VerticalConservative, InvalidInputsAndAliasingAreRejected) {
    auto s = edges({0, 1, 2}), d = edges({0, 1, 2});
    Op op(s, d, 1);
    Op::Field q("q", 1, 2), v("v", 1, 2), bad("bad", 2, 2);
    Kokkos::deep_copy(q, 1.);
    EXPECT_THROW(op.apply(q, q), std::invalid_argument);
    EXPECT_THROW(op.apply(q, bad), std::invalid_argument);
    EXPECT_THROW(op.apply(q, v, Order::LimitedLinear, Norm::DestinationMeasure, v), std::invalid_argument);
    q(0, 0) = NAN;
    EXPECT_THROW(op.apply(q, v), std::invalid_argument);
    auto nonmonotone = edges({0, 2, 1}), nonfinite = edges({0, 1, INFINITY});
    EXPECT_THROW(Op(nonmonotone, d, 1), std::invalid_argument);
    EXPECT_THROW(Op(nonfinite, d, 1), std::invalid_argument);
}
TEST(VerticalConservative, LargeOriginAndOneLayerAndFullyDry) {
    auto s = edges({1e15, 1e15 + 1, 1e15 + 2, 1e15 + 3}), d = edges({1e15 + 1, 1e15 + 1.5});
    Op op(s, d, 1);
    Op::Field q("q", 1, 3), v("v", 1, 1);
    q(0, 0) = 1;
    q(0, 1) = 2;
    q(0, 2) = 3;
    op.apply(q, v, Order::LimitedLinear);
    EXPECT_DOUBLE_EQ(v(0, 0), 1.75);
    auto one = edges({1, 0}), target = edges({0, .5, 1});
    Op single(one, target, 1);
    Op::Field q1("q1", 1, 1), v2("v2", 1, 2);
    q1(0, 0) = 3;
    single.apply(q1, v2, Order::LimitedLinear);
    EXPECT_DOUBLE_EQ(v2(0, 0), 3);
    EXPECT_DOUBLE_EQ(v2(0, 1), 3);
    auto dry = edges({0, 0, 0});
    Op empty(dry, target, 1);
    Op::Field q2("q2", 1, 2);
    Kokkos::deep_copy(q2, NAN);
    empty.apply(q2, v2, Order::LimitedLinear);
    EXPECT_DOUBLE_EQ(v2(0, 0), 0);
    EXPECT_DOUBLE_EQ(empty.covered_source_integral(q2, Order::LimitedLinear)(0), 0);
}
}  // namespace
