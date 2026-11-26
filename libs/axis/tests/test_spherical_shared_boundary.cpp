// SPDX-License-Identifier: Apache-2.0
#include <axis/detail/spherical_clipper.hpp>
#include <gtest/gtest.h>
#include <cmath>

namespace {
using namespace axis::detail;
Vec3 spherical_point(double lon, double lat) {
    constexpr double rad = 0.01745329251994329577;
    lon *= rad;
    lat *= rad;
    return {std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
}
SphericalPolygon<32> cell(double left, double bottom, double right, double top) {
    SphericalPolygon<32> p;
    p.push(spherical_point(left, bottom));
    p.push(spherical_point(right, bottom));
    p.push(spherical_point(right, top));
    p.push(spherical_point(left, top));
    return p;
}
TEST(SphericalSharedBoundary, TouchingEdgeHasZeroArea) {
    auto a = cell(0, 5, 1, 6);
    auto b = cell(1, 5, 2, 6);
    EXPECT_EQ(SphericalClipper::overlap_area(a, b), 0.0);
    EXPECT_EQ(SphericalClipper::overlap_area(b, a), 0.0);
}
TEST(SphericalSharedBoundary, TinyRealOverlapIsRetained) {
    auto a = cell(0, 5, 1, 6);
    auto b = cell(1 - 1e-10, 5, 2 - 1e-10, 6);
    EXPECT_GT(SphericalClipper::overlap_area(a, b), 0.0);
    EXPECT_GT(SphericalClipper::overlap_area(b, a), 0.0);
}
TEST(SphericalSharedBoundary, IdenticalAndCornerTouchingCells) {
    auto a = cell(0, 5, 1, 6);
    auto corner = cell(1, 6, 2, 7);
    EXPECT_GT(SphericalClipper::overlap_area(a, a), 0.0);
    EXPECT_EQ(SphericalClipper::overlap_area(a, corner), 0.0);
    EXPECT_EQ(SphericalClipper::overlap_area(corner, a), 0.0);
}
TEST(SphericalSharedBoundary, SharedVertexWithRealOverlap) {
    auto a = cell(0, 5, 1, 6);
    SphericalPolygon<32> b;
    b.push(spherical_point(1, 6));
    b.push(spherical_point(0.5, 5.5));
    b.push(spherical_point(1.5, 5.5));
    EXPECT_GT(SphericalClipper::overlap_area(a, b), 0.0);
    EXPECT_GT(SphericalClipper::overlap_area(b, a), 0.0);
}
}
