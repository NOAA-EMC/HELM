// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include <queue>

#include "analytical_fields.hpp"

namespace paper {
// Independent integration over radial projections of planar triangles.
// The Jacobian is |det(a,b,c)|/|a+u(b-a)+v(c-a)|^3; a Duffy transform
// maps the integration triangle to [0,1]^2. Compare 6/12-point tensor rules
// and subdivide geodesically when necessary. Estimates are empirical.
template <typename Real>
class FieldQuadrature {
   public:
    using Point = std::array<Real, 3>;
    struct Integral {
        Real value = 0, uncertainty = 0;
    };
    struct Rule {
        std::vector<Real> points, weights;
    };
    static Real dot(Point a, Point b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    static Point midpoint(Point a, Point b) {
        Point p{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
        Real length = std::sqrt(dot(p, p));
        for (auto &x : p)
            x /= length;
        return p;
    }
    static Rule gauss(int n) {
        Rule rule;
        for (int i = 0; i < n; ++i) {
            Real root = std::cos(std::numbers::pi_v<Real> * (i + Real(.75)) / (n + Real(.5)));
            Real derivative = 0;
            for (int iteration = 0; iteration < 50; ++iteration) {
                Real previous = 1, current = root;
                for (int degree = 2; degree <= n; ++degree) {
                    Real next =
                        ((2 * degree - 1) * root * current - (degree - 1) * previous) / degree;
                    previous = current;
                    current = next;
                }
                derivative = n * (root * current - previous) / (root * root - 1);
                Real delta = current / derivative;
                root -= delta;
                if (std::abs(delta) < 8 * std::numeric_limits<Real>::epsilon())
                    break;
            }
            rule.points.push_back((root + 1) / 2);
            rule.weights.push_back(1 / ((1 - root * root) * derivative * derivative));
        }
        return rule;
    }
    static Real evaluate(Point a, Point b, Point c, const std::string &field, const Rule &rule) {
        const Real determinant =
            std::abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
                     a[2] * (b[0] * c[1] - b[1] * c[0]));
        Real sum = 0;
        for (size_t i = 0; i < rule.points.size(); ++i)
            for (size_t j = 0; j < rule.points.size(); ++j) {
                Real u = rule.points[i], v = (1 - u) * rule.points[j];
                Point point;
                for (int k = 0; k < 3; ++k)
                    point[k] = a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]);
                Real length = std::sqrt(dot(point, point));
                for (auto &x : point)
                    x /= length;
                sum += rule.weights[i] * rule.weights[j] * (1 - u) * determinant /
                       (length * length * length) * analytical_value(point, field);
            }
        return sum;
    }
    // Bound the jet phase over a spherical cap containing the triangle.
    // This detects a narrow jet even when both quadrature rules miss it.
    static bool possible_jet(Point a, Point b, Point c) {
        const Real pi = std::numbers::pi_v<Real>, radians = pi / 180;
        Point center{a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]};
        Real length = std::sqrt(dot(center, center));
        for (auto &value : center)
            value /= length;
        Real radius = 0;
        for (Point vertex : {a, b, c})
            radius =
                std::max(radius, std::acos(std::clamp(dot(center, vertex), Real(-1), Real(1))));
        radius += 1e-12;
        Real latitude = std::asin(std::clamp(center[2], Real(-1), Real(1)));
        if (radius >= pi / 2 || std::abs(latitude) + radius >= pi / 2)
            return true;
        Real longitude = std::atan2(center[1], center[0]);
        Real longitude_radius = std::asin(std::min(Real(1), std::sin(radius) / std::cos(latitude)));
        if (longitude - longitude_radius < -pi || longitude + longitude_radius > pi)
            return true;
        Real xmin = longitude - longitude_radius + 80 * radians;
        Real xmax = longitude + longitude_radius + 80 * radians;
        Real ymin = latitude - radius - 25 * radians;
        Real ymax = latitude + radius - 25 * radians;
        auto distance = [](Real low, Real high) {
            return low > 0 ? low : high < 0 ? -high : Real(0);
        };
        Real rmin = std::hypot(distance(xmin, xmax), distance(ymin, ymax));
        Real rmax = 0, amin = pi, amax = -pi;
        for (Real x : {xmin, xmax})
            for (Real y : {ymin, ymax}) {
                rmax = std::max(rmax, std::hypot(x, y));
                amin = std::min(amin, std::atan2(y, x));
                amax = std::max(amax, std::atan2(y, x));
            }
        if (rmin > std::hypot(Real(78.2), Real(25)) * radians)
            return false;
        if (xmin <= 0 && ymin <= 0 && ymax >= 0) {
            amin = -pi;
            amax = pi;
        }
        Real low = Real(.4) * (Real(.5) * rmin + amin) + Real(.37) * pi - Real(.007);
        Real high = Real(.4) * (Real(.5) * rmax + amax) + Real(.37) * pi + Real(.007);
        Real width = std::acos(Real(.999));
        return low <= pi / 2 + width && high >= pi / 2 - width;
    }
    struct Leaf {
        std::array<Point, 3> vertices;
        Integral integral;
        int depth;
        bool operator<(const Leaf &other) const {
            return integral.uncertainty < other.integral.uncertainty;
        }
    };
    static Integral triangle(Point a, Point b, Point c, const std::string &field, Real budget,
                             int depth, size_t &work, size_t max_work) {
        static const Rule coarse = gauss(6), fine = gauss(12);
        auto evaluate_leaf = [&](std::array<Point, 3> vertices, int level) {
            if (++work > max_work)
                throw Error(6, "Analytical field quadrature work limit exceeded");
            Real low = evaluate(vertices[0], vertices[1], vertices[2], field, coarse);
            Real high = evaluate(vertices[0], vertices[1], vertices[2], field, fine);
            Real error = 4 * std::abs(high - low) +
                         32 * std::numeric_limits<Real>::epsilon() * std::abs(high);
            if (field == "gulfstream" && possible_jet(vertices[0], vertices[1], vertices[2])) {
                Real diameter = 0;
                for (int edge = 0; edge < 3; ++edge)
                    diameter =
                        std::max(diameter,
                                 std::acos(std::clamp(dot(vertices[edge], vertices[(edge + 1) % 3]),
                                                      Real(-1), Real(1))));
                // Force spatial sampling through the thin jet before trusting
                // agreement of the two tensor rules. Below this scale, normal
                // adaptive integration resolves its clipped boundaries.
                if (diameter > std::numbers::pi_v<Real> / 1800) {
                    Real determinant = std::abs(dot(
                        vertices[0],
                        Point{vertices[1][1] * vertices[2][2] - vertices[1][2] * vertices[2][1],
                              vertices[1][2] * vertices[2][0] - vertices[1][0] * vertices[2][2],
                              vertices[1][0] * vertices[2][1] - vertices[1][1] * vertices[2][0]}));
                    Real area = 2 * std::atan2(determinant, 1 + dot(vertices[0], vertices[1]) +
                                                                dot(vertices[1], vertices[2]) +
                                                                dot(vertices[2], vertices[0]));
                    error = std::max(error, Real(1.3) * area);
                }
            }
            return Leaf{vertices, {high, error}, level};
        };
        auto initial = evaluate_leaf({a, b, c}, 0);
        std::priority_queue<Leaf> leaves;
        leaves.push(initial);
        Integral total = initial.integral;
        // Allocate a global absolute error budget to this triangle, refining
        // the largest error contribution first. A narrow nonsmooth feature
        // need not satisfy the whole-cell tolerance in every tiny subcell.
        while (total.uncertainty > budget) {
            auto parent = leaves.top();
            leaves.pop();
            if (parent.depth >= depth)
                throw Error(6, "Analytical field quadrature depth limit exceeded");
            auto [x, y, z] = parent.vertices;
            auto xy = midpoint(x, y), yz = midpoint(y, z), zx = midpoint(z, x);
            total.value -= parent.integral.value;
            total.uncertainty -= parent.integral.uncertainty;
            for (auto vertices : std::array<std::array<Point, 3>, 4>{
                     {{x, xy, zx}, {xy, y, yz}, {zx, yz, z}, {xy, yz, zx}}}) {
                auto child = evaluate_leaf(vertices, parent.depth + 1);
                total.value += child.integral.value;
                total.uncertainty += child.integral.uncertainty;
                leaves.push(child);
            }
            total.uncertainty = std::max(Real(0), total.uncertainty);
        }
        return total;
    }
};
}  // namespace paper
