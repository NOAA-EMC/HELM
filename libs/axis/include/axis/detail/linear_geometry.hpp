// SPDX-License-Identifier: Apache-2.0
#ifndef AXIS_DETAIL_LINEAR_GEOMETRY_HPP
#define AXIS_DETAIL_LINEAR_GEOMETRY_HPP

/// @file axis/detail/linear_geometry.hpp
/// @brief Host-side planar and great-circle polygon geometry for linear
///        reconstruction: moments, convex orientation, clipping, and
///        distance-weighted least-squares gradient coefficients.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace axis::detail::linear {
using Point = std::array<double, 3>;
inline Point add(Point a, Point b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline Point sub(Point a, Point b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline Point scale(Point a, double s) {
    return {a[0] * s, a[1] * s, a[2] * s};
}
inline double dot(Point a, Point b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline Point cross(Point a, Point b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(Point a) {
    return std::sqrt(dot(a, a));
}
/// @brief Normalize a point to a unit vector.
/// @throws std::invalid_argument when the vector is too short to define a direction.
inline Point unit(Point a) {
    double n = norm(a);
    if (!(n > 1e-30)) throw std::invalid_argument("linear reconstruction: undefined spherical direction");
    return scale(a, 1 / n);
}

/// @brief Signed area and area-weighted centroid of one polygon.
struct Moments {
    double area{};
    Point mean{};
};

/// @brief Exact area and first moment of a planar polygon.
///
/// Translates before summing to avoid cancellation for projected coordinates
/// far from the origin.
/// @param p Vertices in ring order; fewer than three yields an empty moment.
inline Moments planar_moments(const std::vector<Point> &p) {
    if (p.size() < 3) return {};
    double a2 = 0;
    Point m{};
    for (std::size_t k = 1; k + 1 < p.size(); ++k) {
        auto u = sub(p[k], p[0]), v = sub(p[k + 1], p[0]);
        double c = u[0] * v[1] - u[1] * v[0];
        a2 += c;
        m = add(m, scale(add(u, v), c));
    }
    if (std::abs(a2) < 1e-30) return {};
    return {std::abs(a2) * 0.5, add(p[0], scale(m, 1 / (3 * a2)))};
}

/// @brief Spherical excess area and first moment of a great-circle polygon.
///
/// Uses integral r dOmega = 1/2 sum(theta_edge * n_edge). Signed area and
/// vector moment share the ring orientation. Retaining the unnormalized mean
/// r (inside the sphere) makes mean subtraction conservative.
/// @param p Unit-sphere vertices in ring order; fewer than three yields empty.
inline Moments spherical_moments(const std::vector<Point> &p) {
    if (p.size() < 3) return {};
    double area = 0;
    Point moment{};
    for (std::size_t k = 1; k + 1 < p.size(); ++k) {
        area += 2 * std::atan2(dot(p[0], cross(p[k], p[k + 1])), 1 + dot(p[0], p[k]) + dot(p[k], p[k + 1]) + dot(p[k + 1], p[0]));
    }
    if (std::abs(area) < 1e-30) return {};
    for (std::size_t k = 0; k < p.size(); ++k) {
        auto a = p[k], b = p[(k + 1) % p.size()], n = cross(a, b);
        double s = norm(n);
        if (s > 1e-30) moment = add(moment, scale(n, 0.5 * std::atan2(s, dot(a, b)) / s));
    }
    return {std::abs(area), scale(moment, 1 / area)};
}
/// @brief Orthonormal east/north tangent vectors at a sphere point.
///
/// Picks a helper axis away from the radial direction so the cross products
/// stay well-conditioned at the poles.
inline std::array<Point, 2> tangent_basis(Point mean) {
    auto n = unit(mean);
    Point axis = std::abs(n[2]) < 0.9 ? Point{0, 0, 1} : Point{1, 0, 0};
    auto e = unit(cross(axis, n));
    return {e, cross(n, e)};
}

/// @brief Normalize ring winding and reject geometries this convex method cannot use.
///
/// Reverses the ring when needed so vertices run counterclockwise about the
/// centroid, then verifies convexity edge by edge. Spherical cells must lie in
/// an open hemisphere.
/// @param p        Vertices, reordered in place.
/// @param spherical Use great-circle (true) or planar (false) edge tests.
/// @throws std::invalid_argument for fewer than three vertices, a cell outside
///         an open hemisphere, or a non-convex ring.
inline void orient_convex(std::vector<Point> &p, bool spherical) {
    if (p.size() < 3) throw std::invalid_argument("linear reconstruction requires at least three vertices");
    Point center{};
    for (auto v : p) center = add(center, v);
    center = scale(center, 1.0 / p.size());
    double orientation = 0;
    for (std::size_t k = 0; k < p.size(); ++k) {
        auto a = p[k], b = p[(k + 1) % p.size()];
        orientation += spherical ? dot(cross(a, b), center) : cross(sub(a, center), sub(b, center))[2];
    }
    if (orientation < 0) std::reverse(p.begin(), p.end());
    for (std::size_t k = 0; k < p.size(); ++k) {
        auto a = p[k], b = p[(k + 1) % p.size()];
        auto n = spherical ? cross(a, b) : Point{a[1] - b[1], b[0] - a[0], 0};
        double length = norm(n);
        if (spherical && dot(center, a) <= 0) throw std::invalid_argument("linear reconstruction requires spherical cells in an open hemisphere");
        for (auto v : p) {
            double side = spherical ? dot(n, v) : dot(n, sub(v, a));
            double tolerance = 1e-13 * length * (spherical ? 1.0 : std::max(1.0, norm(sub(v, a))));
            if (side < -tolerance) throw std::invalid_argument("linear reconstruction requires convex cells");
        }
    }
}
/// @brief Sutherland-Hodgman clip of a convex subject by a convex clip polygon.
/// @param subject Convex input ring.
/// @param clip    Convex clip ring.
/// @param out     Intersection, written empty when the rings are disjoint.
/// @param scratch Reusable buffer to avoid per-call allocation.
inline void planar_clip_into(const std::vector<Point> &subject, const std::vector<Point> &clip, std::vector<Point> &out,
                             std::vector<Point> &scratch) {
    out.assign(subject.begin(), subject.end());
    for (std::size_t k = 0; k < clip.size() && !out.empty(); ++k) {
        auto a = clip[k], b = clip[(k + 1) % clip.size()], e = sub(b, a);
        auto side = [&](Point p) {
            auto d = sub(p, a);
            return e[0] * d[1] - e[1] * d[0];
        };
        out.swap(scratch);
        out.clear();
        const auto &input = scratch;
        auto prev = input.back();
        double dp = side(prev);
        for (auto curr : input) {
            double dc = side(curr);
            if ((dc >= 0) != (dp >= 0)) {
                double t = dp / (dp - dc);
                out.push_back(add(prev, scale(sub(curr, prev), t)));
            }
            if (dc >= 0) out.push_back(curr);
            prev = curr;
            dp = dc;
        }
    }
}
/// @brief Convenience wrapper returning the planar intersection by value.
inline std::vector<Point> planar_clip(std::vector<Point> subject, const std::vector<Point> &clip) {
    std::vector<Point> out, scratch;
    planar_clip_into(subject, clip, out, scratch);
    return out;
}

/// @brief Reduce a control-point cloud to its convex hull vertices.
///
/// Duplicate and non-finite inputs are handled conservatively: any non-finite
/// coordinate leaves the set untouched, and fewer than four distinct points is
/// already minimal. Long-double cross products keep the turn test stable.
/// @param points Sorted and replaced in place with the hull in ascending order.
inline void prune_linear_controls(std::vector<std::array<double, 2>> &points) {
    if (points.size() < 4) return;
    for (const auto &p : points)
        if (!std::isfinite(p[0]) || !std::isfinite(p[1])) return;
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() < 4) return;
    auto clockwise = [](const auto &a, const auto &b, const auto &c) {
        long double x = (long double)b[0] - a[0], y = (long double)b[1] - a[1];
        long double u = (long double)c[0] - a[0], v = (long double)c[1] - a[1];
        long double left = x * v, right = y * u;
        return left - right < -64 * std::numeric_limits<long double>::epsilon() * (std::abs(left) + std::abs(right));
    };
    std::vector<std::array<double, 2>> lower, upper;
    for (const auto &p : points) {
        while (lower.size() > 1 && clockwise(lower[lower.size() - 2], lower.back(), p)) lower.pop_back();
        lower.push_back(p);
    }
    for (auto i = points.rbegin(); i != points.rend(); ++i) {
        while (upper.size() > 1 && clockwise(upper[upper.size() - 2], upper.back(), *i)) upper.pop_back();
        upper.push_back(*i);
    }
    lower.pop_back();
    upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    std::sort(lower.begin(), lower.end());
    lower.erase(std::unique(lower.begin(), lower.end()), lower.end());
    points.swap(lower);
}

/// @brief Distance-weighted least-squares gradient stencil coefficients.
///
/// Coefficients multiply (q_neighbor - q_i). Geometry is factored once so the
/// weights can be reused across fields. A rank-deficient stencil returns all-zero
/// coefficients, which safely retains first-order accuracy.
/// @param d Neighbor offsets from the cell centroid in local tangent coordinates.
/// @return One (x, y) coefficient pair per neighbor, in the same order.
inline std::vector<std::array<double, 2>> gradient_coefficients(const std::vector<std::array<double, 2>> &d) {
    double a = 0, b = 0, c = 0;
    for (auto v : d) {
        double r = v[0] * v[0] + v[1] * v[1];
        if (r > 0) {
            a += v[0] * v[0] / r;
            b += v[0] * v[1] / r;
            c += v[1] * v[1] / r;
        }
    }
    std::vector<std::array<double, 2>> coeff(d.size());
    double det = a * c - b * b;
    if (!(det > 1e-12 * (a + c) * (a + c))) return coeff;
    for (std::size_t k = 0; k < d.size(); ++k) {
        auto v = d[k];
        double r = v[0] * v[0] + v[1] * v[1];
        if (r > 0) coeff[k] = {(c * v[0] - b * v[1]) / (det * r), (a * v[1] - b * v[0]) / (det * r)};
    }
    return coeff;
}
}  // namespace axis::detail::linear
#endif
