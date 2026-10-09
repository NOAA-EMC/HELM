// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <array>
#include <numbers>
#include <numeric>

#include "field_quadrature.hpp"
#include "records.hpp"

namespace paper {
// Evaluate the same independent geometry in double and long double. The
// difference provides empirical roundoff evidence, not an interval proof.
// Keeping the scalar type explicit avoids duplicating the scientific formulas.
template <typename Real>
class ReferenceGeometry {
   public:
    static constexpr Real degrees_to_radians = std::numbers::pi_v<Real> / 180;
    struct Vector3 {
        Real x, y, z;
        Vector3 operator+(Vector3 b) const {
            return {x + b.x, y + b.y, z + b.z};
        }
        Vector3 operator-(Vector3 b) const {
            return {x - b.x, y - b.y, z - b.z};
        }
        Vector3 operator*(Real s) const {
            return {x * s, y * s, z * s};
        }
    };
    static Real dot(Vector3 a, Vector3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }
    static Vector3 cross(Vector3 a, Vector3 b) {
        return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
    }
    static Real length(Vector3 a) {
        return std::sqrt(dot(a, a));
    }
    static Vector3 normalized(Vector3 a) {
        return a * (1 / length(a));
    }
    static Vector3 unit_sphere_position(Real lon, Real lat) {
        return {std::cos(lat * degrees_to_radians) * std::cos(lon * degrees_to_radians),
                std::cos(lat * degrees_to_radians) * std::sin(lon * degrees_to_radians),
                std::sin(lat * degrees_to_radians)};
    }
    using SphericalPolygon = std::vector<Vector3>;
    static SphericalPolygon polygon(const Mesh &m, size_t j) {
        SphericalPolygon p;
        for (int k = m.offsets[j]; k < m.offsets[j + 1]; ++k) {
            int n = m.indices[k];
            p.push_back(unit_sphere_position(m.lon[n], m.lat[n]));
        }
        return p;
    }
    static Real area(const SphericalPolygon &p) {
        if (p.size() < 3)
            return 0;
        Real s = 0;
        for (size_t i = 1; i + 1 < p.size(); ++i) {
            auto a = p[0], b = p[i], c = p[i + 1];
            s += 2 * std::atan2(dot(a, cross(b, c)), 1 + dot(a, b) + dot(b, c) + dot(c, a));
        }
        return std::abs(s);
    }
    static Real smooth_field_integral(const SphericalPolygon &p, const std::string &field) {
        Real a = area(p);
        if (field == "constant")
            return a;
        // The boundary integral gives the vector area: integrating x over the
        // spherical polygon is its x component. This is exact for great-circle
        // edges and avoids borrowing either engine's overlap implementation.
        Vector3 v{0, 0, 0};
        for (size_t i = 0; i < p.size(); ++i) {
            Vector3 e = cross(p[i], p[(i + 1) % p.size()]);
            Real l = length(e);
            if (l > 1e-15)
                v = v + e * (.5 * std::atan2(l, dot(p[i], p[(i + 1) % p.size()])) / l);
        }
        if (field == "zero")
            return v.x;
        if (field == "smooth")
            return 2 * a + v.x;
        throw Error(2, "Unknown analytic field");
    }
    struct Point2 {
        Real x, y;
    };
    static Real oriented_side(Point2 a, Point2 b, Point2 c) {
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    }
    struct GnomonicChart {
        Vector3 n, u, v;
        explicit GnomonicChart(const SphericalPolygon &p) {
            n = {0, 0, 0};
            for (auto q : p)
                n = n + q;
            n = normalized(n);
            u = normalized(cross(std::abs(n.z) < .9 ? Vector3{0, 0, 1} : Vector3{1, 0, 0}, n));
            v = cross(n, u);
        }
        Point2 project(Vector3 x) const {
            Real d = dot(n, x);
            if (d <= 1e-8)
                throw Error(6, "Reference polygon outside local hemisphere");
            return {dot(x, u) / d, dot(x, v) / d};
        }
        Vector3 lift(Point2 x) const {
            return normalized(n + u * x.x + v * x.y);
        }
    };
    static std::vector<Point2> projected(const SphericalPolygon &p, const GnomonicChart &c) {
        std::vector<Point2> q;
        for (auto v : p)
            q.push_back(c.project(v));
        Real a = 0;
        for (size_t i = 0; i < q.size(); ++i)
            a += q[i].x * q[(i + 1) % q.size()].y - q[i].y * q[(i + 1) % q.size()].x;
        if (a < 0)
            throw Error(6, "Reference requires CCW polygons");
        for (size_t i = 0; i < q.size(); ++i)
            if (oriented_side(q[i], q[(i + 1) % q.size()], q[(i + 2) % q.size()]) < -1e-12)
                throw Error(4, "Reference requires convex cells; triangulate explicitly");
        return q;
    }
    static std::vector<Point2> clip(std::vector<Point2> q, Point2 a, Point2 b) {
        std::vector<Point2> out;
        if (q.empty())
            return out;
        Point2 prev = q.back();
        Real sp = oriented_side(a, b, prev);
        for (Point2 cur : q) {
            Real sc = oriented_side(a, b, cur);
            if ((sc >= 0) != (sp >= 0)) {
                Real t = sp / (sp - sc);
                out.push_back({prev.x + t * (cur.x - prev.x), prev.y + t * (cur.y - prev.y)});
            }
            if (sc >= 0)
                out.push_back(cur);
            prev = cur;
            sp = sc;
        }
        return out;
    }
    static SphericalPolygon intersect(const SphericalPolygon &a, const SphericalPolygon &b) {
        GnomonicChart c(a);
        auto p = projected(a, c);
        auto q = projected(b, c);
        for (size_t i = 0; i < q.size(); ++i)
            p = clip(p, q[i], q[(i + 1) % q.size()]);
        SphericalPolygon out;
        for (Point2 v : p)
            out.push_back(c.lift(v));
        return out;
    }
    static SphericalPolygon hemisphere(const SphericalPolygon &p) {
        if (p.size() < 3 || area(p) <= 1e-24)
            return {};
        // Avoid projecting collapsed overlap slivers when hemisphere membership is
        // already known from all vertices. Great-circle edges stay in a hemisphere.
        if (std::all_of(p.begin(), p.end(), [](Vector3 vertex) { return vertex.x >= 0; }))
            return p;
        if (std::all_of(p.begin(), p.end(), [](Vector3 vertex) { return vertex.x <= 0; }))
            return {};
        GnomonicChart c(p);
        auto q = projected(p, c);
        std::vector<Point2> o;
        Point2 prev = q.back();
        auto value = [&](Point2 x) { return c.n.x + c.u.x * x.x + c.v.x * x.y; };
        Real sp = value(prev);
        for (Point2 cur : q) {
            Real sc = value(cur);
            if ((sc >= 0) != (sp >= 0)) {
                Real t = sp / (sp - sc);
                o.push_back({prev.x + t * (cur.x - prev.x), prev.y + t * (cur.y - prev.y)});
            }
            if (sc >= 0)
                o.push_back(cur);
            prev = cur;
            sp = sc;
        }
        SphericalPolygon out;
        for (Point2 x : o)
            out.push_back(c.lift(x));
        return out;
    }
    static Real field_integral(const SphericalPolygon &p, const std::string &f) {
        return f == "sharp" ? area(hemisphere(p)) : smooth_field_integral(p, f);
    }
    static Real integrated_field(const SphericalPolygon &polygon, const Case &c, Real &error) {
        if (!valcke_field(c.field))
            return field_integral(polygon, c.field);
        if (polygon.size() < 3)
            return 0;
        Real value = 0;
        size_t work = 0;
        const Real budget = c.tolerance * paper::field_bound(c.field) * area(polygon) / 16;
        for (size_t i = 1; i + 1 < polygon.size(); ++i) {
            auto convert = [](Vector3 p) {
                return std::array<double, 3>{double(p.x), double(p.y), double(p.z)};
            };
            auto integral = FieldQuadrature<double>::triangle(
                convert(polygon[0]), convert(polygon[i]), convert(polygon[i + 1]), c.field,
                budget / (polygon.size() - 2), c.depth, work, c.budget);
            value += integral.value;
            error += integral.uncertainty;
        }
        return value;
    }
    struct Bound {
        Vector3 center;
        Real radius;
    };
    static Bound bound(const SphericalPolygon &p) {
        GnomonicChart c(p);
        Real r = 0;
        for (auto v : p)
            r = std::max(r, std::acos(std::clamp(dot(c.n, v), Real(-1), Real(1))));
        if (r >= std::numbers::pi_v<Real> / 2 - 1e-8)
            throw Error(4, "Cells must lie in an open hemisphere");
        return {c.n, r};
    }
    static bool candidate(Bound a, Bound b) {
        return dot(a.center, b.center) >=
               std::cos(std::min(std::numbers::pi_v<Real>, a.radius + b.radius + 1e-10));
    }
    // Longitude rectangles use exact spherical strip integration; antimeridian
    // cells are unwrapped around their center and periodic copies intersected.
    using R = std::array<Real, 4>;
    static R rectangle(const Mesh &m, size_t j) {
        R r{1e30, -1e30, 1e30, -1e30};
        Real center = m.center_lon[j];
        for (int k = m.offsets[j]; k < m.offsets[j + 1]; ++k) {
            int n = m.indices[k];
            Real lon = center + std::remainder(m.lon[n] - center, 360.);
            r[0] = std::min(r[0], lon * degrees_to_radians);
            r[1] = std::max(r[1], lon * degrees_to_radians);
            r[2] = std::min(r[2], m.lat[n] * degrees_to_radians);
            r[3] = std::max(r[3], m.lat[n] * degrees_to_radians);
        }
        return r;
    }
    static Real rect_area(R r) {
        return std::max(Real(0), r[1] - r[0]) * std::max(Real(0), std::sin(r[3]) - std::sin(r[2]));
    }
    static Real rect_integral(R r, const std::string &f) {
        Real a = rect_area(r);
        if (a <= 0)
            return 0;
        if (f == "constant")
            return a;
        Real x = (std::sin(r[1]) - std::sin(r[0])) *
                 ((r[3] - r[2]) / 2 + (std::sin(2 * r[3]) - std::sin(2 * r[2])) / 4);
        if (f == "smooth")
            return 2 * a + x;
        if (f == "zero")
            return x;
        if (f == "sharp") {
            Real total = 0;
            for (int k = -3; k <= 3; ++k) {
                auto q = r;
                q[0] = std::max(q[0],
                                -std::numbers::pi_v<Real> / 2 + 2 * k * std::numbers::pi_v<Real>);
                q[1] =
                    std::min(q[1], std::numbers::pi_v<Real> / 2 + 2 * k * std::numbers::pi_v<Real>);
                total += rect_area(q);
            }
            return total;
        }
        throw Error(2, "Unknown field");
    }
    static R rect_intersection(R a, R b) {
        return {std::max(a[0], b[0]), std::min(a[1], b[1]), std::max(a[2], b[2]),
                std::min(a[3], b[3])};
    }
    static Reference compute_reference(const Case &c) {
        Reference r;
        size_t ns = c.src.size(), nd = c.dst.size();
        r.src.resize(ns);
        r.source_uncertainty.assign(ns, 0);
        if (valcke_field(c.field) && c.geometry != "great_circle")
            throw Error(4, "Valcke quadrature requires great-circle cells");
        r.dst.assign(nd, 0);
        r.src_area.resize(ns);
        r.dst_area.resize(nd);
        r.src_covered.assign(ns, 0);
        r.dst_covered.assign(nd, 0);
        r.uncertainty.assign(nd, 0);
        r.roundoff_discarded_area.assign(nd, 0);
        std::vector<SphericalPolygon> s, d;
        std::vector<Bound> sb, db;
        std::vector<R> sr, dr;
        for (size_t i = 0; i < ns; ++i) {
            if (c.geometry == "constant_latitude") {
                sr.push_back(rectangle(c.src, i));
                r.src_area[i] = rect_area(sr.back());
                r.src[i] = rect_integral(sr.back(), c.field) / r.src_area[i];
            } else {
                s.push_back(polygon(c.src, i));
                sb.push_back(bound(s.back()));
                projected(s.back(), GnomonicChart(s.back()));
                r.src_area[i] = area(s.back());
                Real error = 0;
                r.src[i] = integrated_field(s.back(), c, error) / r.src_area[i];
                r.source_uncertainty[i] = error / r.src_area[i];
            }
        }
        for (size_t j = 0; j < nd; ++j) {
            if (c.geometry == "constant_latitude") {
                dr.push_back(rectangle(c.dst, j));
                r.dst_area[j] = rect_area(dr.back());
            } else {
                d.push_back(polygon(c.dst, j));
                db.push_back(bound(d.back()));
                projected(d.back(), GnomonicChart(d.back()));
                r.dst_area[j] = area(d.back());
            }
        }
        for (size_t j = 0; j < nd; ++j) {
            if (!c.dst.mask[j])
                continue;
            Real total = 0, quadrature_error = 0;
            std::vector<SphericalPolygon> fragments;
            size_t contributions = 0;
            for (size_t i = 0; i < ns; ++i) {
                if (!c.src.mask[i])
                    continue;
                Real a = 0, f = 0;
                if (c.geometry == "constant_latitude") {
                    for (int k = -1; k <= 1; ++k) {
                        auto shifted = sr[i];
                        shifted[0] += 2 * k * std::numbers::pi_v<Real>;
                        shifted[1] += 2 * k * std::numbers::pi_v<Real>;
                        auto q = rect_intersection(dr[j], shifted);
                        a += rect_area(q);
                        f += rect_integral(q, c.field);
                    }
                } else {
                    if (!candidate(sb[i], db[j]))
                        continue;
                    auto q = intersect(d[j], s[i]);
                    a = area(q);
                    const Real evaluable_area = 64 * std::numeric_limits<double>::epsilon() *
                                                std::min(r.src_area[i], r.dst_area[j]);
                    // Classify slivers before the sharp-field hemisphere clip:
                    // a collapsed polygon need not have a reliable winding.
                    if (a > evaluable_area) {
                        if (valcke_field(c.field))
                            fragments.push_back(std::move(q));
                        else
                            f = field_integral(q, c.field);
                    }
                }
                // Intersections far below double-precision area resolution
                // cannot define a reliable FracArea average. Account for their
                // area instead of dividing by a roundoff-sized denominator.
                const Real sliver_limit = 64 * std::numeric_limits<double>::epsilon() *
                                          std::min(r.src_area[i], r.dst_area[j]);
                if (a > 0 && a <= sliver_limit) {
                    r.roundoff_discarded_area[j] += a;
                    a = 0;
                    f = 0;
                }
                r.src_covered[i] += a;
                r.dst_covered[j] += a;
                total += f;
                if (a > 0)
                    ++contributions;
            }
            if (r.dst_covered[j] > r.dst_area[j] * (1 + 1e-9))
                throw Error(6, "Source cells overlap or reference coverage exceeds destination");
            if (valcke_field(c.field)) {
                // Integrate the destination once if fully covered. Otherwise
                // integrate the independently constructed common-domain pieces.
                if (std::abs(r.dst_covered[j] - r.dst_area[j]) <= c.area_tolerance * r.dst_area[j])
                    total = integrated_field(d[j], c, quadrature_error);
                else
                    for (const auto &fragment : fragments)
                        total += integrated_field(fragment, c, quadrature_error);
            }
            Real denominator = c.norm == "fracarea" ? r.dst_covered[j] : r.dst_area[j];
            if (denominator > 0)
                r.dst[j] = total / denominator;
            // Conservative floating-point estimate, not a quadrature convergence claim.
            r.uncertainty[j] = 64 * std::numeric_limits<double>::epsilon() *
                                   std::max(size_t(1), contributions) *
                                   paper::field_bound(c.field) +
                               (denominator > 0 ? quadrature_error / denominator : 0);
            if (r.uncertainty[j] > c.tolerance * paper::field_bound(c.field))
                throw Error(6, "Analytic reference roundoff estimate exceeds tolerance");
        }
        for (size_t i = 0; i < ns; ++i)
            if (r.src_covered[i] > r.src_area[i] * (1 + 1e-9))
                throw Error(6, "Destination cells overlap");
        return r;
    }
};
}  // namespace paper
