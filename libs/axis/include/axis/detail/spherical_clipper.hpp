// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_SPHERICAL_CLIPPER_HPP
#define AXIS_DETAIL_SPHERICAL_CLIPPER_HPP

/// @file axis/detail/spherical_clipper.hpp
/// @brief GPU-portable spherical polygon clipping via Greiner-Hormann on the
///        unit sphere.
///
/// Provides:
///   - Vec3: minimal 3-D vector with KOKKOS_FUNCTION arithmetic
///   - SphericalPolygon<MaxVerts>: fixed-capacity polygon on the unit sphere
///   - SphericalClipper: Greiner-Hormann clipping adapted for great-circle arcs
///
/// All functions are annotated KOKKOS_FUNCTION for device portability.
/// No heap allocation — safe for use in Kokkos parallel kernels.

#include <Kokkos_Core.hpp>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// Vec3 — minimal 3-D vector for unit-sphere Cartesian coordinates
// ─────────────────────────────────────────────────────────────────────────────

/// @brief A device-portable 3-D vector with basic operations.
struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};

    KOKKOS_FUNCTION Vec3() = default;
    KOKKOS_FUNCTION Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
};

/// @brief Dot product of two vectors.
KOKKOS_INLINE_FUNCTION
double dot(const Vec3 &a, const Vec3 &b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/// @brief Cross product of two vectors.
KOKKOS_INLINE_FUNCTION
Vec3 cross(const Vec3 &a, const Vec3 &b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

/// @brief Squared length of a vector.
KOKKOS_INLINE_FUNCTION
double length_sq(const Vec3 &v) noexcept {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

/// @brief Length of a vector.
KOKKOS_INLINE_FUNCTION
double length(const Vec3 &v) noexcept {
    return Kokkos::sqrt(length_sq(v));
}

/// @brief Normalize a vector to unit length. Returns zero vector if input is
///        near-zero.
KOKKOS_INLINE_FUNCTION
Vec3 normalize(const Vec3 &v) noexcept {
    double len = length(v);
    if (len < 1.0e-300) return {0.0, 0.0, 0.0};
    double inv = 1.0 / len;
    return {v.x * inv, v.y * inv, v.z * inv};
}

/// @brief Negate a vector.
KOKKOS_INLINE_FUNCTION
Vec3 negate(const Vec3 &v) noexcept {
    return {-v.x, -v.y, -v.z};
}

/// @brief Add two vectors.
KOKKOS_INLINE_FUNCTION
Vec3 add(const Vec3 &a, const Vec3 &b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

/// @brief Scale a vector by a scalar.
KOKKOS_INLINE_FUNCTION
Vec3 scale(const Vec3 &v, double s) noexcept {
    return {v.x * s, v.y * s, v.z * s};
}

// ─────────────────────────────────────────────────────────────────────────────
// SphericalPolygon — fixed-capacity polygon on the unit sphere
// ─────────────────────────────────────────────────────────────────────────────

/// @brief A fixed-capacity polygon on the unit sphere.
///
/// Vertices are stored as unit-sphere Cartesian coordinates. No heap allocation,
/// making this safe for use in Kokkos device kernels.
///
/// @tparam MaxVerts Maximum number of vertices this polygon can hold (default 32).
template <int MaxVerts = 32>
struct SphericalPolygon {
    Vec3 verts[MaxVerts];  ///< Unit sphere vertices (x, y, z)
    int n{0};              ///< Current vertex count

    /// @brief Compute the spherical area via spherical excess (Girard's theorem).
    ///
    /// Decomposes the polygon into triangles from vertex 0 and sums the
    /// spherical excess of each using the formula:
    ///   E = 2 * atan2(|a·(b×c)|, 1 + a·b + a·c + b·c)
    ///
    /// @return Area in steradians (non-negative).
    KOKKOS_FUNCTION double area() const noexcept {
        if (n < 3) return 0.0;

        double total = 0.0;
        const Vec3 &a = verts[0];

        for (int i = 1; i < n - 1; ++i) {
            const Vec3 &b = verts[i];
            const Vec3 &c = verts[i + 1];

            // Spherical excess of triangle (a, b, c) via Van Oosterom-Strackee:
            //   tan(E/2) = |a·(b×c)| / (1 + a·b + a·c + b·c)
            double ab = dot(a, b);
            double ac = dot(a, c);
            double bc = dot(b, c);

            Vec3 bxc = cross(b, c);
            double numerator = dot(a, bxc);
            double denominator = 1.0 + ab + ac + bc;

            double tri_area = 2.0 * Kokkos::atan2(numerator, denominator);
            total += tri_area;
        }

        return Kokkos::fabs(total);
    }

    /// @brief Check if the polygon is empty (fewer than 3 vertices).
    KOKKOS_FUNCTION bool empty() const noexcept {
        return n < 3;
    }

    /// @brief Add a vertex to the polygon (clamps at MaxVerts).
    KOKKOS_FUNCTION void push(const Vec3 &v) noexcept {
        if (n < MaxVerts) {
            verts[n] = v;
            ++n;
        }
    }

    /// @brief Clear all vertices.
    KOKKOS_FUNCTION void clear() noexcept {
        n = 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SphericalClipper — Greiner-Hormann clipping on the unit sphere
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Clips spherical polygons against each other using a Sutherland-Hodgman
///        style algorithm adapted for great-circle arcs on the unit sphere.
///
/// The algorithm iterates over each edge of the clip polygon and clips the
/// subject polygon against the half-sphere defined by that great-circle edge.
/// This is equivalent to Greiner-Hormann for convex/concave intersections on
/// the sphere.
///
/// Degenerate cases handled:
///   - Vertex exactly on edge (treat as "inside" — inclusive boundary)
///   - Coincident/near-coincident edges (skip clipping against that edge)
///   - Result with < 3 vertices: returns empty polygon
struct SphericalClipper {
    /// @brief Clip the subject polygon against the clip polygon.
    ///
    /// @tparam M Maximum vertex capacity of the polygons.
    /// @param subject The polygon to be clipped.
    /// @param clip_poly The polygon defining the clipping boundary.
    /// @return The intersection polygon (may be empty).
    template <int M>
    KOKKOS_FUNCTION static SphericalPolygon<M> clip(const SphericalPolygon<M> &subject, const SphericalPolygon<M> &clip_poly) noexcept {
        if (subject.empty() || clip_poly.empty()) {
            return SphericalPolygon<M>{};
        }

        if (separated_at_shared_vertex(subject, clip_poly) || separated_at_shared_vertex(clip_poly, subject)) return SphericalPolygon<M>{};

        // We use two buffers to alternate input/output during iterative clipping.
        SphericalPolygon<M> buf_a;
        SphericalPolygon<M> buf_b;

        // Copy subject into buf_a as initial input.
        buf_a.n = subject.n;
        for (int i = 0; i < subject.n; ++i) {
            buf_a.verts[i] = subject.verts[i];
        }

        // For each edge of clip_poly, clip the current polygon.
        for (int ci = 0; ci < clip_poly.n; ++ci) {
            const Vec3 &edge_a = clip_poly.verts[ci];
            const Vec3 &edge_b = clip_poly.verts[(ci + 1) % clip_poly.n];

            // Compute the great-circle normal for this clip edge.
            // Points "inside" satisfy dot(normal, point) >= 0.
            Vec3 normal = cross(edge_a, edge_b);

            // Check for degenerate clip edge (coincident vertices).
            if (length_sq(normal) < 1.0e-30) {
                // Skip this edge — it's degenerate.
                continue;
            }

            // Clip buf_a against this edge, output to buf_b.
            buf_b.clear();
            clip_against_edge(buf_a, normal, edge_a, edge_b, buf_b);

            // Swap: buf_b becomes the new input for the next edge.
            buf_a.n = buf_b.n;
            for (int i = 0; i < buf_b.n; ++i) {
                buf_a.verts[i] = buf_b.verts[i];
            }

            // Early exit if clipped to nothing.
            if (buf_a.n < 3) {
                return SphericalPolygon<M>{};
            }
        }

        return buf_a;
    }

    /// @brief Compute the overlap area between two spherical polygons.
    ///
    /// Convenience wrapper that clips and then computes the area of the result.
    ///
    /// @tparam M Maximum vertex capacity of the polygons.
    /// @param a First polygon.
    /// @param b Second polygon.
    /// @return Overlap area in steradians (>= 0).
    template <int M>
    KOKKOS_FUNCTION static double overlap_area(const SphericalPolygon<M> &a, const SphericalPolygon<M> &b) noexcept {
        SphericalPolygon<M> result = clip(a, b);
        return result.area();
    }

   private:
    // A shared vertex can be excluded from an edge's half-space test exactly.
    // Require a safely negative signed distance for every other vertex.
    template <int M>
    KOKKOS_FUNCTION static bool separated_at_shared_vertex(const SphericalPolygon<M> &subject, const SphericalPolygon<M> &boundary) noexcept {
        constexpr double roundoff = 16.0 * 2.2204460492503131e-16;
        for (int i = 0; i < boundary.n; ++i) {
            const Vec3 &a = boundary.verts[i];
            const Vec3 &b = boundary.verts[(i + 1) % boundary.n];
            const Vec3 normal = cross(a, b);
            if (length_sq(normal) < 1.0e-30) continue;
            bool shared = false;
            bool separated = true;
            for (int j = 0; j < subject.n; ++j) {
                const Vec3 &v = subject.verts[j];
                if ((v.x == a.x && v.y == a.y && v.z == a.z) || (v.x == b.x && v.y == b.y && v.z == b.z)) {
                    shared = true;
                    continue;
                }
                const double distance = dot(normal, v);
                const double scale = (Kokkos::fabs(a.y * b.z) + Kokkos::fabs(a.z * b.y)) * Kokkos::fabs(v.x) +
                                     (Kokkos::fabs(a.z * b.x) + Kokkos::fabs(a.x * b.z)) * Kokkos::fabs(v.y) +
                                     (Kokkos::fabs(a.x * b.y) + Kokkos::fabs(a.y * b.x)) * Kokkos::fabs(v.z);
                if (distance >= -roundoff * scale) {
                    separated = false;
                    break;
                }
            }
            if (shared && separated) return true;
        }
        return false;
    }

    /// @brief Determine if a point is "inside" a half-sphere defined by a
    ///        great-circle normal.
    ///
    /// A point V is inside edge (A→B) if dot(cross(A, B), V) >= 0.
    /// Vertices exactly on the boundary (dot ≈ 0) are treated as inside
    /// (inclusive boundary) for robustness.
    KOKKOS_FUNCTION
    static double signed_distance(const Vec3 &normal, const Vec3 &point) noexcept {
        return dot(normal, point);
    }

    /// @brief Compute the intersection of a great-circle arc (v1→v2) with the
    ///        great circle defined by (edge_a, edge_b).
    ///
    /// The intersection point lies on the plane defined by (edge_a, edge_b, origin)
    /// and on the arc from v1 to v2.
    ///
    /// Formula: I = normalize(cross(cross(edge_a, edge_b), cross(v1, v2)))
    /// choosing the sign that lies between v1 and v2.
    KOKKOS_FUNCTION
    static Vec3 compute_intersection(const Vec3 &v1, const Vec3 &v2, const Vec3 &edge_a, const Vec3 &edge_b) noexcept {
        // Normal of the clip edge great circle
        Vec3 n1 = cross(edge_a, edge_b);
        // Normal of the arc great circle
        Vec3 n2 = cross(v1, v2);

        // Direction of intersection line
        Vec3 d = cross(n1, n2);
        double d_len_sq = length_sq(d);

        if (d_len_sq < 1.0e-30) {
            // Arcs lie on the same or antipodal great circles.
            // Fallback: return normalized midpoint.
            Vec3 mid = add(v1, v2);
            return normalize(mid);
        }

        // Normalize the candidate
        Vec3 candidate = normalize(d);

        // Choose the sign: the intersection should be "between" v1 and v2.
        // Check via dot products: candidate should have positive dot with
        // both v1 and v2 (or at least with their midpoint direction).
        Vec3 mid_dir = normalize(add(v1, v2));
        if (dot(candidate, mid_dir) < 0.0) {
            candidate = negate(candidate);
        }

        return candidate;
    }

    /// @brief Clip a polygon against a single great-circle edge.
    ///
    /// This is the core Sutherland-Hodgman step: for each edge of the input
    /// polygon, output vertices based on inside/outside classification.
    template <int M>
    KOKKOS_FUNCTION static void clip_against_edge(const SphericalPolygon<M> &input, const Vec3 &normal, const Vec3 &edge_a, const Vec3 &edge_b,
                                                  SphericalPolygon<M> &output) noexcept {
        if (input.n < 1) return;

        // Tolerance for treating a point as "on the boundary" (inclusive).
        constexpr double eps = -1.0e-14;

        for (int i = 0; i < input.n; ++i) {
            int prev_idx = (i + input.n - 1) % input.n;
            const Vec3 &curr = input.verts[i];
            const Vec3 &prev = input.verts[prev_idx];

            double d_curr = signed_distance(normal, curr);
            double d_prev = signed_distance(normal, prev);

            // "Inside" means d >= eps (inclusive boundary with tolerance).
            bool curr_inside = (d_curr >= eps);
            bool prev_inside = (d_prev >= eps);

            if (curr_inside) {
                if (!prev_inside) {
                    // Transition from outside to inside: add intersection.
                    Vec3 inter = compute_intersection(prev, curr, edge_a, edge_b);
                    output.push(inter);
                }
                output.push(curr);
            } else if (prev_inside) {
                // Transition from inside to outside: add intersection.
                Vec3 inter = compute_intersection(prev, curr, edge_a, edge_b);
                output.push(inter);
            }
            // else: both outside — output nothing.
        }
    }
};

}  // namespace axis::detail

#endif  // AXIS_DETAIL_SPHERICAL_CLIPPER_HPP
