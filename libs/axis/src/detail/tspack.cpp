// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/detail/tspack.cpp
/// @brief Tension-spline (TSPACK) evaluation kernels for vertical interpolation.
///
/// Provides the Kokkos-callable cubic and tension spline basis used by the
/// vertical regridder to interpolate between level knots.

#include <axis/detail/tspack.hpp>
#include <cmath>

namespace axis::detail::tspack {

namespace {

/// @brief Crossover between the natural-cubic and hyperbolic-tension branches.
///
/// The tension basis {1, t, cosh(s t), sinh(s t)} has a removable singularity
/// as s -> 0: both the direct and the normalized closed forms suffer
/// catastrophic cancellation for small s, so the natural cubic is used below
/// this threshold (where it is the exact s = 0 limit and is numerically clean).
/// The tension branch is only exercised for s > 1, where it is accurate to
/// better than 1e-8 and the normalized (divide-by-cosh) form is overflow-safe
/// up to s = 1e308.
constexpr double kTensionCrossover = 1.0;

/// @brief Tension recurrence coefficient alpha(s) = (sinh s - s) / (s (cosh s - 1)).
/// @param sigma Non-negative tension parameter.
/// @return alpha in (0, 1/3], with beta = 1 - alpha the companion coefficient.
///
/// The normalized form alpha = (1 - q^2 - 2 s q) / (s (1 - q)^2), q = exp(-s),
/// avoids overflow: every exponential argument is non-positive, and the product
/// s * q is ordered so that s = 1e308, q = 0 yields 0 rather than inf * 0 = nan.
/// alpha -> 1/3 as s -> 0 (matching the natural cubic) and alpha -> 0 as s -> inf
/// (collapsing the spline toward piecewise-linear).
KOKKOS_FUNCTION double tension_alpha(double sigma) {
    const double s = sigma < 0.0 ? -sigma : sigma;
    if (s <= kTensionCrossover) return 1.0 / 3.0;
    const double q = std::exp(-s);
    const double one_minus_q = 1.0 - q;
    return (1.0 - q * q - 2.0 * (s * q)) / (s * one_minus_q * one_minus_q);
}

}  // namespace

KOKKOS_FUNCTION double evaluate_spline(double x, double x1, double x2, double y1, double y2, double d1, double d2, double sigma) {
    const double h = x2 - x1;
    if (h <= 0.0) return y1;
    const double t = (x - x1) / h;
    const double dy = y2 - y1;
    const double s = sigma < 0.0 ? -sigma : sigma;

    if (s <= kTensionCrossover) {
        // Natural cubic Hermite segment (the s -> 0 limit of the tension basis).
        const double om = 1.0 - t;
        const double h00 = (1.0 + 2.0 * t) * om * om;
        const double h10 = t * om * om;
        const double h01 = t * t * (3.0 - 2.0 * t);
        const double h11 = t * t * (t - 1.0);
        return y1 * h00 + d1 * h * h10 + y2 * h01 + d2 * h * h11;
    }

    // Hyperbolic-tension segment, written in the normalized (divide-by-cosh(s))
    // basis so it stays finite for arbitrarily large s. With q = exp(-s),
    // P = exp(-s t), M = exp(-s (1 - t)) (all non-positive exponents) the
    // ratios cosh(s t)/cosh(s) etc. are formed from these directly.
    const double u = d1 * h;
    const double v = d2 * h;
    const double q = std::exp(-s);
    const double P = std::exp(-s * t);
    const double M = std::exp(-s * (1.0 - t));
    const double den1 = 1.0 + q * q;
    const double sech = 2.0 * q / den1;        // sech(s)
    const double th = (1.0 - q * q) / den1;    // tanh(s)
    const double c_st = (M + q * P) / den1;    // cosh(s t) / cosh(s)
    const double s_st = (M - q * P) / den1;    // sinh(s t) / cosh(s)
    const double c_s1mt = (P + q * M) / den1;  // cosh(s(1-t)) / cosh(s)
    const double s_s1m = (q * M - P) / den1;   // sinh(s(t-1)) / cosh(s)
    // Delta / (s cosh s), with one power of s divided out of the numerators via
    // `common` so the (large) s * Delta product never overflows.
    const double Wden = s * th - 2.0 + 2.0 * sech;
    const double common = (th - s_st + s_s1m) / s;
    const double numU = t - t * sech - 1.0 + c_s1mt + common;
    const double numV = -t + t * sech + c_st - sech + common;
    const double U = -numU / Wden;
    const double V = numV / Wden;

    // f(t) = linear + (u - dy) U(t) + (v - dy) V(t). U, V vanish at both ends
    // (exact endpoint values) and satisfy U'(0)=1, V'(1)=1 (C1 with the
    // supplied derivatives). Affine data (u = v = dy) collapse to the linear
    // term exactly for every tension.
    return y1 + dy * t + (u - dy) * U + (v - dy) * V;
}

template <std::size_t MaxLevels>
KOKKOS_FUNCTION void solve_column_spline(const double *x, const double *y, std::size_t n, double sigma, double *d, double *temp_scratch) {
    if (n < 2) return;
    if (n == 2) {
        d[0] = (y[1] - y[0]) / (x[1] - x[0]);
        d[1] = d[0];
        return;
    }

    // Solve for knot first derivatives with natural (zero second derivative)
    // boundary conditions, generalized to the Cooté-Holladay tension spline.
    // In derivative form the system is tridiagonal with coefficients set by
    // alpha = tension_alpha(sigma) and beta = 1 - alpha:
    //   left:    beta d[0]      + alpha d[1]                              = delta[0]
    //   knot k:  alpha h_k d[k-1] + beta (h_k + h_{k-1}) d[k] + alpha h_{k-1} d[k+1]
    //                                                                       = h_k delta[k-1] + h_{k-1} delta[k]
    //   right:   alpha d[n-2]    + beta d[n-1]                             = delta[n-1]
    // At sigma = 0 (alpha = 1/3, beta = 2/3) this is exactly the natural cubic
    // system; as alpha -> 0 the rows decouple toward the secant slopes
    // (piecewise-linear). Solved in place with the Thomas algorithm (no heap).
    const double alpha = tension_alpha(sigma);
    const double beta = 1.0 - alpha;

    const double first_slope = (y[1] - y[0]) / (x[1] - x[0]);
    d[0] = first_slope / beta;
    temp_scratch[0] = alpha / beta;
    for (std::size_t i = 1; i < n - 1; ++i) {
        double dx1 = x[i] - x[i - 1];
        double dx2 = x[i + 1] - x[i];
        if (dx1 <= 0.0) dx1 = 1e-15;
        if (dx2 <= 0.0) dx2 = 1e-15;
        double dy1 = (y[i] - y[i - 1]) / dx1;
        double dy2 = (y[i + 1] - y[i]) / dx2;

        const double lower = alpha * dx2;
        const double diagonal = beta * (dx1 + dx2);
        const double upper = alpha * dx1;
        const double rhs = dx2 * dy1 + dx1 * dy2;
        double denom = diagonal - lower * temp_scratch[i - 1];
        if (denom == 0.0) denom = 1e-15;
        temp_scratch[i] = upper / denom;
        d[i] = (rhs - lower * d[i - 1]) / denom;
    }
    const double last_slope = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    const double last_denom = beta - alpha * temp_scratch[n - 2];
    d[n - 1] = (last_slope - alpha * d[n - 2]) / last_denom;
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        d[i] = d[i] - temp_scratch[i] * d[i + 1];
    }
}

// Explicit template instantiation for common vertical sizes
template KOKKOS_FUNCTION void solve_column_spline<256>(const double *x, const double *y, std::size_t n, double sigma, double *d,
                                                       double *temp_scratch);

}  // namespace axis::detail::tspack
