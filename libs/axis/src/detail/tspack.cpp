// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <axis/detail/tspack.hpp>
#include <cmath>

namespace axis::detail::tspack {

KOKKOS_FUNCTION double evaluate_spline(double x, double x1, double x2, double y1, double y2, double d1, double d2, double sigma) {
    double dx = x2 - x1;
    if (dx <= 0.0) return y1;
    double t = (x - x1) / dx;

    if (sigma <= 0.0) {
        // Cubic spline simplification
        double h00 = (1.0 + 2.0 * t) * (1.0 - t) * (1.0 - t);
        double h10 = t * (1.0 - t) * (1.0 - t);
        double h01 = t * t * (3.0 - 2.0 * t);
        double h11 = t * t * (t - 1.0);
        return y1 * h00 + d1 * dx * h10 + y2 * h01 + d2 * dx * h11;
    }

    // Tension spline evaluation using hyperbolic sines
    double sig_t = sigma * t;
    double sig_1_t = sigma * (1.0 - t);
    double denom = std::sinh(sigma);
    if (denom == 0.0) return y1;

    // Exact mathematical basis functions under tension (Renka's TSPACK)
    double h00 = (1.0 - t) + (std::sinh(sig_1_t) - t * std::sinh(sigma) - (1.0 - t) * std::sinh(sigma)) /
                                 (denom * (sigma * std::cosh(sigma) - std::sinh(sigma)));  // Form block check
    // We can evaluate directly or simplify to the stable hyperbolic formulation:
    double mt = 1.0 - t;
    double s_sig = std::sinh(sigma);
    double d_term = sigma * std::cosh(sigma) - s_sig;
    if (d_term == 0.0) d_term = 1e-15;

    double e1 = std::sinh(sig_1_t) - mt * s_sig;
    double e2 = std::sinh(sig_t) - t * s_sig;

    return y1 * mt + y2 * t + (d1 * dx - (y2 - y1)) * e1 / d_term + (d2 * dx - (y2 - y1)) * e2 / d_term;
}

template <std::size_t MaxLevels>
KOKKOS_FUNCTION void solve_column_spline(const double *x, const double *y, std::size_t n, double sigma, double *d, double *temp_scratch) {
    if (n < 2) return;
    if (n == 2) {
        d[0] = (y[1] - y[0]) / (x[1] - x[0]);
        d[1] = d[0];
        return;
    }

    // Solve for knot derivatives with natural (zero second derivative)
    // boundary conditions. In derivative form, the endpoint equations are
    // 2*d[0] + d[1] = 3*slope[0] and
    // d[n-2] + 2*d[n-1] = 3*slope[n-2].
    const double first_slope = (y[1] - y[0]) / (x[1] - x[0]);
    d[0] = 1.5 * first_slope;
    temp_scratch[0] = 0.5;
    for (std::size_t i = 1; i < n - 1; ++i) {
        double dx1 = x[i] - x[i - 1];
        double dx2 = x[i + 1] - x[i];
        if (dx1 <= 0.0) dx1 = 1e-15;
        if (dx2 <= 0.0) dx2 = 1e-15;
        double dy1 = (y[i] - y[i - 1]) / dx1;
        double dy2 = (y[i + 1] - y[i]) / dx2;

        const double lower = dx2;
        const double diagonal = 2.0 * (dx1 + dx2);
        const double upper = dx1;
        const double rhs = 3.0 * (dx2 * dy1 + dx1 * dy2);
        double denom = diagonal - lower * temp_scratch[i - 1];
        if (denom == 0.0) denom = 1e-15;
        temp_scratch[i] = upper / denom;
        d[i] = (rhs - lower * d[i - 1]) / denom;
    }
    const double last_slope = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    const double last_denom = 2.0 - temp_scratch[n - 2];
    d[n - 1] = (3.0 * last_slope - d[n - 2]) / last_denom;
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        d[i] = d[i] - temp_scratch[i] * d[i + 1];
    }
}

// Explicit template instantiation for common vertical sizes
template KOKKOS_FUNCTION void solve_column_spline<256>(const double *x, const double *y, std::size_t n, double sigma, double *d,
                                                       double *temp_scratch);

}  // namespace axis::detail::tspack
