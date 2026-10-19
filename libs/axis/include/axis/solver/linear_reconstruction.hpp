// SPDX-License-Identifier: Apache-2.0
#ifndef AXIS_SOLVER_LINEAR_RECONSTRUCTION_HPP
#define AXIS_SOLVER_LINEAR_RECONSTRUCTION_HPP

/// @file axis/solver/linear_reconstruction.hpp
/// @brief Second-order (linear) reconstruction applied on top of first-order
///        conservative overlap weights, with an optional limited variant.

#include <Kokkos_Core.hpp>
#include <axis/types.hpp>
#include <stdexcept>

namespace axis::solver {

/// @brief Precomputed geometry that upgrades a conservative operator to linear order.
///
/// Optional field-dependent extension of the base first-order sparse operator.
/// All overlap moments are normalized exactly like their associated area weight.
template <class MemorySpace>
struct LinearReconstruction {
    bool enabled{false};
    bool limited{false};
    Kokkos::View<double *[2], MemorySpace> moments;      // COO order [nnz,2]
    Kokkos::View<double *[2], MemorySpace> csr_moments;  // sorted with CSR weights
    Kokkos::View<index_t *, MemorySpace> neighbor_offsets, neighbors;
    Kokkos::View<double *[2], MemorySpace> gradient_coefficients;
    // Source vertices AND overlap means: limiting here bounds each overlap
    // contribution, hence the covered-area destination average.
    Kokkos::View<index_t *, MemorySpace> control_offsets;
    Kokkos::View<double *[2], MemorySpace> control_points;
};

/// @brief Compute per-source-cell linear gradients, optionally limited.
///
/// Source/DestinationView may be strided subviews for batch application.
/// @tparam Limited Apply a bound-preserving limiter over neighbor and control values.
/// @tparam MemorySpace Kokkos memory space owning the views.
/// @param matrix         Operator carrying the LinearReconstruction payload.
/// @param src            Source cell averages.
/// @param gradients      Output (n_src, 2) gradient workspace, caller-owned.
/// @param limit_factors  Optional per-cell limiter factors; empty to skip.
template <bool Limited, class MemorySpace, class SourceView, class Matrix, class GradientView>
void compute_linear_gradients(const Matrix &matrix, SourceView src, GradientView gradients,
                              Kokkos::View<double *, MemorySpace> limit_factors = {}) {
    using Exec = typename MemorySpace::execution_space;
    const auto &data = matrix.linear_reconstruction();
    if (limit_factors.extent(0) && limit_factors.extent(0) != matrix.n_src())
        throw std::invalid_argument("linear limiter workspace has incorrect shape");
    auto offsets = data.neighbor_offsets;
    auto neighbors = data.neighbors;
    auto coeff = data.gradient_coefficients;
    auto points = data.control_points;
    auto controls = data.control_offsets;

    Kokkos::parallel_for(
        "axis::linear::gradients", Kokkos::RangePolicy<Exec>(0, matrix.n_src()), KOKKOS_LAMBDA(std::size_t i) {
            double g0 = 0, g1 = 0, lo = src(i), hi = src(i);
            for (index_t k = offsets(i); k < offsets(i + 1); ++k) {
                double q = src(neighbors(k)), delta = q - src(i);
                g0 += coeff(k, 0) * delta;
                g1 += coeff(k, 1) * delta;
                if constexpr (Limited) {
                    lo = q < lo ? q : lo;
                    hi = q > hi ? q : hi;
                }
            }
            double phi = 1;
            if constexpr (Limited)
                for (index_t k = controls(i); k < controls(i + 1); ++k) {
                    double change = g0 * points(k, 0) + g1 * points(k, 1);
                    if (change > 0) {
                        double p = (hi - src(i)) / change;
                        phi = p < phi ? p : phi;
                    } else if (change < 0) {
                        double p = (lo - src(i)) / change;
                        phi = p < phi ? p : phi;
                    }
                }
            phi = phi > 0 ? phi : 0;
            if (limit_factors.extent(0)) limit_factors(i) = phi;
            gradients(i, 0) = phi * g0;
            gradients(i, 1) = phi * g1;
        });
 }

/// @brief Accumulate area weights plus moment-dot-gradient terms into dst.
///
/// CSR matrices run branch-free per destination row; COO matrices zero the
/// output then atomically scatter contributions.
/// @param matrix  Operator whose linear payload supplies the overlap moments.
/// @param src     Source cell averages.
/// @param dst     Destination cell averages (written, not accumulated).
/// @param gradients Per-source gradients from compute_linear_gradients.
template <class MemorySpace, class SourceView, class DestinationView, class Matrix, class GradientView>
void apply_linear_gradients(const Matrix &matrix, SourceView src, DestinationView dst, GradientView gradients) {
    using Exec = typename MemorySpace::execution_space;
    const auto &data = matrix.linear_reconstruction();
    if (matrix.is_csr()) {
        auto offsets_csr = matrix.row_ptr();
        auto cols = matrix.col_idx();
        auto weights = matrix.csr_values();
        auto moments = data.csr_moments;
        Kokkos::parallel_for(
            "axis::linear::csr", Kokkos::RangePolicy<Exec>(0, matrix.n_dst()), KOKKOS_LAMBDA(std::size_t j) {
                double value = 0;
                for (index_t k = offsets_csr(j); k < offsets_csr(j + 1); ++k) {
                    auto i = cols(k);
                    value += weights(k) * src(i) + moments(k, 0) * gradients(i, 0) + moments(k, 1) * gradients(i, 1);
                }
                dst(j) = value;
            });
    } else {
        auto rows = matrix.factor_row_view();
        auto cols = matrix.factor_col_view();
        auto weights = matrix.factor_list_view();
        auto moments = data.moments;
        Kokkos::parallel_for("axis::linear::zero", Kokkos::RangePolicy<Exec>(0, matrix.n_dst()), KOKKOS_LAMBDA(std::size_t j) { dst(j) = 0; });
        Kokkos::parallel_for(
            "axis::linear::coo", Kokkos::RangePolicy<Exec>(0, matrix.nnz()), KOKKOS_LAMBDA(std::size_t k) {
                auto i = cols(k);
                double value = weights(k) * src(i) + moments(k, 0) * gradients(i, 0) + moments(k, 1) * gradients(i, 1);
                Kokkos::atomic_add(&dst(rows(k)), value);
            });
    }
    Kokkos::fence("axis::linear::complete");
}
/// @brief Gradient computation followed by accumulation, honoring the limiter flag.
///
/// Caller-owned workspaces are explicit: matrix copies remain immutable and
/// concurrent calls do not share mutable gradient storage.
template <bool Limited, class MemorySpace, class SourceView, class DestinationView, class Matrix, class GradientView>
void apply_linear_impl(const Matrix &matrix, SourceView src, DestinationView dst, GradientView gradients) {
    compute_linear_gradients<Limited, MemorySpace>(matrix, src, gradients);
    apply_linear_gradients<MemorySpace>(matrix, src, dst, gradients);
}

/// @brief Apply linear reconstruction using a caller-supplied gradient workspace.
/// @throws std::invalid_argument when gradients is not (n_src, 2).
template <class MemorySpace, class Matrix, class SourceView, class DestinationView, class GradientView>
void apply_linear_with_workspace(const Matrix &matrix, SourceView src, DestinationView dst, GradientView gradients) {
    if (gradients.extent(0) != matrix.n_src() || gradients.extent(1) != 2)
        throw std::invalid_argument("linear gradient workspace has incorrect shape");
    if (matrix.linear_reconstruction().limited)
        apply_linear_impl<true, MemorySpace>(matrix, src, dst, gradients);
    else
        apply_linear_impl<false, MemorySpace>(matrix, src, dst, gradients);
}
/// @brief Convenience wrapper that allocates its own gradient workspace.
template <class MemorySpace, class SourceView, class DestinationView, class Matrix>
void apply_linear(const Matrix &matrix, SourceView src, DestinationView dst) {
    Kokkos::View<double *[2], MemorySpace> gradients(Kokkos::view_alloc(Kokkos::WithoutInitializing, "axis_linear_gradients"), matrix.n_src());
    apply_linear_with_workspace<MemorySpace>(matrix, src, dst, gradients);
}

/// @brief Tiled batch application over independent variables in column two.
///
/// Gradients are computed per variable tile so the workspace stays small and
/// the stencil is traversed once per source cell for the whole tile.
template <bool Limited, class MemorySpace, class Matrix, class SourceView, class DestinationView>
void batch_apply_linear_impl(const Matrix &matrix, SourceView src, DestinationView dst) {
    using Exec = typename MemorySpace::execution_space;
    constexpr std::size_t tile = 8;
    const std::size_t variables = src.extent(1);
    if (variables == 0) return;
    const auto &data = matrix.linear_reconstruction();
    auto offsets = data.neighbor_offsets;
    auto neighbors = data.neighbors;
    auto coeff = data.gradient_coefficients;
    auto controls = data.control_offsets;
    auto points = data.control_points;
    Kokkos::View<double **[2], Kokkos::LayoutRight, MemorySpace> gradients(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "axis_linear_batch_gradients"), matrix.n_src(), variables);
    Kokkos::parallel_for(
        "axis::linear::batch_gradients", Kokkos::RangePolicy<Exec>(0, matrix.n_src()), KOKKOS_LAMBDA(std::size_t i) {
            for (std::size_t first = 0; first < variables; first += tile) {
                const std::size_t count = variables - first < tile ? variables - first : tile;
                double q[tile], g0[tile]{}, g1[tile]{}, lo[tile], hi[tile], phi[tile];
                for (std::size_t v = 0; v < count; ++v) {
                    q[v] = src(i, first + v);
                    lo[v] = hi[v] = q[v];
                    phi[v] = 1;
                }
                for (index_t k = offsets(i); k < offsets(i + 1); ++k) {
                    const auto n = neighbors(k);
                    const double c0 = coeff(k, 0), c1 = coeff(k, 1);
                    for (std::size_t v = 0; v < count; ++v) {
                        double x = src(n, first + v), delta = x - q[v];
                        g0[v] += c0 * delta;
                        g1[v] += c1 * delta;
                        if constexpr (Limited) {
                            lo[v] = x < lo[v] ? x : lo[v];
                            hi[v] = x > hi[v] ? x : hi[v];
                        }
                    }
                }
                if constexpr (Limited)
                    for (index_t k = controls(i); k < controls(i + 1); ++k) {
                        const double p0 = points(k, 0), p1 = points(k, 1);
                        for (std::size_t v = 0; v < count; ++v) {
                            double change = g0[v] * p0 + g1[v] * p1;
                            if (change > 0) {
                                double p = (hi[v] - q[v]) / change;
                                phi[v] = p < phi[v] ? p : phi[v];
                            } else if (change < 0) {
                                double p = (lo[v] - q[v]) / change;
                                phi[v] = p < phi[v] ? p : phi[v];
                            }
                        }
                    }
                for (std::size_t v = 0; v < count; ++v) {
                    phi[v] = phi[v] > 0 ? phi[v] : 0;
                    gradients(i, first + v, 0) = phi[v] * g0[v];
                    gradients(i, first + v, 1) = phi[v] * g1[v];
                }
            }
        });
    if (matrix.is_csr()) {
        auto row = matrix.row_ptr();
        auto col = matrix.col_idx();
        auto weights = matrix.csr_values();
        auto moments = data.csr_moments;
        Kokkos::parallel_for(
            "axis::linear::batch_csr", Kokkos::RangePolicy<Exec>(0, matrix.n_dst()), KOKKOS_LAMBDA(std::size_t j) {
                for (std::size_t first = 0; first < variables; first += tile) {
                    const std::size_t count = variables - first < tile ? variables - first : tile;
                    double values[tile]{};
                    for (index_t k = row(j); k < row(j + 1); ++k) {
                        const auto i = col(k);
                        const double w = weights(k), m0 = moments(k, 0), m1 = moments(k, 1);
                        for (std::size_t v = 0; v < count; ++v)
                            values[v] += w * src(i, first + v) + m0 * gradients(i, first + v, 0) + m1 * gradients(i, first + v, 1);
                    }
                    for (std::size_t v = 0; v < count; ++v) dst(j, first + v) = values[v];
                }
            });
    } else {
        auto rows = matrix.factor_row_view();
        auto cols = matrix.factor_col_view();
        auto weights = matrix.factor_list_view();
        auto moments = data.moments;
        Kokkos::parallel_for(
            "axis::linear::batch_zero", Kokkos::RangePolicy<Exec>(0, matrix.n_dst()), KOKKOS_LAMBDA(std::size_t j) {
                for (std::size_t v = 0; v < variables; ++v) dst(j, v) = 0;
            });
        Kokkos::parallel_for(
            "axis::linear::batch_coo", Kokkos::RangePolicy<Exec>(0, matrix.nnz()), KOKKOS_LAMBDA(std::size_t k) {
                const auto i = cols(k), j = rows(k);
                for (std::size_t v = 0; v < variables; ++v) {
                    double value = weights(k) * src(i, v) + moments(k, 0) * gradients(i, v, 0) + moments(k, 1) * gradients(i, v, 1);
                    Kokkos::atomic_add(&dst(j, v), value);
                }
            });
    }
    Kokkos::fence("axis::linear::batch_complete");
}
/// @brief Batch entry point that dispatches on the matrix's limiter flag.
template <class MemorySpace, class Matrix, class SourceView, class DestinationView>
void batch_apply_linear(const Matrix &matrix, SourceView src, DestinationView dst) {
    if (matrix.linear_reconstruction().limited)
        batch_apply_linear_impl<true, MemorySpace>(matrix, src, dst);
    else
        batch_apply_linear_impl<false, MemorySpace>(matrix, src, dst);
}
}  // namespace axis::solver
#endif
