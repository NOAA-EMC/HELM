// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_APPLY_HPP
#define AXIS_SOLVER_APPLY_HPP

/// @file axis/solver/apply.hpp
/// @brief Kokkos-parallel sparse matrix-vector apply (SpMV) for field regridding.
///
/// Implements dst = S · src via sparse matrix-vector multiplication.
/// The CSR path delegates to KokkosSparse::spmv for optimized, hardware-tuned
/// SpMV. KokkosKernels is a REQUIRED dependency of AXIS (enforced in
/// libs/axis/CMakeLists.txt — system package or FetchContent, FATAL_ERROR
/// otherwise), so this is the only CSR path; there is no feature macro.
///
/// The COO path uses scatter-add with Kokkos::atomic_add.
///
/// Three overloads are provided:
///   1. Local apply — single-rank, matrix + src + dst
///   2. Distributed apply — matrix + pattern + local_src + gathered_halo_src + dst
///   3. Callback-based distributed apply — uses a user-provided gather functor
///
/// Header-only (template) since it is parameterized on MemorySpace.

#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <Kokkos_Core.hpp>
#include <axis/solver/halo_pattern.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// 1. Local apply: dst = S · src (single-rank)
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the interpolation matrix: dst = S · src.
///
/// Implements, in parallel over k in [0, nnz):
///     Kokkos::atomic_add(&dst(row_k), S(k) * src(col_k));
/// after zero-initializing dst.
///
/// src and dst are non-owning layout_left views whose memory MUST reside in
/// MemorySpace (HELM Law #1: no copy of field data).
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix  Sparse interpolation operator (COO: factorList + row/col)
/// @param src     Source field [n_src]
/// @param dst     Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src() or
///         dst.extent(0) != matrix.n_dst(). dst is NOT written on validation
///         failure.
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const double, 1> src, field_view<double, 1> dst) {
    // ── Extent validation (throw BEFORE touching dst) ────────────────────────
    if (src.extent(0) != matrix.n_src()) {
        throw std::invalid_argument("axis::solver::apply: src.extent(0) (" + std::to_string(src.extent(0)) + ") != matrix.n_src() (" +
                                    std::to_string(matrix.n_src()) + ")");
    }
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument("axis::solver::apply: dst.extent(0) (" + std::to_string(dst.extent(0)) + ") != matrix.n_dst() (" +
                                    std::to_string(matrix.n_dst()) + ")");
    }

    // ── Obtain common metadata ──────────────────────────────────────────────
    const std::size_t n_dst = matrix.n_dst();

    // Wrap the raw dst pointer in a Kokkos unmanaged View for the kernel.
    Kokkos::View<double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst_view(dst.data_handle(), n_dst);

    // Wrap the raw src pointer in a Kokkos unmanaged View for the kernel.
    Kokkos::View<const double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src_view(src.data_handle(), src.extent(0));

    using exec_space = typename MemorySpace::execution_space;

    if (matrix.is_csr()) {
        // ── CSR path: KokkosSparse::spmv (required dependency) ──────────────
        // Hardware-tuned SpMV (cuSPARSE on GPU, MKL on CPU when available, or
        // the KokkosKernels native implementation). The CrsMatrix is a derived
        // operator of the (immutable) CSR weights, so it is built once and
        // cached on the matrix rather than rebuilt per call.
        const auto &A = matrix.kk_crs_matrix();

        // Wrap src/dst as rank-1 Kokkos Views for KokkosSparse::spmv
        Kokkos::View<const double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_view(src.data_handle(), src.extent(0));
        Kokkos::View<double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> y_view(dst.data_handle(), n_dst);

        // y = 1.0 * A * x + 0.0 * y  (i.e., y = A*x, overwriting dst)
        KokkosSparse::spmv("N", 1.0, A, x_view, 0.0, y_view);
    } else {
        // ── COO scatter-add path (original): uses atomics ────────────────────
        const auto &S = matrix.factor_list_view();   // [nnz]
        const auto &row = matrix.factor_row_view();  // [nnz]
        const auto &col = matrix.factor_col_view();  // [nnz]
        const std::size_t nnz = matrix.nnz();

        // Zero-initialize dst before scatter-add
        Kokkos::parallel_for(
            "axis::apply::zero_dst", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const std::size_t j) { dst_view(j) = 0.0; });

        // SpMV scatter-add: dst(row_k) += S(k) * src(col_k)
        Kokkos::parallel_for(
            "axis::apply::spmv", Kokkos::RangePolicy<exec_space>(0, nnz), KOKKOS_LAMBDA(const std::size_t k) {
                const auto r = row(k);
                const auto c = col(k);
                Kokkos::atomic_add(&dst_view(r), S(k) * src_view(c));
            });
    }

    Kokkos::fence("axis::apply::complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Distributed apply: local_src + gathered_halo_src → dst
// ─────────────────────────────────────────────────────────────────────────────

/// Distributed apply (ESMF ASMM). Reads locally-owned source cells from
/// `local_src` and off-rank source cells from `gathered_halo_src` — the
/// contiguous buffer HALO produced by exchanging the published HaloPattern.
///
/// For each nonzero k, the source value is:
///   - local_src(col_k)  if col_k < local_src.extent(0)  (locally-owned)
///   - gathered_halo_src(gather_slot mapping)  otherwise  (off-rank)
///
/// The column indices in the matrix are encoded as:
///   - [0, local_src.extent(0)) → local source cells
///   - [local_src.extent(0), ...) → offset into gathered_halo_src
///
/// AXIS performs NO MPI; it only reads the buffer the caller (via HALO)
/// already gathered in gather_slot order.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix             Sparse interpolation operator
/// @param pattern            HaloPattern describing off-rank dependencies
/// @param local_src          Locally-owned source values [n_local_src]
/// @param gathered_halo_src  Off-rank source values from HALO [num_remote()]
/// @param dst                Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if:
///   - dst.extent(0) != matrix.n_dst()
///   - gathered_halo_src.extent(0) != pattern.num_remote()
///   - local_src extent disagrees with matrix dimensions
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace> &matrix, const HaloPattern &pattern, field_view<const double, 1> local_src,
           field_view<const double, 1> gathered_halo_src, field_view<double, 1> dst) {
    // ── Extent validation (throw BEFORE touching dst) ────────────────────────
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument("axis::solver::apply (distributed): dst.extent(0) (" + std::to_string(dst.extent(0)) + ") != matrix.n_dst() (" +
                                    std::to_string(matrix.n_dst()) + ")");
    }
    if (gathered_halo_src.extent(0) != pattern.num_remote()) {
        throw std::invalid_argument("axis::solver::apply (distributed): gathered_halo_src.extent(0) (" + std::to_string(gathered_halo_src.extent(0)) +
                                    ") != pattern.num_remote() (" + std::to_string(pattern.num_remote()) + ")");
    }

    const std::size_t n_local_src = local_src.extent(0);

    // Validate: local_src + num_remote should cover the matrix's source space
    if (n_local_src + pattern.num_remote() != matrix.n_src()) {
        throw std::invalid_argument("axis::solver::apply (distributed): local_src.extent(0) (" + std::to_string(n_local_src) +
                                    ") + pattern.num_remote() (" + std::to_string(pattern.num_remote()) + ") != matrix.n_src() (" +
                                    std::to_string(matrix.n_src()) + ")");
    }

    // ── Obtain internal Kokkos Views ─────────────────────────────────────────
    const auto &S = matrix.factor_list_view();
    const auto &row = matrix.factor_row_view();
    const auto &col = matrix.factor_col_view();
    const std::size_t nnz = matrix.nnz();
    const std::size_t n_dst = matrix.n_dst();

    // Wrap raw pointers in unmanaged Kokkos Views
    Kokkos::View<double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst_view(dst.data_handle(), n_dst);

    Kokkos::View<const double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> local_src_view(local_src.data_handle(), n_local_src);

    Kokkos::View<const double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> halo_src_view(gathered_halo_src.data_handle(),
                                                                                                     pattern.num_remote());

    // ── Zero-initialize dst ──────────────────────────────────────────────────
    Kokkos::parallel_for(
        "axis::apply_distributed::zero_dst", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_dst),
        KOKKOS_LAMBDA(const std::size_t j) { dst_view(j) = 0.0; });

    // ── SpMV with local/remote source dispatch ───────────────────────────────
    // Column indices < n_local_src refer to locally-owned cells;
    // Column indices >= n_local_src refer to gathered halo buffer.
    const auto n_local = static_cast<index_t>(n_local_src);
    Kokkos::parallel_for(
        "axis::apply_distributed::spmv", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, nnz), KOKKOS_LAMBDA(const std::size_t k) {
            const auto r = row(k);
            const auto c = col(k);
            double src_val;
            if (c < n_local) {
                src_val = local_src_view(c);
            } else {
                src_val = halo_src_view(c - n_local);
            }
            Kokkos::atomic_add(&dst_view(r), S(k) * src_val);
        });

    Kokkos::fence("axis::apply_distributed::complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Callback-based distributed apply
// ─────────────────────────────────────────────────────────────────────────────

/// Distributed apply via a caller-supplied abstract gather callback/functor.
///
/// AXIS invokes `gather` exactly once with the published pattern to obtain the
/// off-rank source values, then proceeds as the gathered-buffer overload.
/// AXIS still performs NO MPI and knows nothing of HALO — the callback owns
/// all communication. Useful for a Python layer or a test stub.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix     Sparse interpolation operator
/// @param pattern    HaloPattern describing off-rank dependencies
/// @param local_src  Locally-owned source values [n_local_src]
/// @param gather     Callback that receives the HaloPattern and returns a
///                   std::vector<double> of size pattern.num_remote() with the
///                   off-rank source values in gather_slot order
/// @param dst        Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if extent validation fails (same as
///         distributed overload)
/// @throws Any exception the gather callback throws
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace> &matrix, const HaloPattern &pattern, field_view<const double, 1> local_src,
           const std::function<std::vector<double>(const HaloPattern &)> &gather, field_view<double, 1> dst) {
    // Invoke the gather callback to obtain off-rank source values
    std::vector<double> halo_buffer = gather(pattern);

    if (halo_buffer.size() != pattern.num_remote()) {
        throw std::invalid_argument("axis::solver::apply (callback): gather returned " + std::to_string(halo_buffer.size()) +
                                    " values but pattern.num_remote() is " + std::to_string(pattern.num_remote()));
    }

    // Wrap the gathered buffer as a field_view and delegate to the buffer overload
    field_view<const double, 1> halo_view(halo_buffer.data(), halo_buffer.size());
    apply(matrix, pattern, local_src, halo_view, dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Batch apply: dst = S · src for rank-2 (cells × variables) views
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the interpolation matrix to multiple variables simultaneously:
///   dst(:, v) = S · src(:, v)  for v in [0, n_vars)
///
/// Both src and dst are rank-2 layout_left (column-major) views where:
///   - extent(0) = number of cells (leading dimension)
///   - extent(1) = number of variables (trailing dimension)
///
/// CSR path: KokkosSparse::spmv over the cached CrsMatrix with rank-2
///   (multivector) views — the same operator the rank-1 apply path caches.
/// COO path: parallel_for over nnz, inner serial loop over variables with
///   atomic accumulation.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix  Sparse interpolation operator
/// @param src     Source fields [n_src, n_vars]
/// @param dst     Destination fields [n_dst, n_vars] — overwritten with result
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src(),
///         dst.extent(0) != matrix.n_dst(), or src.extent(1) != dst.extent(1).
template <class MemorySpace, class Layout = Kokkos::LayoutLeft>
void batch_apply(const InterpolationMatrix<MemorySpace> &matrix,
                 Kokkos::View<const double **, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src,
                 Kokkos::View<double **, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst) {
    // ── Extent validation (throw BEFORE touching dst) ────────────────────────
    if (src.extent(0) != matrix.n_src()) {
        throw std::invalid_argument("axis::solver::batch_apply: src.extent(0) (" + std::to_string(src.extent(0)) + ") != matrix.n_src() (" +
                                    std::to_string(matrix.n_src()) + ")");
    }
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument("axis::solver::batch_apply: dst.extent(0) (" + std::to_string(dst.extent(0)) + ") != matrix.n_dst() (" +
                                    std::to_string(matrix.n_dst()) + ")");
    }
    if (src.extent(1) != dst.extent(1)) {
        throw std::invalid_argument("axis::solver::batch_apply: src.extent(1) (" + std::to_string(src.extent(1)) + ") != dst.extent(1) (" +
                                    std::to_string(dst.extent(1)) + ")");
    }

    // ── Obtain common metadata ──────────────────────────────────────────────
    const std::size_t n_dst = matrix.n_dst();
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_vars = src.extent(1);

    using exec_space = typename MemorySpace::execution_space;

    if (matrix.is_csr()) {
        // ── CSR path: KokkosSparse::spmv with multivector (rank-2) views ────
        // The cached CrsMatrix (built once on first use by the rank-1 apply
        // path) is reused here; KokkosSparse::spmv accepts rank-2
        // (multivector) x/y views against the same operator.
        const auto &A = matrix.kk_crs_matrix();
        KokkosSparse::spmv("N", 1.0, A, src, 0.0, dst);
    } else {
        // ── COO scatter-add path ─────────────────────────────────────────────
        const auto &S = matrix.factor_list_view();   // [nnz]
        const auto &row = matrix.factor_row_view();  // [nnz]
        const auto &col = matrix.factor_col_view();  // [nnz]
        const std::size_t nnz = matrix.nnz();

        // Zero-initialize dst
        Kokkos::parallel_for(
            "axis::batch_apply::zero_dst", Kokkos::RangePolicy<exec_space>(0, n_dst * n_vars), KOKKOS_LAMBDA(const std::size_t idx) {
                const auto j = idx % n_dst;  // cell index (leading dim)
                const auto v = idx / n_dst;  // variable index (trailing dim)
                dst(j, v) = 0.0;
            });

        // Scatter-add: dst(row_k, v) += S(k) * src(col_k, v) for all v
        Kokkos::parallel_for(
            "axis::batch_apply::coo", Kokkos::RangePolicy<exec_space>(0, nnz * n_vars), KOKKOS_LAMBDA(const std::size_t idx) {
                const auto k = idx % nnz;  // nonzero index
                const auto v = idx / nnz;  // variable index
                const auto r = row(k);
                const auto c = col(k);
                Kokkos::atomic_add(&dst(r, v), S(k) * src(c, v));
            });
    }

    Kokkos::fence("axis::batch_apply::complete");
}

template <class MemorySpace>
void batch_apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const double, 2> src, field_view<double, 2> dst) {
    using dst_view_t = Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using src_view_t = Kokkos::View<const double **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    dst_view_t dst_kk(dst.data_handle(), dst.extent(0), dst.extent(1));
    src_view_t src_kk(src.data_handle(), src.extent(0), src.extent(1));
    batch_apply<MemorySpace, Kokkos::LayoutLeft>(matrix, src_kk, dst_kk);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. NaN-aware batch apply: dst = S · src with on-the-fly re-normalization
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the interpolation matrix to multiple variables simultaneously with on-the-fly
/// NaN-aware SpMV re-normalization (fused single-pass).
///
/// Both src and dst are rank-2 layout_left (column-major) views.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix  Sparse interpolation operator
/// @param src     Source fields [n_src, n_vars]
/// @param dst     Destination fields [n_dst, n_vars] — overwritten with result
/// @param na_thres Minimum fraction of valid input contribution required for output cell
template <class MemorySpace, class Layout = Kokkos::LayoutLeft>
void nan_batch_apply(const InterpolationMatrix<MemorySpace> &matrix,
                     Kokkos::View<const double **, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src,
                     Kokkos::View<double **, Layout, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst, double na_thres) {
    if (src.extent(0) != matrix.n_src()) {
        throw std::invalid_argument("axis::solver::nan_batch_apply: src.extent(0) != matrix.n_src()");
    }
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument("axis::solver::nan_batch_apply: dst.extent(0) != matrix.n_dst()");
    }
    if (src.extent(1) != dst.extent(1)) {
        throw std::invalid_argument("axis::solver::nan_batch_apply: src.extent(1) != dst.extent(1)");
    }

    const std::size_t n_dst = matrix.n_dst();
    const std::size_t n_src = matrix.n_src();
    const std::size_t n_vars = src.extent(1);

    using exec_space = typename MemorySpace::execution_space;

    if (!matrix.is_csr()) {
        throw std::runtime_error("axis::solver::nan_batch_apply requires matrix in CSR format. Call to_csr() first.");
    }

    const auto row_ptr = matrix.row_ptr();
    const auto col_idx = matrix.col_idx();
    const auto csr_vals = matrix.csr_values();

    using team_policy = Kokkos::TeamPolicy<exec_space>;
    using member_type = typename team_policy::member_type;

    Kokkos::parallel_for(
        "axis::nan_batch_apply::csr", team_policy(static_cast<int>(n_dst), Kokkos::AUTO), KOKKOS_LAMBDA(const member_type &team) {
            const auto j = team.league_rank();
            const auto start = row_ptr(j);
            const auto end = row_ptr(j + 1);

            // Compute total weight for row j (sum of S)
            double total_row_weight = 0.0;
            for (auto k = start; k < end; ++k) {
                total_row_weight += csr_vals(k);
            }

            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, static_cast<int>(n_vars)), [&](const int v) {
                double weighted_sum_val = 0.0;
                double sum_valid_weights = 0.0;

                for (auto k = start; k < end; ++k) {
                    const double val = src(col_idx(k), v);
                    if (!Kokkos::isnan(val)) {
                        weighted_sum_val += csr_vals(k) * val;
                        sum_valid_weights += csr_vals(k);
                    }
                }

                if (sum_valid_weights == 0.0 || total_row_weight == 0.0) {
                    dst(j, v) = Kokkos::nan("not a number");
                } else {
                    double fraction_valid = sum_valid_weights / total_row_weight;
                    if (fraction_valid < (1.0 - na_thres - 1e-6)) {
                        dst(j, v) = Kokkos::nan("not a number");
                    } else {
                        dst(j, v) = weighted_sum_val / sum_valid_weights;
                    }
                }
            });
        });

    Kokkos::fence("axis::nan_batch_apply::complete");
}

template <class MemorySpace>
void nan_batch_apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const double, 2> src, field_view<double, 2> dst, double na_thres) {
    using dst_view_t = Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using src_view_t = Kokkos::View<const double **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    dst_view_t dst_kk(dst.data_handle(), dst.extent(0), dst.extent(1));
    src_view_t src_kk(src.data_handle(), src.extent(0), src.extent(1));
    nan_batch_apply<MemorySpace, Kokkos::LayoutLeft>(matrix, src_kk, dst_kk, na_thres);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Masked apply: dst = S · src, skipping masked destination cells
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the interpolation matrix with an optional destination cell mask.
///
/// Masked destination cells (mask value == 0) are skipped — their values in
/// dst are left unchanged. This supports Requirement 8.4: land/ocean boundary
/// handling where masked regions should not be written.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @param matrix    Sparse interpolation operator (COO or CSR)
/// @param src       Source field [n_src]
/// @param dst       Destination field [n_dst] — masked cells left unchanged
/// @param dst_mask  Per-cell mask [n_dst] (0=masked/inactive, nonzero=active)
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src() or
///         dst.extent(0) != matrix.n_dst()
template <class MemorySpace>
void apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const double, 1> src, field_view<double, 1> dst,
           field_view<const int, 1> dst_mask) {
    // ── Extent validation (throw BEFORE touching dst) ────────────────────────
    if (src.extent(0) != matrix.n_src()) {
        throw std::invalid_argument("axis::solver::apply (masked): src.extent(0) (" + std::to_string(src.extent(0)) + ") != matrix.n_src() (" +
                                    std::to_string(matrix.n_src()) + ")");
    }
    if (dst.extent(0) != matrix.n_dst()) {
        throw std::invalid_argument("axis::solver::apply (masked): dst.extent(0) (" + std::to_string(dst.extent(0)) + ") != matrix.n_dst() (" +
                                    std::to_string(matrix.n_dst()) + ")");
    }

    const std::size_t n_dst = matrix.n_dst();
    const bool has_mask = (dst_mask.extent(0) == n_dst);

    // Wrap the raw pointers in unmanaged Kokkos Views for the kernel.
    Kokkos::View<double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst_view(dst.data_handle(), n_dst);
    Kokkos::View<const double *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src_view(src.data_handle(), src.extent(0));
    Kokkos::View<const int *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mask_view(dst_mask.data_handle(), dst_mask.extent(0));

    using exec_space = typename MemorySpace::execution_space;

    if (matrix.is_csr()) {
        // ── CSR path with mask ───────────────────────────────────────────────
        const auto row_ptr = matrix.row_ptr();
        const auto col_idx = matrix.col_idx();
        const auto csr_vals = matrix.csr_values();

        using team_policy = Kokkos::TeamPolicy<exec_space>;
        using member_type = typename team_policy::member_type;

        Kokkos::parallel_for(
            "axis::apply::csr_spmv_masked", team_policy(static_cast<int>(n_dst), Kokkos::AUTO), KOKKOS_LAMBDA(const member_type &team) {
                const auto j = team.league_rank();

                // Skip masked destination cells (Req 8.4)
                if (has_mask && mask_view(j) == 0) return;

                const auto start = row_ptr(j);
                const auto end = row_ptr(j + 1);

                double sum = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamVectorRange(team, start, end), [&](const index_t k, double &lsum) { lsum += csr_vals(k) * src_view(col_idx(k)); },
                    sum);

                Kokkos::single(Kokkos::PerTeam(team), [&]() { dst_view(j) = sum; });
            });
    } else {
        // ── COO scatter-add path with mask ───────────────────────────────────
        const auto &S = matrix.factor_list_view();
        const auto &row = matrix.factor_row_view();
        const auto &col = matrix.factor_col_view();
        const std::size_t nnz = matrix.nnz();

        // Zero-initialize only unmasked dst cells
        Kokkos::parallel_for(
            "axis::apply::zero_dst_masked", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const std::size_t j) {
                if (has_mask && mask_view(j) == 0) return;
                dst_view(j) = 0.0;
            });

        // SpMV scatter-add: skip writes to masked rows
        Kokkos::parallel_for(
            "axis::apply::spmv_masked", Kokkos::RangePolicy<exec_space>(0, nnz), KOKKOS_LAMBDA(const std::size_t k) {
                const auto r = row(k);
                // Skip masked destination cells
                if (has_mask && mask_view(r) == 0) return;
                const auto c = col(k);
                Kokkos::atomic_add(&dst_view(r), S(k) * src_view(c));
            });
    }

    Kokkos::fence("axis::apply_masked::complete");
}

}  // namespace axis::solver

#endif  // AXIS_SOLVER_APPLY_HPP
