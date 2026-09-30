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
/// Scalar type: the double paths use the tuned KokkosSparse::spmv (CSR) or
/// atomic scatter-add (COO) kernels. float32 paths (FR-018, R7) accumulate in
/// double per destination row and cast back to the input dtype; they require
/// CSR form (call InterpolationMatrix::to_csr() first — the Python bindings
/// do this automatically) because atomic scatter-add into float32 would be
/// both slower and lossy.
///
/// Overloads are provided for:
///   1. Local apply — single-rank, matrix + src + dst (double / float32)
///   2. Distributed apply — matrix + pattern + local_src + gathered_halo_src + dst
///      (double only; HALO exchange buffers are float64)
///   3. Callback-based distributed apply — uses a user-provided gather functor
///   4. Batch apply — rank-2 (cells × variables) multivector (double / float32)
///   5. Masked local apply — destination cells with mask == 0 left unchanged
///      (double / float32; float32 requires CSR)
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
#include <type_traits>
#include <vector>

namespace axis::solver {

namespace detail {

/// Field dtypes supported by the apply / batch_apply entry points.
template <class T>
struct is_apply_scalar : std::bool_constant<std::is_same_v<T, double> || std::is_same_v<T, float>> {};

template <class T>
inline constexpr bool is_apply_scalar_v = is_apply_scalar<T>::value;

/// @brief Throw a uniform "matrix must be CSR" error for the float32 paths.
///
/// float32 apply accumulates in double per destination row, which needs the
/// row-parallel CSR layout (no atomics on the output).
inline void require_csr(const char *what) {
    throw std::invalid_argument(std::string("axis::solver::") + what +
                                ": float32 apply requires CSR form — call InterpolationMatrix::to_csr() first");
}

/// @brief Row-parallel SpMV over the CSR arrays with double accumulation.
///
/// dst(j, v) = Σ_k csr_vals(k) · src(col_idx(k), v) for k in [row_ptr(j), row_ptr(j+1)).
/// Works for any field dtype T (float32/float64); products accumulate in
/// double and are cast back to T on store. Rank-1 callers pass n_vec == 1.
///
/// @pre matrix.is_csr()
template <class MemorySpace, class T>
void spmv_csr_rows(const InterpolationMatrix<MemorySpace> &matrix,
                   Kokkos::View<const T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src_kk,
                   Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst_kk,
                   Kokkos::View<const int *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> dst_mask, bool has_mask, const char *label) {
    const auto row_ptr = matrix.row_ptr();
    const auto col_idx = matrix.col_idx();
    const auto csr_vals = matrix.csr_values();
    const std::size_t n_dst = matrix.n_dst();
    const std::size_t n_vec = src_kk.extent(1);

    using exec_space = typename MemorySpace::execution_space;
    using point_t = std::int64_t;
    using policy_t = Kokkos::MDRangePolicy<exec_space, Kokkos::Rank<2>, point_t>;
    Kokkos::parallel_for(
        label, policy_t({0, 0}, {static_cast<point_t>(n_dst), static_cast<point_t>(n_vec)}), KOKKOS_LAMBDA(const point_t j, const point_t v) {
            if (has_mask && dst_mask(j) == 0) return;  // masked destination cell left unchanged
            double sum = 0.0;
            for (index_t k = row_ptr(j); k < row_ptr(j + 1); ++k) {
                sum += csr_vals(k) * static_cast<double>(src_kk(col_idx(k), v));
            }
            dst_kk(j, v) = static_cast<T>(sum);
        });
}

}  // namespace detail

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
/// T may be double or float. The float32 path (FR-018) requires the matrix to
/// be in CSR form (to_csr()) and accumulates in double per destination row.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @tparam T           Field element type (double or float)
/// @param matrix  Sparse interpolation operator (COO: factorList + row/col)
/// @param src     Source field [n_src]
/// @param dst     Destination field [n_dst] — overwritten with result
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src() or
///         dst.extent(0) != matrix.n_dst(). dst is NOT written on validation
///         failure.
template <class MemorySpace, class T>
void apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const T, 1> src, field_view<T, 1> dst) {
    static_assert(detail::is_apply_scalar_v<T>, "axis::solver::apply: field dtype must be double or float");

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

    // Wrap the raw src/dst pointers in unmanaged rank-2 Kokkos Views
    // (LayoutLeft, extent(1) == 1) so the tuned double path and the generic
    // row-parallel path share one kernel shape.
    using dst_view_t = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using src_view_t = Kokkos::View<const T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    dst_view_t dst_kk(dst.data_handle(), n_dst, 1);
    src_view_t src_kk(src.data_handle(), matrix.n_src(), 1);

    if constexpr (std::is_same_v<T, double>) {
        using exec_space = typename MemorySpace::execution_space;

        if (matrix.is_csr()) {
            // ── CSR path: KokkosSparse::spmv (required dependency) ─────────────
            // Hardware-tuned SpMV (cuSPARSE on GPU, MKL on CPU when available, or
            // the KokkosKernels native implementation). The CrsMatrix is a derived
            // operator of the (immutable) CSR weights, so it is built once and
            // cached on the matrix rather than rebuilt per call.
            const auto &A = matrix.kk_crs_matrix();
            KokkosSparse::spmv("N", 1.0, A, src_kk, 0.0, dst_kk);
        } else {
            // ── COO scatter-add path (original): uses atomics ──────────────────
            const auto &S = matrix.factor_list_view();   // [nnz]
            const auto &row = matrix.factor_row_view();  // [nnz]
            const auto &col = matrix.factor_col_view();  // [nnz]
            const std::size_t nnz = matrix.nnz();

            // Zero-initialize dst before scatter-add
            Kokkos::parallel_for(
                "axis::apply::zero_dst", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const std::size_t j) { dst_kk(j, 0) = 0.0; });

            // SpMV scatter-add: dst(row_k) += S(k) * src(col_k)
            Kokkos::parallel_for(
                "axis::apply::spmv", Kokkos::RangePolicy<exec_space>(0, nnz), KOKKOS_LAMBDA(const std::size_t k) {
                    const auto r = row(k);
                    const auto c = col(k);
                    Kokkos::atomic_add(&dst_kk(r, 0), S(k) * src_kk(c, 0));
                });
        }
    } else {
        // ── float32 path: CSR row-parallel, double accumulation ───────────────
        if (!matrix.is_csr()) detail::require_csr("apply");
        Kokkos::View<const int *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> no_mask(nullptr, 0);
        detail::spmv_csr_rows<MemorySpace, T>(matrix, src_kk, dst_kk, no_mask, false, "axis::apply::csr_f32");
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
/// CSR path (double): KokkosSparse::spmv over the cached CrsMatrix with rank-2
///   (multivector) views — the same operator the rank-1 apply path caches.
/// COO path (double): parallel_for over nnz, inner serial loop over variables
///   with atomic accumulation.
/// float32 path: CSR row-parallel with double accumulation per (row, variable);
///   requires to_csr() first.
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @tparam T           Field element type (double or float)
/// @param matrix  Sparse interpolation operator
/// @param src     Source fields [n_src, n_vars]
/// @param dst     Destination fields [n_dst, n_vars] — overwritten with result
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src(),
///         dst.extent(0) != matrix.n_dst(), or src.extent(1) != dst.extent(1).
template <class MemorySpace, class T>
void batch_apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const T, 2> src, field_view<T, 2> dst) {
    static_assert(detail::is_apply_scalar_v<T>, "axis::solver::batch_apply: field dtype must be double or float");

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

    // Wrap raw pointers in unmanaged rank-2 Kokkos Views (LayoutLeft = column-major).
    // field_view is layout_left, so src(cell, var) is contiguous along cells.
    using dst_view_t = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using src_view_t = Kokkos::View<const T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    dst_view_t dst_kk(dst.data_handle(), n_dst, n_vars);
    src_view_t src_kk(src.data_handle(), n_src, n_vars);

    if constexpr (std::is_same_v<T, double>) {
        using exec_space = typename MemorySpace::execution_space;

        if (matrix.is_csr()) {
            // ── CSR path: KokkosSparse::spmv with multivector (rank-2) views ────
            // The cached CrsMatrix (built once on first use by the rank-1 apply
            // path) is reused here; KokkosSparse::spmv accepts rank-2
            // (multivector) x/y views against the same operator.
            const auto &A = matrix.kk_crs_matrix();
            KokkosSparse::spmv("N", 1.0, A, src_kk, 0.0, dst_kk);
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
                    dst_kk(j, v) = 0.0;
                });

            // Scatter-add: dst(row_k, v) += S(k) * src(col_k, v) for all v
            Kokkos::parallel_for(
                "axis::batch_apply::coo", Kokkos::RangePolicy<exec_space>(0, nnz * n_vars), KOKKOS_LAMBDA(const std::size_t idx) {
                    const auto k = idx % nnz;  // nonzero index
                    const auto v = idx / nnz;  // variable index
                    const auto r = row(k);
                    const auto c = col(k);
                    Kokkos::atomic_add(&dst_kk(r, v), S(k) * src_kk(c, v));
                });
        }
    } else {
        // ── float32 path: CSR row-parallel, double accumulation ───────────────
        if (!matrix.is_csr()) detail::require_csr("batch_apply");
        Kokkos::View<const int *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> no_mask(nullptr, 0);
        detail::spmv_csr_rows<MemorySpace, T>(matrix, src_kk, dst_kk, no_mask, false, "axis::batch_apply::csr_f32");
    }

    Kokkos::fence("axis::batch_apply::complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Masked apply: dst = S · src, skipping masked destination cells
// ─────────────────────────────────────────────────────────────────────────────

/// Apply the interpolation matrix with an optional destination cell mask.
///
/// Masked destination cells (mask value == 0) are skipped — their values in
/// dst are left unchanged. This supports Requirement 8.4: land/ocean boundary
/// handling where masked regions should not be written.
///
/// The float32 variant requires CSR form (see the local apply note above).
///
/// @tparam MemorySpace Kokkos memory space of the InterpolationMatrix
/// @tparam T           Field element type (double or float)
/// @param matrix    Sparse interpolation operator (COO or CSR)
/// @param src       Source field [n_src]
/// @param dst       Destination field [n_dst] — masked cells left unchanged
/// @param dst_mask  Per-cell mask [n_dst] (0=masked/inactive, nonzero=active)
///
/// @throws std::invalid_argument if src.extent(0) != matrix.n_src() or
///         dst.extent(0) != matrix.n_dst()
template <class MemorySpace, class T>
void apply(const InterpolationMatrix<MemorySpace> &matrix, field_view<const T, 1> src, field_view<T, 1> dst, field_view<const int, 1> dst_mask) {
    static_assert(detail::is_apply_scalar_v<T>, "axis::solver::apply (masked): field dtype must be double or float");

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
    using dst_view_t = Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using src_view_t = Kokkos::View<const T **, Kokkos::LayoutLeft, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    dst_view_t dst_kk(dst.data_handle(), n_dst, 1);
    src_view_t src_kk(src.data_handle(), matrix.n_src(), 1);
    Kokkos::View<const int *, MemorySpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> mask_view(dst_mask.data_handle(), dst_mask.extent(0));

    if constexpr (std::is_same_v<T, double>) {
        using exec_space = typename MemorySpace::execution_space;

        if (matrix.is_csr()) {
            // ── CSR path with mask: row-parallel, double accumulation ──────────
            detail::spmv_csr_rows<MemorySpace, T>(matrix, src_kk, dst_kk, mask_view, has_mask, "axis::apply::csr_masked");
        } else {
            // ── COO scatter-add path with mask ─────────────────────────────────
            const auto &S = matrix.factor_list_view();
            const auto &row = matrix.factor_row_view();
            const auto &col = matrix.factor_col_view();
            const std::size_t nnz = matrix.nnz();

            // Zero-initialize only unmasked dst cells
            Kokkos::parallel_for(
                "axis::apply::zero_dst_masked", Kokkos::RangePolicy<exec_space>(0, n_dst), KOKKOS_LAMBDA(const std::size_t j) {
                    if (has_mask && mask_view(j) == 0) return;
                    dst_kk(j, 0) = 0.0;
                });

            // SpMV scatter-add: skip writes to masked rows
            Kokkos::parallel_for(
                "axis::apply::spmv_masked", Kokkos::RangePolicy<exec_space>(0, nnz), KOKKOS_LAMBDA(const std::size_t k) {
                    const auto r = row(k);
                    // Skip masked destination cells
                    if (has_mask && mask_view(r) == 0) return;
                    const auto c = col(k);
                    Kokkos::atomic_add(&dst_kk(r, 0), S(k) * src_kk(c, 0));
                });
        }
    } else {
        // ── float32 path: CSR row-parallel, double accumulation ───────────────
        if (!matrix.is_csr()) detail::require_csr("apply (masked)");
        detail::spmv_csr_rows<MemorySpace, T>(matrix, src_kk, dst_kk, mask_view, has_mask, "axis::apply::csr_masked_f32");
    }

    Kokkos::fence("axis::apply_masked::complete");
}

}  // namespace axis::solver

#endif  // AXIS_SOLVER_APPLY_HPP
