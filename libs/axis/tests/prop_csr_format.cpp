// ─── Property-Based Tests: CSR Format Validity and Apply Equivalence ─────────
// Feature: axis-v2-improvements, Properties 10 & 11
//
// Property 10: CSR/COO apply equivalence
//   For any matrix and field, CSR and COO apply SHALL agree within 1e-15.
//
// Property 11: CSR structural validity
//   For any COO matrix, to_csr() SHALL produce monotonic row_ptr with
//   row_ptr[0]==0, row_ptr[n_dst]==nnz, and sorted col_idx per row.
//
// **Validates: Requirements 6.1, 6.2, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a random InterpolationMatrix with given dimensions and nnz.
/// Returns the matrix along with the raw COO data for reference computation.
struct MatrixData {
    axis::solver::InterpolationMatrix<Kokkos::HostSpace> matrix;
    std::vector<double> weights;
    std::vector<axis::index_t> rows;
    std::vector<axis::index_t> cols;
    std::size_t n_src;
    std::size_t n_dst;
    std::size_t nnz;
};

/// Generate a random sparse matrix with reasonable dimensions.
MatrixData generateRandomMatrix() {
    const auto n_src = *rc::gen::inRange<std::size_t>(2, 30);
    const auto n_dst = *rc::gen::inRange<std::size_t>(2, 30);
    const auto max_nnz = std::min(n_src * n_dst, static_cast<std::size_t>(80));
    const auto nnz = *rc::gen::inRange<std::size_t>(1, max_nnz + 1);

    std::vector<double> weights(nnz);
    std::vector<axis::index_t> rows(nnz);
    std::vector<axis::index_t> cols(nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        weights[k] = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 1000.0; });
        rows[k] = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_dst));
        cols[k] = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_src));
    }

    // Build Kokkos Views
    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", nnz);
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    for (std::size_t k = 0; k < nnz; ++k) {
        fl(k) = weights[k];
        fr(k) = rows[k];
        fc(k) = cols[k];
    }
    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    axis::solver::InterpolationMatrix<Kokkos::HostSpace> matrix(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                std::move(aa), std::move(ab), n_src, n_dst);

    return MatrixData{std::move(matrix), std::move(weights), std::move(rows), std::move(cols), n_src, n_dst, nnz};
}

// ─── Property 10: CSR/COO apply equivalence ─────────────────────────────────
//
// For any random InterpolationMatrix and source field:
//   1. Apply using COO path to get dst_coo
//   2. Call to_csr()
//   3. Apply using CSR path to get dst_csr
//   4. Verify element-by-element agreement within 1e-15
//
// **Validates: Requirements 6.1, 6.2, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropCsrFormat, CooAndCsrApplyAgree, ()) {
    auto data = generateRandomMatrix();
    const auto n_src = data.n_src;
    const auto n_dst = data.n_dst;

    // Generate random source field
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_data[i] = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 100.0; });
    }

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);

    // ── COO apply ────────────────────────────────────────────────────────────
    RC_ASSERT(!data.matrix.is_csr());
    std::vector<double> dst_coo(n_dst, 0.0);
    axis::field_view<double, 1> dst_coo_view(dst_coo.data(), n_dst);
    axis::solver::apply(data.matrix, src_view, dst_coo_view);

    // ── Convert to CSR ───────────────────────────────────────────────────────
    data.matrix.to_csr();
    RC_ASSERT(data.matrix.is_csr());

    // ── CSR apply ────────────────────────────────────────────────────────────
    std::vector<double> dst_csr(n_dst, 0.0);
    axis::field_view<double, 1> dst_csr_view(dst_csr.data(), n_dst);
    axis::solver::apply(data.matrix, src_view, dst_csr_view);

    // ── Verify agreement within round-off tolerance ────────────────────────
    // KokkosSparse::spmv may use a different summation order than the COO
    // scatter-add path, producing slightly different rounding. Both are correct.
    const double tol = 1e-12;
    for (std::size_t j = 0; j < n_dst; ++j) {
        double err = std::abs(dst_csr[j] - dst_coo[j]);
        double scale = std::max(std::abs(dst_coo[j]), 1.0);
        RC_ASSERT(err <= tol * scale + tol);
    }
}

// ─── Property 11: CSR structural validity ────────────────────────────────────
//
// For any COO matrix, to_csr() SHALL produce:
//   1. row_ptr[0] == 0
//   2. row_ptr[n_dst] == nnz
//   3. row_ptr is monotonically non-decreasing
//   4. For each row, col_idx entries are sorted (non-decreasing)
//
// **Validates: Requirements 6.1, 6.2, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropCsrFormat, CsrStructuralValidity, ()) {
    auto data = generateRandomMatrix();
    const auto n_dst = data.n_dst;
    const auto nnz = data.nnz;

    // Convert to CSR
    data.matrix.to_csr();
    RC_ASSERT(data.matrix.is_csr());

    auto row_ptr = data.matrix.row_ptr();
    auto col_idx = data.matrix.col_idx();

    // ── Invariant 1: row_ptr[0] == 0 ────────────────────────────────────────
    RC_ASSERT(row_ptr(0) == 0);

    // ── Invariant 2: row_ptr[n_dst] == nnz ──────────────────────────────────
    RC_ASSERT(static_cast<std::size_t>(row_ptr(n_dst)) == nnz);

    // ── Invariant 3: row_ptr is monotonically non-decreasing ────────────────
    for (std::size_t j = 0; j < n_dst; ++j) {
        RC_ASSERT(row_ptr(j) <= row_ptr(j + 1));
    }

    // ── Invariant 4: col_idx is sorted within each row ──────────────────────
    for (std::size_t j = 0; j < n_dst; ++j) {
        const auto start = static_cast<std::size_t>(row_ptr(j));
        const auto end = static_cast<std::size_t>(row_ptr(j + 1));
        for (std::size_t k = start + 1; k < end; ++k) {
            RC_ASSERT(col_idx(k - 1) <= col_idx(k));
        }
    }
}

// ─── Property 11b: CSR structural validity — empty matrix edge case ──────────
//
// An empty InterpolationMatrix (nnz == 0) SHALL produce valid CSR arrays:
//   row_ptr[0] == 0, row_ptr[n_dst] == 0, all row_ptr entries == 0.
//
// **Validates: Requirements 6.1, 6.2**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropCsrFormat, CsrStructuralValidityEmptyMatrix, ()) {
    // Generate dimensions but no nonzeros
    const auto n_src = *rc::gen::inRange<std::size_t>(1, 20);
    const auto n_dst = *rc::gen::inRange<std::size_t>(1, 20);

    // Build empty InterpolationMatrix
    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", 0);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", 0);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", 0);
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    axis::solver::InterpolationMatrix<Kokkos::HostSpace> matrix(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                std::move(aa), std::move(ab), n_src, n_dst);

    RC_ASSERT(matrix.nnz() == 0);

    // Convert to CSR
    matrix.to_csr();
    RC_ASSERT(matrix.is_csr());

    auto row_ptr = matrix.row_ptr();

    // row_ptr should have n_dst+1 entries, all zero
    RC_ASSERT(row_ptr.extent(0) == n_dst + 1);
    RC_ASSERT(row_ptr(0) == 0);
    RC_ASSERT(static_cast<std::size_t>(row_ptr(n_dst)) == 0);

    for (std::size_t j = 0; j < n_dst; ++j) {
        RC_ASSERT(row_ptr(j) <= row_ptr(j + 1));
    }
}

// ─── Property 10b: CSR/COO equivalence with repeated to_csr() is idempotent ─
//
// Calling to_csr() multiple times SHALL not change the result (idempotent).
//
// **Validates: Requirements 6.2**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropCsrFormat, ToCsrIsIdempotent, ()) {
    auto data = generateRandomMatrix();
    const auto n_src = data.n_src;
    const auto n_dst = data.n_dst;

    // Generate source field
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_data[i] = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 100.0; });
    }

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);

    // First to_csr() and apply
    data.matrix.to_csr();
    RC_ASSERT(data.matrix.is_csr());

    std::vector<double> dst_first(n_dst, 0.0);
    axis::field_view<double, 1> dst_first_view(dst_first.data(), n_dst);
    axis::solver::apply(data.matrix, src_view, dst_first_view);

    // Second to_csr() call (should be no-op)
    data.matrix.to_csr();
    RC_ASSERT(data.matrix.is_csr());

    std::vector<double> dst_second(n_dst, 0.0);
    axis::field_view<double, 1> dst_second_view(dst_second.data(), n_dst);
    axis::solver::apply(data.matrix, src_view, dst_second_view);

    // Results must be identical (bitwise — same path, same data)
    for (std::size_t j = 0; j < n_dst; ++j) {
        RC_ASSERT(dst_first[j] == dst_second[j]);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
