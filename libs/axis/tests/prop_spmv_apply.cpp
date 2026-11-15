// ─── Property-Based Tests: SpMV Apply Equals Reference Loop ─────────────────
// Feature: helm-axis-microlibrary, Property 17: SpMV Apply Equals Reference Loop
//
// For any manually constructed InterpolationMatrix and source field, verify
// solver::apply produces the same result as the scalar reference loop:
//   dst(row_k) += S(k) * src(col_k)
// within round-off tolerance.
//
// **Validates: Requirements 9.1, 9.2**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

// ─── Property 17: Apply matches scalar reference loop ────────────────────────
// Build a random sparse matrix (small dimensions), fill a source field with
// random values, compare solver::apply result to a scalar reference loop.
//
// **Validates: Requirements 9.1, 9.2**

RC_GTEST_PROP(PropSpmvApply, MatchesReferenceLoop, ()) {
    // Generate dimensions
    const auto n_src = *rc::gen::inRange<std::size_t>(2, 20);
    const auto n_dst = *rc::gen::inRange<std::size_t>(2, 20);
    // Generate number of nonzeros (between 1 and n_src*n_dst capped at 50)
    const auto max_nnz = std::min(n_src * n_dst, static_cast<std::size_t>(50));
    const auto nnz = *rc::gen::inRange<std::size_t>(1, max_nnz + 1);

    // Generate random COO entries
    std::vector<double> weights(nnz);
    std::vector<axis::index_t> rows(nnz);
    std::vector<axis::index_t> cols(nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        weights[k] = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 1000.0; });
        rows[k] = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_dst));
        cols[k] = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_src));
    }

    // Generate random source field
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_data[i] = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 100.0; });
    }

    // Build InterpolationMatrix from the generated data
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

    // Compute reference result via scalar loop
    std::vector<double> ref_dst(n_dst, 0.0);
    for (std::size_t k = 0; k < nnz; ++k) {
        auto r = static_cast<std::size_t>(rows[k]);
        auto c = static_cast<std::size_t>(cols[k]);
        ref_dst[r] += weights[k] * src_data[c];
    }

    // Compute via solver::apply
    std::vector<double> dst_data(n_dst, -999.0);  // sentinel to verify overwrite
    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    axis::solver::apply(matrix, src_view, dst_view);

    // Compare: must match within round-off
    const double tol = 1e-12;
    for (std::size_t j = 0; j < n_dst; ++j) {
        double err = std::abs(dst_data[j] - ref_dst[j]);
        double scale = std::max(std::abs(ref_dst[j]), 1.0);
        RC_ASSERT(err < tol * scale + tol);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
