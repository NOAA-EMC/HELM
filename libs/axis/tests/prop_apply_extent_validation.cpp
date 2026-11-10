// ─── Property-Based Tests: Apply Extent Validation ───────────────────────────
// Feature: helm-axis-microlibrary, Property 18: Apply Extent Validation
//
// Call apply with mismatched src/dst extents, verify std::invalid_argument
// is thrown and dst is NOT modified (remains at sentinel value).
//
// **Validates: Requirements 9.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

/// Build a trivial InterpolationMatrix with given dimensions.
axis::solver::InterpolationMatrix<Kokkos::HostSpace> build_trivial_matrix(std::size_t n_src, std::size_t n_dst) {
    // One nonzero entry: weight=1.0, row=0, col=0
    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", 1);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", 1);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", 1);
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    fl(0) = 1.0;
    fr(0) = 0;
    fc(0) = 0;
    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    return axis::solver::InterpolationMatrix<Kokkos::HostSpace>(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                std::move(aa), std::move(ab), n_src, n_dst);
}

// ─── Property 18a: Mismatched src extent throws and dst untouched ────────────

RC_GTEST_PROP(PropApplyExtentValidation, MismatchedSrcThrows, ()) {
    const auto n_src = *rc::gen::inRange<std::size_t>(3, 20);
    const auto n_dst = *rc::gen::inRange<std::size_t>(3, 20);

    // Wrong src extent: off by at least 1
    const auto wrong_src = *rc::gen::suchThat(rc::gen::inRange<std::size_t>(1, 30), [n_src](std::size_t v) { return v != n_src; });

    auto matrix = build_trivial_matrix(n_src, n_dst);

    // Create wrong-sized source and correct-sized dst
    std::vector<double> src_data(wrong_src, 1.0);
    const double sentinel = -42.42;
    std::vector<double> dst_data(n_dst, sentinel);

    axis::field_view<const double, 1> src_view(src_data.data(), wrong_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), n_dst);

    // Should throw std::invalid_argument
    bool threw = false;
    try {
        axis::solver::apply(matrix, src_view, dst_view);
    } catch (const std::invalid_argument &) {
        threw = true;
    }

    RC_ASSERT(threw);

    // dst must be untouched (sentinel preserved)
    for (std::size_t j = 0; j < n_dst; ++j) {
        RC_ASSERT(dst_data[j] == sentinel);
    }
}

// ─── Property 18b: Mismatched dst extent throws and dst untouched ────────────

RC_GTEST_PROP(PropApplyExtentValidation, MismatchedDstThrows, ()) {
    const auto n_src = *rc::gen::inRange<std::size_t>(3, 20);
    const auto n_dst = *rc::gen::inRange<std::size_t>(3, 20);

    // Wrong dst extent: off by at least 1
    const auto wrong_dst = *rc::gen::suchThat(rc::gen::inRange<std::size_t>(1, 30), [n_dst](std::size_t v) { return v != n_dst; });

    auto matrix = build_trivial_matrix(n_src, n_dst);

    // Correct-sized source and wrong-sized dst
    std::vector<double> src_data(n_src, 1.0);
    const double sentinel = -42.42;
    std::vector<double> dst_data(wrong_dst, sentinel);

    axis::field_view<const double, 1> src_view(src_data.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_data.data(), wrong_dst);

    // Should throw std::invalid_argument
    bool threw = false;
    try {
        axis::solver::apply(matrix, src_view, dst_view);
    } catch (const std::invalid_argument &) {
        threw = true;
    }

    RC_ASSERT(threw);

    // dst must be untouched (sentinel preserved)
    for (std::size_t j = 0; j < wrong_dst; ++j) {
        RC_ASSERT(dst_data[j] == sentinel);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
