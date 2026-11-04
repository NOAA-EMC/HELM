// AXIS unit test: SpMV apply equals reference scalar loop + extent validation
// Verifies correctness of Kokkos parallel apply against a manual reference loop
// and validates that extent mismatches throw std::invalid_argument.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>

#include "common/test_kokkos_env.hpp"

namespace {}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Helper: build a small hand-crafted InterpolationMatrix for testing.
// 3 source cells, 2 destination cells, 4 nonzero weights:
//   dst[0] = 0.5 * src[0] + 0.3 * src[1]
//   dst[1] = 0.2 * src[1] + 1.0 * src[2]
static solver::InterpolationMatrix<MemSpace> make_test_matrix() {
    constexpr std::size_t nnz = 4;
    constexpr std::size_t n_src = 3;
    constexpr std::size_t n_dst = 2;

    Kokkos::View<double *, MemSpace> weights("weights", nnz);
    Kokkos::View<index_t *, MemSpace> rows("rows", nnz);
    Kokkos::View<index_t *, MemSpace> cols("cols", nnz);
    Kokkos::View<double *, MemSpace> frac_a("frac_a", n_src);
    Kokkos::View<double *, MemSpace> frac_b("frac_b", n_dst);
    Kokkos::View<double *, MemSpace> area_a("area_a", n_src);
    Kokkos::View<double *, MemSpace> area_b("area_b", n_dst);

    // dst[0] += 0.5 * src[0]
    weights(0) = 0.5;
    rows(0) = 0;
    cols(0) = 0;
    // dst[0] += 0.3 * src[1]
    weights(1) = 0.3;
    rows(1) = 0;
    cols(1) = 1;
    // dst[1] += 0.2 * src[1]
    weights(2) = 0.2;
    rows(2) = 1;
    cols(2) = 1;
    // dst[1] += 1.0 * src[2]
    weights(3) = 1.0;
    rows(3) = 1;
    cols(3) = 2;

    for (std::size_t i = 0; i < n_src; ++i) {
        frac_a(i) = 1.0;
        area_a(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        frac_b(j) = 1.0;
        area_b(j) = 1.0;
    }

    return solver::InterpolationMatrix<MemSpace>(std::move(weights), std::move(rows), std::move(cols), std::move(frac_a), std::move(frac_b),
                                                 std::move(area_a), std::move(area_b), n_src, n_dst);
}

// Test: apply matches the expected reference-loop result
TEST(SpmvApply, MatchesReferenceLoop) {
    auto matrix = make_test_matrix();

    // src = {2.0, 4.0, 6.0}
    std::vector<double> src_data = {2.0, 4.0, 6.0};
    std::vector<double> dst_data(2, 0.0);

    field_view<const double, 1> src_view(src_data.data(), 3);
    field_view<double, 1> dst_view(dst_data.data(), 2);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // Reference:
    //   dst[0] = 0.5*2.0 + 0.3*4.0 = 1.0 + 1.2 = 2.2
    //   dst[1] = 0.2*4.0 + 1.0*6.0 = 0.8 + 6.0 = 6.8
    EXPECT_NEAR(dst_data[0], 2.2, 1e-14);
    EXPECT_NEAR(dst_data[1], 6.8, 1e-14);
}

// Test: apply zeroes the destination before accumulating
TEST(SpmvApply, ZerosDestinationFirst) {
    auto matrix = make_test_matrix();

    std::vector<double> src_data = {1.0, 1.0, 1.0};
    // Pre-fill dst with garbage
    std::vector<double> dst_data = {999.0, -999.0};

    field_view<const double, 1> src_view(src_data.data(), 3);
    field_view<double, 1> dst_view(dst_data.data(), 2);

    solver::apply<MemSpace>(matrix, src_view, dst_view);

    // dst[0] = 0.5*1 + 0.3*1 = 0.8
    // dst[1] = 0.2*1 + 1.0*1 = 1.2
    EXPECT_NEAR(dst_data[0], 0.8, 1e-14);
    EXPECT_NEAR(dst_data[1], 1.2, 1e-14);
}

// Test: mismatched src extent throws std::invalid_argument
TEST(SpmvApply, SrcExtentMismatchThrows) {
    auto matrix = make_test_matrix();

    std::vector<double> bad_src(5, 1.0);  // n_src is 3, giving 5 → mismatch
    std::vector<double> dst_data(2, 0.0);

    field_view<const double, 1> src_view(bad_src.data(), 5);
    field_view<double, 1> dst_view(dst_data.data(), 2);

    EXPECT_THROW(solver::apply<MemSpace>(matrix, src_view, dst_view), std::invalid_argument);
    // dst must be untouched
    EXPECT_EQ(dst_data[0], 0.0);
    EXPECT_EQ(dst_data[1], 0.0);
}

// Test: mismatched dst extent throws std::invalid_argument
TEST(SpmvApply, DstExtentMismatchThrows) {
    auto matrix = make_test_matrix();

    std::vector<double> src_data(3, 1.0);
    std::vector<double> bad_dst(5, 0.0);  // n_dst is 2, giving 5 → mismatch

    field_view<const double, 1> src_view(src_data.data(), 3);
    field_view<double, 1> dst_view(bad_dst.data(), 5);

    EXPECT_THROW(solver::apply<MemSpace>(matrix, src_view, dst_view), std::invalid_argument);
}

}  // namespace axis::test
