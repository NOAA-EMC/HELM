// AXIS unit test: HaloPattern + gathered-buffer distributed apply equivalence
// Verifies that the distributed apply overload (local_src + gathered_halo_src)
// produces a result identical to a single-rank apply over the full source field.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/halo_pattern.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Build a test scenario where:
//   - 3 local src cells (indices 0,1,2) + 2 remote cells (indices 3,4)
//   - 2 dst cells
//   - Matrix: dst[0] = 0.5*src[0] + 0.5*src[3]  (src[3] is remote)
//             dst[1] = 0.3*src[1] + 0.7*src[4]  (src[4] is remote)

// Test: distributed apply equals single-rank apply over full source
TEST(DistributedApply, EquivalentToSingleRank) {
    constexpr std::size_t n_local_src = 3;
    constexpr std::size_t n_remote = 2;
    constexpr std::size_t n_src = n_local_src + n_remote;
    constexpr std::size_t n_dst = 2;
    constexpr std::size_t nnz = 4;

    // Build the matrix as if running distributed: col indices 0,1,2 are local,
    // col indices 3,4 map to gathered_halo_src[0], gathered_halo_src[1]
    Kokkos::View<double *, MemSpace> weights("weights", nnz);
    Kokkos::View<index_t *, MemSpace> rows("rows", nnz);
    Kokkos::View<index_t *, MemSpace> cols("cols", nnz);
    Kokkos::View<double *, MemSpace> frac_a("frac_a", n_src);
    Kokkos::View<double *, MemSpace> frac_b("frac_b", n_dst);
    Kokkos::View<double *, MemSpace> area_a("area_a", n_src);
    Kokkos::View<double *, MemSpace> area_b("area_b", n_dst);

    weights(0) = 0.5;
    rows(0) = 0;
    cols(0) = 0;  // dst[0] += 0.5*local[0]
    weights(1) = 0.5;
    rows(1) = 0;
    cols(1) = 3;  // dst[0] += 0.5*remote[0]
    weights(2) = 0.3;
    rows(2) = 1;
    cols(2) = 1;  // dst[1] += 0.3*local[1]
    weights(3) = 0.7;
    rows(3) = 1;
    cols(3) = 4;  // dst[1] += 0.7*remote[1]

    for (std::size_t i = 0; i < n_src; ++i) {
        frac_a(i) = 1.0;
        area_a(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        frac_b(j) = 1.0;
        area_b(j) = 1.0;
    }

    solver::InterpolationMatrix<MemSpace> matrix(std::move(weights), std::move(rows), std::move(cols), std::move(frac_a), std::move(frac_b),
                                                 std::move(area_a), std::move(area_b), n_src, n_dst);

    // HaloPattern: 2 remote cells from 1 neighbor (rank 1)
    solver::HaloPattern pattern;
    pattern.source_ranks = {1};
    pattern.rank_offsets = {0, 2};
    pattern.needed_global_src_ids = {100, 200};  // global ids (arbitrary)
    pattern.gather_slot = {0, 1};

    // Source data
    std::vector<double> local_src = {2.0, 4.0, 6.0};
    std::vector<double> halo_src = {8.0, 10.0};  // remote values

    // Distributed apply
    std::vector<double> dst_dist(n_dst, 0.0);
    field_view<const double, 1> local_view(local_src.data(), n_local_src);
    field_view<const double, 1> halo_view(halo_src.data(), n_remote);
    field_view<double, 1> dst_dist_view(dst_dist.data(), n_dst);

    solver::apply<MemSpace>(matrix, pattern, local_view, halo_view, dst_dist_view);

    // Single-rank reference: concatenate local + remote as full source
    std::vector<double> full_src = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::vector<double> dst_single(n_dst, 0.0);
    field_view<const double, 1> full_view(full_src.data(), n_src);
    field_view<double, 1> dst_single_view(dst_single.data(), n_dst);

    solver::apply<MemSpace>(matrix, full_view, dst_single_view);

    // Both should match exactly
    // dst[0] = 0.5*2 + 0.5*8 = 1 + 4 = 5.0
    // dst[1] = 0.3*4 + 0.7*10 = 1.2 + 7.0 = 8.2
    EXPECT_NEAR(dst_dist[0], dst_single[0], 1e-14);
    EXPECT_NEAR(dst_dist[1], dst_single[1], 1e-14);
    EXPECT_NEAR(dst_dist[0], 5.0, 1e-14);
    EXPECT_NEAR(dst_dist[1], 8.2, 1e-14);
}

// Test: distributed apply with mismatched gathered_halo_src size throws
TEST(DistributedApply, HaloSizeMismatchThrows) {
    constexpr std::size_t n_src = 4;
    constexpr std::size_t n_dst = 2;
    constexpr std::size_t nnz = 2;

    Kokkos::View<double *, MemSpace> weights("w", nnz);
    Kokkos::View<index_t *, MemSpace> rows("r", nnz);
    Kokkos::View<index_t *, MemSpace> cols("c", nnz);
    Kokkos::View<double *, MemSpace> frac_a("fa", n_src);
    Kokkos::View<double *, MemSpace> frac_b("fb", n_dst);
    Kokkos::View<double *, MemSpace> area_a("aa", n_src);
    Kokkos::View<double *, MemSpace> area_b("ab", n_dst);

    weights(0) = 1.0;
    rows(0) = 0;
    cols(0) = 0;
    weights(1) = 1.0;
    rows(1) = 1;
    cols(1) = 1;

    solver::InterpolationMatrix<MemSpace> matrix(std::move(weights), std::move(rows), std::move(cols), std::move(frac_a), std::move(frac_b),
                                                 std::move(area_a), std::move(area_b), n_src, n_dst);

    solver::HaloPattern pattern;
    pattern.source_ranks = {1};
    pattern.rank_offsets = {0, 2};
    pattern.needed_global_src_ids = {10, 20};  // expects 2 remote values
    pattern.gather_slot = {0, 1};

    std::vector<double> local_src(2, 1.0);
    std::vector<double> bad_halo(5, 1.0);  // should be 2, not 5
    std::vector<double> dst(n_dst, 0.0);

    field_view<const double, 1> lv(local_src.data(), 2);
    field_view<const double, 1> hv(bad_halo.data(), 5);
    field_view<double, 1> dv(dst.data(), n_dst);

    EXPECT_THROW(solver::apply<MemSpace>(matrix, pattern, lv, hv, dv), std::invalid_argument);
}

}  // namespace axis::test
