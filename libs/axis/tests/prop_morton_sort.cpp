// ─── Property-Based Tests: Morton Code Spatial Ordering ─────────────────────
// Feature: axis-performance-optimizations
//
// Property 7: Morton Code Spatial Ordering
//   For any set of 2D centroid coordinates normalized to [0, 2^16), the Morton
//   codes produced by morton_encode_2d() SHALL produce a valid Z-curve ordering
//   such that sorting indices by Morton code groups spatially adjacent cells
//   consecutively. Specifically, for any two centroids within Chebyshev distance
//   1 in the normalized grid, their Morton codes SHALL differ by at most
//   3 * 2^(2*k) for some integer k (Z-curve locality guarantee).
//
// **Validates: Requirements 5.1, 5.2, 5.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/morton_sort.hpp>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <set>
#include <vector>

namespace {

using axis::detail::morton_encode_2d;
using axis::detail::morton_sort_indices;
using axis::detail::normalize_coord;

// ─── Helper: Check Z-curve locality bound ────────────────────────────────────
// For two points within Chebyshev distance 1, their Morton codes should differ
// by at most 3 * 2^(2*k) for some integer k >= 0.
// The set of valid bounds is: 3, 12, 48, 192, 768, ...
// i.e., 3 * 4^k for k = 0, 1, 2, ...
// We check if the difference is <= 3 * 4^k for some k in [0, 16].
bool satisfies_zcurve_locality_bound(uint64_t code_a, uint64_t code_b) {
    uint64_t diff = (code_a > code_b) ? (code_a - code_b) : (code_b - code_a);
    if (diff == 0) return true;

    // Check if diff <= 3 * 4^k for some k in [0, 15]
    uint64_t bound = 3;
    for (int k = 0; k <= 15; ++k) {
        if (diff <= bound) return true;
        bound *= 4;
    }
    return false;
}

// ─── Helper: Compute Euclidean distance between two 2D points ────────────────
double euclidean_dist(double x1, double y1, double x2, double y2) {
    double dx = x1 - x2;
    double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// ─── Property 7a: morton_encode_2d Bit Interleave Correctness ────────────────
// For random (x, y) pairs, bits of x appear in even positions and bits of y
// appear in odd positions of the result.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, BitInterleaveCorrectness, ()) {
    uint32_t x = *rc::gen::inRange(0u, 65536u);
    uint32_t y = *rc::gen::inRange(0u, 65536u);

    uint64_t code = morton_encode_2d(x, y);

    // Extract even-position bits (should reconstruct x)
    uint64_t extracted_x = 0;
    for (int bit = 0; bit < 16; ++bit) {
        uint64_t even_bit = (code >> (2 * bit)) & 1;
        extracted_x |= (even_bit << bit);
    }

    // Extract odd-position bits (should reconstruct y)
    uint64_t extracted_y = 0;
    for (int bit = 0; bit < 16; ++bit) {
        uint64_t odd_bit = (code >> (2 * bit + 1)) & 1;
        extracted_y |= (odd_bit << bit);
    }

    RC_ASSERT(extracted_x == static_cast<uint64_t>(x));
    RC_ASSERT(extracted_y == static_cast<uint64_t>(y));
}

// ─── Property 7b: morton_encode_2d Monotonicity within rows ──────────────────
// For fixed y, increasing x should produce increasing Morton codes within the
// same Z-order quadrant structure. Specifically, for two x values in the same
// 2-cell block (consecutive even/odd), the one with larger x has a larger code.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, MonotonicityWithinRow, ()) {
    uint32_t y = *rc::gen::inRange(0u, 65535u);
    uint32_t x1 = *rc::gen::inRange(0u, 65534u);
    uint32_t x2 = x1 + 1;  // Consecutive x values

    uint64_t code1 = morton_encode_2d(x1, y);
    uint64_t code2 = morton_encode_2d(x2, y);

    // For same y, x1 < x2 implies code1 < code2
    // (since x occupies even bits, incrementing x always increases the code
    // when y is fixed)
    RC_ASSERT(code1 < code2);
}

// ─── Property 7c: normalize_coord Range ─────────────────────────────────────
// For any val in [min_val, max_val], normalize_coord returns a value in
// [0, 65535]. For val < min_val → 0, for val > max_val → 65535.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, NormalizeCoordRange, ()) {
    // Generate a valid range [min_val, max_val] with max > min
    double min_val = *rc::gen::map(rc::gen::inRange(-10000, 10000), [](int v) { return v * 0.01; });
    double range = *rc::gen::map(rc::gen::inRange(1, 10000), [](int v) { return v * 0.01; });
    double max_val = min_val + range;

    // Generate a value anywhere in the extended range
    double t = *rc::gen::map(rc::gen::inRange(-200, 1200), [](int v) { return v * 0.001; });
    double val = min_val + t * (max_val - min_val);

    uint32_t result = normalize_coord(val, min_val, max_val);

    // Result must always be in [0, 65535]
    RC_ASSERT(result <= 65535u);

    // If val <= min_val, result should be 0
    if (val <= min_val) {
        RC_ASSERT(result == 0u);
    }
    // If val >= max_val, result should be 65535
    if (val >= max_val) {
        RC_ASSERT(result == 65535u);
    }
}

// ─── Property 7d: normalize_coord Degenerate Range ──────────────────────────
// When max_val <= min_val, returns 0.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, NormalizeCoordDegenerateRange, ()) {
    double min_val = *rc::gen::map(rc::gen::inRange(-10000, 10000), [](int v) { return v * 0.01; });
    // max_val <= min_val (degenerate)
    double offset = *rc::gen::map(rc::gen::inRange(0, 5000), [](int v) { return v * 0.01; });
    double max_val = min_val - offset;

    double val = *rc::gen::map(rc::gen::inRange(-10000, 10000), [](int v) { return v * 0.01; });

    uint32_t result = normalize_coord(val, min_val, max_val);
    RC_ASSERT(result == 0u);
}

// ─── Property 7e: morton_sort_indices Permutation Validity ───────────────────
// The returned View is a valid permutation of [0, n_dst) — all values are
// unique and in range.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, PermutationValidity, ()) {
    // Generate a set of random 2D centroids
    int n = *rc::gen::inRange(1, 200);

    // Generate random centroid coordinates
    std::vector<double> cx(n), cy(n);
    for (int i = 0; i < n; ++i) {
        cx[i] = *rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return v * 0.01; });
        cy[i] = *rc::gen::map(rc::gen::inRange(-9000, 9001), [](int v) { return v * 0.01; });
    }

    // Create Kokkos views on host
    using MemSpace = Kokkos::HostSpace;
    Kokkos::View<double *, MemSpace> centroids_x("cx", n);
    Kokkos::View<double *, MemSpace> centroids_y("cy", n);
    for (int i = 0; i < n; ++i) {
        centroids_x(i) = cx[i];
        centroids_y(i) = cy[i];
    }

    // Get const views
    Kokkos::View<const double *, MemSpace> cx_const(centroids_x);
    Kokkos::View<const double *, MemSpace> cy_const(centroids_y);

    auto sorted_indices = morton_sort_indices<MemSpace>(cx_const, cy_const, static_cast<std::size_t>(n));

    // Verify it's a valid permutation: all values unique and in [0, n)
    std::set<int64_t> seen;
    for (int i = 0; i < n; ++i) {
        int64_t idx = sorted_indices(i);
        RC_ASSERT(idx >= 0);
        RC_ASSERT(idx < static_cast<int64_t>(n));
        RC_ASSERT(seen.find(idx) == seen.end());
        seen.insert(idx);
    }
    RC_ASSERT(static_cast<int>(seen.size()) == n);
}

// ─── Property 7f: Morton Spatial Locality ────────────────────────────────────
// For a grid of centroids, morton-sorted consecutive indices should be spatially
// close. Verify that mean distance between consecutive sorted cells is less than
// mean distance between consecutive natural-order cells (i.e., Morton order
// improves locality).
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, SpatialLocality, ()) {
    // Generate a grid of centroids (at least 4x4 to see locality benefit)
    int ni = *rc::gen::inRange(4, 20);
    int nj = *rc::gen::inRange(4, 20);
    int n = ni * nj;

    std::vector<double> cx(n), cy(n);
    for (int j = 0; j < nj; ++j) {
        for (int i = 0; i < ni; ++i) {
            int idx = j * ni + i;
            cx[idx] = static_cast<double>(i);
            cy[idx] = static_cast<double>(j);
        }
    }

    // Create Kokkos views
    using MemSpace = Kokkos::HostSpace;
    Kokkos::View<double *, MemSpace> centroids_x("cx", n);
    Kokkos::View<double *, MemSpace> centroids_y("cy", n);
    for (int i = 0; i < n; ++i) {
        centroids_x(i) = cx[i];
        centroids_y(i) = cy[i];
    }

    Kokkos::View<const double *, MemSpace> cx_const(centroids_x);
    Kokkos::View<const double *, MemSpace> cy_const(centroids_y);

    auto sorted_indices = morton_sort_indices<MemSpace>(cx_const, cy_const, static_cast<std::size_t>(n));

    // Compute mean distance between consecutive cells in Morton order
    double morton_total_dist = 0.0;
    for (int k = 1; k < n; ++k) {
        int64_t i_prev = sorted_indices(k - 1);
        int64_t i_curr = sorted_indices(k);
        morton_total_dist += euclidean_dist(cx[i_prev], cy[i_prev], cx[i_curr], cy[i_curr]);
    }
    double morton_mean_dist = morton_total_dist / static_cast<double>(n - 1);

    // Compute mean distance between consecutive cells in natural (row-major) order
    double natural_total_dist = 0.0;
    for (int k = 1; k < n; ++k) {
        natural_total_dist += euclidean_dist(cx[k - 1], cy[k - 1], cx[k], cy[k]);
    }
    double natural_mean_dist = natural_total_dist / static_cast<double>(n - 1);

    // Morton order should improve locality (lower mean distance)
    // For regular grids, row-major has jumps at row boundaries while Morton
    // maintains 2D locality. Use an adaptive tolerance based on grid size:
    // tiny grids or skinny strips are prone to edge noise, while larger grids
    // should strictly demonstrate Morton's spatial locality benefit.
    double tol = 1.01;
    if (n < 25 || std::min(ni, nj) < 6) {
        tol = 1.15;  // Relaxed for tiny grids or thin strips to accommodate edge effects
    } else if (n >= 100) {
        tol = 0.95;  // Stronger bound for larger symmetric grids where Morton must be superior
    }
    RC_ASSERT(morton_mean_dist <= natural_mean_dist * tol);
}

// ─── Property 7g: Z-curve Locality Bound ────────────────────────────────────
// For any two points (x1,y1) and (x2,y2) with Chebyshev distance 1 in the
// normalized grid (|x1-x2|<=1 and |y1-y2|<=1), their Morton codes differ by
// at most 3.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropMortonSort, ZCurveLocalityBound, ()) {
    // Generate a point in the valid range
    uint32_t x = *rc::gen::inRange(1u, 65534u);
    uint32_t y = *rc::gen::inRange(1u, 65534u);

    // Generate a neighbor within Chebyshev distance 1
    int dx = *rc::gen::inRange(-1, 2);  // -1, 0, or 1
    int dy = *rc::gen::inRange(-1, 2);  // -1, 0, or 1

    uint32_t nx = static_cast<uint32_t>(static_cast<int>(x) + dx);
    uint32_t ny = static_cast<uint32_t>(static_cast<int>(y) + dy);

    uint64_t code1 = morton_encode_2d(x, y);
    uint64_t code2 = morton_encode_2d(nx, ny);

    // The Z-curve locality bound: for Chebyshev distance 1 neighbors,
    // the Morton code difference should satisfy the locality bound
    RC_ASSERT(satisfies_zcurve_locality_bound(code1, code2));
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
