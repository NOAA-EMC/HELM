// ─── Property-Based Tests: Rule-Based Generation Matches Resolution ──────────
// Feature: helm-axis-microlibrary, Property 6: Rule-Based Generation Matches Resolution
//
// Uses RapidCheck to verify that for any valid RegularLatLon GridRulesParams,
// the RuleGenerator produces a mesh whose cell count exactly equals
// floor((max_x - min_x) / r_x) * floor((max_y - min_y) / r_y), and whose
// node count equals (ni + 1) * (nj + 1). Also verifies that invalid parameters
// (zero resolution, bbox max < min) throw std::invalid_argument.
//
// All tests execute on Kokkos::HostSpace only (host-only property test).
//
// **Validates: Requirements 7.1, 7.2**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/rule_generator.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "common/test_kokkos_env.hpp"

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a valid RegularLatLon GridRulesParams with constrained ranges:
///   min_x in [-180, 0], max_x in (min_x+1, 180]
///   min_y in [-90, 0], max_y in (min_y+1, 90]
///   r_x in (0.5, 10], r_y in (0.5, 10]
rc::Gen<axis::ingest::GridRulesParams> genRegularLatLonParams() {
    return rc::gen::exec([]() {
        axis::ingest::GridRulesParams params;
        params.kind = "RegularLatLon";

        // min_x in [-180, 0] (integer steps for simplicity)
        params.min_x = *rc::gen::inRange(-180, 1);

        // max_x in (min_x + 1, 180] — ensure at least 1 degree span
        const int min_max_x = static_cast<int>(params.min_x) + 1;
        params.max_x = *rc::gen::inRange(min_max_x, 181);

        // min_y in [-90, 0]
        params.min_y = *rc::gen::inRange(-90, 1);

        // max_y in (min_y + 1, 90]
        const int min_max_y = static_cast<int>(params.min_y) + 1;
        params.max_y = *rc::gen::inRange(min_max_y, 91);

        // r_x in (0.5, 10] — use integer tenths then convert
        // Generate integer in [5, 100] representing tenths of a degree
        const int rx_tenths = *rc::gen::inRange(5, 101);
        params.r_x = static_cast<double>(rx_tenths) / 10.0;

        // r_y in (0.5, 10]
        const int ry_tenths = *rc::gen::inRange(5, 101);
        params.r_y = static_cast<double>(ry_tenths) / 10.0;

        return params;
    });
}

// ─── Property 6a: Cell count matches expected formula ────────────────────────
// Generate random RegularLatLon GridRulesParams, call RuleGenerator::generate,
// verify: mesh.n_cells() == floor((max_x - min_x) / r_x) * floor((max_y - min_y) / r_y)
//
// **Validates: Requirements 7.1, 7.2**

RC_GTEST_PROP(PropRuleGeneration, CellCountMatchesResolution, ()) {
    const auto params = *genRegularLatLonParams();

    const auto ni = static_cast<std::size_t>(std::floor((params.max_x - params.min_x) / params.r_x));
    const auto nj = static_cast<std::size_t>(std::floor((params.max_y - params.min_y) / params.r_y));

    // Skip degenerate cases where resolution is too large for the bbox
    RC_PRE(ni > 0 && nj > 0);

    const std::size_t expected_cells = ni * nj;

    auto mesh = axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(params);

    RC_ASSERT(mesh.n_cells() == expected_cells);
}

// ─── Property 6b: Node count matches expected formula ────────────────────────
// For a RegularLatLon grid with ni cells in x and nj cells in y, the node count
// must be (ni + 1) * (nj + 1) since each cell is a quad sharing boundary nodes.
//
// **Validates: Requirements 7.1, 7.2**

RC_GTEST_PROP(PropRuleGeneration, NodeCountMatchesResolution, ()) {
    const auto params = *genRegularLatLonParams();

    const auto ni = static_cast<std::size_t>(std::floor((params.max_x - params.min_x) / params.r_x));
    const auto nj = static_cast<std::size_t>(std::floor((params.max_y - params.min_y) / params.r_y));

    // Skip degenerate cases
    RC_PRE(ni > 0 && nj > 0);

    const std::size_t expected_nodes = (ni + 1) * (nj + 1);

    auto mesh = axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(params);

    RC_ASSERT(mesh.n_nodes() == expected_nodes);
}

// ─── Property 6c: Generated mesh has valid CSR connectivity ──────────────────
// For any valid RegularLatLon GridRulesParams, the produced mesh must satisfy
// the fundamental CSR invariants: offsets[0]==0, offsets non-decreasing,
// offsets[n_cells]==indices.size(), all indices in [0, n_nodes).
//
// **Validates: Requirements 7.1, 7.2**

RC_GTEST_PROP(PropRuleGeneration, GeneratedMeshHasValidCSR, ()) {
    const auto params = *genRegularLatLonParams();

    const auto ni = static_cast<std::size_t>(std::floor((params.max_x - params.min_x) / params.r_x));
    const auto nj = static_cast<std::size_t>(std::floor((params.max_y - params.min_y) / params.r_y));

    // Skip degenerate cases
    RC_PRE(ni > 0 && nj > 0);

    auto mesh = axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(params);

    const std::size_t n_cells = mesh.n_cells();
    const std::size_t n_nodes = mesh.n_nodes();

    auto offsets = mesh.conn_offsets();
    auto indices = mesh.conn_indices();

    // offsets must have n_cells + 1 entries
    RC_ASSERT(offsets.extent(0) == n_cells + 1);

    // offsets[0] == 0
    RC_ASSERT(offsets[0] == 0);

    // offsets is non-decreasing
    for (std::size_t c = 0; c < n_cells; ++c) {
        RC_ASSERT(offsets[c] <= offsets[c + 1]);
    }

    // offsets[n_cells] == indices.extent(0)
    RC_ASSERT(static_cast<std::size_t>(offsets[n_cells]) == indices.extent(0));

    // All indices in [0, n_nodes)
    const std::size_t nnz = indices.extent(0);
    for (std::size_t k = 0; k < nnz; ++k) {
        RC_ASSERT(indices[k] >= 0);
        RC_ASSERT(static_cast<std::size_t>(indices[k]) < n_nodes);
    }
}

// ─── Property 6d: Zero resolution throws std::invalid_argument ───────────────
// When r_x or r_y is zero (or negative), RuleGenerator must throw.
//
// **Validates: Requirements 7.5**

RC_GTEST_PROP(PropRuleGeneration, ZeroResolutionThrows, ()) {
    auto params = *genRegularLatLonParams();

    // Choose which resolution to make invalid
    const bool zero_rx = *rc::gen::arbitrary<bool>();

    if (zero_rx) {
        // Set r_x to zero or negative
        const int neg_tenths = *rc::gen::inRange(-10, 1);  // [-10, 0] → [-1.0, 0.0]
        params.r_x = static_cast<double>(neg_tenths) / 10.0;
    } else {
        // Set r_y to zero or negative
        const int neg_tenths = *rc::gen::inRange(-10, 1);
        params.r_y = static_cast<double>(neg_tenths) / 10.0;
    }

    RC_ASSERT_THROWS_AS((void)axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(params), std::invalid_argument);
}

// ─── Property 6e: Bounding box max < min throws std::invalid_argument ────────
// When max_x <= min_x or max_y <= min_y, RuleGenerator must throw.
//
// **Validates: Requirements 7.5**

RC_GTEST_PROP(PropRuleGeneration, BboxMaxLessThanMinThrows, ()) {
    axis::ingest::GridRulesParams params;
    params.kind = "RegularLatLon";
    params.r_x = 1.0;
    params.r_y = 1.0;

    // Choose which bbox constraint to violate
    const bool violate_x = *rc::gen::arbitrary<bool>();

    if (violate_x) {
        // Make max_x <= min_x
        params.min_x = *rc::gen::inRange(0, 181);
        const int max_x_offset = *rc::gen::inRange(-10, 1);  // [-10, 0]
        params.max_x = params.min_x + static_cast<double>(max_x_offset);

        // Keep y valid
        params.min_y = -45.0;
        params.max_y = 45.0;
    } else {
        // Make max_y <= min_y
        params.min_y = *rc::gen::inRange(0, 91);
        const int max_y_offset = *rc::gen::inRange(-10, 1);  // [-10, 0]
        params.max_y = params.min_y + static_cast<double>(max_y_offset);

        // Keep x valid
        params.min_x = -90.0;
        params.max_x = 90.0;
    }

    RC_ASSERT_THROWS_AS((void)axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(params), std::invalid_argument);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// RapidCheck/GTest property tests need Kokkos initialized for View allocation.

}  // namespace
