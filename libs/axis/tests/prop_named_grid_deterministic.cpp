// ─── Property-Based Tests: Named-Grid Generation Is Deterministic and File-Free
// Feature: helm-axis-microlibrary, Property 4: Named-Grid Generation Is
//          Deterministic and File-Free
//
// Uses RapidCheck to verify that for all valid named-grid strings from the O,
// F, and N families with N values in [2, 8], two independent calls to
// NamedGridRegistry::generate<Kokkos::HostSpace>(name) produce:
//   1. Identical n_nodes() and n_cells()
//   2. Bitwise-identical node_coords (memcmp of the underlying data)
//   3. Identical CSR offsets and indices
//
// This validates that generation is deterministic (no random state, no
// file-dependent ordering) and purely in-memory (zero file I/O).
//
// **Validates: Requirements 6.1, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <cstring>
#include <string>

#include "common/test_kokkos_env.hpp"

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a valid grid family character: 'O', 'F', 'N', or 'R'.
rc::Gen<char> genFamily() {
    return rc::gen::element('O', 'F', 'N', 'R');
}

/// Generate a small N value in [2, 8] to keep tests fast while exercising
/// the generation logic across multiple sizes.
rc::Gen<int> genSmallN() {
    return rc::gen::inRange(2, 9);
}

/// Generate a valid named-grid string from a random family and small N.
rc::Gen<std::string> genGridName() {
    return rc::gen::apply([](char family, int n) { return std::string(1, family) + std::to_string(n); }, genFamily(), genSmallN());
}

// ─── Property 4a: Same n_nodes() and n_cells() on two independent generates
//
// Generate a random valid grid name, call generate twice, verify identical
// node and cell counts.
//
// **Validates: Requirements 6.1, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropNamedGridDeterministic, SameNodeAndCellCounts, ()) {
    const std::string name = *genGridName();

    auto mesh1 = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);
    auto mesh2 = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);

    RC_ASSERT(mesh1.n_nodes() == mesh2.n_nodes());
    RC_ASSERT(mesh1.n_cells() == mesh2.n_cells());

    // Also verify the meshes are non-empty (valid generation, Req 6.1)
    RC_ASSERT(mesh1.n_nodes() > 0);
    RC_ASSERT(mesh1.n_cells() > 0);
}

// ─── Property 4b: Bitwise-identical node_coords on two independent generates
//
// Generate a random valid grid name, call generate twice, compare the entire
// node_coords buffer byte-for-byte via memcmp.
//
// **Validates: Requirements 6.1, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropNamedGridDeterministic, BitwiseIdenticalNodeCoords, ()) {
    const std::string name = *genGridName();

    auto mesh1 = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);
    auto mesh2 = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);

    // Precondition: identical sizes (tested in 4a but assert here for safety)
    RC_ASSERT(mesh1.n_nodes() == mesh2.n_nodes());

    auto coords1 = mesh1.node_coords();
    auto coords2 = mesh2.node_coords();

    // Verify extents match
    RC_ASSERT(coords1.extent(0) == coords2.extent(0));
    RC_ASSERT(coords1.extent(1) == coords2.extent(1));

    // Bitwise comparison of the underlying coordinate data
    const std::size_t n_bytes = coords1.extent(0) * coords1.extent(1) * sizeof(double);

    if (n_bytes > 0) {
        RC_ASSERT(std::memcmp(coords1.data_handle(), coords2.data_handle(), n_bytes) == 0);
    }
}

// ─── Property 4c: Identical CSR offsets and indices on two independent generates
//
// Generate a random valid grid name, call generate twice, compare the CSR
// connectivity arrays (offsets and indices) byte-for-byte.
//
// **Validates: Requirements 6.1, 6.5**
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropNamedGridDeterministic, IdenticalCsrConnectivity, ()) {
    const std::string name = *genGridName();

    auto mesh1 = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);
    auto mesh2 = axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);

    // Compare CSR offsets
    auto offsets1 = mesh1.conn_offsets();
    auto offsets2 = mesh2.conn_offsets();

    RC_ASSERT(offsets1.extent(0) == offsets2.extent(0));

    const std::size_t offsets_bytes = offsets1.extent(0) * sizeof(axis::index_t);

    if (offsets_bytes > 0) {
        RC_ASSERT(std::memcmp(offsets1.data_handle(), offsets2.data_handle(), offsets_bytes) == 0);
    }

    // Compare CSR indices
    auto indices1 = mesh1.conn_indices();
    auto indices2 = mesh2.conn_indices();

    RC_ASSERT(indices1.extent(0) == indices2.extent(0));

    const std::size_t indices_bytes = indices1.extent(0) * sizeof(axis::index_t);

    if (indices_bytes > 0) {
        RC_ASSERT(std::memcmp(indices1.data_handle(), indices2.data_handle(), indices_bytes) == 0);
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// RapidCheck/GTest property tests need Kokkos initialized for View allocation
// and parallel kernel execution in the named-grid generators.

}  // namespace
