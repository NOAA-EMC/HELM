// ─── Property-Based Tests: layout_left Field View Round-Trip ─────────────────
// Feature: helm-axis-microlibrary, Property 1: layout_left Field View Round-Trip
//
// Uses RapidCheck to verify that for any column-major (std::layout_left) field
// array owned by a caller, wrapping it as a field_view and passing it through
// detail::to_view then detail::to_mdspan yields a view addressing the identical
// memory with identical extents and identical element values, performing no copy
// of the underlying buffer.
//
// The round-trip is tested for both Rank-1 and Rank-2 field_view instances with
// randomly generated sizes and element values. Each property verifies:
//   1. Same data pointer (memory address identity)
//   2. Same extents in each dimension
//   3. Same element values (bitwise, since no copy occurs)
//
// All tests execute on Kokkos::HostSpace only (host-only property test).
//
// **Validates: Requirements 1.4, 1.5, 1.6**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/mdspan_interop.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <cstring>
#include <vector>

namespace {

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a valid array size in [1, 512]. Avoids zero-length arrays (which
/// would have a null data pointer) while exercising a reasonable range.
rc::Gen<std::size_t> genExtent() {
    return rc::gen::inRange<std::size_t>(1, 513);
}

/// Generate a vector of doubles with a given size. Values span the full finite
/// double range to stress bitwise equality checks.
rc::Gen<std::vector<double>> genDoubleVector(std::size_t n) {
    return rc::gen::container<std::vector<double>>(n, rc::gen::arbitrary<double>());
}

// ─── Property 1a: Rank-1 field_view round-trip ───────────────────────────────
// Generate a random 1-D array, wrap as field_view<double, 1>, pass through
// to_view → to_mdspan, verify pointer, extent, and values match exactly.
//
// **Validates: Requirements 1.4, 1.5, 1.6**

RC_GTEST_PROP(FieldViewRoundtripProperty1, Rank1RoundTrip, ()) {
    // Generate random extent and data
    const std::size_t n = *genExtent();
    auto data = *genDoubleVector(n);

    // Wrap raw array as a Rank-1 field_view (layout_left mdspan)
    axis::field_view<double, 1> fv(data.data(), n);

    // Round-trip: field_view → Kokkos::View → field_view
    auto view = axis::detail::to_view<double, Kokkos::HostSpace, 1>(fv);
    auto result = axis::detail::to_mdspan(view);

    // 1. Same data pointer (memory address identity)
    RC_ASSERT(result.data_handle() == data.data());

    // 2. Same extent
    RC_ASSERT(result.extent(0) == n);

    // 3. Same element values (bitwise — no copy occurred)
    //    Since we verified the pointer is the same, checking values is redundant
    //    but provides a strong correctness signal for the full chain.
    for (std::size_t i = 0; i < n; ++i) {
        // Use memcmp to verify bitwise identity (handles NaN correctly)
        double original = data[i];
        double roundtripped = result.data_handle()[i];
        RC_ASSERT(std::memcmp(&original, &roundtripped, sizeof(double)) == 0);
    }
}

// ─── Property 1b: Rank-2 field_view round-trip ───────────────────────────────
// Generate a random 2-D array (stored column-major as a flat vector), wrap as
// field_view<double, 2>, pass through to_view → to_mdspan, verify pointer,
// extents, and values match exactly.
//
// **Validates: Requirements 1.4, 1.5, 1.6**

RC_GTEST_PROP(FieldViewRoundtripProperty1, Rank2RoundTrip, ()) {
    // Generate random extents for both dimensions
    const std::size_t ni = *genExtent();
    const std::size_t nj = *rc::gen::inRange<std::size_t>(1, 65);
    const std::size_t total = ni * nj;

    auto data = *genDoubleVector(total);

    // Wrap raw array as a Rank-2 field_view (layout_left: column-major)
    axis::field_view<double, 2> fv(data.data(), ni, nj);

    // Round-trip: field_view → Kokkos::View → field_view
    auto view = axis::detail::to_view<double, Kokkos::HostSpace, 2>(fv);
    auto result = axis::detail::to_mdspan(view);

    // 1. Same data pointer (memory address identity)
    RC_ASSERT(result.data_handle() == data.data());

    // 2. Same extents in both dimensions
    RC_ASSERT(result.extent(0) == ni);
    RC_ASSERT(result.extent(1) == nj);

    // 3. Same element values accessed via 2-D indexing (bitwise identity)
    //    Use the flat underlying buffer for comparison since both views share it.
    //    Column-major (layout_left): element (i,j) is at offset i + ni*j.
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            const std::size_t offset = i + ni * j;
            double original = data[offset];
            double roundtripped = result.data_handle()[offset];
            RC_ASSERT(std::memcmp(&original, &roundtripped, sizeof(double)) == 0);
        }
    }
}

// ─── Property 1c: to_view preserves pointer (intermediate check) ─────────────
// Verify that to_view itself wraps the same pointer — the Kokkos::View data()
// matches the original array address. This confirms zero-copy in the first half
// of the round-trip independently.
//
// **Validates: Requirements 1.4**

RC_GTEST_PROP(FieldViewRoundtripProperty1, ToViewPreservesPointer, ()) {
    const std::size_t n = *genExtent();
    auto data = *genDoubleVector(n);

    axis::field_view<double, 1> fv(data.data(), n);
    auto view = axis::detail::to_view<double, Kokkos::HostSpace, 1>(fv);

    // The Kokkos::View must wrap the exact same pointer
    RC_ASSERT(view.data() == data.data());

    // Extent must match
    RC_ASSERT(static_cast<std::size_t>(view.extent(0)) == n);
}

// ─── Property 1d: to_mdspan preserves pointer (intermediate check) ───────────
// Verify that to_mdspan wraps the same pointer the View holds. This confirms
// zero-copy in the second half of the round-trip independently.
//
// **Validates: Requirements 1.5**

RC_GTEST_PROP(FieldViewRoundtripProperty1, ToMdspanPreservesPointer, ()) {
    const std::size_t n = *genExtent();
    auto data = *genDoubleVector(n);

    // Create an unmanaged Kokkos::View manually (simulating what to_view returns)
    using view_type = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
    view_type view(data.data(), n);

    auto result = axis::detail::to_mdspan(view);

    // The mdspan must wrap the exact same pointer
    RC_ASSERT(result.data_handle() == data.data());
    RC_ASSERT(result.extent(0) == n);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// RapidCheck/GTest property tests need Kokkos initialized for View allocation.



}  // namespace
