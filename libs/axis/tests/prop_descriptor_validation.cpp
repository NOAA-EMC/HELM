// ─── Property-Based Tests: Descriptor Field Validation ───────────────────────
// Feature: helm-axis-microlibrary, Property 24: Descriptor Field Validation
//
// Uses RapidCheck to generate deliberately malformed GridDescriptors (unknown
// kind, missing field, inconsistent extents, null buffers) and verifies that
// MeshFactory::from_descriptor throws std::invalid_argument with a what()
// message naming the offending field.
//
// Test cases cover all ConventionKind variants plus unknown kinds:
//   1.  CF with ni=0 or nj=0 → throws mentioning 'ni' or 'nj'
//   2.  CF with center_x extent != ni*nj → throws mentioning 'center_x'
//   3.  CF with null center_x data_handle → throws mentioning 'center_x'
//   4.  UGRID with empty node_coords → throws mentioning 'node_coords'
//   5.  UGRID with empty conn_offsets → throws mentioning 'conn_offsets'
//   6.  UGRID with empty conn_indices → throws mentioning 'conn_indices'
//   7.  GRIB with empty grid_type → throws mentioning 'grid_type'
//   8.  GRIB with ni<=0 → throws mentioning 'ni'
//   9.  Projected with empty proj_string → throws mentioning 'proj_string'
//  10.  NamedGrid with empty name → throws mentioning 'name'
//  11.  GridRules with empty kind → throws mentioning 'kind'
//
// **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using axis::field_view;
using axis::index_t;
using axis::ingest::BufferViews;
using axis::ingest::CfParams;
using axis::ingest::ConventionKind;
using axis::ingest::GribParams;
using axis::ingest::GridDescriptor;
using axis::ingest::GridRulesParams;
using axis::ingest::NamedGridParams;
using axis::ingest::ProjectedParams;
using axis::ingest::UgridParams;

// ─── Helper: verify that from_descriptor throws std::invalid_argument
// containing the expected substring in what(). ────────────────────────────────

void expect_throws_naming(const GridDescriptor &desc, const std::string &field) {
    try {
        (void)axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);
        RC_FAIL("Expected std::invalid_argument but no exception was thrown");
    } catch (const std::invalid_argument &e) {
        std::string msg = e.what();
        RC_ASSERT(msg.find(field) != std::string::npos);
    } catch (const std::exception &e) {
        RC_FAIL("Expected std::invalid_argument but got different exception: " + std::string(e.what()));
    }
}

// ─── RapidCheck Generators ───────────────────────────────────────────────────

/// Generate a small positive extent for structured grid dimensions.
rc::Gen<std::size_t> genPositiveExtent() {
    return rc::gen::inRange<std::size_t>(1, 33);
}

// ─── Property 24a: CF with ni=0 or nj=0 throws mentioning 'ni' or 'nj' ──────
// A CF descriptor with zero ni or nj is invalid. Validation must catch this
// and throw std::invalid_argument naming the missing dimension.
//
// **Validates: Requirements 5.2**

RC_GTEST_PROP(PropDescriptorValidation, CfZeroDimension, ()) {
    // Generate valid ni and nj, then zero one of them
    const std::size_t ni = *genPositiveExtent();
    const std::size_t nj = *genPositiveExtent();
    const bool zero_ni = *rc::gen::arbitrary<bool>();

    // Build a CF descriptor with one zeroed dimension
    GridDescriptor desc{};
    desc.kind = ConventionKind::CF;

    // Provide valid buffer data for center_x/y (won't matter—validation fails
    // on the zero dimension first)
    const std::size_t n = ni * nj;
    std::vector<double> buf(n, 1.0);

    desc.buffers.ni = zero_ni ? 0 : ni;
    desc.buffers.nj = zero_ni ? nj : 0;
    desc.buffers.center_x = field_view<const double, 1>(buf.data(), n);
    desc.buffers.center_y = field_view<const double, 1>(buf.data(), n);

    const std::string expected_field = zero_ni ? "ni" : "nj";
    expect_throws_naming(desc, expected_field);
}

// ─── Property 24b: CF with center_x extent != ni*nj throws mentioning
// 'center_x' or 'extents' ─────────────────────────────────────────────────────
//
// **Validates: Requirements 5.3**

RC_GTEST_PROP(PropDescriptorValidation, CfInconsistentExtents, ()) {
    const std::size_t ni = *genPositiveExtent();
    const std::size_t nj = *genPositiveExtent();
    const std::size_t expected = ni * nj;

    // Generate a wrong extent (different from ni*nj)
    std::size_t wrong_extent = *rc::gen::inRange<std::size_t>(1, 1025);
    RC_PRE(wrong_extent != expected);

    std::vector<double> correct_buf(expected, 1.0);
    std::vector<double> wrong_buf(wrong_extent, 1.0);

    GridDescriptor desc{};
    desc.kind = ConventionKind::CF;
    desc.buffers.ni = ni;
    desc.buffers.nj = nj;
    desc.buffers.center_x = field_view<const double, 1>(wrong_buf.data(), wrong_extent);
    desc.buffers.center_y = field_view<const double, 1>(correct_buf.data(), expected);

    expect_throws_naming(desc, "center_x");
}

// ─── Property 24c: CF with null center_x data_handle throws mentioning
// 'center_x' ──────────────────────────────────────────────────────────────────
//
// **Validates: Requirements 5.4**

RC_GTEST_PROP(PropDescriptorValidation, CfNullCenterX, ()) {
    const std::size_t ni = *genPositiveExtent();
    const std::size_t nj = *genPositiveExtent();
    const std::size_t n = ni * nj;

    std::vector<double> buf(n, 1.0);

    GridDescriptor desc{};
    desc.kind = ConventionKind::CF;
    desc.buffers.ni = ni;
    desc.buffers.nj = nj;
    // center_x is default (null data_handle, extent 0)
    desc.buffers.center_y = field_view<const double, 1>(buf.data(), n);

    expect_throws_naming(desc, "center_x");
}

// ─── Property 24d: UGRID with empty node_coords throws mentioning
// 'node_coords' ───────────────────────────────────────────────────────────────
//
// **Validates: Requirements 5.4**

RC_GTEST_PROP(PropDescriptorValidation, UgridEmptyNodeCoords, ()) {
    const std::size_t n_cells = *rc::gen::inRange<std::size_t>(1, 33);

    // Build valid conn_offsets and conn_indices
    std::vector<index_t> offsets(n_cells + 1);
    for (std::size_t i = 0; i <= n_cells; ++i) {
        offsets[i] = static_cast<index_t>(i * 3);
    }
    std::vector<index_t> indices(n_cells * 3, 0);

    GridDescriptor desc{};
    desc.kind = ConventionKind::UGRID;
    // node_coords left default (empty)
    desc.buffers.conn_offsets = field_view<const index_t, 1>(offsets.data(), offsets.size());
    desc.buffers.conn_indices = field_view<const index_t, 1>(indices.data(), indices.size());

    expect_throws_naming(desc, "node_coords");
}

// ─── Property 24e: UGRID with empty conn_offsets or conn_indices throws
// appropriately ────────────────────────────────────────────────────────────────
//
// **Validates: Requirements 5.4**

RC_GTEST_PROP(PropDescriptorValidation, UgridEmptyConnectivity, ()) {
    const std::size_t n_nodes = *rc::gen::inRange<std::size_t>(3, 65);
    const std::size_t n_cells = *rc::gen::inRange<std::size_t>(1, 33);
    const bool empty_offsets = *rc::gen::arbitrary<bool>();

    // Valid node_coords [n_nodes, 2]
    std::vector<double> coords(n_nodes * 2, 0.0);
    // Valid connectivity
    std::vector<index_t> offsets(n_cells + 1);
    for (std::size_t i = 0; i <= n_cells; ++i) {
        offsets[i] = static_cast<index_t>(i * 3);
    }
    std::vector<index_t> indices(n_cells * 3, 0);

    GridDescriptor desc{};
    desc.kind = ConventionKind::UGRID;
    desc.buffers.node_coords = field_view<const double, 2>(coords.data(), n_nodes, 2);

    if (empty_offsets) {
        // Leave conn_offsets default (empty), provide conn_indices
        desc.buffers.conn_indices = field_view<const index_t, 1>(indices.data(), indices.size());
    } else {
        // Provide conn_offsets, leave conn_indices default (empty)
        desc.buffers.conn_offsets = field_view<const index_t, 1>(offsets.data(), offsets.size());
    }

    const std::string expected_field = empty_offsets ? "conn_offsets" : "conn_indices";
    expect_throws_naming(desc, expected_field);
}

// ─── Property 24f: GRIB with empty grid_type throws mentioning 'grid_type' ──
//
// **Validates: Requirements 5.2**

RC_GTEST_PROP(PropDescriptorValidation, GribEmptyGridType, ()) {
    const std::int64_t ni = *rc::gen::inRange<std::int64_t>(1, 33);
    const std::int64_t nj = *rc::gen::inRange<std::int64_t>(1, 33);
    const std::size_t n = static_cast<std::size_t>(ni * nj);

    std::vector<double> buf(n, 1.0);

    GridDescriptor desc{};
    desc.kind = ConventionKind::GRIB;
    desc.grib.grid_type = "";  // deliberately empty
    desc.grib.ni = ni;
    desc.grib.nj = nj;
    desc.buffers.center_x = field_view<const double, 1>(buf.data(), n);
    desc.buffers.center_y = field_view<const double, 1>(buf.data(), n);

    expect_throws_naming(desc, "grid_type");
}

// ─── Property 24g: GRIB with ni<=0 throws mentioning 'ni' ───────────────────
//
// **Validates: Requirements 5.2**

RC_GTEST_PROP(PropDescriptorValidation, GribInvalidNi, ()) {
    // Generate ni <= 0
    const std::int64_t ni = *rc::gen::inRange<std::int64_t>(-10, 1);
    const std::int64_t nj = *rc::gen::inRange<std::int64_t>(1, 33);

    GridDescriptor desc{};
    desc.kind = ConventionKind::GRIB;
    desc.grib.grid_type = "regular_ll";
    desc.grib.ni = ni;
    desc.grib.nj = nj;
    // Buffers don't matter — ni validation fails first

    expect_throws_naming(desc, "ni");
}

// ─── Property 24h: Projected with empty proj_string throws mentioning
// 'proj_string' ───────────────────────────────────────────────────────────────
//
// **Validates: Requirements 5.2**

RC_GTEST_PROP(PropDescriptorValidation, ProjectedEmptyProjString, ()) {
    const std::size_t ni = *genPositiveExtent();
    const std::size_t nj = *genPositiveExtent();
    const std::size_t n = ni * nj;

    std::vector<double> buf(n, 1.0);

    GridDescriptor desc{};
    desc.kind = ConventionKind::Projected;
    desc.projected.proj_string = "";  // deliberately empty
    desc.buffers.ni = ni;
    desc.buffers.nj = nj;
    desc.buffers.center_x = field_view<const double, 1>(buf.data(), n);
    desc.buffers.center_y = field_view<const double, 1>(buf.data(), n);

    expect_throws_naming(desc, "proj_string");
}

// ─── Property 24i: NamedGrid with empty name throws mentioning 'name' ────────
//
// **Validates: Requirements 5.2**

RC_GTEST_PROP(PropDescriptorValidation, NamedGridEmptyName, ()) {
    GridDescriptor desc{};
    desc.kind = ConventionKind::NamedGrid;
    desc.named_grid.name = "";  // deliberately empty

    expect_throws_naming(desc, "name");
}

// ─── Property 24j: GridRules with empty kind throws mentioning 'kind' ────────
//
// **Validates: Requirements 5.2**

RC_GTEST_PROP(PropDescriptorValidation, GridRulesEmptyKind, ()) {
    GridDescriptor desc{};
    desc.kind = ConventionKind::GridRules;
    desc.grid_rules.kind = "";  // deliberately empty

    expect_throws_naming(desc, "kind");
}

// ─── Property 24k: Unknown ConventionKind throws mentioning the kind value ───
// A descriptor with a ConventionKind value outside the defined enum range is
// invalid. Validation must catch this and throw std::invalid_argument.
//
// **Validates: Requirements 5.1**

RC_GTEST_PROP(PropDescriptorValidation, UnknownConventionKind, ()) {
    // Generate an integer outside the valid ConventionKind range [0, 5].
    const int raw_kind = *rc::gen::inRange(7, 255);

    GridDescriptor desc{};
    desc.kind = static_cast<ConventionKind>(static_cast<std::uint8_t>(raw_kind));

    // Should throw naming the kind or "ConventionKind"
    try {
        (void)axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);
        RC_FAIL("Expected std::invalid_argument but no exception was thrown");
    } catch (const std::invalid_argument &e) {
        // Verify we get a meaningful error about the unknown kind
        std::string msg = e.what();
        RC_ASSERT(!msg.empty());
        // The message should mention "ConventionKind" or the numeric value
        RC_ASSERT(msg.find("ConventionKind") != std::string::npos || msg.find(std::to_string(raw_kind)) != std::string::npos);
    } catch (const std::exception &e) {
        RC_FAIL("Expected std::invalid_argument but got different exception: " + std::string(e.what()));
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────
// Property tests require Kokkos initialized for MeshFactory operations.



}  // namespace
