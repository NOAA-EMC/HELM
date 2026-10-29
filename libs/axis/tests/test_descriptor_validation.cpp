// AXIS unit test: malformed descriptors throw correctly
// Verifies that from_descriptor rejects invalid descriptors with
// std::invalid_argument and produces no partial mesh.

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/types.hpp>

namespace {

}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Test: GRIB descriptor with ni=0 → throw std::invalid_argument
TEST(DescriptorValidation, GribWithZeroNiThrows) {
    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::GRIB;
    desc.grib.grid_type = "regular_ll";
    desc.grib.ni = 0;  // invalid
    desc.grib.nj = 10;

    EXPECT_THROW(topology::MeshFactory::from_descriptor<MemSpace>(desc), std::invalid_argument);
}

// Test: Projected with empty proj_string → throw std::invalid_argument
TEST(DescriptorValidation, ProjectedEmptyProjStringThrows) {
    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::Projected;
    desc.projected.proj_string = "";  // missing required field

    EXPECT_THROW(topology::MeshFactory::from_descriptor<MemSpace>(desc), std::invalid_argument);
}

// Test: UGRID with empty node_coords → throw std::invalid_argument
TEST(DescriptorValidation, UgridEmptyNodeCoordsThrows) {
    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::UGRID;
    desc.ugrid.topology_dimension = 2;
    // node_coords is default-constructed (empty)
    // conn_offsets and conn_indices are empty too

    EXPECT_THROW(topology::MeshFactory::from_descriptor<MemSpace>(desc), std::invalid_argument);
}

// Test: Unknown ConventionKind value (cast an invalid integer)
TEST(DescriptorValidation, UnknownConventionKindThrows) {
    ingest::GridDescriptor desc;
    desc.kind = static_cast<ingest::ConventionKind>(255);  // invalid

    EXPECT_THROW(topology::MeshFactory::from_descriptor<MemSpace>(desc), std::invalid_argument);
}

// Test: CF descriptor with mismatched buffer extents (center_x size != ni*nj)
TEST(DescriptorValidation, CfInconsistentExtentsThrows) {
    // Provide center_x of size 4 but ni=3, nj=2 (expects 6)
    std::vector<double> lons = {0.5, 1.5, 2.5, 3.5};
    std::vector<double> lats = {0.5, 1.5, 2.5, 3.5};

    ingest::GridDescriptor desc;
    desc.kind = ingest::ConventionKind::CF;
    desc.cf.grid_mapping_name = "latitude_longitude";
    desc.buffers.center_x = field_view<const double, 1>(lons.data(), 4);
    desc.buffers.center_y = field_view<const double, 1>(lats.data(), 4);
    desc.buffers.ni = 3;
    desc.buffers.nj = 2;  // expects center_x.extent(0) == 6, but got 4

    EXPECT_THROW(topology::MeshFactory::from_descriptor<MemSpace>(desc), std::invalid_argument);
}

}  // namespace axis::test
