// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
namespace axis::distributed::detail {
using HostMesh = topology::UnstructuredMesh<Kokkos::HostSpace>;
struct SearchGeometry {
    Kokkos::View<float *[3], Kokkos::HostSpace> points;
    Kokkos::View<float *[6], Kokkos::HostSpace> boxes;
};
// Implemented beside the native weight generator, reusing its centroid and
// bounding-box routines so discovery uses identical numerical conventions.
SearchGeometry search_geometry(const HostMesh &, const solver::RegridConfig &, bool wet_search = false);
}  // namespace axis::distributed::detail
