// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_MESH_BUILDER_HPP
#define AXIS_TOPOLOGY_MESH_BUILDER_HPP

/// @file axis/topology/mesh_builder.hpp
/// @brief Validated construction of the common CSR unstructured mesh.

#include <Kokkos_Core.hpp>
#include <axis/topology/enums.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace axis::topology {

/// Construct the common mesh by validating and adopting prebuilt views.
///
/// The function accepts mixed-arity polygons. CSR offsets are zero based,
/// monotone, and terminate at the number of connectivity indices. Every cell
/// must contain at least three valid node indices. Spherical coordinates use
/// (longitude, latitude); Cartesian3D uses (x, y, z).
///
/// Validation reads views in their owning memory space using Kokkos kernels;
/// no host mirror or implicit copy is used. The supplied views are then moved
/// into the result without copying.
///
/// @throws std::invalid_argument for malformed extents or connectivity.
template <class MemorySpace>
UnstructuredMesh<MemorySpace> make_unstructured(Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords,
                                                Kokkos::View<index_t *, MemorySpace> conn_offsets, Kokkos::View<index_t *, MemorySpace> conn_indices,
                                                CoordinateSystem coord_sys, Kokkos::View<double *, MemorySpace> areas = {},
                                                Kokkos::View<int *, MemorySpace> mask = {}, GeometryMetadata geometry = {});

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_MESH_BUILDER_HPP
