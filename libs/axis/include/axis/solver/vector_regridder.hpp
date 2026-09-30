// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_VECTOR_REGRIDDER_HPP
#define AXIS_SOLVER_VECTOR_REGRIDDER_HPP

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <utility>

namespace axis::solver {

/// @struct GridRotation
/// @brief Struct holding local grid rotation angles (in radians) for curvilinear grids.
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
struct GridRotation {
    Kokkos::View<const double *, MemorySpace> alpha;  ///< Local grid rotation angle per grid cell
};

/// @class VectorWeightGenerator
/// @brief Generates pre-assembled, coupled interpolation matrices for 2D vector fields.
///
/// This generator mathematically couples local grid-relative coordinate frame rotations
/// with standard scalar spatial interpolation weights into pre-assembled coupled matrices,
/// allowing vector remapping to be performed in a single high-performance SpMV step.
///
/// @tparam MemorySpace The Kokkos memory space (e.g., Kokkos::HostSpace, Kokkos::CudaSpace).
template <typename MemorySpace>
class VectorWeightGenerator {
   public:
    /// @brief Generate coupled weight matrices for u and v vector components.
    /// @param src_mesh      Source unstructured mesh.
    /// @param dst_mesh      Destination unstructured mesh.
    /// @param src_rotation  Rotation angles at source grid cells.
    /// @param dst_rotation  Rotation angles at destination grid cells.
    /// @param config        Scalar regridding configuration.
    /// @return A pair of InterpolationMatrix objects: first is W_u, second is W_v.
    static std::pair<InterpolationMatrix<MemorySpace>, InterpolationMatrix<MemorySpace>> generate(
        const topology::UnstructuredMesh<MemorySpace> &src_mesh, const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
        const GridRotation<MemorySpace> &src_rotation, const GridRotation<MemorySpace> &dst_rotation, const RegridConfig &config);
};

/// @brief Compute per-cell vector rotation angles (radians) for a mesh.
///
/// Returns, for each cell, the counter-clockwise angle from geographic east
/// to the cell's local +i direction — the direction of the cell's first
/// connectivity edge (vertex 0 → vertex 1). This is the convention the
/// coupled rotation in VectorWeightGenerator::generate expects: with
/// diff = alpha_dst - alpha_src, u_dst = cos(diff)·u + sin(diff)·v and
/// v_dst = -sin(diff)·u + cos(diff)·v transports a vector correctly between
/// two locally-oriented grids.
///
/// Per-cell rules (fit-time metadata; host-side computation):
///   - Spherical meshes with quadrilateral cells (cubed-sphere tiles,
///     curvilinear structured grids, regular lat-lon): α = π/2 − bearing of
///     the great-circle edge 0→1. A lat-lon grid yields α = 0 everywhere;
///     a cubed-sphere tile yields the tile's spatially varying rotation.
///   - Non-quadrilateral cells (MPAS/ICON polygons): vertex order carries no
///     i-axis meaning and cell-centered vectors are conventionally stored on
///     the geographic east/north basis — α = 0 (identity rotation).
///   - Projected (planar) meshes: no geographic east — α = 0; u/v is assumed
///     expressed in the projection's x/y basis on both grids.
///
/// @tparam MemorySpace Kokkos memory space of the mesh.
/// @param mesh Source or destination mesh.
/// @return View<double*, MemorySpace> of length mesh.n_cells(), in radians.
template <typename MemorySpace>
Kokkos::View<double *, MemorySpace> compute_rotation_angles(const topology::UnstructuredMesh<MemorySpace> &mesh);

}  // namespace axis::solver

#endif  // AXIS_SOLVER_VECTOR_REGRIDDER_HPP
