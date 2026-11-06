// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_MULTI_FACE_GRID_HPP
#define AXIS_TOPOLOGY_MULTI_FACE_GRID_HPP

/// @file axis/topology/multi_face_grid.hpp
/// @brief MultiFaceGrid — generic structured multi-face topology (cubed sphere
///        and related layouts) convertible to the common unstructured mesh.

#include <Kokkos_Core.hpp>
#include <axis/topology/enums.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <vector>

namespace axis::topology {

/// @brief Edges of a logically rectangular face, ordered counterclockwise.
enum class FaceEdge : std::uint8_t { South, East, North, West };

/// @brief A shared edge between two faces with an optional index reversal.
///
/// edge_a runs from its start corner to its end corner; reversed records
/// whether face_b's corresponding edge traverses in the opposite direction,
/// which determines how vertex and cell indices align across the seam.
struct FaceConnection {
    std::size_t face_a{0};
    FaceEdge edge_a{FaceEdge::East};
    std::size_t face_b{0};
    FaceEdge edge_b{FaceEdge::West};
    bool reversed{false};
};

/// @brief Identifies one corner node by face and (i, j) corner index.
///
/// Used to declare equivalence classes of nodes that coincide geometrically
/// but live on different faces (or repeated corners of one face).
struct FaceVertexRef {
    std::size_t face{0};
    std::size_t i{0};
    std::size_t j{0};
};

/// @brief One logically rectangular face of a multi-face grid.
template <class MemorySpace = Kokkos::HostSpace>
struct StructuredFace {
    std::size_t ni{0};
    std::size_t nj{0};
    /// Explicit face-corner coordinates, laid out as [(ni+1)*(nj+1), ndim].
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords;
    /// Set when the face's cell-coordinate ordering is clockwise.
    bool reverse_cell_orientation{false};
};

/// Generic structured multi-face topology. Face contacts and optional
/// equivalence classes are explicit; geographic coordinate equality is never
/// used to invent adjacency. Face/node iteration defines deterministic IDs.
template <class MemorySpace = Kokkos::HostSpace>
class MultiFaceGrid {
   public:
    /// @brief Assemble faces, edge contacts, and node equivalences.
    /// @param faces               Faces in deterministic iteration order.
    /// @param connections         Edge-to-edge contacts between faces.
    /// @param coordinate_system   Spherical or Cartesian interpretation.
    /// @param vertex_equivalences Optional groups of corner references that
    ///                            denote the same underlying vertex.
    /// @throws std::invalid_argument for out-of-range face indices, malformed
    ///         corner arrays, or inconsistent connection endpoints.
    MultiFaceGrid(std::vector<StructuredFace<MemorySpace>> faces, std::vector<FaceConnection> connections, CoordinateSystem coordinate_system,
                  std::vector<std::vector<FaceVertexRef>> vertex_equivalences = {});

    /// @brief Merge equivalent nodes and emit every cell as a quadrilateral.
    ///
    /// Node IDs follow face/node iteration order, so the result is stable for
    /// a given input. Cell orientation follows each face's
    /// reverse_cell_orientation flag.
    [[nodiscard]] UnstructuredMesh<MemorySpace> to_unstructured() const;

   private:
    std::vector<StructuredFace<MemorySpace>> faces_;
    std::vector<FaceConnection> connections_;
    CoordinateSystem coordinate_system_;
    std::vector<std::vector<FaceVertexRef>> vertex_equivalences_;
};

extern template class MultiFaceGrid<Kokkos::HostSpace>;
#ifdef KOKKOS_ENABLE_CUDA
extern template class MultiFaceGrid<Kokkos::CudaSpace>;
#endif
#ifdef KOKKOS_ENABLE_HIP
extern template class MultiFaceGrid<Kokkos::HIPSpace>;
#endif

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_MULTI_FACE_GRID_HPP
