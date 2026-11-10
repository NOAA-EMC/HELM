// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_MULTI_FACE_GRID_HPP
#define AXIS_TOPOLOGY_MULTI_FACE_GRID_HPP

#include <Kokkos_Core.hpp>
#include <axis/topology/enums.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <vector>

namespace axis::topology {

enum class FaceEdge : std::uint8_t { South, East, North, West };

struct FaceConnection {
    std::size_t face_a{0};
    FaceEdge edge_a{FaceEdge::East};
    std::size_t face_b{0};
    FaceEdge edge_b{FaceEdge::West};
    bool reversed{false};
};

struct FaceVertexRef {
    std::size_t face{0};
    std::size_t i{0};
    std::size_t j{0};
};

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
    MultiFaceGrid(std::vector<StructuredFace<MemorySpace>> faces, std::vector<FaceConnection> connections, CoordinateSystem coordinate_system,
                  std::vector<std::vector<FaceVertexRef>> vertex_equivalences = {});

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
