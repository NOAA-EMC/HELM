// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_UNSTRUCTURED_MESH_HPP
#define AXIS_TOPOLOGY_UNSTRUCTURED_MESH_HPP

/// @file axis/topology/unstructured_mesh.hpp
/// @brief The common internal finite-element unstructured mesh representation.
///
/// UnstructuredMesh is the single internal format that ALL source grids convert
/// into and that the solver operates on exclusively. It owns Kokkos Views in a
/// single memory space; all public accessors return field_view (non-owning
/// std::mdspan<layout_left>) so consumers never copy.
///
/// Templated on a Kokkos MemorySpace (HELM Law #2: explicit placement, no UVM).
/// Light inline accessors live in this header; the heavy compute_areas kernel
/// is implemented in the .cpp with explicit template instantiations.

#include <Kokkos_Core.hpp>
#include <axis/topology/enums.hpp>
#include <axis/types.hpp>
#include <cstddef>

namespace axis::topology {

/// @brief Finite-element unstructured mesh representation.
///
/// This class represents an unstructured mesh containing nodes (vertices) and elements (cells) with
/// arbitrary, potentially mixed element types. The cell-to-node connectivity is stored in the
/// Compressed Sparse Row (CSR) format. It owns Kokkos Views allocated in a single memory space.
/// All public accessors return non-owning `field_view` (std::mdspan with layout_left) wrapping
/// the underlying data pointers, eliminating unnecessary copying.
///
/// This class is explicitly templated on a Kokkos memory space to enforce explicit memory placement
/// guidelines (no Unified Virtual Memory). Lightweight inline accessors are provided in this header,
/// while heavy compute kernels (such as `compute_areas()`) are implemented in the source file with
/// explicit template instantiations.
///
/// @tparam MemorySpace The Kokkos memory space used for internal array storage (e.g., Kokkos::HostSpace, Kokkos::CudaSpace, Kokkos::HIPSpace).
template <class MemorySpace = Kokkos::HostSpace>
class UnstructuredMesh {
   public:
    /// @brief Type alias for the memory space template parameter.
    using memory_space = MemorySpace;

    /// @brief Default constructor.
    ///
    /// Constructs an empty UnstructuredMesh instance with default-initialized internal Views.
    UnstructuredMesh() = default;

    /// @brief Constructs an UnstructuredMesh by adopting pre-built Kokkos arrays.
    ///
    /// This constructor adopts the provided Kokkos::View objects, moving them into
    /// the mesh instance without performing deep copies.
    ///
    /// @param node_coords Node coordinates represented as a rank-2 Kokkos::View of dimensions [n_nodes, ndim] in Column-Major Layout
    /// (Kokkos::LayoutLeft).
    /// @param conn_offsets Cell-to-node connectivity offsets represented as a rank-1 Kokkos::View of size [n_cells + 1] in Compressed Sparse Row
    /// (CSR) format.
    /// @param conn_indices Cell-to-node connectivity indices represented as a rank-1 Kokkos::View of size [nnz] containing flattened node indices for
    /// all cells.
    /// @param coord_sys The CoordinateSystem enum value specifying the coordinate system (e.g., Cartesian, SphericalDeg) used by the node
    /// coordinates.
    /// @param areas Optional precomputed cell areas represented as a rank-1 Kokkos::View of size [n_cells]. Defaults to an empty View.
    /// @param mask Optional cell mask represented as a rank-1 Kokkos::View of size [n_cells] where 1 indicates an active cell and 0 indicates a
    /// masked/inactive cell. Defaults to an empty View.
    UnstructuredMesh(Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords, Kokkos::View<index_t *, MemorySpace> conn_offsets,
                     Kokkos::View<index_t *, MemorySpace> conn_indices, CoordinateSystem coord_sys, Kokkos::View<double *, MemorySpace> areas = {},
                     Kokkos::View<int *, MemorySpace> mask = {}, GeometryMetadata geometry = {})
        : node_coords_(std::move(node_coords)),
          conn_offsets_(std::move(conn_offsets)),
          conn_indices_(std::move(conn_indices)),
          coord_sys_(coord_sys),
          cell_areas_(std::move(areas)),
          cell_mask_(std::move(mask)),
          geometry_(geometry) {}

    // ─────────────────────────────────────────────────────────────────────────
    // Scalar accessors
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Gets the number of nodes (vertices) in the mesh.
    /// @return The total number of nodes as a std::size_t.
    [[nodiscard]] std::size_t n_nodes() const noexcept {
        return node_coords_.extent(0);
    }

    /// @brief Gets the number of cells (elements) in the mesh.
    /// @return The total number of cells as a std::size_t, derived from the CSR offsets array size.
    [[nodiscard]] std::size_t n_cells() const noexcept {
        // CSR offsets array has length n_cells + 1
        return conn_offsets_.extent(0) > 0 ? conn_offsets_.extent(0) - 1 : 0;
    }

    /// @brief Gets the coordinate system for this mesh's node data.
    /// @return The CoordinateSystem enum value representing the mesh's coordinate system.
    [[nodiscard]] CoordinateSystem coord_system() const noexcept {
        return coord_sys_;
    }

    /// @brief Query how boundary coordinates and cell areas should be interpreted.
    [[nodiscard]] GeometryMetadata geometry_metadata() const noexcept {
        return geometry_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Non-owning field_view accessors (zero-copy, layout_left)
    //
    // These return std::mdspan<const T, dextents, layout_left> directly
    // wrapping the internal Kokkos::View data pointers. No allocation or copy.
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Gets the node coordinates as a non-owning rank-2 field_view.
    ///
    /// This returns a zero-copy field_view wrapping the internal Kokkos::View node coordinate data.
    ///
    /// @return A non-owning rank-2 field_view of dimensions [n_nodes, ndim].
    [[nodiscard]] field_view<const double, 2> node_coords() const noexcept {
        return field_view<const double, 2>{node_coords_.data(), node_coords_.extent(0), node_coords_.extent(1)};
    }

    /// @brief Gets the Compressed Sparse Row (CSR) connectivity offsets.
    ///
    /// The node indices for a specific cell index `c` are located in the range
    /// `conn_indices[conn_offsets[c] .. conn_offsets[c+1])`.
    ///
    /// @return A non-owning rank-1 field_view of size [n_cells + 1] containing the offsets.
    [[nodiscard]] field_view<const index_t, 1> conn_offsets() const noexcept {
        return field_view<const index_t, 1>{conn_offsets_.data(), conn_offsets_.extent(0)};
    }

    /// @brief Gets the Compressed Sparse Row (CSR) connectivity indices.
    ///
    /// This represents the flattened node indices for all cells (the non-zero entries).
    ///
    /// @return A non-owning rank-1 field_view of size [nnz] containing the connectivity indices.
    [[nodiscard]] field_view<const index_t, 1> conn_indices() const noexcept {
        return field_view<const index_t, 1>{conn_indices_.data(), conn_indices_.extent(0)};
    }

    /// @brief Gets the per-cell areas of the mesh.
    ///
    /// This returns a non-owning view of the computed cell areas. If `compute_areas()`
    /// has not been called, or if precomputed areas were not provided at construction,
    /// this returns an empty view of size 0.
    ///
    /// @return A non-owning rank-1 field_view containing cell areas.
    [[nodiscard]] field_view<const double, 1> cell_areas() const noexcept {
        return field_view<const double, 1>{cell_areas_.data(), cell_areas_.extent(0)};
    }

    /// @brief Gets the optional per-cell mask.
    ///
    /// Mask values are 0 for masked/inactive cells and 1 for active cells. If no mask
    /// was provided at construction, this returns an empty view of size 0.
    ///
    /// @return A non-owning rank-1 field_view representing cell activity.
    [[nodiscard]] field_view<const int, 1> cell_mask() const noexcept {
        return field_view<const int, 1>{cell_mask_.data(), cell_mask_.extent(0)};
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Internal Kokkos::View accessors (for topology builders that need
    // direct View access, e.g., MeshFactory, GmshWriter, WeightGenerator)
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Gets a reference to the internal rank-2 Kokkos::View of node coordinates.
    /// @return A const reference to the Kokkos::View containing the node coordinates.
    [[nodiscard]] const auto &node_coords_view() const noexcept {
        return node_coords_;
    }

    /// @brief Gets a reference to the internal rank-1 Kokkos::View of connectivity offsets.
    /// @return A const reference to the Kokkos::View containing the offsets.
    [[nodiscard]] const auto &conn_offsets_view() const noexcept {
        return conn_offsets_;
    }

    /// @brief Gets a reference to the internal rank-1 Kokkos::View of connectivity indices.
    /// @return A const reference to the Kokkos::View containing the indices.
    [[nodiscard]] const auto &conn_indices_view() const noexcept {
        return conn_indices_;
    }

    /// @brief Gets a reference to the internal rank-1 Kokkos::View of cell areas.
    /// @return A const reference to the Kokkos::View containing the cell areas.
    [[nodiscard]] const auto &cell_areas_view() const noexcept {
        return cell_areas_;
    }

    /// @brief Gets a reference to the internal rank-1 Kokkos::View of cell mask values.
    /// @return A const reference to the Kokkos::View containing the cell mask.
    [[nodiscard]] const auto &cell_mask_view() const noexcept {
        return cell_mask_;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Area computation (heavy kernel — implemented in .cpp)
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Computes cell areas using a high-performance Kokkos parallel kernel.
    ///
    /// The actual computation is performed on the device or host associated with the
    /// template's `MemorySpace`.
    ///
    /// For SphericalDeg and SphericalRad coordinate systems, the area is calculated
    /// using the spherical excess formula (Girard's theorem for triangles, generalized
    /// to arbitrary spherical polygons via spherical excess summation). The computed
    /// areas are represented in steradians on the unit sphere.
    ///
    /// For Cartesian3D coordinate systems, the standard planar polygon area is
    /// calculated via 3D cross-product summation (generalized shoelace formula).
    ///
    /// After computation, the results are cached internally and can be retrieved
    /// via the `cell_areas()` accessor.
    void compute_areas();

   private:
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords_;
    Kokkos::View<index_t *, MemorySpace> conn_offsets_;
    Kokkos::View<index_t *, MemorySpace> conn_indices_;
    CoordinateSystem coord_sys_{CoordinateSystem::SphericalDeg};
    Kokkos::View<double *, MemorySpace> cell_areas_;
    Kokkos::View<int *, MemorySpace> cell_mask_;
    GeometryMetadata geometry_{};
};

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_UNSTRUCTURED_MESH_HPP
