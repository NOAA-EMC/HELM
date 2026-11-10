// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <axis/topology/mesh_builder.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace axis::topology {

namespace {

template <class MemorySpace>
void validate_mesh_views(const Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> &node_coords,
                         const Kokkos::View<index_t *, MemorySpace> &conn_offsets, const Kokkos::View<index_t *, MemorySpace> &conn_indices,
                         CoordinateSystem coord_sys, const Kokkos::View<double *, MemorySpace> &areas, const Kokkos::View<int *, MemorySpace> &mask) {
    const std::size_t expected_ndim = (coord_sys == CoordinateSystem::Cartesian3D) ? 3 : 2;
    if (node_coords.extent(1) != expected_ndim) {
        throw std::invalid_argument("make_unstructured: node coordinate dimension does not match coordinate system");
    }
    if (conn_offsets.extent(0) == 0) {
        throw std::invalid_argument("make_unstructured: CSR offsets must contain at least the initial offset");
    }

    const std::size_t n_cells = conn_offsets.extent(0) - 1;
    const std::size_t n_nodes = node_coords.extent(0);
    const std::size_t nnz = conn_indices.extent(0);
    constexpr auto max_index = static_cast<std::uint64_t>(std::numeric_limits<index_t>::max());
    if (static_cast<std::uint64_t>(n_nodes) > max_index || static_cast<std::uint64_t>(nnz) > max_index) {
        throw std::overflow_error("make_unstructured: node or connectivity extent exceeds index_t range");
    }
    if (areas.extent(0) != 0 && areas.extent(0) != n_cells) {
        throw std::invalid_argument("make_unstructured: optional cell areas must be empty or have one value per cell");
    }
    if (mask.extent(0) != 0 && mask.extent(0) != n_cells) {
        throw std::invalid_argument("make_unstructured: optional cell mask must be empty or have one value per cell");
    }

    // CSR values may reside in device-only memory. Validate them on the owning
    // execution space and return only the scalar error count to the host.
    using exec_space = typename MemorySpace::execution_space;
    std::size_t invalid_cells = 0;
    auto offsets = conn_offsets;
    auto indices = conn_indices;
    const auto n_nodes_i = static_cast<index_t>(n_nodes);
    const auto nnz_i = static_cast<index_t>(nnz);
    using policy = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_reduce(
        "make_unstructured_validate_csr", policy(0, n_cells == 0 ? 1 : n_cells),
        KOKKOS_LAMBDA(const std::size_t cell, std::size_t &errors) {
            if (n_cells == 0) {
                if (offsets(0) != 0 || nnz != 0) ++errors;
                return;
            }
            const index_t begin = offsets(cell);
            const index_t end = offsets(cell + 1);
            bool invalid = begin < 0 || end < begin || end > nnz_i || end - begin < 3;
            if (cell == 0 && begin != 0) invalid = true;
            if (cell + 1 == n_cells && end != nnz_i) invalid = true;
            if (!invalid) {
                for (index_t k = begin; k < end; ++k) {
                    const index_t node = indices(static_cast<std::size_t>(k));
                    if (node < 0 || node >= n_nodes_i) {
                        invalid = true;
                        break;
                    }
                }
            }
            if (invalid) ++errors;
        },
        invalid_cells);

    // The zero-cell case still has a meaningful offsets[0] sentinel.
    if (invalid_cells != 0) {
        if (n_cells == 0) {
            throw std::invalid_argument("make_unstructured: empty mesh must have offsets [0] and no indices");
        }
        throw std::invalid_argument("make_unstructured: malformed CSR offsets, polygon arity, or node index");
    }
}

}  // namespace

template <class MemorySpace>
UnstructuredMesh<MemorySpace> make_unstructured(Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords,
                                                Kokkos::View<index_t *, MemorySpace> conn_offsets, Kokkos::View<index_t *, MemorySpace> conn_indices,
                                                CoordinateSystem coord_sys, Kokkos::View<double *, MemorySpace> areas,
                                                Kokkos::View<int *, MemorySpace> mask, GeometryMetadata geometry) {
    validate_mesh_views(node_coords, conn_offsets, conn_indices, coord_sys, areas, mask);
    return UnstructuredMesh<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), coord_sys, std::move(areas),
                                         std::move(mask), geometry);
}

template UnstructuredMesh<Kokkos::HostSpace> make_unstructured<Kokkos::HostSpace>(Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>,
                                                                                  Kokkos::View<index_t *, Kokkos::HostSpace>,
                                                                                  Kokkos::View<index_t *, Kokkos::HostSpace>, CoordinateSystem,
                                                                                  Kokkos::View<double *, Kokkos::HostSpace>,
                                                                                  Kokkos::View<int *, Kokkos::HostSpace>, GeometryMetadata);

#ifdef KOKKOS_ENABLE_CUDA
template UnstructuredMesh<Kokkos::CudaSpace> make_unstructured<Kokkos::CudaSpace>(Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::CudaSpace>,
                                                                                  Kokkos::View<index_t *, Kokkos::CudaSpace>,
                                                                                  Kokkos::View<index_t *, Kokkos::CudaSpace>, CoordinateSystem,
                                                                                  Kokkos::View<double *, Kokkos::CudaSpace>,
                                                                                  Kokkos::View<int *, Kokkos::CudaSpace>, GeometryMetadata);
#endif

#ifdef KOKKOS_ENABLE_HIP
template UnstructuredMesh<Kokkos::HIPSpace> make_unstructured<Kokkos::HIPSpace>(Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HIPSpace>,
                                                                                Kokkos::View<index_t *, Kokkos::HIPSpace>,
                                                                                Kokkos::View<index_t *, Kokkos::HIPSpace>, CoordinateSystem,
                                                                                Kokkos::View<double *, Kokkos::HIPSpace>,
                                                                                Kokkos::View<int *, Kokkos::HIPSpace>, GeometryMetadata);
#endif

}  // namespace axis::topology
