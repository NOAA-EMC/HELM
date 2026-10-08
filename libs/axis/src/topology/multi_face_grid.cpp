// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <Kokkos_Core.hpp>
#include <axis/detail/memory_traits.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/multi_face_grid.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace axis::topology {

namespace {

std::size_t checked_add(std::size_t a, std::size_t b, const char *what) {
    if (b > std::numeric_limits<std::size_t>::max() - a) throw std::overflow_error(std::string("MultiFaceGrid: overflow computing ") + what);
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char *what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(std::string("MultiFaceGrid: overflow computing ") + what);
    return a * b;
}

class DisjointSet {
   public:
    explicit DisjointSet(std::size_t n) : parent_(n), rank_(n, 0) {
        std::iota(parent_.begin(), parent_.end(), std::size_t{0});
    }
    std::size_t find(std::size_t x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void unite(std::size_t a, std::size_t b) {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent_[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
    }

   private:
    std::vector<std::size_t> parent_;
    std::vector<unsigned char> rank_;
};

template <class MemorySpace>
std::size_t face_node_index(const StructuredFace<MemorySpace> &face, FaceEdge edge, std::size_t k) {
    const std::size_t stride = face.ni + 1;
    switch (edge) {
        case FaceEdge::South:
            return k;
        case FaceEdge::North:
            return k + face.nj * stride;
        case FaceEdge::West:
            return k * stride;
        case FaceEdge::East:
            return face.ni + k * stride;
    }
    throw std::invalid_argument("MultiFaceGrid: unknown face edge");
}

template <class MemorySpace>
std::size_t edge_node_count(const StructuredFace<MemorySpace> &face, FaceEdge edge) {
    return (edge == FaceEdge::East || edge == FaceEdge::West) ? face.nj + 1 : face.ni + 1;
}

}  // namespace

template <class MemorySpace>
MultiFaceGrid<MemorySpace>::MultiFaceGrid(std::vector<StructuredFace<MemorySpace>> faces, std::vector<FaceConnection> connections,
                                          CoordinateSystem coordinate_system, std::vector<std::vector<FaceVertexRef>> vertex_equivalences)
    : faces_(std::move(faces)),
      connections_(std::move(connections)),
      coordinate_system_(coordinate_system),
      vertex_equivalences_(std::move(vertex_equivalences)) {
    if (faces_.empty()) throw std::invalid_argument("MultiFaceGrid: at least one face is required");
    const std::size_t ndim = coordinate_system_ == CoordinateSystem::Cartesian3D ? 3 : 2;
    for (const auto &face : faces_) {
        if (face.ni == 0 || face.nj == 0) throw std::invalid_argument("MultiFaceGrid: face dimensions must be positive");
        const std::size_t expected = checked_mul(checked_add(face.ni, 1, "ni+1"), checked_add(face.nj, 1, "nj+1"), "face node count");
        if (face.node_coords.extent(0) != expected || face.node_coords.extent(1) != ndim) {
            throw std::invalid_argument("MultiFaceGrid: face node-coordinate extent does not match its dimensions or coordinate system");
        }
    }
    for (const auto &connection : connections_) {
        if (connection.face_a >= faces_.size() || connection.face_b >= faces_.size() || connection.face_a == connection.face_b) {
            throw std::invalid_argument("MultiFaceGrid: face connection references an invalid or identical face");
        }
        if (edge_node_count(faces_[connection.face_a], connection.edge_a) != edge_node_count(faces_[connection.face_b], connection.edge_b)) {
            throw std::invalid_argument("MultiFaceGrid: connected edges must have the same number of vertices");
        }
    }
}

template <class MemorySpace>
UnstructuredMesh<MemorySpace> MultiFaceGrid<MemorySpace>::to_unstructured() const {
    std::vector<std::size_t> face_node_offsets(faces_.size() + 1, 0);
    std::vector<std::size_t> face_cell_offsets(faces_.size() + 1, 0);
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        const auto &face = faces_[f];
        const std::size_t nodes = checked_mul(face.ni + 1, face.nj + 1, "face node count");
        const std::size_t cells = checked_mul(face.ni, face.nj, "face cell count");
        face_node_offsets[f + 1] = checked_add(face_node_offsets[f], nodes, "total local node count");
        face_cell_offsets[f + 1] = checked_add(face_cell_offsets[f], cells, "total cell count");
    }
    const std::size_t local_nodes = face_node_offsets.back();
    const std::size_t n_cells = face_cell_offsets.back();
    const std::size_t ndim = coordinate_system_ == CoordinateSystem::Cartesian3D ? 3 : 2;
    const std::size_t nnz = checked_mul(n_cells, 4, "connectivity size");
    if (local_nodes > static_cast<std::size_t>(std::numeric_limits<index_t>::max()) ||
        nnz > static_cast<std::size_t>(std::numeric_limits<index_t>::max())) {
        throw std::overflow_error("MultiFaceGrid: mesh extents exceed index_t range");
    }

    DisjointSet sets(local_nodes);
    auto flat_local = [&](std::size_t f, std::size_t local_node) { return face_node_offsets[f] + local_node; };
    for (const auto &connection : connections_) {
        const auto &a = faces_[connection.face_a];
        const auto &b = faces_[connection.face_b];
        const std::size_t count = edge_node_count(a, connection.edge_a);
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t kb = connection.reversed ? count - 1 - k : k;
            sets.unite(flat_local(connection.face_a, face_node_index(a, connection.edge_a, k)),
                       flat_local(connection.face_b, face_node_index(b, connection.edge_b, kb)));
        }
    }
    auto validate_ref = [&](const FaceVertexRef &ref) {
        if (ref.face >= faces_.size() || ref.i > faces_[ref.face].ni || ref.j > faces_[ref.face].nj) {
            throw std::invalid_argument("MultiFaceGrid: explicit vertex equivalence references an invalid face vertex");
        }
        return flat_local(ref.face, ref.i + ref.j * (faces_[ref.face].ni + 1));
    };
    for (const auto &group : vertex_equivalences_) {
        if (group.size() < 2) throw std::invalid_argument("MultiFaceGrid: vertex-equivalence groups must contain at least two vertices");
        const std::size_t first = validate_ref(group.front());
        for (std::size_t k = 1; k < group.size(); ++k) sets.unite(first, validate_ref(group[k]));
    }

    // IDs follow first occurrence in face order, then local row-major node order.
    std::vector<index_t> local_to_global(local_nodes);
    std::vector<std::size_t> representative;
    std::vector<std::size_t> root_to_global(local_nodes, std::numeric_limits<std::size_t>::max());
    for (std::size_t local = 0; local < local_nodes; ++local) {
        const std::size_t root = sets.find(local);
        if (root_to_global[root] == std::numeric_limits<std::size_t>::max()) {
            root_to_global[root] = representative.size();
            representative.push_back(local);
        }
        local_to_global[local] = static_cast<index_t>(root_to_global[root]);
    }
    const std::size_t n_nodes = representative.size();

    std::vector<index_t> host_offsets(n_cells + 1);
    std::vector<index_t> host_indices(nnz);
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        const auto &face = faces_[f];
        const std::size_t stride = face.ni + 1;
        for (std::size_t cell = 0; cell < face.ni * face.nj; ++cell) {
            const std::size_t i = cell % face.ni;
            const std::size_t j = cell / face.ni;
            const std::size_t local_ids[4] = {i + j * stride, i + 1 + j * stride, i + 1 + (j + 1) * stride, i + (j + 1) * stride};
            const std::size_t global_cell = face_cell_offsets[f] + cell;
            host_offsets[global_cell] = static_cast<index_t>(global_cell * 4);
            for (std::size_t k = 0; k < 4; ++k) {
                const std::size_t source_k = face.reverse_cell_orientation ? (4 - k) % 4 : k;
                host_indices[global_cell * 4 + k] = local_to_global[flat_local(f, local_ids[source_k])];
            }
        }
    }
    host_offsets[n_cells] = static_cast<index_t>(nnz);

    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> local_coords("MultiFaceGrid::local_coords", local_nodes, ndim);
    using exec_space = typename detail::exec_space_t<MemorySpace>;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    for (std::size_t f = 0; f < faces_.size(); ++f) {
        auto source = faces_[f].node_coords;
        auto destination = local_coords;
        const std::size_t begin = face_node_offsets[f];
        const std::size_t count = face_node_offsets[f + 1] - begin;
        Kokkos::parallel_for(
            "MultiFaceGrid::copy_face_coordinates", Range(exec, begin, begin + count), KOKKOS_LAMBDA(const std::size_t local) {
                const std::size_t source_node = local - begin;
                for (std::size_t d = 0; d < ndim; ++d) destination(local, d) = source(source_node, d);
            });
    }
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("MultiFaceGrid::node_coords", n_nodes, ndim);
    auto representative_view = Kokkos::View<std::size_t *, MemorySpace>("MultiFaceGrid::representatives", n_nodes);
    auto rep_host = Kokkos::create_mirror_view(representative_view);
    for (std::size_t n = 0; n < n_nodes; ++n) rep_host(n) = representative[n];
    Kokkos::deep_copy(representative_view, rep_host);
    auto reps = representative_view;
    auto output_coords = node_coords;
    Kokkos::parallel_for(
        "MultiFaceGrid::resolve_shared_coordinates", Range(exec, 0, n_nodes), KOKKOS_LAMBDA(const std::size_t node) {
            const std::size_t source_node = reps(node);
            for (std::size_t d = 0; d < ndim; ++d) output_coords(node, d) = local_coords(source_node, d);
        });

    Kokkos::View<index_t *, MemorySpace> offsets("MultiFaceGrid::offsets", n_cells + 1);
    Kokkos::View<index_t *, MemorySpace> indices("MultiFaceGrid::indices", nnz);
    auto offsets_host = Kokkos::create_mirror_view(offsets);
    auto indices_host = Kokkos::create_mirror_view(indices);
    for (std::size_t k = 0; k < host_offsets.size(); ++k) offsets_host(k) = host_offsets[k];
    for (std::size_t k = 0; k < host_indices.size(); ++k) indices_host(k) = host_indices[k];
    Kokkos::deep_copy(offsets, offsets_host);
    Kokkos::deep_copy(indices, indices_host);

    std::size_t invalid_coordinates = 0;
    auto local_map = Kokkos::View<index_t *, MemorySpace>("MultiFaceGrid::local_to_global", local_nodes);
    auto map_host = Kokkos::create_mirror_view(local_map);
    for (std::size_t k = 0; k < local_nodes; ++k) map_host(k) = local_to_global[k];
    Kokkos::deep_copy(local_map, map_host);
    auto map = local_map;
    const double period = coordinate_system_ == CoordinateSystem::SphericalRad ? 2.0 * std::acos(-1.0) : 360.0;
    const double latitude_limit = coordinate_system_ == CoordinateSystem::SphericalRad ? 0.5 * std::acos(-1.0) : 90.0;
    const CoordinateSystem coordinate_system = coordinate_system_;
    Kokkos::parallel_reduce(
        "MultiFaceGrid::validate_equivalent_coordinates", Range(exec, 0, local_nodes),
        KOKKOS_LAMBDA(const std::size_t local, std::size_t &errors) {
            const std::size_t global = static_cast<std::size_t>(map(local));
            const std::size_t representative_local = reps(global);
            for (std::size_t d = 0; d < ndim; ++d) {
                double difference = Kokkos::abs(local_coords(local, d) - local_coords(representative_local, d));
                if (d == 0 && coordinate_system != CoordinateSystem::Cartesian3D) {
                    difference = Kokkos::fmin(difference, Kokkos::abs(period - difference));
                    if (ndim == 2 && Kokkos::abs(local_coords(local, 1)) >= latitude_limit - 1e-12) difference = 0.0;
                }
                const double scale =
                    Kokkos::fmax(1.0, Kokkos::fmax(Kokkos::abs(local_coords(local, d)), Kokkos::abs(local_coords(representative_local, d))));
                if (!Kokkos::isfinite(local_coords(local, d)) || difference > 1e-10 * scale) {
                    ++errors;
                    break;
                }
            }
        },
        invalid_coordinates);
    exec.fence("MultiFaceGrid::assemble_fence");
    if (invalid_coordinates != 0) throw std::invalid_argument("MultiFaceGrid: explicitly equivalent seam vertices have inconsistent coordinates");
    return make_unstructured(std::move(node_coords), std::move(offsets), std::move(indices), coordinate_system_, {}, {},
                             GeometryMetadata{GeometryProvenance::SourceAuthoritative, BoundaryModel::Unspecified, AreaModel::Unspecified});
}

template class MultiFaceGrid<Kokkos::HostSpace>;
#ifdef KOKKOS_ENABLE_CUDA
template class MultiFaceGrid<Kokkos::CudaSpace>;
#endif
#ifdef KOKKOS_ENABLE_HIP
template class MultiFaceGrid<Kokkos::HIPSpace>;
#endif

}  // namespace axis::topology
