// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <axis/distributed/point_locator.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
template <class T>
nb::ndarray<nb::numpy, T> point_result_array(const std::vector<T> &values) {
    auto *data = new T[values.size()];
    std::copy(values.begin(), values.end(), data);
    nb::capsule owner(data, [](void *ptr) noexcept { delete[] static_cast<T *>(ptr); });
    return nb::ndarray<nb::numpy, T>(data, {values.size()}, std::move(owner));
}

axis::solver::PointLocationOptions point_options(double tolerance, std::size_t max_import_bytes, std::size_t max_message_bytes,
                                                 std::size_t query_batch_size) {
    axis::solver::PointLocationOptions options;
    options.tolerance = tolerance;
    options.max_import_bytes = max_import_bytes;
    options.max_message_bytes = max_message_bytes;
    options.query_batch_size = query_batch_size;
    return options;
}

nb::dict point_results(const std::vector<axis::solver::PointLocation> &results) {
    std::vector<axis::index_t> point_ids, cell_ids, owner_local_indices;
    std::vector<int> owner_ranks, status;
    point_ids.reserve(results.size());
    cell_ids.reserve(results.size());
    owner_local_indices.reserve(results.size());
    owner_ranks.reserve(results.size());
    status.reserve(results.size());
    for (const auto &result : results) {
        point_ids.push_back(result.point_id);
        cell_ids.push_back(result.cell_id);
        owner_local_indices.push_back(result.owner_local_index);
        owner_ranks.push_back(result.owner_rank);
        status.push_back(static_cast<int>(result.status));
    }
    nb::dict output;
    output["point_ids"] = point_result_array(point_ids);
    output["cell_ids"] = point_result_array(cell_ids);
    output["owner_local_indices"] = point_result_array(owner_local_indices);
    output["owner_ranks"] = point_result_array(owner_ranks);
    output["status"] = point_result_array(status);
    return output;
}

nb::ndarray<nb::numpy, double> mesh_cell_centroids(const HostMesh &mesh) {
    const auto nodes = mesh.node_coords();
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();
    const auto n_cells = mesh.n_cells();
    if (offsets.extent(0) != n_cells + 1) throw std::invalid_argument("point locator: mesh has invalid connectivity offsets");

    const auto coordinate_system = mesh.coord_system();
    const bool is_spherical = coordinate_system == axis::topology::CoordinateSystem::SphericalDeg ||
                              coordinate_system == axis::topology::CoordinateSystem::SphericalRad;
    if (!is_spherical && coordinate_system != axis::topology::CoordinateSystem::Cartesian3D)
        throw std::invalid_argument("point locator: mesh has unsupported coordinates");
    const double angle_factor = coordinate_system == axis::topology::CoordinateSystem::SphericalDeg ? std::numbers::pi / 180.0 : 1.0;

    std::vector<double> centroids(n_cells * 3, 0.0);
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        if (offsets[cell] < 0 || offsets[cell + 1] < offsets[cell] || offsets[cell + 1] > static_cast<axis::index_t>(indices.extent(0)))
            throw std::invalid_argument("point locator: mesh has invalid cell connectivity");
        const auto vertex_count = static_cast<std::size_t>(offsets[cell + 1] - offsets[cell]);
        if (vertex_count == 0) throw std::invalid_argument("point locator: mesh has empty cell connectivity");
        double *center = centroids.data() + cell * 3;
        for (axis::index_t position = offsets[cell]; position < offsets[cell + 1]; ++position) {
            const auto node = indices[position];
            if (node < 0 || static_cast<std::size_t>(node) >= mesh.n_nodes()) throw std::invalid_argument("point locator: mesh has invalid vertex index");
            const double x = nodes(node, 0);
            const double y = nodes(node, 1);
            const double z = nodes.extent(1) > 2 ? nodes(node, 2) : 0.0;
            if (is_spherical) {
                const double lon = x * angle_factor;
                const double lat = y * angle_factor;
                center[0] += std::cos(lat) * std::cos(lon);
                center[1] += std::cos(lat) * std::sin(lon);
                center[2] += std::sin(lat);
            } else {
                center[0] += x;
                center[1] += y;
                center[2] += z;
            }
        }
        if (is_spherical) {
            const double length = std::hypot(std::hypot(center[0], center[1]), center[2]);
            if (!(length > 0.0) || !std::isfinite(length)) throw std::invalid_argument("point locator: spherical cell has no canonical centroid");
            for (int dimension = 0; dimension < 3; ++dimension) center[dimension] /= length;
        } else {
            for (int dimension = 0; dimension < 3; ++dimension) center[dimension] /= static_cast<double>(vertex_count);
        }
    }

    auto *data = new double[centroids.size()];
    std::copy(centroids.begin(), centroids.end(), data);
    nb::capsule owner(data, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
    const std::size_t shape[2] = {n_cells, 3};
    return nb::ndarray<nb::numpy, double>(data, 2, shape, std::move(owner));
}

nb::dict mesh_extract_cells(const HostMesh &mesh, nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> cell_rows) {
    const auto nodes = mesh.node_coords();
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();
    if (offsets.extent(0) != mesh.n_cells() + 1) throw std::invalid_argument("point locator: mesh has invalid connectivity offsets");

    std::vector<bool> seen(mesh.n_cells(), false);
    std::vector<axis::index_t> out_offsets(cell_rows.shape(0) + 1, 0);
    std::vector<axis::index_t> out_indices;
    std::vector<axis::index_t> old_nodes;
    std::unordered_map<axis::index_t, axis::index_t> new_node_index;
    new_node_index.reserve(cell_rows.shape(0) * 4);

    for (std::size_t local_cell = 0; local_cell < cell_rows.shape(0); ++local_cell) {
        const auto row = cell_rows.data()[local_cell * cell_rows.stride(0)];
        if (row < 0 || static_cast<std::size_t>(row) >= mesh.n_cells()) throw std::invalid_argument("point locator: cell row is out of range");
        if (seen[static_cast<std::size_t>(row)]) throw std::invalid_argument("point locator: duplicate cell row");
        seen[static_cast<std::size_t>(row)] = true;
        if (offsets[row] < 0 || offsets[row + 1] < offsets[row] || offsets[row + 1] > static_cast<axis::index_t>(indices.extent(0)))
            throw std::invalid_argument("point locator: mesh has invalid cell connectivity");
        for (axis::index_t position = offsets[row]; position < offsets[row + 1]; ++position) {
            const auto old_node = indices[position];
            if (old_node < 0 || static_cast<std::size_t>(old_node) >= mesh.n_nodes())
                throw std::invalid_argument("point locator: mesh has invalid vertex index");
            auto [entry, inserted] = new_node_index.emplace(old_node, static_cast<axis::index_t>(old_nodes.size()));
            if (inserted) old_nodes.push_back(old_node);
            out_indices.push_back(entry->second);
        }
        out_offsets[local_cell + 1] = static_cast<axis::index_t>(out_indices.size());
    }

    std::vector<double> out_nodes(old_nodes.size() * nodes.extent(1));
    for (std::size_t local_node = 0; local_node < old_nodes.size(); ++local_node)
        for (std::size_t dimension = 0; dimension < nodes.extent(1); ++dimension)
            out_nodes[local_node * nodes.extent(1) + dimension] = nodes(old_nodes[local_node], dimension);

    auto *node_data = new double[out_nodes.size()];
    std::copy(out_nodes.begin(), out_nodes.end(), node_data);
    nb::capsule node_owner(node_data, [](void *ptr) noexcept { delete[] static_cast<double *>(ptr); });
    const std::size_t node_shape[2] = {old_nodes.size(), nodes.extent(1)};

    nb::dict result;
    result["node_coords"] = nb::ndarray<nb::numpy, double>(node_data, 2, node_shape, std::move(node_owner));
    result["conn_offsets"] = point_result_array(out_offsets);
    result["conn_indices"] = point_result_array(out_indices);
    return result;
}

axis::solver::PointSetView point_set_view(nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> ids,
                                          nb::ndarray<nb::numpy, const double, nb::ndim<2>> coordinates,
                                          axis::topology::CoordinateSystem coordinate_system) {
    if (coordinates.shape(1) != 2 && coordinates.shape(1) != 3) throw std::invalid_argument("point locator: coordinates must have 2 or 3 columns");
    if (coordinates.shape(0) > 1 && coordinates.stride(0) != 1) throw std::invalid_argument("point locator: coordinates must be Fortran contiguous");
    if (coordinates.shape(0) > 1 && coordinates.stride(1) != static_cast<std::int64_t>(coordinates.shape(0)))
        throw std::invalid_argument("point locator: coordinates must be Fortran contiguous");
    return {{ids.data(), ids.shape(0)},
            axis::field_view<const double, 2>(coordinates.data(), coordinates.shape(0), coordinates.shape(1)),
            coordinate_system};
}

MPI_Comm python_communicator(std::int64_t fortran_handle) {
    int initialized = 0, finalized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) throw std::runtime_error("DistributedPointLocator requires initialized MPI");
    MPI_Finalized(&finalized);
    if (finalized) throw std::runtime_error("DistributedPointLocator cannot be used after MPI_Finalize");
    if (fortran_handle < std::numeric_limits<MPI_Fint>::min() || fortran_handle > std::numeric_limits<MPI_Fint>::max())
        throw std::invalid_argument("DistributedPointLocator: invalid MPI communicator handle");
    MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_handle));
    if (comm == MPI_COMM_NULL) throw std::invalid_argument("DistributedPointLocator: MPI communicator is null");
    return comm;
}
}  // namespace

inline void bind_point_locator(nb::module_ &m) {
    nb::class_<axis::solver::PointLocator>(m, "_PointLocator")
        .def(
            "__init__",
            [](axis::solver::PointLocator *self, const HostMesh &mesh, nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> cell_ids,
               double tolerance, std::size_t max_import_bytes, std::size_t max_message_bytes, std::size_t query_batch_size) {
                ensure_kokkos();
                auto options = point_options(tolerance, max_import_bytes, max_message_bytes, query_batch_size);
                new (self) axis::solver::PointLocator(mesh, {cell_ids.data(), cell_ids.shape(0)}, options);
            },
            "mesh"_a, "cell_ids"_a, "tolerance"_a, "max_import_bytes"_a, "max_message_bytes"_a, "query_batch_size"_a)
        .def(
            "locate",
            [](const axis::solver::PointLocator &locator, nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> point_ids,
               nb::ndarray<nb::numpy, const double, nb::ndim<2>> coordinates, axis::topology::CoordinateSystem coordinate_system) {
                auto points = point_set_view(point_ids, coordinates, coordinate_system);
                return point_results(locator.locate(points));
            },
            "point_ids"_a, "coordinates"_a, "coordinate_system"_a)
        .def(
            "bounds",
            [](const axis::solver::PointLocator &locator) {
                const auto bounds = locator.bounds();
                return point_result_array(std::vector<double>(bounds.begin(), bounds.end()));
            });

    nb::class_<axis::distributed::DistributedPointLocator>(m, "_DistributedPointLocator")
        .def(
            "__init__",
            [](axis::distributed::DistributedPointLocator *self, const HostMesh &mesh,
               nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> cell_ids, std::int64_t comm_f, double tolerance, std::size_t max_import_bytes,
               std::size_t max_message_bytes, std::size_t query_batch_size) {
                ensure_kokkos();
                auto options = point_options(tolerance, max_import_bytes, max_message_bytes, query_batch_size);
                new (self)
                    axis::distributed::DistributedPointLocator(mesh, {cell_ids.data(), cell_ids.shape(0)}, python_communicator(comm_f), options);
            },
            "mesh"_a, "cell_ids"_a, "comm_f"_a, "tolerance"_a, "max_import_bytes"_a, "max_message_bytes"_a, "query_batch_size"_a)
        .def(
            "locate",
            [](axis::distributed::DistributedPointLocator &locator, nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> point_ids,
               nb::ndarray<nb::numpy, const double, nb::ndim<2>> coordinates, axis::topology::CoordinateSystem coordinate_system) {
                auto points = point_set_view(point_ids, coordinates, coordinate_system);
                return point_results(locator.locate(points));
            },
            "point_ids"_a, "coordinates"_a, "coordinate_system"_a);

    m.def(
        "_mesh_cell_centroids",
        [](const HostMesh &mesh) {
            ensure_kokkos();
            return mesh_cell_centroids(mesh);
        },
        "mesh"_a);
    m.def(
        "_mesh_extract_cells",
        [](const HostMesh &mesh, nb::ndarray<nb::numpy, const axis::index_t, nb::ndim<1>> cell_rows) {
            ensure_kokkos();
            return mesh_extract_cells(mesh, cell_rows);
        },
        "mesh"_a, "cell_rows"_a);
}
