// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <axis/distributed/point_locator.hpp>
#include <limits>
#include <stdexcept>
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
            "point_ids"_a, "coordinates"_a, "coordinate_system"_a);

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
}
