// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <algorithm>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/reduced_gaussian_grid.hpp>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace axis::topology {

namespace {

constexpr double pi = 3.14159265358979323846;

std::size_t checked_add(std::size_t a, std::size_t b, const char *what) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw std::overflow_error(std::string("ReducedGaussianGrid: size overflow computing ") + what);
    }
    return a + b;
}

std::size_t checked_multiply(std::size_t a, std::size_t b, const char *what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("ReducedGaussianGrid: size overflow computing ") + what);
    }
    return a * b;
}

template <class MemorySpace, class View, class Values>
void copy_from_host_values(const View &destination, const Values &values) {
    auto mirror = Kokkos::create_mirror_view(destination);
    for (std::size_t i = 0; i < values.size(); ++i) mirror(i) = values[i];
    Kokkos::deep_copy(destination, mirror);
}

}  // namespace

ReducedGaussianGrid::ReducedGaussianGrid(std::vector<double> latitude_boundaries, std::vector<ReducedGaussianRow> rows,
                                         CoordinateSystem coordinate_system, LongitudePeriodicity seam)
    : latitude_boundaries_(std::move(latitude_boundaries)), rows_(std::move(rows)), coordinate_system_(coordinate_system), seam_(seam) {
    if (coordinate_system_ != CoordinateSystem::SphericalDeg && coordinate_system_ != CoordinateSystem::SphericalRad) {
        throw std::invalid_argument("ReducedGaussianGrid: only spherical degree or radian coordinates are supported");
    }
    if (rows_.empty() || latitude_boundaries_.size() != rows_.size() + 1) {
        throw std::invalid_argument("ReducedGaussianGrid: latitude boundaries must contain one more entry than the nonempty row list");
    }
    if (seam_.periodic && (!std::isfinite(seam_.period) || seam_.period <= 0.0)) {
        throw std::invalid_argument("ReducedGaussianGrid: periodic longitude period must be finite and positive");
    }
    const double lat_limit = coordinate_system_ == CoordinateSystem::SphericalDeg ? 90.0 : pi / 2.0;
    for (std::size_t j = 0; j < latitude_boundaries_.size(); ++j) {
        if (!std::isfinite(latitude_boundaries_[j]) || std::abs(latitude_boundaries_[j]) > lat_limit + 1e-12) {
            throw std::invalid_argument("ReducedGaussianGrid: latitude boundary is non-finite or outside the poles");
        }
        if (j > 0 && !(latitude_boundaries_[j] > latitude_boundaries_[j - 1])) {
            throw std::invalid_argument("ReducedGaussianGrid: latitude boundaries must be strictly increasing");
        }
    }

    row_offsets_.assign(rows_.size() + 1, 0);
    for (std::size_t j = 0; j < rows_.size(); ++j) {
        const auto &bounds = rows_[j].longitude_boundaries;
        if (bounds.size() < 2) throw std::invalid_argument("ReducedGaussianGrid: every row needs at least one longitude cell");
        for (std::size_t i = 0; i < bounds.size(); ++i) {
            if (!std::isfinite(bounds[i])) throw std::invalid_argument("ReducedGaussianGrid: longitude boundary is non-finite");
            if (i > 0 && !(bounds[i] > bounds[i - 1])) {
                throw std::invalid_argument("ReducedGaussianGrid: longitude boundaries must be strictly increasing");
            }
        }
        if (seam_.periodic) {
            const double span = bounds.back() - bounds.front();
            const double tolerance = 1e-10 * std::max(1.0, seam_.period);
            if (std::abs(span - seam_.period) > tolerance) {
                throw std::invalid_argument("ReducedGaussianGrid: periodic row boundaries must span the declared period");
            }
        }
        const std::size_t nlon = bounds.size() - 1;
        const std::size_t next = checked_add(static_cast<std::size_t>(row_offsets_[j]), nlon, "cell count");
        if (next > static_cast<std::size_t>(std::numeric_limits<index_t>::max())) {
            throw std::overflow_error("ReducedGaussianGrid: cell count exceeds index_t range");
        }
        row_offsets_[j + 1] = static_cast<index_t>(next);
    }
}

template <class MemorySpace>
UnstructuredMesh<MemorySpace> ReducedGaussianGrid::to_unstructured(bool conforming) const {
    const std::size_t n_cells = static_cast<std::size_t>(row_offsets_.back());
    std::vector<double> host_node_lon;
    std::vector<double> host_node_lat;
    std::vector<double> host_areas;
    std::vector<std::size_t> host_arities;
    std::vector<index_t> host_indices;
    host_areas.reserve(n_cells);
    host_arities.reserve(n_cells);

    using NodeKey = std::pair<double, double>;
    std::map<NodeKey, index_t> node_ids;
    auto canonical_lon = [&](double lon) {
        if (!seam_.periodic) return lon;
        double value = std::fmod(lon, seam_.period);
        if (value < 0.0) value += seam_.period;
        const double tolerance = 8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, seam_.period);
        if (std::abs(value) <= tolerance || std::abs(value - seam_.period) <= tolerance) value = 0.0;
        return value;
    };
    auto get_node = [&](double lon, double lat) -> index_t {
        const double x = canonical_lon(lon);
        const NodeKey key{x, lat};
        auto found = node_ids.find(key);
        if (found != node_ids.end()) return found->second;
        if (host_node_lon.size() >= static_cast<std::size_t>(std::numeric_limits<index_t>::max())) {
            throw std::overflow_error("ReducedGaussianGrid: node count exceeds index_t range");
        }
        const auto id = static_cast<index_t>(host_node_lon.size());
        node_ids.emplace(key, id);
        host_node_lon.push_back(x);
        host_node_lat.push_back(lat);
        return id;
    };
    const double scale = coordinate_system_ == CoordinateSystem::SphericalDeg ? pi / 180.0 : 1.0;
    const double latitude_tolerance = 1e-11;

    auto edge_splits = [&](std::size_t adjacent_row, double west, double east) {
        std::vector<double> result;
        const auto &bounds = rows_[adjacent_row].longitude_boundaries;
        for (double boundary : bounds) {
            if (seam_.periodic) {
                const double base_shift = std::floor((west - boundary) / seam_.period);
                for (int shift = -1; shift <= 2; ++shift) {
                    const double candidate = boundary + (base_shift + static_cast<double>(shift)) * seam_.period;
                    if (candidate > west + latitude_tolerance && candidate < east - latitude_tolerance) result.push_back(candidate);
                }
            } else if (boundary > west + latitude_tolerance && boundary < east - latitude_tolerance) {
                result.push_back(boundary);
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end(), [](double a, double b) { return std::abs(a - b) <= 1e-11; }), result.end());
        return result;
    };

    for (std::size_t j = 0; j < rows_.size(); ++j) {
        const auto &bounds = rows_[j].longitude_boundaries;
        const double south = latitude_boundaries_[j];
        const double north = latitude_boundaries_[j + 1];
        if (conforming && j > 0 && !seam_.periodic) {
            const auto &previous = rows_[j - 1].longitude_boundaries;
            if (std::abs(previous.front() - bounds.front()) > 1e-11 || std::abs(previous.back() - bounds.back()) > 1e-11) {
                throw std::invalid_argument("ReducedGaussianGrid: conforming nonperiodic rows must share longitude extent");
            }
        }

        for (std::size_t i = 0; i + 1 < bounds.size(); ++i) {
            const double west = bounds[i];
            const double east = bounds[i + 1];
            std::vector<index_t> polygon;
            polygon.push_back(get_node(west, south));
            if (conforming && j > 0) {
                for (double lon : edge_splits(j - 1, west, east)) polygon.push_back(get_node(lon, south));
            }
            polygon.push_back(get_node(east, south));
            polygon.push_back(get_node(east, north));
            if (conforming && j + 1 < rows_.size()) {
                auto split = edge_splits(j + 1, west, east);
                for (auto it = split.rbegin(); it != split.rend(); ++it) polygon.push_back(get_node(*it, north));
            }
            polygon.push_back(get_node(west, north));

            // Remove consecutive duplicate IDs, which can occur at an exactly
            // aligned seam after canonicalization.
            polygon.erase(std::unique(polygon.begin(), polygon.end()), polygon.end());
            if (polygon.size() < 3) throw std::invalid_argument("ReducedGaussianGrid: generated a degenerate cell polygon");
            host_arities.push_back(polygon.size());
            host_indices.insert(host_indices.end(), polygon.begin(), polygon.end());

            const double area = (east - west) * scale * (std::sin(north * scale) - std::sin(south * scale));
            if (!std::isfinite(area) || area <= 0.0) throw std::invalid_argument("ReducedGaussianGrid: cell has invalid spherical area");
            host_areas.push_back(area);
        }
    }

    const std::size_t n_nodes = host_node_lon.size();
    const std::size_t nnz = host_indices.size();
    (void)checked_multiply(n_cells, sizeof(index_t), "offset storage");
    if (nnz > static_cast<std::size_t>(std::numeric_limits<index_t>::max())) {
        throw std::overflow_error("ReducedGaussianGrid: connectivity size exceeds index_t range");
    }

    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("ReducedGaussianGrid::node_coords", n_nodes, 2);
    auto host_coords = Kokkos::create_mirror_view(node_coords);
    for (std::size_t i = 0; i < n_nodes; ++i) {
        host_coords(i, 0) = host_node_lon[i];
        host_coords(i, 1) = host_node_lat[i];
    }
    Kokkos::deep_copy(node_coords, host_coords);

    Kokkos::View<std::size_t *, MemorySpace> arities("ReducedGaussianGrid::arities", n_cells);
    copy_from_host_values<MemorySpace>(arities, host_arities);
    Kokkos::View<index_t *, MemorySpace> offsets("ReducedGaussianGrid::offsets", n_cells + 1);
    using exec_space = typename MemorySpace::execution_space;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    std::size_t total_entries = 0;
    Kokkos::parallel_scan(
        "ReducedGaussianGrid::csr_offsets", Range(exec, 0, n_cells),
        KOKKOS_LAMBDA(const std::size_t cell, std::size_t &update, bool final) {
            if (final) offsets(cell) = static_cast<index_t>(update);
            update += arities(cell);
        },
        total_entries);
    if (total_entries != nnz) throw std::logic_error("ReducedGaussianGrid: CSR scan total disagrees with connectivity size");
    Kokkos::parallel_for(
        "ReducedGaussianGrid::csr_sentinel", Range(exec, 0, 1),
        KOKKOS_LAMBDA(const std::size_t) { offsets(n_cells) = static_cast<index_t>(total_entries); });

    Kokkos::View<index_t *, MemorySpace> indices("ReducedGaussianGrid::indices", nnz);
    copy_from_host_values<MemorySpace>(indices, host_indices);
    Kokkos::View<double *, MemorySpace> areas("ReducedGaussianGrid::areas", n_cells);
    copy_from_host_values<MemorySpace>(areas, host_areas);
    exec.fence("ReducedGaussianGrid::finish_topology");
    return make_unstructured(
        std::move(node_coords), std::move(offsets), std::move(indices), coordinate_system_, std::move(areas), {},
        GeometryMetadata{GeometryProvenance::DeclaredGridModel, BoundaryModel::ConstantLatitude, AreaModel::ConstantLatitudeStrip, seam_});
}

template UnstructuredMesh<Kokkos::HostSpace> ReducedGaussianGrid::to_unstructured<Kokkos::HostSpace>(bool) const;
#ifdef KOKKOS_ENABLE_CUDA
template UnstructuredMesh<Kokkos::CudaSpace> ReducedGaussianGrid::to_unstructured<Kokkos::CudaSpace>(bool) const;
#endif
#ifdef KOKKOS_ENABLE_HIP
template UnstructuredMesh<Kokkos::HIPSpace> ReducedGaussianGrid::to_unstructured<Kokkos::HIPSpace>(bool) const;
#endif

}  // namespace axis::topology
