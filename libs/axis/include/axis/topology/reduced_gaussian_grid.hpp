// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_REDUCED_GAUSSIAN_GRID_HPP
#define AXIS_TOPOLOGY_REDUCED_GAUSSIAN_GRID_HPP

/// @file axis/topology/reduced_gaussian_grid.hpp
/// @brief Ragged-row Gaussian latitude/longitude geometry and topology.

#include <Kokkos_Core.hpp>
#include <axis/topology/enums.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <vector>

namespace axis::topology {

/// Longitude boundaries for one reduced-Gaussian latitude row. They are
/// strictly increasing and contain nlon+1 values; the row origin is the first
/// boundary. A periodic row spans exactly the declared longitude period.
struct ReducedGaussianRow {
    std::vector<double> longitude_boundaries;
};

/// Ragged Gaussian latitude/longitude grid input. The cells in row j retain
/// their order at `[row_offsets[j], row_offsets[j+1])`.
class ReducedGaussianGrid {
   public:
    ReducedGaussianGrid(std::vector<double> latitude_boundaries, std::vector<ReducedGaussianRow> rows, CoordinateSystem coordinate_system,
                        LongitudePeriodicity seam);

    [[nodiscard]] std::size_t n_rows() const noexcept {
        return rows_.size();
    }
    [[nodiscard]] const std::vector<index_t> &row_offsets() const noexcept {
        return row_offsets_;
    }

    /// Convert to the common CSR mesh. Conforming mode splits cell edges at
    /// boundary points from adjacent rows. Nonconforming mode retains each
    /// source cell's four declared corners.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] UnstructuredMesh<MemorySpace> to_unstructured(bool conforming = true) const;

   private:
    std::vector<double> latitude_boundaries_;
    std::vector<ReducedGaussianRow> rows_;
    std::vector<index_t> row_offsets_;
    CoordinateSystem coordinate_system_;
    LongitudePeriodicity seam_;
};

extern template UnstructuredMesh<Kokkos::HostSpace> ReducedGaussianGrid::to_unstructured<Kokkos::HostSpace>(bool) const;
#ifdef KOKKOS_ENABLE_CUDA
extern template UnstructuredMesh<Kokkos::CudaSpace> ReducedGaussianGrid::to_unstructured<Kokkos::CudaSpace>(bool) const;
#endif
#ifdef KOKKOS_ENABLE_HIP
extern template UnstructuredMesh<Kokkos::HIPSpace> ReducedGaussianGrid::to_unstructured<Kokkos::HIPSpace>(bool) const;
#endif

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_REDUCED_GAUSSIAN_GRID_HPP
