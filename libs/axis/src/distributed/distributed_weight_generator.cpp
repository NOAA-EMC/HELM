// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/distributed/distributed_weight_generator.cpp
/// @brief MPI-distributed horizontal weight generation for AXIS regridding.
///
/// Builds the destination-owned sparse weight matrix across ranks, borrowing
/// the caller's communicator through HALO and falling back to local generation
/// on a single rank.

#include <algorithm>
#include <axis/distributed/distributed_weight_generator.hpp>
#include <axis/solver/weight_generator.hpp>
#include <halo/collectives.hpp>
#include <halo/communicator.hpp>
#include <stdexcept>
#include <vector>

namespace axis::distributed {

template <typename MemorySpace>
solver::InterpolationMatrix<MemorySpace> DistributedWeightGenerator<MemorySpace>::generate(
    const topology::UnstructuredMesh<MemorySpace> &local_src_mesh, const topology::UnstructuredMesh<MemorySpace> &local_dst_mesh,
    const solver::RegridConfig &config, MPI_Comm comm) {
    // HALO duplicates the borrowed handle without adopting the caller's comm.
    auto h_comm = halo::Communicator::duplicate_of(comm);

    const int rank = h_comm.rank();
    const int size = h_comm.size();

    if (size == 1) {
        // Fallback to standard local weight generation for single-rank run
        return solver::WeightGenerator::template generate<MemorySpace>(local_src_mesh, local_dst_mesh, config);
    }

    if (config.method == solver::InterpolationMethod::Conservative2ndOrder)
        throw std::invalid_argument("multi-rank linear reconstruction requires gradient-stencil halos");

    const std::size_t n_src_local = local_src_mesh.n_cells();
    const std::size_t n_dst_local = local_dst_mesh.n_cells();

    // 2. Gather all destination mesh global cell offsets to map global indices
    int local_dst_count = static_cast<int>(n_dst_local);

    auto dst_counts = halo::allgather(h_comm, &local_dst_count, 1);

    std::vector<std::size_t> dst_offsets(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        dst_offsets[r + 1] = dst_offsets[r] + dst_counts[r];
    }

    // 3. Invoke standard local weights generation for local partition boundaries
    auto local_matrix = solver::WeightGenerator::template generate<MemorySpace>(local_src_mesh, local_dst_mesh, config);

    const std::size_t nnz = local_matrix.nnz();
    auto rows = local_matrix.factor_row_view();
    auto cols = local_matrix.factor_col_view();
    auto vals = local_matrix.factor_list_view();

    auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);
    auto h_cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cols);
    auto h_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vals);

    // Map row indices to their global destination indices
    std::size_t global_dst_offset = dst_offsets[rank];
    Kokkos::View<index_t *, Kokkos::HostSpace> h_global_rows("h_global_rows", nnz);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_global_cols("h_global_cols", nnz);
    Kokkos::View<double *, Kokkos::HostSpace> h_global_vals("h_global_vals", nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_global_rows(k) = h_rows(k) + global_dst_offset;
        h_global_cols(k) = h_cols(k);  // Local column indices mapped natively
        h_global_vals(k) = h_vals(k);
    }

    auto dev_rows = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_global_rows);
    auto dev_cols = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_global_cols);
    auto dev_vals = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_global_vals);

    return solver::InterpolationMatrix<MemorySpace>(dev_vals, dev_rows, dev_cols, local_matrix.frac_a_view(), local_matrix.frac_b_view(),
                                                    local_matrix.area_a_view(), local_matrix.area_b_view(), local_src_mesh.n_cells(),
                                                    dst_offsets[size]);
}

template class DistributedWeightGenerator<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class DistributedWeightGenerator<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template class DistributedWeightGenerator<Kokkos::HIPSpace>;
#endif

}  // namespace axis::distributed
