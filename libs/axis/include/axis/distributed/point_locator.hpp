// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <mpi.h>

#include <axis/solver/point_locator.hpp>
namespace axis::distributed {
using solver::PointLocationOptions;
/// Collective CPU point location with independently partitioned mesh and points.
/// Mesh and point IDs must each be globally unique. Input mesh excludes ghosts.
/// Results stay in local input order. Rank/local indices describe only this
/// locator's ownership epoch; stable cell IDs survive repartitioning.
/// Owns a duplicated communicator. Calls are collective, including empty ranks;
/// do not call concurrently on one instance. Destroy before MPI_Finalize.
class DistributedPointLocator {
    struct Impl;
    std::unique_ptr<Impl> impl_;

   public:
    DistributedPointLocator(const topology::UnstructuredMesh<Kokkos::HostSpace> &mesh, std::span<const index_t> cell_ids, MPI_Comm comm,
                            PointLocationOptions options = {});
    ~DistributedPointLocator();
    DistributedPointLocator(DistributedPointLocator &&) noexcept;
    DistributedPointLocator &operator=(DistributedPointLocator &&) noexcept;
    DistributedPointLocator(const DistributedPointLocator &) = delete;
    DistributedPointLocator &operator=(const DistributedPointLocator &) = delete;
    [[nodiscard]] std::vector<solver::PointLocation> locate(solver::PointSetView points);
};
}  // namespace axis::distributed
