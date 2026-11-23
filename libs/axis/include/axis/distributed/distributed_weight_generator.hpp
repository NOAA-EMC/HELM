// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DISTRIBUTED_WEIGHT_GENERATOR_HPP
#define AXIS_DISTRIBUTED_WEIGHT_GENERATOR_HPP

/// @file axis/distributed/distributed_weight_generator.hpp
/// @brief Legacy MPI weight-generation adapter (see DistributedWeightGenerator).

#include <mpi.h>

#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>

namespace axis::distributed {

/// @class DistributedWeightGenerator
/// @brief Legacy export of global destination row numbers.
///
/// This adapter does not discover remote source geometry. Callers must provide
/// complete source geometry (e.g. replicated source meshes). For partitioned
/// source ownership and HALO-backed field exchange use DistributedRegridder
/// from axis/distributed/regridder.hpp. Its matrix rows are local, unlike this
/// legacy row-export format. Linear reconstruction uses the new operator.
template <typename MemorySpace>
class DistributedWeightGenerator {
   public:
    /// @brief Generate an InterpolationMatrix partitioned across MPI ranks.
    /// @param local_src_mesh Source mesh partition owned by this rank.
    /// @param local_dst_mesh Destination mesh partition owned by this rank.
    /// @param config         Regridding configuration.
    /// @param comm           The MPI communicator.
    /// @return A partitioned InterpolationMatrix representing this rank's contribution.
    static solver::InterpolationMatrix<MemorySpace> generate(const topology::UnstructuredMesh<MemorySpace> &local_src_mesh,
                                                             const topology::UnstructuredMesh<MemorySpace> &local_dst_mesh,
                                                             const solver::RegridConfig &config, MPI_Comm comm);
};

}  // namespace axis::distributed

#endif  // AXIS_DISTRIBUTED_WEIGHT_GENERATOR_HPP
