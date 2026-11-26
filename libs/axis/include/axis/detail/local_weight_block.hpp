// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <axis/solver/weight_generator.hpp>

namespace axis::detail {

// Both meshes contain only the cells needed by this partition. Keep the
// matrix indices local to those meshes and use the same path on every worker.
inline solver::InterpolationMatrix<Kokkos::HostSpace> generate_local_weight_block(const topology::UnstructuredMesh<Kokkos::HostSpace> &source,
                                                                                  const topology::UnstructuredMesh<Kokkos::HostSpace> &destination,
                                                                                  solver::RegridConfig config) {
    config.use_structured_fast_paths = false;
    return solver::WeightGenerator::generate(source, destination, config);
}

}  // namespace axis::detail
