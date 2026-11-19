// SPDX-License-Identifier: Apache-2.0
/// @file src/distributed/selective_geometry.hpp
/// @brief Exchange only the source cells a rank actually needs for its halo.
///
/// Selects candidate source geometry through proximity queries and bounded
/// peer exchange, returning an imported host mesh with global cell ids and
/// discovery statistics.
#pragma once
#include <axis/distributed/regridder.hpp>
#include <halo/communicator.hpp>

#include "search_geometry.hpp"
namespace axis::distributed::detail {
struct ImportedGeometry {
    HostMesh mesh;
    std::vector<long long> global_cells;
    GeometryStatistics statistics;
};
ImportedGeometry selective_geometry(const HostMesh &, const HostMesh &, std::span<const long long>, const solver::RegridConfig &,
                                    const halo::Communicator &, const DistributedOptions &);
}  // namespace axis::distributed::detail
