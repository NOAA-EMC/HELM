// SPDX-License-Identifier: Apache-2.0
#ifndef AXIS_DISTRIBUTED_REGRIDDER_HPP
#define AXIS_DISTRIBUTED_REGRIDDER_HPP

/// @file axis/distributed/regridder.hpp
/// @brief DistributedRegridder — MPI-distributed regridding over HALO exchange.

#include <mpi.h>

#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <memory>
#include <span>
#include <vector>

namespace axis::distributed {

/// @brief How source geometry reaches ranks that query it.
///
/// Selective imports only spatial candidates and reconstruction neighbors
/// through HALO. ReplicatedReference is an explicit opt-in that exchanges the
/// full reference geometry for comparison and debugging.
enum class GeometryStrategy { Selective, ReplicatedReference };

/// @brief Counters describing how much remote geometry a setup imported.
struct GeometryStatistics {
    std::size_t imported_cells{}, imported_vertices{}, estimated_import_bytes{};
    std::size_t peak_received_message_bytes{}, directory_incidents{};
};

/// @brief Tunable limits and batching for a distributed setup.
struct DistributedOptions {
    GeometryStrategy geometry = GeometryStrategy::Selective;
    // Caps apply to imported geometry and each temporary transfer, not caller
    // meshes, matrix storage, or allocator overhead. Fail collectively on excess.
    std::size_t max_import_bytes = 512ULL * 1024 * 1024;
    std::size_t max_message_bytes = 8ULL * 1024 * 1024;
    std::size_t query_batch_size = 128;
};
/// CPU distributed operator. Each rank supplies uniquely owned cells
/// and retains local destination rows. HALO owns every communication operation.
/// Setup imports only spatial candidates and reconstruction neighbors through
/// HALO. Replicated geometry is an explicit opt-in reference strategy.
/// Supply stable global source node IDs for linear reconstruction so adjacency
/// across partitions is exact. Cell IDs are rank-contiguous in owned-cell order.
/// Calls, including apply, are collective; empty partitions must participate.
/// A single instance must not be applied concurrently. Distinct instances own
/// independent communicators. Destroy instances before MPI_Finalize.
class DistributedRegridder {
    struct Impl;
    std::unique_ptr<Impl> impl_;

   public:
    /// @brief Build weights for this rank's owned source / local destination cells.
    ///
    /// Collective. Imports candidate geometry through HALO according to
    /// options.geometry and retains only local destination rows.
    /// @param owned_source          Cells uniquely owned by this rank.
    /// @param local_destination     Destination rows retained on this rank.
    /// @param config                Regridding method, masking, and extrapolation.
    /// @param comm                  MPI communicator shared by all participants.
    /// @param global_source_node_ids Stable global IDs for source vertices, required
    ///                               for exact cross-partition linear reconstruction.
    /// @param options               Geometry strategy and transfer caps.
    /// @throws std::invalid_argument for malformed meshes, mismatched ownership,
    ///         or missing global node IDs when reconstruction needs them.
    DistributedRegridder(const topology::UnstructuredMesh<Kokkos::HostSpace> &owned_source,
                         const topology::UnstructuredMesh<Kokkos::HostSpace> &local_destination, const solver::RegridConfig &config, MPI_Comm comm,
                         std::span<const long long> global_source_node_ids = {}, DistributedOptions options = {});
    ~DistributedRegridder();
    /// @brief Move construction keeps the communicator and imported geometry intact.
    DistributedRegridder(DistributedRegridder &&) noexcept;
    /// @brief Move assignment; the moved-from instance becomes empty.
    DistributedRegridder &operator=(DistributedRegridder &&) noexcept;
    DistributedRegridder(const DistributedRegridder &) = delete;
    DistributedRegridder &operator=(const DistributedRegridder &) = delete;
    /// @brief Apply the retained weights to one owned scalar field (collective).
    /// @param owned_source       Values on this rank's owned source cells.
    /// @param local_destination  Output on this rank's destination rows.
    void apply(field_view<const double, 1> owned_source, field_view<double, 1> local_destination);
    /// @brief Apply to multiple independent fields/levels packed along columns (collective).
    ///
    /// Columns are independent fields/levels, stored in AXIS layout_left.
    void apply(field_view<const double, 2> owned_source, field_view<double, 2> local_destination);
    /// @brief Regrid a (u, v) vector pair through the common east/north frame (collective).
    ///
    /// Rotate source components into the common east/north frame, apply the
    /// selected scalar reconstruction independently, rotate into destination
    /// frames. For limited reconstruction this is a componentwise limiter.
    void apply_vector(field_view<const double, 1> u, field_view<const double, 1> v, field_view<const double, 1> source_angles,
                      field_view<const double, 1> destination_angles, field_view<double, 1> destination_u, field_view<double, 1> destination_v);
    /// @brief Geometry imported by the last setup.
    [[nodiscard]] const GeometryStatistics &geometry_statistics() const noexcept;
    /// @brief Owned source cells on this rank.
    [[nodiscard]] std::size_t local_source_size() const noexcept;
    /// @brief Retained destination rows on this rank.
    [[nodiscard]] std::size_t local_destination_size() const noexcept;
    /// @brief This rank's destination-row slice of the operator (read-only).
    [[nodiscard]] const solver::InterpolationMatrix<Kokkos::HostSpace> &local_matrix() const noexcept;
    /// @brief Global source IDs whose values this rank must receive on apply.
    [[nodiscard]] const std::vector<long long> &requested_source_ids() const noexcept;
    /// Global participating fraction, reduced to uniquely owned source cells.
    /// Conservative methods only; other methods return an empty vector.
    [[nodiscard]] const std::vector<double> &owned_source_fractions() const noexcept;
};
}  // namespace axis::distributed
#endif
