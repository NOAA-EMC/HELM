// SPDX-License-Identifier: Apache-2.0
#ifndef AXIS_DISTRIBUTED_REGRIDDER_HPP
#define AXIS_DISTRIBUTED_REGRIDDER_HPP
#include <mpi.h>

#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <memory>
#include <span>
#include <vector>

namespace axis::distributed {
enum class GeometryStrategy { Selective, ReplicatedReference };
struct GeometryStatistics {
    std::size_t imported_cells{}, imported_vertices{}, estimated_import_bytes{};
    std::size_t peak_received_message_bytes{}, directory_incidents{};
};
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
    DistributedRegridder(const topology::UnstructuredMesh<Kokkos::HostSpace> &owned_source,
                         const topology::UnstructuredMesh<Kokkos::HostSpace> &local_destination, const solver::RegridConfig &config, MPI_Comm comm,
                         std::span<const long long> global_source_node_ids = {}, DistributedOptions options = {});
    ~DistributedRegridder();
    DistributedRegridder(DistributedRegridder &&) noexcept;
    DistributedRegridder &operator=(DistributedRegridder &&) noexcept;
    DistributedRegridder(const DistributedRegridder &) = delete;
    DistributedRegridder &operator=(const DistributedRegridder &) = delete;
    void apply(field_view<const double, 1> owned_source, field_view<double, 1> local_destination);
    /// Columns are independent fields/levels, stored in AXIS layout_left.
    void apply(field_view<const double, 2> owned_source, field_view<double, 2> local_destination);
    /// Rotate source components into the common east/north frame, apply the
    /// selected scalar reconstruction independently, rotate into destination
    /// frames. For limited reconstruction this is a componentwise limiter.
    void apply_vector(field_view<const double, 1> u, field_view<const double, 1> v, field_view<const double, 1> source_angles,
                      field_view<const double, 1> destination_angles, field_view<double, 1> destination_u, field_view<double, 1> destination_v);
    [[nodiscard]] const GeometryStatistics &geometry_statistics() const noexcept;
    [[nodiscard]] std::size_t local_source_size() const noexcept;
    [[nodiscard]] std::size_t local_destination_size() const noexcept;
    [[nodiscard]] const solver::InterpolationMatrix<Kokkos::HostSpace> &local_matrix() const noexcept;
    [[nodiscard]] const std::vector<long long> &requested_source_ids() const noexcept;
    /// Global participating fraction, reduced to uniquely owned source cells.
    /// Conservative methods only; other methods return an empty vector.
    [[nodiscard]] const std::vector<double> &owned_source_fractions() const noexcept;
};
}  // namespace axis::distributed
#endif
