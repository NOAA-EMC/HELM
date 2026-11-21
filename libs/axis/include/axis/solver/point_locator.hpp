// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <array>
#include <axis/topology/unstructured_mesh.hpp>
#include <memory>
#include <span>
#include <vector>

namespace axis::distributed {
class DistributedPointLocator;
}

namespace axis::solver {
/// Caller-owned point rows. IDs must be nonnegative and unique; coordinates
/// are [point, dimension], LayoutLeft. Cartesian accepts 2 or 3 dimensions;
/// spherical accepts longitude/latitude in degrees or radians (third ignored).
struct PointSetView {
    std::span<const index_t> ids;
    field_view<const double, 2> coordinates;
    topology::CoordinateSystem coordinate_system = topology::CoordinateSystem::SphericalDeg;
};
enum class PointLocationStatus { Unmapped, Interior, Boundary };
struct PointLocation {
    index_t point_id = -1, cell_id = -1;
    int owner_rank = -1;
    index_t owner_local_index = -1;
    PointLocationStatus status = PointLocationStatus::Unmapped;
};
/// Tolerance is a Cartesian distance or spherical angle in radians.
struct PointLocationOptions {
    double tolerance = 1e-10;
    std::size_t max_import_bytes = 512ULL * 1024 * 1024;
    std::size_t max_message_bytes = 8ULL * 1024 * 1024;
    std::size_t query_batch_size = 128;
};
/// CPU locator with retained geometry; rebuild after mesh changes. Convex
/// planar/great-circle polygons and constant-latitude rectangles only.
/// Explicit boundary metadata is required. Masks do not change containment.
class PointLocator {
    struct Impl;
    std::shared_ptr<const Impl> impl_;

   public:
    PointLocator(const topology::UnstructuredMesh<Kokkos::HostSpace> &mesh, std::span<const index_t> cell_ids, PointLocationOptions options = {});
    [[nodiscard]] std::vector<PointLocation> locate(PointSetView points) const;
    /// Conservative canonical xyz bounds, including tolerance; empty if no cells.
    [[nodiscard]] std::array<double, 6> bounds() const;

   private:
    friend class axis::distributed::DistributedPointLocator;
    /// Canonical query representation: Cartesian xyz or unit-sphere xyz.
    [[nodiscard]] std::vector<std::array<double, 3>> canonical_points(PointSetView points) const;
    /// Internal routing primitive; returns the smallest containing stable ID.
    [[nodiscard]] std::vector<PointLocation> locate_canonical(std::span<const std::array<double, 3>> points) const;
};
}  // namespace axis::solver
