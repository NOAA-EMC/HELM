// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_WEIGHT_GENERATOR_HPP
#define AXIS_SOLVER_WEIGHT_GENERATOR_HPP

/// @file axis/solver/weight_generator.hpp
/// @brief Static interface for computing interpolation weights from source and
///        destination meshes.
///
/// WeightGenerator is a stateless facade (no instance state). It provides:
///   - generate() (single-rank): produces an InterpolationMatrix from two
///     UnstructuredMesh instances and a RegridConfig.
///   - generate() (distributed): additionally accepts global_ids, owner maps,
///     and outputs a HaloPattern for off-rank source gather.
///
/// Supported interpolation methods:
///   - Bilinear: inverse-distance / barycentric weights within the containing
///     source cell. Source cell areas (area_a) are set to 0.0 (ESMF convention).
///   - Conservative1stOrder: area-weighted overlap. Weights satisfy first-order
///     conservation and are non-negative. Supports DstArea and FracArea
///     normalization.
///
/// @tparam MemorySpace Kokkos memory space for internal computation and output.

#include <Kokkos_Core.hpp>
#include <axis/solver/halo_pattern.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <utility>
#include <vector>

namespace axis::solver {

/// A collection of point samples, independent of cell boundary geometry.
/// Coordinates are stored as [n_samples, 2] in (x/lon, y/lat) order.
struct SampleGrid {
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> positions;
    /// Optional per-sample activity flags: 1 is active and 0 is masked.
    /// Nearest-neighbor honors both source and destination masks. Bilinear and
    /// bicubic currently reject masked stencils because their stencil kernels
    /// do not remove masked center samples.
    Kokkos::View<int *, Kokkos::HostSpace> mask;
    /// Optional stable identifiers. On the source grid these become matrix
    /// column indices in [0, global_source_count); on the destination grid
    /// they become matrix row indices in [0, global_destination_count). Each
    /// list must be unique. Empty uses local row-major indices.
    Kokkos::View<index_t *, Kokkos::HostSpace> global_indices;
    std::size_t global_source_count{0};
    std::size_t global_destination_count{0};
    /// Optional logical shape. Both values must be zero (unstructured samples)
    /// or both positive with nx*ny equal to the number of positions.
    std::size_t nx{0};
    std::size_t ny{0};
};

/// Stateless weight-generation facade.
///
/// All methods are static — no instance state is needed. This mirrors the
/// factory pattern used by MeshFactory in the topology namespace.
class WeightGenerator {
   public:
    WeightGenerator() = delete;

    /// Generate point interpolation weights without requiring polygon cells.
    /// Supports nearest, bilinear and bicubic host-space methods. The centers
    /// in SampleGrid are the interpolation points; no cell corners are inferred.
    static InterpolationMatrix<Kokkos::HostSpace> generate_from_samples(const SampleGrid &src, const SampleGrid &dst,
                                                                        topology::CoordinateSystem coordinate_system, const RegridConfig &config);

    // ─────────────────────────────────────────────────────────────────────────
    // Single-rank generate
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute interpolation weights between a source and destination mesh.
    ///
    /// @tparam MemorySpace Kokkos memory space (default Kokkos::HostSpace)
    /// @param src_mesh  Source unstructured mesh
    /// @param dst_mesh  Destination unstructured mesh
    /// @param config    Regridding configuration (method, norm, unmapped, etc.)
    /// @return          Sparse interpolation matrix in COO form
    ///
    /// @throws std::invalid_argument if src/dst CoordinateSystem values differ
    /// @throws std::runtime_error    if unmapped==Error and any dst cell is uncovered
    template <class MemorySpace = Kokkos::HostSpace>
    static InterpolationMatrix<MemorySpace> generate(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                     const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config);

    // ─────────────────────────────────────────────────────────────────────────
    // Distributed generate (publishes HaloPattern for off-rank gather)
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute interpolation weights in distributed mode.
    ///
    /// In addition to producing a local InterpolationMatrix, this overload
    /// inspects factor columns against local ownership and publishes a
    /// HaloPattern describing which off-rank source cells are needed.
    ///
    /// @tparam MemorySpace   Kokkos memory space
    /// @param src_mesh       Local partition of the source mesh
    /// @param dst_mesh       Local partition of the destination mesh
    /// @param config         Regridding configuration
    /// @param src_global_ids Global cell IDs for local source cells [n_src_local]
    /// @param dst_global_ids Global cell IDs for local destination cells [n_dst_local]
    /// @param owner_of_src   Mapping from global source cell ID to owning rank
    ///
    /// @return Pair of (InterpolationMatrix, HaloPattern)
    ///
    /// @throws std::invalid_argument if src/dst CoordinateSystem values differ
    /// @throws std::runtime_error    if unmapped==Error and any dst cell is uncovered
    template <class MemorySpace = Kokkos::HostSpace>
    static std::pair<InterpolationMatrix<MemorySpace>, HaloPattern> generate(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                             const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                                             const RegridConfig &config,
                                                                             Kokkos::View<const index_t *, MemorySpace> src_global_ids,
                                                                             Kokkos::View<const index_t *, MemorySpace> dst_global_ids,
                                                                             const std::vector<int> &owner_of_src);

   private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal dispatch helpers
    // ─────────────────────────────────────────────────────────────────────────

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace> generate_bilinear(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                              const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace> generate_nearest(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                             const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace> generate_bicubic(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                             const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace> generate_patch(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                           const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace> generate_conservative(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                  const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                                  const RegridConfig &config);

    template <class MemorySpace>
    static InterpolationMatrix<MemorySpace> generate_conservative_2nd_order(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                            const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                                            const RegridConfig &config);
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_WEIGHT_GENERATOR_HPP
