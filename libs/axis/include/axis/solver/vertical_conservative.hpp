// SPDX-License-Identifier: Apache-2.0
#pragma once

/// @file axis/solver/vertical_conservative.hpp
/// @brief VerticalConservativeRegridder — finite-volume remap of layer means
///        between interface grids in a shared cumulative measure.

#include <Kokkos_Core.hpp>
#include <cstddef>

namespace axis::solver {

/// @brief Reconstruction order used inside each destination layer.
///
/// First remaps piecewise-constant layers; LimitedLinear integrates a
/// mean-preserving, bound-preserving linear reconstruction, flattening the
/// slope in boundary layers and layers touching a dry or masked neighbor.
enum class VerticalConservativeOrder { First, LimitedLinear };

/// @brief Which measure normalizes the remapped layer mean.
///
/// DestinationMeasure divides by the whole destination layer measure; the
/// result is the average over the destination layer. CoveredMeasure divides
/// by only the portion that received source fluid, so a partial overlap does
/// not dilute the value.
enum class VerticalConservativeNorm { DestinationMeasure, CoveredMeasure };

/// Reusable finite-volume remap of layer means in a common cumulative measure.
/// Interfaces: (1,N_layer+1) for shared geometry, or (N_column,N_layer+1).
/// Use volume/thickness coordinates for volume concentration, and carrier mass
/// coordinates for mass mixing ratio. The implementation never infers density.
/// Interfaces must be finite and monotone; duplicate interfaces are dry layers.
/// Masks: (N_column,N_layer), nonzero is wet. Geometry/masks are owned snapshots.
/// DestinationMeasure divides by whole-layer measure; CoveredMeasure divides
/// by participating measure. Unmapped/dry output and coverage are zero.
/// First preserves covered integrals to roundoff. LimitedLinear integrates a
/// mean-preserving, bounded linear reconstruction, flattening boundary layers
/// and layers adjacent to a dry/masked neighbor. Full-column conservation needs
/// equal wet domains. covered_source_integral includes partial-cell slope terms.
template <class MemorySpace>
class VerticalConservativeRegridder {
   public:
    using Field = Kokkos::View<double **, MemorySpace>;
    using ConstField = Kokkos::View<const double **, MemorySpace>;
    using Mask = Kokkos::View<const int **, MemorySpace>;
    using Integral = Kokkos::View<double *, MemorySpace>;
    /// @brief Compute the layer-to-layer overlap geometry once and cache it.
    /// @param source_interfaces      (1, N_src+1) shared or (columns, N_src+1) per-column.
    /// @param destination_interfaces (1, N_dst+1) shared or (columns, N_dst+1) per-column.
    /// @param columns                Number of vertical columns.
    /// @param source_mask            (columns, N_src) wet/dry flags; zero means all wet.
    /// @param destination_mask       (columns, N_dst) wet/dry flags; zero means all wet.
    /// @throws std::invalid_argument for non-monotone or non-finite interfaces,
    ///         inconsistent column counts, or shapes that disagree with columns.
    VerticalConservativeRegridder(ConstField source_interfaces, ConstField destination_interfaces, std::size_t columns, Mask source_mask = {},
                                  Mask destination_mask = {});
    /// @brief Remap layer means from the source to the destination grid.
    ///
    /// Thread-safe and device-resident: the geometry built at construction is
    /// read-only here, so one instance may be reused across columns and calls.
    /// @param source          (columns, N_src) source layer means.
    /// @param destination     (columns, N_dst) output layer means.
    /// @param order           Reconstruction order; see VerticalConservativeOrder.
    /// @param norm            Normalizing measure; see VerticalConservativeNorm.
    /// @param slope_workspace Optional (columns, N_src) scratch reused across calls.
    void apply(ConstField source, Field destination, VerticalConservativeOrder order = VerticalConservativeOrder::First,
               VerticalConservativeNorm norm = VerticalConservativeNorm::DestinationMeasure, Field slope_workspace = {}) const;
    /// @brief Per-column integral actually received from the source grid.
    ///
    /// Includes partial-cell slope terms under LimitedLinear, so it is the
    /// correct left-hand side when checking conservation against a
    /// CoveredMeasure destination.
    Integral covered_source_integral(ConstField source, VerticalConservativeOrder order = VerticalConservativeOrder::First,
                                     Field slope_workspace = {}) const;
    /// @brief Fraction of each source layer that reached a destination (0..1).
    ConstField source_coverage() const {
        return source_fraction_;
    }
    /// @brief Fraction of each destination layer supplied by a source (0..1).
    ConstField destination_coverage() const {
        return destination_fraction_;
    }
    /// @brief Measure of each source layer that participated in the transfer.
    ConstField source_covered_measure() const {
        return source_covered_;
    }
    /// @brief Measure of each destination layer that received source fluid.
    ConstField destination_covered_measure() const {
        return destination_covered_;
    }
    /// @brief Number of vertical columns handled by this instance.
    std::size_t columns() const {
        return columns_;
    }
    /// @brief Number of source layers.
    std::size_t source_layers() const {
        return ns_;
    }
    /// @brief Number of destination layers.
    std::size_t destination_layers() const {
        return nd_;
    }

   private:
    std::size_t columns_, ns_, nd_, source_rows_, destination_rows_, geometry_rows_;
    Field source_width_, destination_width_, lower_source_, lower_destination_;
    Field overlap_, moment_, source_covered_, destination_covered_, source_moment_;
    Field source_fraction_, destination_fraction_;
    Kokkos::View<int **, MemorySpace> source_mask_, destination_mask_;
    Kokkos::View<int *, MemorySpace> source_direction_;
    Kokkos::View<std::size_t **, MemorySpace> source_index_, destination_index_;
    Kokkos::View<std::size_t *, MemorySpace> counts_;
    void validate_field(ConstField source) const;
    Field slopes(ConstField source, VerticalConservativeOrder order, Field workspace) const;
};
}  // namespace axis::solver
