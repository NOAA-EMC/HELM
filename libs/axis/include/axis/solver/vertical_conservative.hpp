// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <Kokkos_Core.hpp>
#include <cstddef>

namespace axis::solver {
enum class VerticalConservativeOrder { First, LimitedLinear };
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
    VerticalConservativeRegridder(ConstField source_interfaces, ConstField destination_interfaces, std::size_t columns, Mask source_mask = {},
                                  Mask destination_mask = {});
    void apply(ConstField source, Field destination, VerticalConservativeOrder order = VerticalConservativeOrder::First,
               VerticalConservativeNorm norm = VerticalConservativeNorm::DestinationMeasure, Field slope_workspace = {}) const;
    Integral covered_source_integral(ConstField source, VerticalConservativeOrder order = VerticalConservativeOrder::First,
                                     Field slope_workspace = {}) const;
    ConstField source_coverage() const {
        return source_fraction_;
    }
    ConstField destination_coverage() const {
        return destination_fraction_;
    }
    ConstField source_covered_measure() const {
        return source_covered_;
    }
    ConstField destination_covered_measure() const {
        return destination_covered_;
    }
    std::size_t columns() const {
        return columns_;
    }
    std::size_t source_layers() const {
        return ns_;
    }
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
