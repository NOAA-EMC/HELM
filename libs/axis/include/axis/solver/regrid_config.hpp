// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_REGRID_CONFIG_HPP
#define AXIS_SOLVER_REGRID_CONFIG_HPP

/// @file axis/solver/regrid_config.hpp
/// @brief Solver configuration enumerations and the RegridConfig struct.
///
/// Defines InterpolationMethod, NormType, LineType, and UnmappedAction enums
/// along with the RegridConfig struct that captures all runtime options for
/// weight generation. Header-only — no associated .cpp compilation unit.

#include <axis/types.hpp>
#include <cstdint>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// InterpolationMethod — the spatial interpolation algorithm.
// ─────────────────────────────────────────────────────────────────────────────

/// Selects the interpolation method used by WeightGenerator.
enum class InterpolationMethod : std::uint8_t {
    Bilinear,              ///< True bilinear: point-in-cell location + barycentric weights
    NearestNeighbor,       ///< Nearest source cell centroid to each destination point (weight = 1.0)
    Bicubic,               ///< Bicubic: 4×4 stencil cubic interpolation (CDO remapbic equivalent)
    Patch,                 ///< Patch recovery: least-squares polynomial fit over local patch (ESMF REGRID_METHOD_PATCH)
    Conservative1stOrder,  ///< Area-weighted first-order conservative (overlap-based via ArborX BVH + Sutherland-Hodgman)
    Conservative2ndOrder   ///< Second-order conservative: linear gradient reconstruction + overlap integrals (requires GradientReconstructor)
};

// ─────────────────────────────────────────────────────────────────────────────
// NormType — normalization semantics for conservative weights.
// ─────────────────────────────────────────────────────────────────────────────

/// Controls how partially-covered destination cells are normalized.
/// Matches ESMF normalization semantics.
enum class NormType : std::uint8_t {
    DstArea,  ///< Unnormalized: dst_raw = frac_b * dst_true (default)
    FracArea  ///< Fraction baked in: apply yields true value directly
};

// ─────────────────────────────────────────────────────────────────────────────
// LineType — geometry of paths between grid points.
// ─────────────────────────────────────────────────────────────────────────────

/// Determines how lines between grid points are computed for overlap detection.
enum class LineType : std::uint8_t {
    Cartesian,   ///< Straight lines in projected (Cartesian) space
    GreatCircle  ///< Geodesic arcs on the sphere
};

// ─────────────────────────────────────────────────────────────────────────────
// UnmappedAction — behavior for destination cells with no source coverage.
// ─────────────────────────────────────────────────────────────────────────────

/// Controls what happens when a destination cell has no source overlap.
enum class UnmappedAction : std::uint8_t {
    Error,   ///< Throw std::runtime_error identifying the unmapped index
    Ignore,  ///< Leave destination value at zero; no entry in the matrix
    Mask     ///< Like Ignore (zero row), but the matrix records an unmapped_mask
             ///< (per-dst 0/1) so callers can distinguish "no data" from "zero".
             ///< The destination is never silently filled with a fake value.
};

// ─────────────────────────────────────────────────────────────────────────────
// ExtrapolationAction — behavior for completely dry destination cells.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Controls extrapolation behavior for completely masked destination points.
enum class ExtrapolationAction : std::uint8_t {
    None,       ///< Leave row zero (unmapped) or trigger error policy.
    NearestWet  ///< Find closest wet source cell and map it with a weight of 1.0 (Default).
};

// ─────────────────────────────────────────────────────────────────────────────
// RegridConfig — aggregate configuration for weight generation.
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration struct capturing all runtime options for weight generation.
/// Passed by value to WeightGenerator::generate.
struct RegridConfig {
    InterpolationMethod method = InterpolationMethod::Bilinear;
    NormType norm_type = NormType::DstArea;
    LineType line_type = LineType::GreatCircle;
    UnmappedAction unmapped = UnmappedAction::Ignore;
    bool use_limiter = false;  ///< Enable Barth-Jespersen monotonicity limiter (Conservative2ndOrder only)

    /// @brief Extrapolation method when a destination cell has zero wet source overlaps.
    ExtrapolationAction extrap_method = ExtrapolationAction::NearestWet;

    /// @brief Explicit global-longitude periodicity override for the source grid.
    ///
    /// The rectilinear fast paths auto-detect periodicity when the source
    /// longitude span is ≈ 360°. This tri-state lets a caller override that
    /// detection when the span is ambiguous (e.g. a global grid whose edge
    /// columns are duplicated, or a regional grid that spans 360° but must
    /// NOT wrap). Values: 0 = auto-detect (default); 1 = force periodic;
    /// -1 = force non-periodic.
    std::int8_t periodic = 0;

    /// @brief Optional fit-time destination-cell mask [n_dst] (non-owning).
    ///
    /// When extent(0) == n_dst, destination cells with dst_mask == 0 have
    /// every matrix row zeroed (all COO entries dropped) and are recorded in
    /// the matrix's unmapped mask — they are "no data", never silent zeros.
    /// Empty view (default) = no destination masking.
    field_view<const int, 1> dst_mask{};
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_REGRID_CONFIG_HPP
