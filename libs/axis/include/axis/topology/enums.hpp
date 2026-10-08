// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_ENUMS_HPP
#define AXIS_TOPOLOGY_ENUMS_HPP

/// @file axis/topology/enums.hpp
/// @brief Defines fundamental topology-level enumerations used throughout the AXIS library.
///
/// This header contains the core enumerations (ElementType, StaggerLoc, CoordinateSystem)
/// which classify cell geometric shapes, field stagger locations, and coordinate reference systems.
/// CoordinateSystem is defined identically in axis/ingest/grid_descriptor.hpp to keep the
/// ingest contract self-contained without a topology dependency. Both share identical underlying
/// integer values and semantics. Topology-level code should prefer axis::topology::CoordinateSystem.

#include <cstdint>

/// @namespace axis::topology
/// @brief Contains the core spatial topology, grid, and mesh representations for the AXIS solver.
namespace axis::topology {

// ─────────────────────────────────────────────────────────────────────────────
// ElementType — cell topology classification for mixed-element CSR meshes.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Classification of cell element geometric types for mixed-element CSR meshes.
///
/// This enumeration defines the geometric topologies supported for cells within
/// an unstructured mesh, primarily focusing on 2-D surface meshes. Mixed-element
/// meshes store these classifications per-cell alongside the compressed sparse row
/// (CSR) connectivity.
enum class ElementType : std::uint8_t {
    /// @brief A 3-node triangular cell element.
    Triangle,
    /// @brief A 4-node quadrilateral cell element.
    Quadrilateral,
    /// @brief A general n-node polygonal cell element with arbitrary vertex count.
    Polygon
};

// ─────────────────────────────────────────────────────────────────────────────
// StaggerLoc — stagger location (center vs corner vs edge) within a cell.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Represents field stagger locations relative to cells or elements.
///
/// Used to designate where physical fields are discretized (e.g., cell center, cell vertex,
/// or edge midpoints) for modeling variables in finite volume (FV) or staggered grid methods.
enum class StaggerLoc : std::uint8_t {
    /// @brief Located at the cell centroid or center of mass (default for FV fields).
    Center,
    /// @brief Located at a cell vertex or corner node.
    Corner,
    /// @brief Located at a cell edge midpoint.
    Edge
};

// ─────────────────────────────────────────────────────────────────────────────
// CoordinateSystem — coordinate system for node/cell coordinates.
//
// Also defined identically in axis::ingest (grid_descriptor.hpp) to keep the
// ingest contract self-contained. Both definitions have the same underlying
// values (SphericalDeg=0, SphericalRad=1, Cartesian3D=2).
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Classification of spatial coordinate systems for node or cell coordinates.
///
/// Identifies the coordinate system and units used by spatial coordinate vectors inside
/// grid and mesh objects. For compatibility and to avoid cyclic dependencies, an identical
/// enumeration with matching values (SphericalDeg=0, SphericalRad=1, Cartesian3D=2) is also
/// defined in axis::ingest.
enum class CoordinateSystem : std::uint8_t {
    /// @brief Geographic longitude and latitude coordinates in degrees.
    SphericalDeg,
    /// @brief Geographic longitude and latitude coordinates in radians.
    SphericalRad,
    /// @brief Three-dimensional Cartesian coordinates (x, y, z) in meters.
    Cartesian3D
};

/// Controls how a structured grid obtains its cell corners when none were set.
enum class CornerPolicy : std::uint8_t { RequireExplicit, GaussianLatLon, RectilinearMidpoint, CurvilinearApproximate };

/// Explicit longitude seam metadata shared by grid topology builders.
struct LongitudePeriodicity {
    bool periodic{false};
    double period{360.0};
    bool increasing{true};
};

/// Describes how mesh boundary coordinates were obtained.
enum class GeometryProvenance : std::uint8_t { Unspecified, SourceAuthoritative, DeclaredGridModel, Reconstructed, Approximate };

/// Mathematical interpretation of polygon edges.
enum class BoundaryModel : std::uint8_t { Unspecified, GreatCircle, ConstantLatitude, CartesianStraight };

/// Mathematical model associated with the mesh's current cell-area values.
enum class AreaModel : std::uint8_t { Unspecified, SphericalExcess, ConstantLatitudeStrip, PlanarPolygon, SourceSupplied };

/// Queryable geometry semantics retained with the canonical CSR mesh.
struct GeometryMetadata {
    GeometryProvenance provenance{GeometryProvenance::Unspecified};
    BoundaryModel boundary_model{BoundaryModel::Unspecified};
    AreaModel area_model{AreaModel::Unspecified};
    LongitudePeriodicity longitude_periodicity{};
};

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_ENUMS_HPP
