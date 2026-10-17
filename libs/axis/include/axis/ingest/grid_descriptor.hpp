// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_INGEST_GRID_DESCRIPTOR_HPP
#define AXIS_INGEST_GRID_DESCRIPTOR_HPP

/// @file axis/ingest/grid_descriptor.hpp
/// @brief The public plain-data ingest contract for AXIS.
///
/// GridDescriptor is the single, standard, plain-data value type through which
/// every file-backed grid enters AXIS. A producer (AMIO today, Python/pybind11
/// tomorrow) populates this descriptor; MeshFactory::from_descriptor consumes it.
///
/// The descriptor holds only plain enums, plain scalar fields, standard library
/// types, and non-owning std::mdspan<layout_left> buffer views. It contains
/// NO Kokkos type, NO AMIO type, NO eckit type, NO file handle, and NO file path.
/// This "trivially mirrorable in Python" property is an explicit design goal.
///
/// Header-only: no associated .cpp compilation unit.

#include <axis/types.hpp>
#include <cstddef>
#include <cstdint>
#include <string>

namespace axis::ingest {

// ─────────────────────────────────────────────────────────────────────────────
// ConventionKind — how to interpret the descriptor's metadata and buffers.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief How to interpret the descriptor's metadata and buffers.
///
/// The producer sets this after detecting the file's grid convention; AXIS branches
/// on it ONCE inside MeshFactory::from_descriptor.
enum class ConventionKind : std::uint8_t {
    CF,              ///< CF-conventions structured grid (grid_mapping + coord vars)
    UGRID,           ///< UGRID unstructured mesh (node/edge/face topology)
    GRIB,            ///< GRIB2 grid-description keys (gridType, Ni/Nj, Gaussian N, ...)
    Projected,       ///< Regular grid in a PROJ/proj4 projection (AXIS transforms via PROJ)
    NamedGrid,       ///< A registry token (e.g. "O1280"); AXIS generates it in-memory
    GridRules,       ///< Rule parameters (kind, bbox, resolution, gaussian_n); AXIS generates
    ReducedGaussian  ///< Ragged latitude/longitude rows with per-row longitude boundaries
};

// ─────────────────────────────────────────────────────────────────────────────
// CoordinateSystem — shared with topology; defined here so the ingest contract
// is self-contained (no dependency on topology headers).
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Coordinate system for node/cell coordinates.
enum class CoordinateSystem : std::uint8_t {
    SphericalDeg,  ///< Geographic lon/lat in degrees
    SphericalRad,  ///< Geographic lon/lat in radians
    Cartesian3D    ///< 3-D Cartesian (x, y, z)
};

// ─────────────────────────────────────────────────────────────────────────────
// Convention-specific metadata structs — plain data only.
// Only the member matching GridDescriptor::convention is meaningful.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief CF grid_mapping parameters (cf-conventions ch.5.6).
struct CfParams {
    /// @brief Name of the grid mapping variable projection (e.g. "latitude_longitude", "lambert_conformal_conic").
    std::string grid_mapping_name;
    /// @brief Optional sphere Earth radius in meters (0.0 if unset).
    double earth_radius{0.0};
    /// @brief Optional semi-major axis of the ellipsoid in meters (0.0 if unset).
    double semi_major{0.0};
    /// @brief Optional semi-minor axis of the ellipsoid in meters (0.0 if unset).
    double semi_minor{0.0};
    /// @brief Optional inverse flattening of the ellipsoid (0.0 if unset).
    double inverse_flattening{0.0};
};

/// @brief UGRID topology attributes (ugrid-conventions).
struct UgridParams {
    /// @brief Topology dimension of the mesh, usually 2 (surface) or 3 (volume).
    int topology_dimension{2};
    /// @brief Connectivity array index offset (0-based or 1-based index).
    int start_index{0};
    /// @brief Name of the mesh topology container variable.
    std::string mesh_name;
};

/// @brief GRIB grid-description keys (as surfaced by AMIO's g2c driver).
struct GribParams {
    /// @brief GRIB grid type name (e.g. "regular_ll", "regular_gg", "reduced_gg").
    std::string grid_type;
    /// @brief Number of grid points along a parallel of latitude (Ni).
    std::int64_t ni{0};
    /// @brief Number of grid points along a meridian of longitude (Nj).
    std::int64_t nj{0};
    /// @brief Gaussian truncation parameter N (number of latitude rows between a pole and the equator; 0 if not Gaussian).
    std::int64_t gaussian_n{0};
    /// @brief GRIB scanning mode flag byte.
    std::uint8_t scanning_mode{0};
    /// @brief GRIB shape of the earth code identifier.
    int earth_shape_code{0};
};

/// @brief Projected-grid parameters. AXIS transforms via PROJ (its only optional dep).
struct ProjectedParams {
    /// @brief PROJ/proj4 spatial projection definition string (e.g., "+proj=lcc +lat_1=33 ...").
    std::string proj_string;
};

/// @brief Named-grid parameters: just the registry token. AXIS generates in-memory.
struct NamedGridParams {
    /// @brief Registered grid name token (e.g. "O1280", "F128", "N320").
    std::string name;
};

/// Metadata for a reduced-Gaussian descriptor. The row count determines the
/// required latitude-boundary and row-cell-offset lengths.
struct ReducedGaussianParams {
    std::size_t n_rows{0};
};

/// @brief Rule-based parameters (a GridRules YAML is parsed by the PRODUCER into these
/// plain fields; AXIS never parses YAML or JSON).
struct GridRulesParams {
    /// @brief Rule kind identifier (e.g., "RegularLatLon", "GaussianRegular").
    std::string kind;
    /// @brief Minimum coordinate value in the X (longitude) direction.
    double min_x{-180.0};
    /// @brief Maximum coordinate value in the X (longitude) direction.
    double max_x{180.0};
    /// @brief Minimum coordinate value in the Y (latitude) direction.
    double min_y{-90.0};
    /// @brief Maximum coordinate value in the Y (latitude) direction.
    double max_y{90.0};
    /// @brief Grid resolution or spacing increment in the X direction.
    double r_x{1.0};
    /// @brief Grid resolution or spacing increment in the Y direction.
    double r_y{1.0};
    /// @brief Gaussian truncation parameter N (0 if not Gaussian).
    std::int64_t gaussian_n{0};
    /// @brief Optional proj4 spatial projection definition string used when kind is "Projected".
    std::string proj_string;
};

// ─────────────────────────────────────────────────────────────────────────────
// BufferViews — non-owning layout_left mdspan views over decoded arrays.
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Non-owning layout_left views over decoded source arrays.
///
/// The consumer/producer populates these; AXIS copies out of them into the target Kokkos memory space.
/// Unused views should be left default-constructed with size 0.
struct BufferViews {
    // ── Structured (CF / GRIB): 1-D center coordinate arrays ────────────────
    /// @brief Structured Grid longitude/X centers array of shape [n_points] or similar 1-D layout.
    field_view<const double, 1> center_x{};
    /// @brief Structured Grid latitude/Y centers array of shape [n_points] or similar 1-D layout.
    field_view<const double, 1> center_y{};

    // ── Structured: corner coordinate arrays (optional) ─────────────────────
    /// @brief Structured Grid corner vertex longitude/X coordinates.
    field_view<const double, 1> corner_x{};
    /// @brief Structured Grid corner vertex latitude/Y coordinates.
    field_view<const double, 1> corner_y{};
    /// Authoritative separable structured-cell bounds (ni+1 and nj+1).
    field_view<const double, 1> longitude_bounds{};
    field_view<const double, 1> latitude_bounds{};
    /// Optional Gaussian quadrature weights ordered with structured rows; sum to 2.
    field_view<const double, 1> gaussian_latitude_weights{};
    /// Reduced Gaussian ragged row cell offsets (n_rows+1) and flattened
    /// per-row longitude boundaries (total_cells+n_rows).
    field_view<const index_t, 1> row_cell_offsets{};
    field_view<const double, 1> row_longitude_boundaries{};

    // ── Unstructured (UGRID): node coordinates + CSR connectivity ───────────
    /// @brief Unstructured Grid (UGRID) node coordinates of shape [n_nodes, ndim].
    field_view<const double, 2> node_coords{};
    /// @brief Unstructured Grid (UGRID) CSR connectivity offsets of length [n_cells + 1].
    field_view<const index_t, 1> conn_offsets{};
    /// @brief Unstructured Grid (UGRID) CSR column node indices mapping cells to nodes.
    field_view<const index_t, 1> conn_indices{};

    // ── Optional per-cell metadata (any convention) ─────────────────────────
    /// @brief Precomputed cell areas, size [n_cells] (optional).
    field_view<const double, 1> cell_areas{};
    /// @brief Cell active/validity mask (0 = masked, 1 = active), size [n_cells] (optional).
    field_view<const int, 1> cell_mask{};

    // ── Structured grid dimensions (for CF / GRIB) ──────────────────────────
    /// @brief Number of grid points along structured dimension i (0 for unstructured meshes).
    std::size_t ni{0};
    /// @brief Number of grid points along structured dimension j (0 for unstructured meshes).
    std::size_t nj{0};
};

// ─────────────────────────────────────────────────────────────────────────────
// GridDescriptor — THE INGEST CONTRACT
// ─────────────────────────────────────────────────────────────────────────────

/// @brief THE INGEST CONTRACT.
///
/// A producer fills this; MeshFactory::from_descriptor consumes it.
/// Pure data: copyable, no destructor logic, no owned resources.
///
/// Only the convention-specific params member matching `kind` is meaningful;
/// the rest are default-constructed. This flat layout (instead of std::variant)
/// is chosen so a Python layer can populate by name without variant gymnastics.
struct GridDescriptor {
    /// @brief Which grid convention the producer detected in the file.
    ConventionKind kind{ConventionKind::CF};

    /// @brief Coordinate system for the buffer data.
    CoordinateSystem coord_system{CoordinateSystem::SphericalDeg};

    /// Explicit longitude seam declaration for structured geographic inputs.
    /// Periodicity is never inferred from center spacing. Set the period in the
    /// coordinate unit (360 for degrees, 2*pi for radians). A zero value uses
    /// that unit-specific default.
    bool longitude_periodic{false};
    double longitude_period{0.0};
    bool longitude_increasing{true};
    /// Declare that structured centers are curvilinear rather than separable.
    /// Their reconstructed corners are approximate and should not be used for
    /// strict conservative remapping without authoritative geometry.
    bool curvilinear_centers{false};

    // ── Convention-specific metadata (only one is meaningful per `kind`) ─────
    /// @brief CF-conventions structured grid metadata params.
    CfParams cf{};
    /// @brief UGRID unstructured mesh metadata params.
    UgridParams ugrid{};
    /// @brief GRIB2 grid-description metadata params.
    GribParams grib{};
    /// @brief Projected grid parameters.
    ProjectedParams projected{};
    /// @brief Named grid metadata parameter token.
    NamedGridParams named_grid{};
    ReducedGaussianParams reduced_gaussian{};
    /// @brief Rule-based mesh generation parameters.
    GridRulesParams grid_rules{};

    // ── Decoded buffers (empty for NamedGrid / GridRules, generated by AXIS) ─
    /// @brief Plain non-owning views of the decoded buffers.
    BufferViews buffers{};
};

}  // namespace axis::ingest

#endif  // AXIS_INGEST_GRID_DESCRIPTOR_HPP
