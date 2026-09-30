// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP
#define AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP

/// @file axis/topology/named_grid_registry.hpp
/// @brief NamedGridRegistry — on-the-fly generation of standard global weather
///        grids (O-octahedral, F-regular, N-reduced Gaussian families) via
///        Kokkos parallel kernels with zero file I/O.
///
/// Generates standard ECMWF-style Gaussian grids by name string (e.g.
/// "O1280", "F128", "N320"). Each family has a registered generator that
/// builds an UnstructuredMesh entirely in memory using Kokkos parallel
/// kernels — no file access, no third-party parsing.
///
/// Grid families:
///   O — Octahedral reduced Gaussian (ECMWF convention): 2N latitude circles
///       total (N per hemisphere). Latitude circle j (0-indexed from nearest
///       pole) has 20 + 4*j points. Total points = 4*N*(N+9).
///   F — Regular (full) Gaussian: 2N latitude circles, each with 4N points.
///       Total points = 2N * 4N = 8N^2.
///   N — Reduced Gaussian: same as O (octahedral pattern, ECMWF convention).
///
/// All generation is deterministic: two calls with the same name produce
/// bitwise-identical results (Requirement 6.5).

#include <axis/topology/unstructured_mesh.hpp>
#include <string>
#include <vector>

namespace axis::topology {

/// @class NamedGridRegistry
/// @brief Registry of hardcoded mathematical generators for standard global weather
///        grids. Purely static interface — no instance state.
class NamedGridRegistry {
   public:
    /// @struct ParsedName
    /// @brief Parsed grid name containing family and grid number.
    ///
    /// For example, "O1280" parses into family 'O' and number 1280.
    struct ParsedName {
        char family;  ///< Family prefix: 'O', 'F', 'N', or 'R' (standard weather grid families).
        int number;   ///< Grid number (a positive integer, typically representing the Gaussian number N).
    };

    /// @brief Parse and validate a named-grid string.
    ///
    /// @param name The std::string representing the grid name to parse (e.g., "O1280", "F128", "N320").
    /// @return NamedGridRegistry::ParsedName A structure containing the parsed family character and grid number.
    /// @throw std::invalid_argument If the family prefix is unknown, the grid number is non-positive,
    ///                              or the string is otherwise malformed.
    [[nodiscard]] static ParsedName parse(const std::string &name);

    /// @brief Check whether a name string corresponds to a registered grid generator.
    ///
    /// This method returns false and does not throw if the grid name is malformed or if
    /// the grid family prefix is unregistered.
    ///
    /// @param name The std::string representing the grid name to check.
    /// @return bool True if a grid generator is registered for the specified name, false otherwise.
    [[nodiscard]] static bool is_registered(const std::string &name) noexcept;

    /// @brief Generate the named grid as an UnstructuredMesh in the specified MemorySpace.
    ///
    /// The mesh is built entirely via Kokkos parallel kernels with zero file
    /// I/O. Generation is deterministic: two calls with the same name produce
    /// bitwise-identical results.
    ///
    /// @tparam MemorySpace The Kokkos memory space in which the generated mesh should reside. Defaults to Kokkos::HostSpace.
    /// @param name The std::string representing the valid grid name string (e.g., "O1280").
    /// @return UnstructuredMesh<MemorySpace> The generated UnstructuredMesh consisting of quadrilateral cells.
    /// @throw std::invalid_argument If the grid name is unknown or malformed.
    template <class MemorySpace = Kokkos::HostSpace>
    [[nodiscard]] static UnstructuredMesh<MemorySpace> generate(const std::string &name);

    /// @brief Enumerate all registered family prefixes.
    ///
    /// @return std::vector<char> A sorted vector of registered grid family characters (currently {'F', 'G', 'N', 'O', 'R'}).
    [[nodiscard]] static std::vector<char> registered_families();

    /// @struct Layout
    /// @brief Structured (row/column) description of a named grid's cell ordering.
    ///
    /// The generators emit cells in a deterministic order that is *rectangular*
    /// for several families even though the mesh itself is stored unstructured
    /// (CSR). This descriptor lets clients reshape the flat ``ncol`` axis back
    /// into its natural 2-D (or 3-D, for cubed-sphere tiles) layout instead of
    /// shipping regridded data as an opaque cell list.
    ///
    /// @see cell_centers() for the authoritative per-cell coordinates.
    struct Layout {
        char family = '?';        ///< Family prefix as produced by parse().
        int number = 0;           ///< Grid number as produced by parse().
        std::size_t ni = 0;       ///< Cells per latitude row (0 => not rectangular).
        std::size_t nj = 0;       ///< Latitude rows (0 => not rectangular).
        std::size_t n_tiles = 1;  ///< Leading tile count (6 for cubed-sphere, else 1).
        bool row_uniform_lon = false;  ///< Every row shares the same longitudes (1-D coords valid).
        bool projected = false;   ///< NOAA GRIB projected grid (centers are x/y metres).
        std::string proj_string;  ///< PROJ string when projected, else empty.

        /// @brief True when cells fill a (n_tiles, nj, ni) rectangular array.
        [[nodiscard]] bool structured() const noexcept {
            return ni > 0 && nj > 0;
        }
    };

    /// @brief Describe the cell ordering of a named grid.
    ///
    /// @param name A valid grid name (e.g. "F128", "C96", "O64").
    /// @return Layout Structured descriptor; ``structured()`` is false for
    ///         reduced Gaussian (O/N) and projected GRIB grids, whose cells have
    ///         no rectangular arrangement.
    /// @throw std::invalid_argument If the grid name is unknown or malformed.
    [[nodiscard]] static Layout layout(const std::string &name);

    /// @brief Compute the geographic center of every cell, in engine cell order.
    ///
    /// Returned as a flat row-major array of ``(lon, lat)`` degree pairs, i.e.
    /// ``out.size() == 2 * n_cells`` with ``out[2*c]`` the longitude and
    /// ``out[2*c + 1]`` the latitude of cell ``c``. The ordering matches
    /// generate() exactly, which makes it the authority for reshaping a flat
    /// ``ncol`` result into (lat, lon) / (tile, j, i) coordinates.
    ///
    /// @param name A valid grid name (e.g. "F128", "C96", "O64").
    /// @return std::vector<double> 2 x n_cells interleaved lon/lat degrees.
    /// @throw std::invalid_argument If the grid name is unknown or malformed.
    /// @throw std::runtime_error If a projected grid is requested without PROJ support.
    [[nodiscard]] static std::vector<double> cell_centers(const std::string &name);
};

}  // namespace axis::topology

#endif  // AXIS_TOPOLOGY_NAMED_GRID_REGISTRY_HPP
