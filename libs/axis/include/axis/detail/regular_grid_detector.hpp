// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_REGULAR_GRID_DETECTOR_HPP
#define AXIS_DETAIL_REGULAR_GRID_DETECTOR_HPP

/// @file axis/detail/regular_grid_detector.hpp
/// @brief Detect whether an UnstructuredMesh originated from a regular lat-lon
///        grid with uniform spacing.
///
/// Provides:
///   - RegularGridInfo: device-portable struct holding grid dimensions and spacing
///   - detect_regular_grid(): host-side detection examining mesh connectivity and
///     coordinate uniformity
///
/// Detection runs on host but RegularGridInfo is device-portable (POD struct).
/// No heap allocation in the struct itself (HELM Law #2).

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/topology/unstructured_mesh.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// RegularGridInfo — POD struct describing a regular lat-lon grid
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Describes a regular lat-lon grid detected from an UnstructuredMesh.
///
/// If is_regular is true, the mesh cells form a logically rectangular ni × nj
/// grid with uniform spacing delta_lon and delta_lat. This enables the
/// rectangle fast-path and trig cache optimizations.
///
/// This struct is a POD type — safe to copy to device memory spaces.
struct RegularGridInfo {
    bool is_regular{false};
    double lon_min{0.0};
    double lon_max{0.0};
    double delta_lon{0.0};
    double lat_min{0.0};
    double lat_max{0.0};
    double delta_lat{0.0};
    std::size_t ni{0};  ///< Number of unique longitude cell centers
    std::size_t nj{0};  ///< Number of unique latitude cell centers
};

// ─────────────────────────────────────────────────────────────────────────────
// RectilinearGridInfo — Describes a non-uniform or uniform rectilinear grid
// ─────────────────────────────────────────────────────────────────────────────

struct RectilinearGridInfo {
    bool is_rectilinear{false};
    std::size_t ni{0};
    std::size_t nj{0};
    Kokkos::View<double *, Kokkos::HostSpace> unique_lons;
    Kokkos::View<double *, Kokkos::HostSpace> unique_lats;
};

/// @brief Verify that an all-quad mesh uses the canonical structured connectivity.
///
/// A logically rectangular `ni × nj` grid produced by `to_unstructured()` stores
/// cell `c` at row-major position `i + j * ni` (with `i = c % ni`, `j = c / ni`)
/// and its four corners in counter-clockwise order starting from the
/// south-west node `i + j * (ni + 1)`. The arithmetic fast paths enabled by
/// `detect_regular_grid()` and `detect_rectilinear_grid()` index source cells
/// and nodes with exactly this convention, so a mesh whose coordinate sets look
/// regular but whose connectivity is permuted, reversed, or contains duplicated
/// cells must be rejected: the coordinate-set checks alone cannot see such a
/// defect, but the fast path would silently read the wrong neighbours.
///
/// @param offsets CSR cell-to-node offsets of length `n_cells + 1`.
/// @param indices CSR cell-to-node indices of length `n_cells * 4`.
/// @param n_cells Number of cells.
/// @param ni      Number of columns (cells along x).
/// @param nj      Number of rows (cells along y).
/// @return true when every cell follows the canonical node ordering.
template <class MemorySpace>
bool has_canonical_connectivity(const Kokkos::View<index_t *, MemorySpace> &offsets, const Kokkos::View<index_t *, MemorySpace> &indices,
                                std::size_t n_cells, std::size_t ni, std::size_t nj) {
    for (std::size_t c = 0; c < n_cells; ++c) {
        if (offsets(c) != static_cast<index_t>(c * 4)) return false;
        const std::size_t i = c % ni;
        const std::size_t j = c / ni;
        const std::size_t nip1 = ni + 1;
        const index_t expected[4] = {static_cast<index_t>(i + j * nip1),             // bottom-left
                                     static_cast<index_t>((i + 1) + j * nip1),       // bottom-right
                                     static_cast<index_t>((i + 1) + (j + 1) * nip1),  // top-right
                                     static_cast<index_t>(i + (j + 1) * nip1)};       // top-left
        for (std::size_t k = 0; k < 4; ++k) {
            if (indices(c * 4 + k) != expected[k]) return false;
        }
    }
    return offsets(n_cells) == static_cast<index_t>(n_cells * 4);
}

template <class MemorySpace>
RectilinearGridInfo detect_rectilinear_grid(const topology::UnstructuredMesh<MemorySpace> &mesh) {
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible,
                  "detect_rectilinear_grid() requires a host-accessible mesh");

    RectilinearGridInfo info;

    const auto n_cells = mesh.n_cells();
    if (n_cells == 0) {
        return info;
    }

    const auto &offsets = mesh.conn_offsets_view();
    const auto &indices = mesh.conn_indices_view();
    const auto &coords = mesh.node_coords_view();

    for (std::size_t c = 0; c < n_cells; ++c) {
        auto start = offsets(c);
        auto end = offsets(c + 1);
        if ((end - start) != 4) {
            return info;  // Must be all quads
        }
    }

    const auto n_nodes = mesh.n_nodes();
    std::vector<double> all_lons;
    std::vector<double> all_lats;
    all_lons.reserve(n_nodes);
    all_lats.reserve(n_nodes);

    for (std::size_t i = 0; i < n_nodes; ++i) {
        all_lons.push_back(coords(i, 0));
        all_lats.push_back(coords(i, 1));
    }

    std::sort(all_lons.begin(), all_lons.end());
    std::sort(all_lats.begin(), all_lats.end());

    constexpr double unique_tol = 1.0e-12;
    auto unique_filter = [&](std::vector<double> &sorted) -> std::vector<double> {
        std::vector<double> unique_vals;
        if (sorted.empty()) return unique_vals;
        unique_vals.push_back(sorted[0]);
        for (std::size_t i = 1; i < sorted.size(); ++i) {
            if (std::abs(sorted[i] - unique_vals.back()) > unique_tol) {
                unique_vals.push_back(sorted[i]);
            }
        }
        return unique_vals;
    };

    auto unique_lons = unique_filter(all_lons);
    auto unique_lats = unique_filter(all_lats);

    if (unique_lons.size() < 2 || unique_lats.size() < 2) {
        return info;
    }

    const std::size_t ni = unique_lons.size() - 1;
    const std::size_t nj = unique_lats.size() - 1;

    if (ni * nj != n_cells) {
        return info;  // Not a rectilinear structured layout
    }

    // The coordinate sets alone cannot detect a permuted, reversed, or
    // duplicated connectivity layout, which would make the arithmetic fast
    // paths read the wrong neighbours. Require the canonical ordering.
    if (!has_canonical_connectivity(offsets, indices, n_cells, ni, nj)) {
        return info;
    }

    // Populate RectilinearGridInfo
    info.is_rectilinear = true;
    info.ni = ni;
    info.nj = nj;

    info.unique_lons = Kokkos::View<double *, Kokkos::HostSpace>("unique_lons", unique_lons.size());
    info.unique_lats = Kokkos::View<double *, Kokkos::HostSpace>("unique_lats", unique_lats.size());

    for (std::size_t i = 0; i < unique_lons.size(); ++i) {
        info.unique_lons(i) = unique_lons[i];
    }
    for (std::size_t j = 0; j < unique_lats.size(); ++j) {
        info.unique_lats(j) = unique_lats[j];
    }

    return info;
}

// ─────────────────────────────────────────────────────────────────────────────
// detect_regular_grid — host-side detection algorithm
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Detect whether an UnstructuredMesh originated from a regular lat-lon
///        grid.
///
/// Checks:
///   1. All cells are quads (4 vertices each)
///   2. Extract unique sorted lon and lat values from cell vertices
///   3. Compute deltas between consecutive unique values
///   4. Verify max(delta) - min(delta) < 1e-10 * mean(delta) for both axes
///
/// Detection runs on host. For device-space meshes, the caller must first
/// mirror to host before calling this function.
///
/// @tparam MemorySpace Kokkos memory space of the mesh (must be host-accessible)
/// @param mesh The unstructured mesh to examine
/// @return RegularGridInfo with is_regular=true if the mesh is a uniform grid
template <class MemorySpace>
RegularGridInfo detect_regular_grid(const topology::UnstructuredMesh<MemorySpace> &mesh) {
    static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, MemorySpace>::accessible, "detect_regular_grid() requires a host-accessible mesh");

    RegularGridInfo info;

    const auto n_cells = mesh.n_cells();
    if (n_cells == 0) {
        return info;
    }

    // Access raw Kokkos views (host-accessible)
    const auto &offsets = mesh.conn_offsets_view();
    const auto &indices = mesh.conn_indices_view();
    const auto &coords = mesh.node_coords_view();

    // ── Step 1: Verify all cells are quads (4 vertices each) ──
    for (std::size_t c = 0; c < n_cells; ++c) {
        auto start = offsets(c);
        auto end = offsets(c + 1);
        if ((end - start) != 4) {
            return info;  // Not all quads → not a regular grid
        }
    }

    // ── Step 2: Extract unique sorted lon and lat values from cell vertices ──
    // Collect all unique longitude and latitude values from the node coordinates.
    // For a regular grid with ni×nj cells, there are (ni+1) unique lon values
    // and (nj+1) unique lat values at cell boundaries.

    const auto n_nodes = mesh.n_nodes();
    std::vector<double> all_lons;
    std::vector<double> all_lats;
    all_lons.reserve(n_nodes);
    all_lats.reserve(n_nodes);

    for (std::size_t i = 0; i < n_nodes; ++i) {
        all_lons.push_back(coords(i, 0));  // lon is column 0
        all_lats.push_back(coords(i, 1));  // lat is column 1
    }

    // Sort and extract unique values with a tolerance for floating-point equality
    std::sort(all_lons.begin(), all_lons.end());
    std::sort(all_lats.begin(), all_lats.end());

    // Remove near-duplicates (tolerance-based unique)
    constexpr double unique_tol = 1.0e-12;

    auto unique_filter = [&](std::vector<double> &sorted) -> std::vector<double> {
        std::vector<double> unique_vals;
        if (sorted.empty()) return unique_vals;
        unique_vals.push_back(sorted[0]);
        for (std::size_t i = 1; i < sorted.size(); ++i) {
            if (std::abs(sorted[i] - unique_vals.back()) > unique_tol) {
                unique_vals.push_back(sorted[i]);
            }
        }
        return unique_vals;
    };

    auto unique_lons = unique_filter(all_lons);
    auto unique_lats = unique_filter(all_lats);

    // For a regular grid with ni×nj cells, we expect (ni+1) unique lon values
    // and (nj+1) unique lat values (cell boundary coordinates).
    if (unique_lons.size() < 2 || unique_lats.size() < 2) {
        return info;
    }

    const std::size_t n_lon_bounds = unique_lons.size();  // ni + 1
    const std::size_t n_lat_bounds = unique_lats.size();  // nj + 1
    const std::size_t ni = n_lon_bounds - 1;
    const std::size_t nj = n_lat_bounds - 1;

    // Verify cell count matches ni * nj
    if (ni * nj != n_cells) {
        return info;
    }

    // The coordinate sets alone cannot detect a permuted, reversed, or
    // duplicated connectivity layout, which would make the arithmetic fast
    // paths read the wrong neighbours. Require the canonical ordering.
    if (!has_canonical_connectivity(offsets, indices, n_cells, ni, nj)) {
        return info;
    }

    // ── Step 3: Compute deltas between consecutive unique values ──
    std::vector<double> lon_deltas(ni);
    std::vector<double> lat_deltas(nj);

    for (std::size_t i = 0; i < ni; ++i) {
        lon_deltas[i] = unique_lons[i + 1] - unique_lons[i];
    }
    for (std::size_t j = 0; j < nj; ++j) {
        lat_deltas[j] = unique_lats[j + 1] - unique_lats[j];
    }

    // ── Step 4: Verify uniformity — max(delta) - min(delta) < 1e-10 * mean(delta) ──
    auto check_uniform = [](const std::vector<double> &deltas) -> bool {
        if (deltas.empty()) return false;

        double min_d = deltas[0];
        double max_d = deltas[0];
        double sum = 0.0;

        for (const auto &d : deltas) {
            min_d = std::min(min_d, d);
            max_d = std::max(max_d, d);
            sum += d;
        }

        // All deltas must be positive
        if (min_d <= 0.0) return false;

        double mean_d = sum / static_cast<double>(deltas.size());
        double range = max_d - min_d;

        return range < 1.0e-10 * mean_d;
    };

    if (!check_uniform(lon_deltas) || !check_uniform(lat_deltas)) {
        return info;
    }

    // ── All checks passed — fill the RegularGridInfo struct ──
    double lon_sum = 0.0;
    for (const auto &d : lon_deltas) lon_sum += d;
    double lat_sum = 0.0;
    for (const auto &d : lat_deltas) lat_sum += d;

    info.is_regular = true;
    info.lon_min = unique_lons.front();
    info.lon_max = unique_lons.back();
    info.delta_lon = lon_sum / static_cast<double>(ni);
    info.lat_min = unique_lats.front();
    info.lat_max = unique_lats.back();
    info.delta_lat = lat_sum / static_cast<double>(nj);
    info.ni = ni;
    info.nj = nj;

    return info;
}

// ─────────────────────────────────────────────────────────────────────────────
// Tripolar Grid Detection
// ─────────────────────────────────────────────────────────────────────────────

/// @struct TripolarGridInfo
/// @brief Holds metadata for detected folded tripolar grids.
struct TripolarGridInfo {
    bool is_tripolar{false};      ///< True if the mesh has a folded tripolar northern boundary
    std::size_t ni{0};            ///< Number of columns (cells along x)
    std::size_t nj{0};            ///< Number of rows (cells along y)
    double seam_lat{0.0};         ///< Latitude of the folded seam boundary (degrees)
    double seam_lon_center{0.0};  ///< Longitudinal reflection midpoint of the folded boundary
};

/// @brief Detect whether an unstructured mesh represents a folded tripolar grid.
/// @param mesh The unstructured mesh to check.
/// @param ni   Expected number of columns.
/// @param nj   Expected number of rows.
/// @return TripolarGridInfo metadata with is_tripolar set to true if folded symmetry is detected.
template <class MemorySpace>
inline TripolarGridInfo detect_tripolar_grid(const topology::UnstructuredMesh<MemorySpace> &mesh, std::size_t ni, std::size_t nj) {
    TripolarGridInfo info;
    if (mesh.n_nodes() == 0 || ni == 0 || nj == 0) return info;

    // A folded tripolar grid has folded node symmetry along its northernmost boundary row (j = nj)
    // node(i, nj) == node(ni - i, nj)
    auto coords = mesh.node_coords_view();
    auto h_coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coords);

    std::size_t n_nodes_x = ni + 1;
    std::size_t top_row_start = nj * n_nodes_x;

    if (top_row_start + ni >= mesh.n_nodes()) return info;

    bool folded = true;
    for (std::size_t i = 0; i <= ni / 2; ++i) {
        std::size_t idx1 = top_row_start + i;
        std::size_t idx2 = top_row_start + (ni - i);
        double lat1 = h_coords(idx1, 1);
        double lat2 = h_coords(idx2, 1);

        // Folded symmetry check: northernmost row latitudes must match perfectly
        if (std::abs(lat1 - lat2) > 1.0e-9) {
            folded = false;
            break;
        }
    }

    if (folded) {
        info.is_tripolar = true;
        info.ni = ni;
        info.nj = nj;
        info.seam_lat = h_coords(top_row_start, 1);
        // Set reflection center longitude as the midpoint longitude of the northern seam
        info.seam_lon_center = h_coords(top_row_start + ni / 2, 0);
    }

    return info;
}

}  // namespace axis::detail

#endif  // AXIS_DETAIL_REGULAR_GRID_DETECTOR_HPP
