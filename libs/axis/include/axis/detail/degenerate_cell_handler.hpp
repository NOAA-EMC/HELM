// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_DEGENERATE_CELL_HANDLER_HPP
#define AXIS_DETAIL_DEGENERATE_CELL_HANDLER_HPP

/// @file axis/detail/degenerate_cell_handler.hpp
/// @brief GPU-portable degenerate cell detection for unstructured meshes.
///
/// Provides:
///   - DegenerateType: classification enum for degenerate cell conditions
///   - DegenerateCellReport: diagnostic record of excluded cells
///   - DegenerateCellHandler: classify individual cells (device) and scan
///     entire meshes (host) for degenerate conditions
///
/// The classify() function is annotated KOKKOS_FUNCTION for device portability.
/// The scan() function runs on the host and builds a diagnostic report using
/// std::vector. Warnings are emitted via a caller-supplied callback, not through
/// LOGS (preserving Tier 1 isolation).

#include <Kokkos_Core.hpp>
#include <axis/detail/spherical_clipper.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// DegenerateType — classification of degenerate cell conditions
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Enumeration of degenerate cell condition types.
///
/// Used to classify cells detected by DegenerateCellHandler. A cell may have
/// multiple degenerate properties; the handler reports the first/most-severe
/// condition found.
enum class DegenerateType : uint8_t {
    None = 0,             ///< Cell is valid (no degenerate condition)
    ZeroArea = 1,         ///< Cell area below minimum threshold
    CollapsedEdge = 2,    ///< Two or more coincident vertices (edge length ≈ 0)
    SelfIntersecting = 3  ///< Cell boundary crosses itself
};

// ─────────────────────────────────────────────────────────────────────────────
// DegenerateCellReport — diagnostic record of excluded cells
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Diagnostic report containing indices and types of degenerate cells.
///
/// Populated by DegenerateCellHandler::scan() after scanning all cells in a
/// mesh. Consumers can attach this report to the InterpolationMatrix for
/// downstream diagnostic inspection.
struct DegenerateCellReport {
    std::vector<index_t> excluded_indices;       ///< Cell indices flagged as degenerate
    std::vector<DegenerateType> excluded_types;  ///< Corresponding degenerate types
    std::size_t total_cells{0};                  ///< Total number of cells scanned
    std::size_t degenerate_count{0};             ///< Number of degenerate cells found

    /// @brief Fraction of degenerate cells relative to total.
    [[nodiscard]] double degenerate_fraction() const noexcept {
        return total_cells > 0 ? static_cast<double>(degenerate_count) / static_cast<double>(total_cells) : 0.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// DegenerateCellHandler — detection logic
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Detects degenerate cells in unstructured meshes.
///
/// Provides two levels of usage:
///   1. classify(): Device-portable, classifies a single SphericalPolygon.
///      Suitable for use within Kokkos parallel kernels.
///   2. scan(): Host function that iterates over all cells in an
///      UnstructuredMesh, builds a DegenerateCellReport, and optionally
///      invokes a warning callback if >1% of cells are degenerate.
///
/// Detection checks (in order):
///   - Collapsed edges: coincident vertices with edge length < edge_threshold
///   - Zero area: cell area < area_threshold steradians
///   - Self-intersecting boundary: any pair of non-adjacent edges cross
struct DegenerateCellHandler {
    /// Default minimum area threshold in steradians.
    static constexpr double DEFAULT_AREA_THRESHOLD = 1e-25;

    /// Default minimum edge length threshold (normalized coordinates).
    static constexpr double DEFAULT_EDGE_THRESHOLD = 1e-15;

    /// Default warning fraction: warn when degenerate cells exceed this ratio.
    static constexpr double DEFAULT_WARNING_FRACTION = 0.01;

    // ─────────────────────────────────────────────────────────────────────────
    // Device-portable classification
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Classify a single spherical polygon cell as degenerate or valid.
    ///
    /// Checks are performed in order of computational cost:
    ///   1. Collapsed edges (cheapest — just distance checks)
    ///   2. Zero area (moderate — spherical excess computation)
    ///   3. Self-intersecting boundary (most expensive — edge-pair tests)
    ///
    /// Returns the first degenerate condition found, or DegenerateType::None.
    ///
    /// @tparam MaxVerts Maximum vertex capacity of the polygon.
    /// @param cell The spherical polygon to classify.
    /// @param area_threshold Minimum area in steradians (default 1e-25).
    /// @param edge_threshold Minimum edge length (default 1e-15).
    /// @return The degenerate type, or None if the cell is valid.
    template <int MaxVerts>
    KOKKOS_FUNCTION static DegenerateType classify(const SphericalPolygon<MaxVerts> &cell, double area_threshold = DEFAULT_AREA_THRESHOLD,
                                                   double edge_threshold = DEFAULT_EDGE_THRESHOLD) noexcept {
        // A polygon with fewer than 3 vertices is trivially degenerate.
        if (cell.n < 3) {
            return DegenerateType::ZeroArea;
        }

        // Check 1: Zero area.
        double a = cell.area();
        if (a < area_threshold) {
            return DegenerateType::ZeroArea;
        }

        // Check 2: Collapsed edges only if fewer than 3 unique vertices remain.
        // Polar cap quads on regular lat-lon grids have 2 coincident vertices at the pole,
        // leaving 3 distinct vertices forming a valid spherical triangle with positive area.
        if (count_unique_vertices(cell, edge_threshold) < 3) {
            return DegenerateType::CollapsedEdge;
        }

        // Check 3: Self-intersecting boundary.
        if (is_self_intersecting(cell)) {
            return DegenerateType::SelfIntersecting;
        }

        return DegenerateType::None;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Host-side mesh scan
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Scan all cells in a mesh and build a degenerate cell report.
    ///
    /// Iterates over each cell in the mesh, constructs a SphericalPolygon from
    /// the mesh connectivity, and classifies it. Cells flagged as degenerate
    /// are recorded in the returned report.
    ///
    /// If the degenerate fraction exceeds 1% (configurable via
    /// DEFAULT_WARNING_FRACTION), and a warning_cb is provided, the callback
    /// is invoked with (degenerate_count, total_cells).
    ///
    /// @tparam MemorySpace Kokkos memory space of the input mesh.
    /// @param mesh The unstructured mesh to scan.
    /// @param area_threshold Minimum area threshold in steradians.
    /// @param edge_threshold Minimum edge length threshold.
    /// @param warning_cb Optional callback invoked when >1% degenerate.
    /// @return A DegenerateCellReport with excluded indices and types.
    template <class MemorySpace>
    static DegenerateCellReport scan(const topology::UnstructuredMesh<MemorySpace> &mesh, double area_threshold = DEFAULT_AREA_THRESHOLD,
                                     double edge_threshold = DEFAULT_EDGE_THRESHOLD,
                                     std::function<void(std::size_t degenerate, std::size_t total)> warning_cb = nullptr) {
        DegenerateCellReport report;
        report.total_cells = mesh.n_cells();
        report.degenerate_count = 0;

        if (report.total_cells == 0) {
            return report;
        }

        // Access mesh data — mirror to host if necessary.
        auto coords_view = mesh.node_coords();    // [n_nodes, ndim]
        auto offsets_view = mesh.conn_offsets();  // [n_cells + 1]
        auto indices_view = mesh.conn_indices();  // [nnz]

        const std::size_t n_cells = report.total_cells;
        const std::size_t ndim = coords_view.extent(1);
        const bool cartesian = mesh.coord_system() == topology::CoordinateSystem::Cartesian3D;

        for (std::size_t ci = 0; ci < n_cells; ++ci) {
            // Build a SphericalPolygon from the mesh cell connectivity.
            index_t start = offsets_view[ci];
            index_t end = offsets_view[ci + 1];
            int n_verts = static_cast<int>(end - start);

            // Cartesian3D stores coordinates in physical x/y/z units, not
            // unit-sphere vectors. Classifying those values with SphericalPolygon
            // can discard valid planar cells (and suppress UnmappedAction::Error).
            if (cartesian) {
                if (n_verts < 3) {
                    report.excluded_indices.push_back(static_cast<index_t>(ci));
                    report.excluded_types.push_back(DegenerateType::ZeroArea);
                    ++report.degenerate_count;
                    continue;
                }
                double nx = 0.0, ny = 0.0, nz = 0.0;
                int unique_vertices = 0;
                for (int vi = 0; vi < n_verts; ++vi) {
                    const std::size_t node = static_cast<std::size_t>(indices_view[start + vi]);
                    const std::size_t next_node = static_cast<std::size_t>(indices_view[start + (vi + 1) % n_verts]);
                    const double x = coords_view(node, 0), y = coords_view(node, 1), z = coords_view(node, 2);
                    const double xn = coords_view(next_node, 0), yn = coords_view(next_node, 1), zn = coords_view(next_node, 2);
                    nx += y * zn - z * yn;
                    ny += z * xn - x * zn;
                    nz += x * yn - y * xn;

                    bool duplicate = false;
                    for (int previous = 0; previous < vi; ++previous) {
                        const std::size_t prior_node = static_cast<std::size_t>(indices_view[start + previous]);
                        const double dx = x - coords_view(prior_node, 0);
                        const double dy = y - coords_view(prior_node, 1);
                        const double dz = z - coords_view(prior_node, 2);
                        if (dx * dx + dy * dy + dz * dz < edge_threshold * edge_threshold) {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate) ++unique_vertices;
                }
                const double area = 0.5 * Kokkos::sqrt(nx * nx + ny * ny + nz * nz);
                const DegenerateType dtype =
                    area < area_threshold ? DegenerateType::ZeroArea : (unique_vertices < 3 ? DegenerateType::CollapsedEdge : DegenerateType::None);
                if (dtype != DegenerateType::None) {
                    report.excluded_indices.push_back(static_cast<index_t>(ci));
                    report.excluded_types.push_back(dtype);
                    ++report.degenerate_count;
                }
                continue;
            }

            // Use MaxVerts = 32 (matches the default throughout AXIS).
            constexpr int MaxVerts = 32;
            SphericalPolygon<MaxVerts> poly;

            for (int vi = 0; vi < n_verts && vi < MaxVerts; ++vi) {
                index_t node_idx = indices_view[start + vi];

                Vec3 v;
                if (ndim >= 3) {
                    // Already in Cartesian (x, y, z) on the unit sphere.
                    v.x = coords_view(node_idx, 0);
                    v.y = coords_view(node_idx, 1);
                    v.z = coords_view(node_idx, 2);
                } else {
                    // Spherical coordinates (lon, lat) in degrees — convert to
                    // unit-sphere Cartesian.
                    double lon_deg = coords_view(node_idx, 0);
                    double lat_deg = coords_view(node_idx, 1);
                    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                    double lon = lon_deg * deg2rad;
                    double lat = lat_deg * deg2rad;
                    double cos_lat = Kokkos::cos(lat);
                    v.x = cos_lat * Kokkos::cos(lon);
                    v.y = cos_lat * Kokkos::sin(lon);
                    v.z = Kokkos::sin(lat);
                }

                poly.push(v);
            }

            // Classify the cell.
            DegenerateType dtype = classify(poly, area_threshold, edge_threshold);

            if (dtype != DegenerateType::None) {
                report.excluded_indices.push_back(static_cast<index_t>(ci));
                report.excluded_types.push_back(dtype);
                ++report.degenerate_count;
            }
        }

        // Emit warning via callback if degenerate fraction exceeds threshold.
        if (warning_cb && report.degenerate_fraction() > DEFAULT_WARNING_FRACTION) {
            warning_cb(report.degenerate_count, report.total_cells);
        }

        return report;
    }

   private:
    // ─────────────────────────────────────────────────────────────────────────
    // Internal detection helpers — all KOKKOS_FUNCTION for device portability
    // ─────────────────────────────────────────────────────────────────────────

    /// @brief Count number of distinct/unique vertices in a polygon.
    ///
    /// Vertices closer than threshold are treated as coincident.
    template <int MaxVerts>
    KOKKOS_FUNCTION static int count_unique_vertices(const SphericalPolygon<MaxVerts> &cell, double threshold) noexcept {
        if (cell.n < 1) return 0;
        double thresh_sq = threshold * threshold;
        int unique_count = 1;

        for (int i = 1; i < cell.n; ++i) {
            bool duplicate = false;
            for (int j = 0; j < i; ++j) {
                Vec3 diff{cell.verts[i].x - cell.verts[j].x, cell.verts[i].y - cell.verts[j].y, cell.verts[i].z - cell.verts[j].z};
                if (length_sq(diff) < thresh_sq) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) {
                unique_count++;
            }
        }
        return unique_count;
    }

    /// @brief Detect collapsed edges (coincident adjacent vertices).
    ///
    /// An edge is "collapsed" if the Euclidean distance between its endpoints
    /// is below the threshold. This indicates degenerate geometry where two
    /// vertices have merged.
    ///
    /// @tparam MaxVerts Maximum vertex capacity.
    /// @param cell The polygon to check.
    /// @param threshold Minimum edge length.
    /// @return true if any edge is collapsed.
    template <int MaxVerts>
    KOKKOS_FUNCTION static bool has_collapsed_edge(const SphericalPolygon<MaxVerts> &cell, double threshold) noexcept {
        double thresh_sq = threshold * threshold;

        for (int i = 0; i < cell.n; ++i) {
            int j = (i + 1) % cell.n;
            Vec3 diff{cell.verts[j].x - cell.verts[i].x, cell.verts[j].y - cell.verts[i].y, cell.verts[j].z - cell.verts[i].z};
            if (length_sq(diff) < thresh_sq) {
                return true;
            }
        }
        return false;
    }

    /// @brief Detect self-intersecting polygon boundaries.
    ///
    /// Tests all pairs of non-adjacent edges for intersection. Two edges
    /// intersect on the sphere if their great-circle arcs cross each other.
    /// Uses the sign of the cross-product dot product to determine if endpoints
    /// of one edge lie on opposite sides of the other edge's great circle.
    ///
    /// @tparam MaxVerts Maximum vertex capacity.
    /// @param cell The polygon to check.
    /// @return true if any non-adjacent edge pair intersects.
    template <int MaxVerts>
    KOKKOS_FUNCTION static bool is_self_intersecting(const SphericalPolygon<MaxVerts> &cell) noexcept {
        if (cell.n < 4) {
            // Triangles cannot self-intersect.
            return false;
        }

        // Test all pairs of non-adjacent edges.
        for (int i = 0; i < cell.n; ++i) {
            int i_next = (i + 1) % cell.n;

            for (int j = i + 2; j < cell.n; ++j) {
                // Skip if edges share a vertex (adjacent).
                int j_next = (j + 1) % cell.n;
                if (j_next == i) continue;

                // Check if edge (i, i_next) and edge (j, j_next) intersect.
                if (edges_intersect(cell.verts[i], cell.verts[i_next], cell.verts[j], cell.verts[j_next])) {
                    return true;
                }
            }
        }
        return false;
    }

    /// @brief Test if two great-circle arcs intersect on the unit sphere.
    ///
    /// Two arcs (A→B) and (C→D) intersect if:
    ///   1. C and D lie on opposite sides of the great circle through A,B
    ///   2. A and B lie on opposite sides of the great circle through C,D
    ///
    /// "Side" is determined by the sign of dot(cross(P1,P2), Q).
    ///
    /// @param a Start of first arc.
    /// @param b End of first arc.
    /// @param c Start of second arc.
    /// @param d End of second arc.
    /// @return true if the arcs properly cross.
    KOKKOS_FUNCTION
    static bool edges_intersect(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &d) noexcept {
        // Normal of the great circle containing arc (a, b).
        Vec3 n_ab = cross(a, b);

        // Signed distances of c and d from the great circle (a, b).
        double dc = dot(n_ab, c);
        double dd = dot(n_ab, d);

        // c and d must be on opposite sides (strict crossing).
        if (dc * dd >= 0.0) {
            return false;
        }

        // Normal of the great circle containing arc (c, d).
        Vec3 n_cd = cross(c, d);

        // Signed distances of a and b from the great circle (c, d).
        double da = dot(n_cd, a);
        double db = dot(n_cd, b);

        // a and b must be on opposite sides (strict crossing).
        if (da * db >= 0.0) {
            return false;
        }

        return true;
    }
};

}  // namespace axis::detail

#endif  // AXIS_DETAIL_DEGENERATE_CELL_HANDLER_HPP
