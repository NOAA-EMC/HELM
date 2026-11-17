// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/weight_generator_conservative_rect.cpp
/// @brief Regular-grid rectangle-rectangle conservative fast-path.
///
/// When both source and destination meshes are detected as regular lat-lon grids
/// (uniform spacing), this path computes conservative overlap weights as simple
/// axis-aligned rectangle intersections. It bypasses BVH construction entirely —
/// overlapping source cells for each destination cell are determined analytically
/// via integer index arithmetic.
///
/// This is the primary CDO-competitive path for large regular grids (e.g.,
/// 3600×1800 → 1440×720).
///
/// Requirements: 2.2, 2.3, 2.4, 2.7

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// rect_overlap — axis-aligned rectangle intersection area
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the intersection area of two axis-aligned rectangles.
///
/// Source rectangle: [s_lo_x, s_hi_x] × [s_lo_y, s_hi_y]
/// Destination rectangle: [d_lo_x, d_hi_x] × [d_lo_y, d_hi_y]
///
/// Returns max(0, overlap_x) * max(0, overlap_y).
/// Result is always non-negative (Req 2.4).
KOKKOS_INLINE_FUNCTION
double rect_overlap(double s_lo_x, double s_hi_x, double s_lo_y, double s_hi_y, double d_lo_x, double d_hi_x, double d_lo_y, double d_hi_y) noexcept {
    double dx = Kokkos::fmax(0.0, Kokkos::fmin(s_hi_x, d_hi_x) - Kokkos::fmax(s_lo_x, d_lo_x));
    double dy = Kokkos::fmax(0.0, Kokkos::fmin(s_hi_y, d_hi_y) - Kokkos::fmax(s_lo_y, d_lo_y));
    return dx * dy;
}

KOKKOS_INLINE_FUNCTION
double rect_overlap_spherical(double s_lo_x, double s_hi_x, double s_lo_y, double s_hi_y, double d_lo_x, double d_hi_x, double d_lo_y, double d_hi_y,
                              bool is_degrees) noexcept {
    double dx = Kokkos::fmax(0.0, Kokkos::fmin(s_hi_x, d_hi_x) - Kokkos::fmax(s_lo_x, d_lo_x));
    double lo_y = Kokkos::fmax(s_lo_y, d_lo_y);
    double hi_y = Kokkos::fmin(s_hi_y, d_hi_y);

    if (hi_y <= lo_y || dx <= 0.0) return 0.0;

    if (is_degrees) {
        constexpr double deg2rad = 3.14159265358979323846 / 180.0;
        dx *= deg2rad;
        lo_y *= deg2rad;
        hi_y *= deg2rad;
    }

    return dx * (Kokkos::sin(hi_y) - Kokkos::sin(lo_y));
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_conservative_rect — analytic rectangle overlap path
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Generate conservative interpolation weights for two regular lat-lon
///        grids using analytic rectangle intersection.
///
/// This function bypasses the BVH entirely. For each destination cell (id, jd),
/// it computes the range of overlapping source cells via simple index arithmetic,
/// then computes the exact overlap area as an axis-aligned rectangle intersection.
///
/// @tparam MemorySpace Kokkos memory space (must be host-accessible)
/// @param src_mesh       Source unstructured mesh (regular grid)
/// @param dst_mesh       Destination unstructured mesh (regular grid)
/// @param config         Regridding configuration
/// @param src_grid_info  RegularGridInfo for the source mesh
/// @param dst_grid_info  RegularGridInfo for the destination mesh
/// @return InterpolationMatrix with conservative weights
template <class MemorySpace>
InterpolationMatrix<MemorySpace> generate_conservative_rect(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                            const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config,
                                                            const detail::RegularGridInfo &src_grid_info,
                                                            const detail::RegularGridInfo &dst_grid_info) {
    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    const std::size_t src_ni = src_grid_info.ni;
    const std::size_t src_nj = src_grid_info.nj;
    const std::size_t dst_ni = dst_grid_info.ni;
    const std::size_t dst_nj = dst_grid_info.nj;

    const double src_lon_min = src_grid_info.lon_min;
    const double src_lat_min = src_grid_info.lat_min;
    const double src_delta_lon = src_grid_info.delta_lon;
    const double src_delta_lat = src_grid_info.delta_lat;

    const double dst_lon_min = dst_grid_info.lon_min;
    const double dst_lat_min = dst_grid_info.lat_min;
    const double dst_delta_lon = dst_grid_info.delta_lon;
    const double dst_delta_lat = dst_grid_info.delta_lat;

    // ── Dateline wrap-around detection ──
    // If the source grid spans close to 360° in longitude, cells near the
    // edges may wrap around. Detect this by checking if the source longitude
    // range is close to 360°. If there's a wrap (source range < 360 but
    // destination cells extend beyond source bounds), we fall back to the
    // standard BVH path. For now, detect and handle the simple case where
    // both grids cover the same longitude domain.
    const auto csys = src_mesh.coord_system();
    const bool is_degrees = (csys == topology::CoordinateSystem::SphericalDeg);
    const double wrap_threshold = is_degrees ? 350.0 : 6.0;

    // Compute full periodic ranges as ni * delta_lon (accounts for synthesized corners)
    const double src_full_range = static_cast<double>(src_ni) * src_delta_lon;
    const double dst_full_range = static_cast<double>(dst_ni) * dst_delta_lon;

    // Check if either grid wraps around the dateline
    // A grid is "global" if it spans nearly 360 degrees (or 2*pi radians)
    const bool src_is_global = (src_full_range > wrap_threshold);
    const bool dst_is_global = (dst_full_range > wrap_threshold);

    // If destination extends beyond source longitude bounds and neither is
    // global, fall back — caller should use the BVH path.
    // This is a conservative check: if the grids don't align well in longitude,
    // we don't try to handle wrapping here.
    if (!src_is_global && !dst_is_global) {
        // Non-global grids: check if destination fits within source bounds
        if (dst_grid_info.lon_min < src_grid_info.lon_min - src_delta_lon * 0.5 ||
            dst_grid_info.lon_max > src_grid_info.lon_max + src_delta_lon * 0.5) {
            // Destination extends beyond source — possible wrap-around.
            // Return empty matrix to signal fallback to caller.
            // Note: The integration dispatch (Task 3.1) should handle this by
            // falling through to the standard BVH path.
            return InterpolationMatrix<MemorySpace>();
        }
    }

    // ── Compute cell areas ──
    // For a regular grid, cell area = delta_lon * delta_lat (in coordinate units).
    // This is exact for the planar (Cartesian) interpretation.
    // Note: The mesh may have precomputed areas; use those if available for
    // consistency with the non-optimized path.
    auto mesh_src_areas = src_mesh.cell_areas();
    auto mesh_dst_areas = dst_mesh.cell_areas();

    const bool has_src_areas = (mesh_src_areas.extent(0) == n_src);
    const bool has_dst_areas = (mesh_dst_areas.extent(0) == n_dst);

    // ── Retrieve optional cell masks ──
    auto src_mask = src_mesh.cell_mask();
    auto dst_mask = dst_mesh.cell_mask();
    const bool has_src_mask = (src_mask.extent(0) == n_src);
    const bool has_dst_mask = (dst_mask.extent(0) == n_dst);

    // ── First pass: count nonzeros per destination cell ──
    // We use a two-pass approach: first count entries, then fill.
    // This avoids dynamic allocation in the inner loop.

    // For regular grids, each destination cell overlaps at most
    // (ceil(dst_delta_lon / src_delta_lon) + 1) × (ceil(dst_delta_lat / src_delta_lat) + 1)
    // source cells.

    // ── Accumulate COO entries ──
    // Use std::vector for host-space accumulation (same pattern as generate_conservative).
    std::vector<double> weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    // Pre-estimate capacity: typical overlap count per dst cell
    const std::size_t est_overlap_lon = static_cast<std::size_t>(std::ceil(dst_delta_lon / src_delta_lon)) + 2;
    const std::size_t est_overlap_lat = static_cast<std::size_t>(std::ceil(dst_delta_lat / src_delta_lat)) + 2;
    const std::size_t est_nnz = n_dst * est_overlap_lon * est_overlap_lat;
    weights_vec.reserve(std::min(est_nnz, n_src * n_dst));
    rows_vec.reserve(std::min(est_nnz, n_src * n_dst));
    cols_vec.reserve(std::min(est_nnz, n_src * n_dst));

    const bool is_spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);

    // ── Conservation bookkeeping arrays ──
    std::vector<double> frac_a_acc(n_src, 0.0);
    std::vector<double> frac_b_acc(n_dst, 0.0);
    std::vector<double> src_areas(n_src);
    std::vector<double> dst_areas(n_dst);

    // Compute source cell areas
    for (std::size_t c = 0; c < n_src; ++c) {
        if (has_src_areas) {
            src_areas[c] = mesh_src_areas[c];
        } else {
            if (is_spherical) {
                std::size_t js = c / src_ni;
                double s_lo_y = src_lat_min + static_cast<double>(js) * src_delta_lat;
                double s_hi_y = s_lo_y + src_delta_lat;
                double dlon = src_delta_lon;
                double s_lo_y_r = s_lo_y;
                double s_hi_y_r = s_hi_y;
                if (is_degrees) {
                    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                    dlon *= deg2rad;
                    s_lo_y_r *= deg2rad;
                    s_hi_y_r *= deg2rad;
                }
                src_areas[c] = dlon * (std::sin(s_hi_y_r) - std::sin(s_lo_y_r));
            } else {
                src_areas[c] = src_delta_lon * src_delta_lat;
            }
        }
    }

    // Compute destination cell areas
    for (std::size_t c = 0; c < n_dst; ++c) {
        if (has_dst_areas) {
            dst_areas[c] = mesh_dst_areas[c];
        } else {
            if (is_spherical) {
                std::size_t jd = c / dst_ni;
                double d_lo_y = dst_lat_min + static_cast<double>(jd) * dst_delta_lat;
                double d_hi_y = d_lo_y + dst_delta_lat;
                double dlon = dst_delta_lon;
                double d_lo_y_r = d_lo_y;
                double d_hi_y_r = d_hi_y;
                if (is_degrees) {
                    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                    dlon *= deg2rad;
                    d_lo_y_r *= deg2rad;
                    d_hi_y_r *= deg2rad;
                }
                dst_areas[c] = dlon * (std::sin(d_hi_y_r) - std::sin(d_lo_y_r));
            } else {
                dst_areas[c] = dst_delta_lon * dst_delta_lat;
            }
        }
    }

    // ── Main loop over destination cells ──
    // For each destination cell (id, jd), determine the range of overlapping
    // source cells via index arithmetic, then compute exact overlap areas.
    for (std::size_t jd = 0; jd < dst_nj; ++jd) {
        for (std::size_t id = 0; id < dst_ni; ++id) {
            const std::size_t dst_cell_idx = jd * dst_ni + id;

            // Skip masked destination cells
            if (has_dst_mask && dst_mask[dst_cell_idx] == 0) continue;

            // Destination cell bounds
            const double d_lo_x = dst_lon_min + static_cast<double>(id) * dst_delta_lon;
            const double d_hi_x = d_lo_x + dst_delta_lon;
            const double d_lo_y = dst_lat_min + static_cast<double>(jd) * dst_delta_lat;
            const double d_hi_y = d_lo_y + dst_delta_lat;

            // Normalize destination bounds when mapping a global periodic grid
            double d_lo_x_norm = d_lo_x;
            double d_hi_x_norm = d_hi_x;
            if (src_is_global) {
                double shift = 360.0 * std::floor((d_lo_x - src_lon_min) / 360.0);
                d_lo_x_norm = d_lo_x - shift;
                d_hi_x_norm = d_hi_x - shift;
            }

            // Area of destination cell
            const double area_dst = dst_areas[dst_cell_idx];
            if (area_dst <= 0.0) continue;

            // ── Determine overlapping source cell index range ──
            // Map destination cell bounds into source grid index space.
            int is_lo = static_cast<int>(std::floor((d_lo_x_norm - src_lon_min) / src_delta_lon));
            int is_hi = static_cast<int>(std::floor((d_hi_x_norm - src_lon_min) / src_delta_lon));
            int js_lo = static_cast<int>(std::floor((d_lo_y - src_lat_min) / src_delta_lat));
            int js_hi = static_cast<int>(std::floor((d_hi_y - src_lat_min) / src_delta_lat));

            // Clamp to valid source grid range [0, ni-1] × [0, nj-1]
            if (!src_is_global) {
                is_lo = std::max(0, is_lo);
                is_hi = std::min(static_cast<int>(src_ni) - 1, is_hi);
            }
            js_lo = std::max(0, js_lo);
            js_hi = std::min(static_cast<int>(src_nj) - 1, js_hi);

            // Handle global grids with wrap-around in longitude
            // If source is global and destination cell wraps, we handle it by
            // extending the index range and using modular arithmetic.
            bool lon_wraps = false;
            if (src_is_global && is_lo > is_hi) {
                // Destination cell extends past the source grid's right edge
                // and wraps around to the beginning
                lon_wraps = true;
            }

            if (!lon_wraps && is_lo > is_hi) continue;  // No overlap in x
            if (js_lo > js_hi) continue;                // No overlap in y

            // Iterate over overlapping source cells
            for (int js = js_lo; js <= js_hi; ++js) {
                for (int is = is_lo; is <= is_hi; ++is) {
                    // Handle wrap-around: use modular index for global grids
                    int is_actual = is;
                    if (src_is_global) {
                        is_actual = ((is % static_cast<int>(src_ni)) + static_cast<int>(src_ni)) % static_cast<int>(src_ni);
                    }

                    const std::size_t src_cell_idx = static_cast<std::size_t>(js) * src_ni + static_cast<std::size_t>(is_actual);

                    if (src_cell_idx >= n_src) continue;

                    // Skip masked source cells
                    if (has_src_mask && src_mask[src_cell_idx] == 0) continue;

                    const double area_src = src_areas[src_cell_idx];
                    if (area_src <= 0.0) continue;

                    // Source cell bounds
                    const double s_lo_x = src_lon_min + static_cast<double>(is) * src_delta_lon;
                    const double s_hi_x = s_lo_x + src_delta_lon;
                    const double s_lo_y = src_lat_min + static_cast<double>(js) * src_delta_lat;
                    const double s_hi_y = s_lo_y + src_delta_lat;

                    // Compute rectangle overlap area
                    double overlap_area = 0.0;
                    if (is_spherical) {
                        overlap_area = rect_overlap_spherical(s_lo_x, s_hi_x, s_lo_y, s_hi_y, d_lo_x_norm, d_hi_x_norm, d_lo_y, d_hi_y, is_degrees);
                    } else {
                        overlap_area = rect_overlap(s_lo_x, s_hi_x, s_lo_y, s_hi_y, d_lo_x_norm, d_hi_x_norm, d_lo_y, d_hi_y);
                    }

                    if (overlap_area <= 0.0) continue;

                    // Weight = overlap_area / dst_area (Req 2.3: conservation)
                    double w_ij = overlap_area / area_dst;
                    // Ensure non-negative (Req 2.4) — should always be true
                    // given the max(0, ...) in rect_overlap, but belt-and-suspenders.
                    w_ij = std::max(w_ij, 0.0);

                    weights_vec.push_back(w_ij);
                    rows_vec.push_back(static_cast<index_t>(dst_cell_idx));
                    cols_vec.push_back(static_cast<index_t>(src_cell_idx));

                    // Conservation bookkeeping
                    frac_a_acc[src_cell_idx] += overlap_area / area_src;
                    frac_b_acc[dst_cell_idx] += overlap_area / area_dst;
                }
            }
        }
    }

    // ── Apply FracArea normalization if configured ──
    if (config.norm_type == NormType::FracArea) {
        for (std::size_t k = 0; k < weights_vec.size(); ++k) {
            auto j = static_cast<std::size_t>(rows_vec[k]);
            if (frac_b_acc[j] > 0.0) {
                weights_vec[k] /= frac_b_acc[j];
            }
        }
    }

    // Clamp fractions to [0, 1]
    for (std::size_t i = 0; i < n_src; ++i) {
        frac_a_acc[i] = std::min(frac_a_acc[i], 1.0);
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        frac_b_acc[j] = std::min(frac_b_acc[j], 1.0);
    }

    // ── Pack into InterpolationMatrix ──
    const std::size_t nnz = weights_vec.size();

    Kokkos::View<double *, MemorySpace> factor_list("factor_list", nnz);
    Kokkos::View<index_t *, MemorySpace> factor_row("factor_row", nnz);
    Kokkos::View<index_t *, MemorySpace> factor_col("factor_col", nnz);
    Kokkos::View<double *, MemorySpace> frac_a("frac_a", n_src);
    Kokkos::View<double *, MemorySpace> frac_b("frac_b", n_dst);
    Kokkos::View<double *, MemorySpace> area_a("area_a", n_src);
    Kokkos::View<double *, MemorySpace> area_b("area_b", n_dst);

    auto h_factor_list = Kokkos::create_mirror_view(factor_list);
    auto h_factor_row = Kokkos::create_mirror_view(factor_row);
    auto h_factor_col = Kokkos::create_mirror_view(factor_col);
    auto h_frac_a = Kokkos::create_mirror_view(frac_a);
    auto h_frac_b = Kokkos::create_mirror_view(frac_b);
    auto h_area_a = Kokkos::create_mirror_view(area_a);
    auto h_area_b = Kokkos::create_mirror_view(area_b);

    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k) = rows_vec[k];
        h_factor_col(k) = cols_vec[k];
    }

    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = src_areas[i];
        h_frac_a(i) = frac_a_acc[i];
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = dst_areas[j];
        h_frac_b(j) = frac_b_acc[j];
    }

    Kokkos::deep_copy(factor_list, h_factor_list);
    Kokkos::deep_copy(factor_row, h_factor_row);
    Kokkos::deep_copy(factor_col, h_factor_col);
    Kokkos::deep_copy(frac_a, h_frac_a);
    Kokkos::deep_copy(frac_b, h_frac_b);
    Kokkos::deep_copy(area_a, h_area_a);
    Kokkos::deep_copy(area_b, h_area_b);

    return InterpolationMatrix<MemorySpace>(std::move(factor_list), std::move(factor_row), std::move(factor_col), std::move(frac_a),
                                            std::move(frac_b), std::move(area_a), std::move(area_b), n_src, n_dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiation — HostSpace only
// (Device-space dispatch handled by generate_conservative_device in
//  weight_generator.cpp; rectangle fast-path is host-only for now.)
// ─────────────────────────────────────────────────────────────────────────────

template InterpolationMatrix<Kokkos::HostSpace> generate_conservative_rect<Kokkos::HostSpace>(const topology::UnstructuredMesh<Kokkos::HostSpace> &,
                                                                                              const topology::UnstructuredMesh<Kokkos::HostSpace> &,
                                                                                              const RegridConfig &, const detail::RegularGridInfo &,
                                                                                              const detail::RegularGridInfo &);

}  // namespace axis::solver
