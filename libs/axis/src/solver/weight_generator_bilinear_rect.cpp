// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/weight_generator_bilinear_rect.cpp
/// @brief Regular-grid bilinear interpolation fast-path.
///
/// When both source and destination meshes are detected as regular lat-lon grids
/// (uniform spacing), this path computes bilinear weights via O(1) index
/// arithmetic and simple fractional-position products. It bypasses BVH
/// construction and gnomonic projection entirely — the containing source cell
/// for each destination centroid is found analytically via floor division on
/// normalized coordinates.
///
/// Requirements: 1.1, 2.1–2.8, 3.1–3.5, 4.1–4.6, 7.1–7.4

#include <Kokkos_Core.hpp>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace axis::solver {

// ─────────────────────────────────────────────────────────────────────────────
// BilinearRectKernel — device-portable per-cell computation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Internal helper struct encapsulating the bilinear weight computation
///        for a single destination point on a regular source grid.
struct BilinearRectKernel {
    double lon_min;
    double lat_min;
    double delta_lon;
    double delta_lat;
    std::size_t ni;
    std::size_t nj;
    bool is_periodic;

    /// @brief Compute bilinear weights and source cell indices for a destination
    ///        point at (lon_d, lat_d).
    ///
    /// The interpolation is performed on cell-centered data. Cell centers are
    /// located at: lon_min + (i + 0.5) * delta_lon, lat_min + (j + 0.5) * delta_lat.
    /// We compute fractional positions relative to these cell centers, finding
    /// the 2×2 stencil of source cell centers that surround the query point.
    ///
    /// @param[in]  lon_d  Destination centroid longitude (degrees)
    /// @param[in]  lat_d  Destination centroid latitude (degrees)
    /// @param[out] idx    Array of 4 flat source cell indices
    /// @param[out] wts    Array of 4 bilinear weights
    /// @return true if the point is mappable, false if outside bounds
    KOKKOS_INLINE_FUNCTION
    bool compute_weights(double lon_d, double lat_d, std::size_t *idx, double *wts) const noexcept {
        // ── Longitude normalization for periodic grids ──
        if (is_periodic) {
            // Normalize longitude into [lon_min, lon_min + 360) using a single
            // floor-based operation for numerical stability with large offsets.
            double offset = lon_d - lon_min;
            offset = offset - Kokkos::floor(offset / 360.0) * 360.0;
            lon_d = lon_min + offset;
        }

        // ── Cell-center-relative index arithmetic ──
        // Cell centers are at lon_min + (i + 0.5) * delta_lon for i in [0, ni).
        // The continuous cell-center index is:
        //   fi = (lon_d - lon_min) / delta_lon - 0.5
        // so that fi = 0.0 corresponds exactly to the center of cell 0,
        // fi = 1.0 to cell 1, etc.
        double fi = (lon_d - lon_min) / delta_lon - 0.5;
        double fj = (lat_d - lat_min) / delta_lat - 0.5;

        // ── Bounds checking ──
        // A destination centroid that lands exactly on a grid boundary can fall
        // a fraction of a ULP outside the [-0.5, n-0.5] range because of
        // floating-point round-off in the centroid averaging. Such points are
        // meant to be accepted and then clamped, so admit a small tolerance
        // before rejecting a point as unmapped.
        const double bound_tol = 1.0e-9;
        const double lat_lo = -0.5 - bound_tol;
        const double lat_hi = static_cast<double>(nj) - 0.5 + bound_tol;
        if (!is_periodic) {
            // The valid interpolation range is fi ∈ [-0.5, ni-0.5]
            // (anything outside the grid boundary is unmapped)
            const double lon_lo = -0.5 - bound_tol;
            const double lon_hi = static_cast<double>(ni) - 0.5 + bound_tol;
            if (fi < lon_lo || fi > lon_hi || fj < lat_lo || fj > lat_hi) {
                return false;  // Outside bounds
            }
        } else {
            // For periodic grids, only check latitude bounds
            if (fj < lat_lo || fj > lat_hi) {
                return false;  // Outside latitude bounds
            }
        }

        // ── Compute integer cell indices ──
        // i is the index of the "left" cell center in the stencil
        int i = static_cast<int>(Kokkos::floor(fi));
        int j = static_cast<int>(Kokkos::floor(fj));

        // ── Boundary clamping for latitude ──
        // Clamp j so that both j and j+1 are valid cell indices [0, nj-1]
        if (j < 0) j = 0;
        if (j >= static_cast<int>(nj) - 1) j = static_cast<int>(nj) - 2;
        if (j < 0) j = 0;  // Safety for nj == 1

        // ── Boundary clamping for longitude ──
        if (!is_periodic) {
            if (i < 0) i = 0;
            if (i >= static_cast<int>(ni) - 1) i = static_cast<int>(ni) - 2;
            if (i < 0) i = 0;  // Safety for ni == 1
        } else {
            // For periodic grids, wrap i into [0, ni)
            i = ((i % static_cast<int>(ni)) + static_cast<int>(ni)) % static_cast<int>(ni);
        }

        // ── Fractional position relative to cell center (i, j) ──
        // tx = fi - i gives fraction between cell center i and cell center i+1
        double tx = fi - static_cast<double>(i);
        double ty = fj - static_cast<double>(j);

        // Clamp tx, ty to [0, 1] for numerical safety at boundaries
        if (tx < 0.0) tx = 0.0;
        if (tx > 1.0) tx = 1.0;
        if (ty < 0.0) ty = 0.0;
        if (ty > 1.0) ty = 1.0;

        // ── Compute i+1 with periodic wraparound ──
        int i1 = i + 1;
        if (is_periodic) {
            if (i1 >= static_cast<int>(ni)) {
                i1 = 0;  // Wrap around
            }
        } else {
            // Already clamped i so i+1 <= ni-1
            if (i1 >= static_cast<int>(ni)) {
                i1 = static_cast<int>(ni) - 1;
            }
        }

        int j1 = j + 1;
        if (j1 >= static_cast<int>(nj)) {
            j1 = static_cast<int>(nj) - 1;
        }

        // ── Flat source cell indices: j * ni + i ──
        idx[0] = static_cast<std::size_t>(j) * ni + static_cast<std::size_t>(i);
        idx[1] = static_cast<std::size_t>(j) * ni + static_cast<std::size_t>(i1);
        idx[2] = static_cast<std::size_t>(j1) * ni + static_cast<std::size_t>(i);
        idx[3] = static_cast<std::size_t>(j1) * ni + static_cast<std::size_t>(i1);

        // ── Bilinear weights ──
        wts[0] = (1.0 - tx) * (1.0 - ty);  // w00
        wts[1] = tx * (1.0 - ty);          // w10
        wts[2] = (1.0 - tx) * ty;          // w01
        wts[3] = tx * ty;                  // w11

        return true;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// generate_bilinear_rect — analytic bilinear fast-path
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Generate bilinear interpolation weights for two regular lat-lon grids
///        using analytic index arithmetic — no BVH, no projection.
///
/// @tparam MemorySpace Kokkos memory space (must be host-accessible)
/// @param src_mesh       Source unstructured mesh (regular grid)
/// @param dst_mesh       Destination unstructured mesh (regular grid)
/// @param config         Regridding configuration (unmapped policy)
/// @param src_grid_info  RegularGridInfo for source mesh
/// @param dst_grid_info  RegularGridInfo for destination mesh (unused but kept for API consistency)
/// @return InterpolationMatrix with bilinear weights (exactly 4 per interior dst cell)
template <class MemorySpace>
InterpolationMatrix<MemorySpace> generate_bilinear_rect(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                        const topology::UnstructuredMesh<MemorySpace> &dst_mesh, const RegridConfig &config,
                                                        const detail::RegularGridInfo &src_grid_info,
                                                        [[maybe_unused]] const detail::RegularGridInfo &dst_grid_info) {
    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    const std::size_t ni = src_grid_info.ni;
    const std::size_t nj = src_grid_info.nj;

    // ── Degenerate grid check — return empty matrix as fallback signal ──
    if (ni == 0 || nj == 0) {
        return InterpolationMatrix<MemorySpace>();
    }

    const double lon_min = src_grid_info.lon_min;
    const double lat_min = src_grid_info.lat_min;
    const double delta_lon = src_grid_info.delta_lon;
    const double delta_lat = src_grid_info.delta_lat;

    // ── Periodicity detection ──
    // Source grid is periodic if its longitude span ≈ 360° within tolerance.
    // RegridConfig::periodic overrides auto-detection: +1 forces periodic,
    // -1 forces non-periodic, 0 keeps span-based detection.
    const double lon_span = src_grid_info.lon_max - src_grid_info.lon_min;
    const double period_tol = 1.0e-10 * delta_lon;
    bool is_periodic = std::abs(lon_span - 360.0) < period_tol;
    if (config.periodic > 0) {
        is_periodic = true;
    } else if (config.periodic < 0) {
        is_periodic = false;
    }

    // ── Construct the compute kernel ──
    BilinearRectKernel kernel{lon_min, lat_min, delta_lon, delta_lat, ni, nj, is_periodic};

    // ── Access destination mesh connectivity and coordinates for centroid computation ──
    const auto &offsets = dst_mesh.conn_offsets_view();
    const auto &indices = dst_mesh.conn_indices_view();
    const auto &coords = dst_mesh.node_coords_view();

    // ── Accumulate COO entries ──
    // Reserve for 4 entries per destination cell (bilinear stencil)
    std::vector<double> weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    weights_vec.reserve(n_dst * 4);
    rows_vec.reserve(n_dst * 4);
    cols_vec.reserve(n_dst * 4);

    // ── Main loop over destination cells ──
    for (std::size_t c = 0; c < n_dst; ++c) {
        // ── Compute destination cell centroid (arithmetic mean of vertices) ──
        auto start = static_cast<std::size_t>(offsets(c));
        auto end = static_cast<std::size_t>(offsets(c + 1));
        std::size_t n_verts = end - start;

        double lon_sum = 0.0;
        double lat_sum = 0.0;
        for (std::size_t v = start; v < end; ++v) {
            auto node_idx = static_cast<std::size_t>(indices(v));
            lon_sum += coords(node_idx, 0);  // lon is column 0
            lat_sum += coords(node_idx, 1);  // lat is column 1
        }

        double lon_d = lon_sum / static_cast<double>(n_verts);
        double lat_d = lat_sum / static_cast<double>(n_verts);

        // ── Compute bilinear weights ──
        std::size_t src_idx[4];
        double wts[4];

        bool mapped = kernel.compute_weights(lon_d, lat_d, src_idx, wts);

        if (!mapped) {
            // Point is outside source grid bounds (non-periodic)
            if (config.unmapped == UnmappedAction::Error) {
                throw std::runtime_error("Unmapped destination cell " + std::to_string(c) + " in bilinear regular-grid fast-path");
            }
            // UnmappedAction::Ignore — skip this cell (zero-row)
            continue;
        }

        // ── Emit 4 COO entries for this destination cell ──
        for (int k = 0; k < 4; ++k) {
            weights_vec.push_back(wts[k]);
            rows_vec.push_back(static_cast<index_t>(c));
            cols_vec.push_back(static_cast<index_t>(src_idx[k]));
        }
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

    // Fill COO entries
    for (std::size_t k = 0; k < nnz; ++k) {
        h_factor_list(k) = weights_vec[k];
        h_factor_row(k) = rows_vec[k];
        h_factor_col(k) = cols_vec[k];
    }

    // frac_a = 1.0 for all source cells (bilinear convention)
    for (std::size_t i = 0; i < n_src; ++i) {
        h_frac_a(i) = 1.0;
    }

    // frac_b = 1.0 for mapped destination cells, 0.0 for unmapped
    // Track which destination cells were mapped
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_frac_b(j) = 0.0;
    }
    for (std::size_t k = 0; k < nnz; ++k) {
        h_frac_b(static_cast<std::size_t>(rows_vec[k])) = 1.0;
    }

    // area_a and area_b = 0.0 (ESMF bilinear convention — no area weighting)
    for (std::size_t i = 0; i < n_src; ++i) {
        h_area_a(i) = 0.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        h_area_b(j) = 0.0;
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
// (Device-space dispatch handled by generate_bilinear_device in
//  weight_generator.cpp; bilinear rect fast-path is host-only for now.)
// ─────────────────────────────────────────────────────────────────────────────

template InterpolationMatrix<Kokkos::HostSpace> generate_bilinear_rect<Kokkos::HostSpace>(const topology::UnstructuredMesh<Kokkos::HostSpace> &,
                                                                                          const topology::UnstructuredMesh<Kokkos::HostSpace> &,
                                                                                          const RegridConfig &, const detail::RegularGridInfo &,
                                                                                          const detail::RegularGridInfo &);

}  // namespace axis::solver
