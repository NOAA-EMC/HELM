// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/weight_generator_conservative_rect_nonuniform.cpp
/// @brief Non-uniform rectilinear grid conservative interpolation fast-path.

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

namespace {

KOKKOS_INLINE_FUNCTION
double rect_overlap_nonuniform(double s_lo_x, double s_hi_x, double s_lo_y, double s_hi_y, double d_lo_x, double d_hi_x, double d_lo_y,
                               double d_hi_y) noexcept {
    double dx = Kokkos::fmax(0.0, Kokkos::fmin(s_hi_x, d_hi_x) - Kokkos::fmax(s_lo_x, d_lo_x));
    double dy = Kokkos::fmax(0.0, Kokkos::fmin(s_hi_y, d_hi_y) - Kokkos::fmax(s_lo_y, d_lo_y));
    return dx * dy;
}

KOKKOS_INLINE_FUNCTION
double rect_overlap_spherical_nonuniform(double s_lo_x, double s_hi_x, double s_lo_y, double s_hi_y, double d_lo_x, double d_hi_x, double d_lo_y,
                                         double d_hi_y, bool is_degrees) noexcept {
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

}  // namespace

template <class MemorySpace>
InterpolationMatrix<MemorySpace> generate_conservative_rect_nonuniform(const topology::UnstructuredMesh<MemorySpace> &src_mesh,
                                                                       const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
                                                                       const RegridConfig &config, const detail::RectilinearGridInfo &src_rect_info,
                                                                       const detail::RectilinearGridInfo &dst_rect_info) {
    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    const std::size_t src_ni = src_rect_info.ni;
    const std::size_t src_nj = src_rect_info.nj;
    const std::size_t dst_ni = dst_rect_info.ni;
    const std::size_t dst_nj = dst_rect_info.nj;

    const auto csys = src_mesh.coord_system();
    const bool is_spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);
    const bool is_degrees = (csys == topology::CoordinateSystem::SphericalDeg);

    auto src_lons = src_rect_info.unique_lons;
    auto src_lats = src_rect_info.unique_lats;
    auto dst_lons = dst_rect_info.unique_lons;
    auto dst_lats = dst_rect_info.unique_lats;

    const double src_lon_min = src_lons(0);
    const double src_lon_max = src_lons(src_ni);
    const double src_full_range = src_lon_max - src_lon_min;
    const double wrap_threshold = is_degrees ? 350.0 : 6.0;
    const bool src_is_global = (src_full_range > wrap_threshold);

    auto mesh_src_areas = src_mesh.cell_areas();
    auto mesh_dst_areas = dst_mesh.cell_areas();

    const bool has_src_areas = (mesh_src_areas.extent(0) == n_src);
    const bool has_dst_areas = (mesh_dst_areas.extent(0) == n_dst);

    auto src_mask = src_mesh.cell_mask();
    auto dst_mask = dst_mesh.cell_mask();
    const bool has_src_mask = (src_mask.extent(0) == n_src);
    const bool has_dst_mask = (dst_mask.extent(0) == n_dst);

    std::vector<double> weights_vec;
    std::vector<index_t> rows_vec;
    std::vector<index_t> cols_vec;

    std::vector<double> frac_a_acc(n_src, 0.0);
    std::vector<double> frac_b_acc(n_dst, 0.0);

    for (std::size_t jd = 0; jd < dst_nj; ++jd) {
        for (std::size_t id = 0; id < dst_ni; ++id) {
            const std::size_t c_dst = id + jd * dst_ni;

            if (has_dst_mask && dst_mask(c_dst) == 0) continue;

            double d_lo_x = dst_lons(id);
            double d_hi_x = dst_lons(id + 1);
            double d_lo_y = dst_lats(jd);
            double d_hi_y = dst_lats(jd + 1);

            double d_lo_x_norm = d_lo_x;
            double d_hi_x_norm = d_hi_x;
            if (src_is_global) {
                double shift = 360.0 * std::floor((d_lo_x - src_lon_min) / 360.0);
                d_lo_x_norm = d_lo_x - shift;
                d_hi_x_norm = d_hi_x - shift;
            }

            double area_dst = 0.0;
            if (has_dst_areas) {
                area_dst = mesh_dst_areas(c_dst);
            } else {
                if (is_spherical) {
                    double dlon = d_hi_x_norm - d_lo_x_norm;
                    double d_lo_y_r = d_lo_y;
                    double d_hi_y_r = d_hi_y;
                    if (is_degrees) {
                        constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                        dlon *= deg2rad;
                        d_lo_y_r *= deg2rad;
                        d_hi_y_r *= deg2rad;
                    }
                    area_dst = dlon * (std::sin(d_hi_y_r) - std::sin(d_lo_y_r));
                } else {
                    area_dst = (d_hi_x_norm - d_lo_x_norm) * (d_hi_y - d_lo_y);
                }
            }
            if (area_dst <= 0.0) continue;

            // Find overlapping source cell index ranges using binary search
            auto src_lon_start_it = std::lower_bound(src_lons.data(), src_lons.data() + src_lons.extent(0), d_lo_x_norm);
            auto src_lon_end_it = std::upper_bound(src_lons.data(), src_lons.data() + src_lons.extent(0), d_hi_x_norm);

            int is_start = std::max(0, static_cast<int>(std::distance(src_lons.data(), src_lon_start_it) - 1));
            int is_end = static_cast<int>(std::distance(src_lons.data(), src_lon_end_it));

            if (src_is_global) {
                if (d_hi_x_norm <= src_lon_max) {
                    is_end = std::min(static_cast<int>(src_ni) - 1, is_end);
                }
            } else {
                is_end = std::min(static_cast<int>(src_ni) - 1, is_end);
            }

            auto src_lat_start_it = std::lower_bound(src_lats.data(), src_lats.data() + src_lats.extent(0), d_lo_y);
            auto src_lat_end_it = std::upper_bound(src_lats.data(), src_lats.data() + src_lats.extent(0), d_hi_y);

            int js_start = std::max(0, static_cast<int>(std::distance(src_lats.data(), src_lat_start_it) - 1));
            int js_end = std::min(static_cast<int>(src_nj) - 1, static_cast<int>(std::distance(src_lats.data(), src_lat_end_it)));

            for (int js = js_start; js <= js_end; ++js) {
                for (int is = is_start; is <= is_end; ++is) {
                    int is_actual = is;
                    if (src_is_global) {
                        is_actual = ((is % static_cast<int>(src_ni)) + static_cast<int>(src_ni)) % static_cast<int>(src_ni);
                    }
                    const std::size_t c_src = is_actual + js * src_ni;

                    if (has_src_mask && src_mask(c_src) == 0) continue;

                    double s_lo_x = src_lons(is_actual);
                    double s_hi_x = src_lons(is_actual + 1);
                    if (src_is_global && is >= static_cast<int>(src_ni)) {
                        s_lo_x += 360.0;
                        s_hi_x += 360.0;
                    }
                    double s_lo_y = src_lats(js);
                    double s_hi_y = src_lats(js + 1);

                    double overlap_area = 0.0;
                    if (is_spherical) {
                        overlap_area =
                            rect_overlap_spherical_nonuniform(s_lo_x, s_hi_x, s_lo_y, s_hi_y, d_lo_x_norm, d_hi_x_norm, d_lo_y, d_hi_y, is_degrees);
                    } else {
                        overlap_area = rect_overlap_nonuniform(s_lo_x, s_hi_x, s_lo_y, s_hi_y, d_lo_x_norm, d_hi_x_norm, d_lo_y, d_hi_y);
                    }

                    if (overlap_area > 1e-12) {
                        double area_src = 0.0;
                        if (has_src_areas) {
                            area_src = mesh_src_areas(c_src);
                        } else {
                            if (is_spherical) {
                                double dlon = s_hi_x - s_lo_x;
                                double s_lo_y_r = s_lo_y;
                                double s_hi_y_r = s_hi_y;
                                if (is_degrees) {
                                    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                                    dlon *= deg2rad;
                                    s_lo_y_r *= deg2rad;
                                    s_hi_y_r *= deg2rad;
                                }
                                area_src = dlon * (std::sin(s_hi_y_r) - std::sin(s_lo_y_r));
                            } else {
                                area_src = (s_hi_x - s_lo_x) * (s_hi_y - s_lo_y);
                            }
                        }
                        if (area_src <= 0.0) continue;

                        double weight = overlap_area / area_dst;

                        weights_vec.push_back(weight);
                        rows_vec.push_back(static_cast<index_t>(c_dst));
                        cols_vec.push_back(static_cast<index_t>(c_src));

                        frac_a_acc[c_src] += overlap_area / area_src;
                        frac_b_acc[c_dst] += overlap_area / area_dst;
                    }
                }
            }
        }
    }

    // Apply FracArea normalization if configured
    if (config.norm_type == NormType::FracArea) {
        for (std::size_t k = 0; k < weights_vec.size(); ++k) {
            auto j = static_cast<std::size_t>(rows_vec[k]);
            if (frac_b_acc[j] > 0.0) {
                weights_vec[k] /= frac_b_acc[j];
            }
        }
    }

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
        double s_lo_x = src_lons(i % src_ni);
        double s_hi_x = src_lons((i % src_ni) + 1);
        double s_lo_y = src_lats(i / src_ni);
        double s_hi_y = src_lats((i / src_ni) + 1);
        if (has_src_areas) {
            h_area_a(i) = mesh_src_areas(i);
        } else {
            if (is_spherical) {
                double dlon = s_hi_x - s_lo_x;
                double s_lo_y_r = s_lo_y;
                double s_hi_y_r = s_hi_y;
                if (is_degrees) {
                    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                    dlon *= deg2rad;
                    s_lo_y_r *= deg2rad;
                    s_hi_y_r *= deg2rad;
                }
                h_area_a(i) = dlon * (std::sin(s_hi_y_r) - std::sin(s_lo_y_r));
            } else {
                h_area_a(i) = (s_hi_x - s_lo_x) * (s_hi_y - s_lo_y);
            }
        }
        h_frac_a(i) = std::min(frac_a_acc[i], 1.0);
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        double d_lo_x = dst_lons(j % dst_ni);
        double d_hi_x = dst_lons((j % dst_ni) + 1);
        double d_lo_y = dst_lats(j / dst_ni);
        double d_hi_y = dst_lats((j / dst_ni) + 1);
        if (has_dst_areas) {
            h_area_b(j) = mesh_dst_areas(j);
        } else {
            if (is_spherical) {
                double dlon = d_hi_x - d_lo_x;
                double d_lo_y_r = d_lo_y;
                double d_hi_y_r = d_hi_y;
                if (is_degrees) {
                    constexpr double deg2rad = 3.14159265358979323846 / 180.0;
                    dlon *= deg2rad;
                    d_lo_y_r *= deg2rad;
                    d_hi_y_r *= deg2rad;
                }
                h_area_b(j) = dlon * (std::sin(d_hi_y_r) - std::sin(d_lo_y_r));
            } else {
                h_area_b(j) = (d_hi_x - d_lo_x) * (d_hi_y - d_lo_y);
            }
        }
        h_frac_b(j) = std::min(frac_b_acc[j], 1.0);
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

template InterpolationMatrix<Kokkos::HostSpace> generate_conservative_rect_nonuniform<Kokkos::HostSpace>(
    const topology::UnstructuredMesh<Kokkos::HostSpace> &, const topology::UnstructuredMesh<Kokkos::HostSpace> &, const RegridConfig &,
    const detail::RectilinearGridInfo &, const detail::RectilinearGridInfo &);

}  // namespace axis::solver
