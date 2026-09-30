// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <axis/detail/memory_traits.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/solver/weight_generator.hpp>
#include <cmath>
#include <vector>

namespace axis::solver {

template <typename MemorySpace>
std::pair<InterpolationMatrix<MemorySpace>, InterpolationMatrix<MemorySpace>> VectorWeightGenerator<MemorySpace>::generate(
    const topology::UnstructuredMesh<MemorySpace> &src_mesh, const topology::UnstructuredMesh<MemorySpace> &dst_mesh,
    const GridRotation<MemorySpace> &src_rotation, const GridRotation<MemorySpace> &dst_rotation, const RegridConfig &config) {
    // 1. Generate the standard scalar spatial interpolation weight matrix
    auto W_scalar = WeightGenerator::generate<MemorySpace>(src_mesh, dst_mesh, config);

    const std::size_t n_src = W_scalar.n_src();
    const std::size_t n_dst = W_scalar.n_dst();
    const std::size_t nnz_scalar = W_scalar.nnz();

    // 2. We extract the scalar sparse matrix underlying Views directly
    const auto &rows = W_scalar.factor_row_view();
    const auto &cols = W_scalar.factor_col_view();
    const auto &vals = W_scalar.factor_list_view();

    // Each scalar entry W_ji yields exactly 2 entries in W_u and 2 entries in W_v
    std::size_t nnz_vector = nnz_scalar * 2;

    Kokkos::View<index_t *, MemorySpace> dev_u_rows("dev_u_rows", nnz_vector);
    Kokkos::View<index_t *, MemorySpace> dev_u_cols("dev_u_cols", nnz_vector);
    Kokkos::View<double *, MemorySpace> dev_u_vals("dev_u_vals", nnz_vector);

    Kokkos::View<index_t *, MemorySpace> dev_v_rows("dev_v_rows", nnz_vector);
    Kokkos::View<index_t *, MemorySpace> dev_v_cols("dev_v_cols", nnz_vector);
    Kokkos::View<double *, MemorySpace> dev_v_vals("dev_v_vals", nnz_vector);

    // Create stacked fraction and area views matching double n_src
    Kokkos::View<double *, MemorySpace> dev_frac_a("frac_a", n_src * 2);
    Kokkos::View<double *, MemorySpace> dev_area_a("area_a", n_src * 2);

    if constexpr (axis::detail::is_device_space_v<MemorySpace>) {
        using exec_space = typename MemorySpace::execution_space;

        Kokkos::parallel_for(
            "AssembleVectorWeightsDevice", Kokkos::RangePolicy<exec_space>(0, nnz_scalar), KOKKOS_LAMBDA(const std::size_t k) {
                index_t j = rows(k);  // dst cell index
                index_t i = cols(k);  // src cell index
                double w = vals(k);

                double a_src = src_rotation.alpha(i);
                double a_dst = dst_rotation.alpha(j);
                double diff_alpha = a_dst - a_src;

                double cos_d = Kokkos::cos(diff_alpha);
                double sin_d = Kokkos::sin(diff_alpha);

                std::size_t idx0 = k * 2;
                std::size_t idx1 = k * 2 + 1;

                // --- Coupled Weights for W_u ---
                dev_u_rows(idx0) = j;
                dev_u_cols(idx0) = i;
                dev_u_vals(idx0) = w * cos_d;

                dev_u_rows(idx1) = j;
                dev_u_cols(idx1) = i + n_src;  // Stacked v component index
                dev_u_vals(idx1) = w * sin_d;

                // --- Coupled Weights for W_v ---
                dev_v_rows(idx0) = j;
                dev_v_cols(idx0) = i;
                dev_v_vals(idx0) = -w * sin_d;

                dev_v_rows(idx1) = j;
                dev_v_cols(idx1) = i + n_src;  // Stacked v component index
                dev_v_vals(idx1) = w * cos_d;
            });

        auto dev_orig_frac_a = W_scalar.frac_a_view();
        auto dev_orig_area_a = W_scalar.area_a_view();

        Kokkos::parallel_for(
            "CopyFractionsDevice", Kokkos::RangePolicy<exec_space>(0, n_src), KOKKOS_LAMBDA(const std::size_t i) {
                dev_frac_a(i) = dev_orig_frac_a(i);
                dev_frac_a(i + n_src) = dev_orig_frac_a(i);
                dev_area_a(i) = dev_orig_area_a(i);
                dev_area_a(i + n_src) = dev_orig_area_a(i);
            });
    } else {
        // Host mirrors for mapping and trigonometric calculations
        auto h_rows = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rows);
        auto h_cols = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cols);
        auto h_vals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vals);
        auto h_src_alpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src_rotation.alpha);
        auto h_dst_alpha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dst_rotation.alpha);

        std::vector<index_t> u_rows, u_cols;
        std::vector<double> u_vals;
        u_rows.reserve(nnz_vector);
        u_cols.reserve(nnz_vector);
        u_vals.reserve(nnz_vector);

        std::vector<index_t> v_rows, v_cols;
        std::vector<double> v_vals;
        v_rows.reserve(nnz_vector);
        v_cols.reserve(nnz_vector);
        v_vals.reserve(nnz_vector);

        for (std::size_t k = 0; k < nnz_scalar; ++k) {
            index_t j = h_rows(k);  // dst cell index
            index_t i = h_cols(k);  // src cell index
            double w = h_vals(k);

            double a_src = h_src_alpha(i);
            double a_dst = h_dst_alpha(j);
            double diff_alpha = a_dst - a_src;

            double cos_d = std::cos(diff_alpha);
            double sin_d = std::sin(diff_alpha);

            // --- Coupled Weights for W_u ---
            // u_dst_j += w * cos(a_dst - a_src) * u_src_i
            u_rows.push_back(j);
            u_cols.push_back(i);
            u_vals.push_back(w * cos_d);

            // u_dst_j += w * sin(a_dst - a_src) * v_src_i
            u_rows.push_back(j);
            u_cols.push_back(i + n_src);  // Stacked v component index
            u_vals.push_back(w * sin_d);

            // --- Coupled Weights for W_v ---
            // v_dst_j += w * -sin(a_dst - a_src) * u_src_i
            v_rows.push_back(j);
            v_cols.push_back(i);
            v_vals.push_back(-w * sin_d);

            // v_dst_j += w * cos(a_dst - a_src) * v_src_i
            v_rows.push_back(j);
            v_cols.push_back(i + n_src);  // Stacked v component index
            v_vals.push_back(w * cos_d);
        }

        // Build the final host-space coupled vector matrices
        Kokkos::View<index_t *, Kokkos::HostSpace> host_u_rows("u_rows", nnz_vector);
        Kokkos::View<index_t *, Kokkos::HostSpace> host_u_cols("u_cols", nnz_vector);
        Kokkos::View<double *, Kokkos::HostSpace> host_u_vals("u_vals", nnz_vector);

        Kokkos::View<index_t *, Kokkos::HostSpace> host_v_rows("v_rows", nnz_vector);
        Kokkos::View<index_t *, Kokkos::HostSpace> host_v_cols("v_cols", nnz_vector);
        Kokkos::View<double *, Kokkos::HostSpace> host_v_vals("v_vals", nnz_vector);

        for (std::size_t k = 0; k < nnz_vector; ++k) {
            host_u_rows(k) = u_rows[k];
            host_u_cols(k) = u_cols[k];
            host_u_vals(k) = u_vals[k];

            host_v_rows(k) = v_rows[k];
            host_v_cols(k) = v_cols[k];
            host_v_vals(k) = v_vals[k];
        }

        // Copy to targeted MemorySpace
        Kokkos::deep_copy(dev_u_rows, host_u_rows);
        Kokkos::deep_copy(dev_u_cols, host_u_cols);
        Kokkos::deep_copy(dev_u_vals, host_u_vals);

        Kokkos::deep_copy(dev_v_rows, host_v_rows);
        Kokkos::deep_copy(dev_v_cols, host_v_cols);
        Kokkos::deep_copy(dev_v_vals, host_v_vals);

        // Copy the original fractions and areas
        auto dev_orig_frac_a = W_scalar.frac_a_view();
        auto dev_orig_area_a = W_scalar.area_a_view();

        Kokkos::parallel_for(
            "CopyFractionsHost", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_src), KOKKOS_LAMBDA(const std::size_t i) {
                dev_frac_a(i) = dev_orig_frac_a(i);
                dev_frac_a(i + n_src) = dev_orig_frac_a(i);
                dev_area_a(i) = dev_orig_area_a(i);
                dev_area_a(i + n_src) = dev_orig_area_a(i);
            });
    }

    InterpolationMatrix<MemorySpace> W_u(dev_u_vals, dev_u_rows, dev_u_cols, dev_frac_a, W_scalar.frac_b_view(), dev_area_a, W_scalar.area_b_view(),
                                         n_src * 2, n_dst);

    InterpolationMatrix<MemorySpace> W_v(dev_v_vals, dev_v_rows, dev_v_cols, dev_frac_a, W_scalar.frac_b_view(), dev_area_a, W_scalar.area_b_view(),
                                         n_src * 2, n_dst);

    return {std::move(W_u), std::move(W_v)};
}

template class VectorWeightGenerator<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class VectorWeightGenerator<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template class VectorWeightGenerator<Kokkos::HIPSpace>;
#endif

// ─────────────────────────────────────────────────────────────────────────────
// compute_rotation_angles — per-cell east-vector orientation (R8)
// ─────────────────────────────────────────────────────────────────────────────

template <typename MemorySpace>
Kokkos::View<double *, MemorySpace> compute_rotation_angles(const topology::UnstructuredMesh<MemorySpace> &mesh) {
    const std::size_t n_cells = mesh.n_cells();

    Kokkos::View<double *, Kokkos::HostSpace> h_alpha("rotation_angles", n_cells);

    // Projected (planar) meshes have no geographic east: identity rotation,
    // u/v is assumed expressed in the projection x/y basis on both grids.
    const auto csys = mesh.coord_system();
    const bool spherical = (csys == topology::CoordinateSystem::SphericalDeg || csys == topology::CoordinateSystem::SphericalRad);
    if (!spherical) {
        auto dev_alpha = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_alpha);
        return dev_alpha;  // all-zero
    }

    const double pi = 3.14159265358979323846;
    const double deg2rad = pi / 180.0;
    const double scale = (csys == topology::CoordinateSystem::SphericalRad) ? 1.0 : deg2rad;

    auto h_coords = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.node_coords_view());
    auto h_offsets = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.conn_offsets_view());
    auto h_indices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, mesh.conn_indices_view());

    // Unit vector (x, y, z) from lon/lat in radians.
    auto to_unit = [](double lon_rad, double lat_rad) {
        const double cl = std::cos(lat_rad);
        return std::array<double, 3>{cl * std::cos(lon_rad), cl * std::sin(lon_rad), std::sin(lat_rad)};
    };

    for (std::size_t c = 0; c < n_cells; ++c) {
        const auto start = static_cast<std::size_t>(h_offsets(c));
        const auto end = static_cast<std::size_t>(h_offsets(c + 1));

        // Only quadrilateral cells carry a meaningful local i-axis.
        // Polygons (MPAS/ICON) store cell vectors on the geographic basis.
        if (end - start != 4) {
            h_alpha(c) = 0.0;
            continue;
        }

        // CCW winding (n00, n10, n11, n01): +i axis runs west→east, along the
        // grid line through the west- and east-edge midpoints.
        const std::size_t v0 = static_cast<std::size_t>(h_indices(start + 0));
        const std::size_t v1 = static_cast<std::size_t>(h_indices(start + 1));
        const std::size_t v2 = static_cast<std::size_t>(h_indices(start + 2));
        const std::size_t v3 = static_cast<std::size_t>(h_indices(start + 3));

        auto unit_at = [&](std::size_t vi) { return to_unit(h_coords(vi, 0) * scale, h_coords(vi, 1) * scale); };
        auto renorm = [](std::array<double, 3> m) {
            const double r = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
            return std::array<double, 3>{m[0] / r, m[1] / r, m[2] / r};
        };

        const auto u0 = unit_at(v0);
        const auto u1 = unit_at(v1);
        const auto u2 = unit_at(v2);
        const auto u3 = unit_at(v3);

        // Cell-center approximation: normalized mean of the four unit vertices.
        auto rc = renorm({(u0[0] + u1[0] + u2[0] + u3[0]) / 4.0, (u0[1] + u1[1] + u2[1] + u3[1]) / 4.0, (u0[2] + u1[2] + u2[2] + u3[2]) / 4.0});

        // Local east / north basis at the cell center (ẑ × r, r × east).
        const std::array<double, 3> east_raw{-rc[1], rc[0], 0.0};
        const double en = std::sqrt(east_raw[0] * east_raw[0] + east_raw[1] * east_raw[1]);
        if (en < 1.0e-300) {  // at a pole: any direction is east; identity rotation
            h_alpha(c) = 0.0;
            continue;
        }
        const std::array<double, 3> e{east_raw[0] / en, east_raw[1] / en, 0.0};
        const std::array<double, 3> north{rc[1] * e[2] - rc[2] * e[1], rc[2] * e[0] - rc[0] * e[2], rc[0] * e[1] - rc[1] * e[0]};

        // Local +i direction: tangent, at the cell center, to the great circle
        // through the west- and east-edge midpoints. This is exact for lat-lon
        // (the great circle peaks at the center → tangent ∥ east → α = 0) and
        // for gnomonic cubed-sphere tiles (the a-line IS that great circle and
        // the vertex-mean center lies on it by symmetry).
        const auto wmid = renorm({u0[0] + u3[0], u0[1] + u3[1], u0[2] + u3[2]});
        const auto emid = renorm({u1[0] + u2[0], u1[1] + u2[1], u1[2] + u2[2]});
        std::array<double, 3> gnorm{wmid[1] * emid[2] - wmid[2] * emid[1], wmid[2] * emid[0] - wmid[0] * emid[2],
                                    wmid[0] * emid[1] - wmid[1] * emid[0]};
        const double gn = std::sqrt(gnorm[0] * gnorm[0] + gnorm[1] * gnorm[1] + gnorm[2] * gnorm[2]);
        if (gn < 1.0e-300) {  // midpoints coincide/antipodal: degenerate cell
            h_alpha(c) = 0.0;
            continue;
        }
        gnorm[0] /= gn;
        gnorm[1] /= gn;
        gnorm[2] /= gn;

        // Project the center onto the great-circle plane, then take the
        // travel-direction tangent (gnorm × p) from west to east.
        const double cd = rc[0] * gnorm[0] + rc[1] * gnorm[1] + rc[2] * gnorm[2];
        auto p = renorm({rc[0] - cd * gnorm[0], rc[1] - cd * gnorm[1], rc[2] - cd * gnorm[2]});
        const std::array<double, 3> t{gnorm[1] * p[2] - gnorm[2] * p[1], gnorm[2] * p[0] - gnorm[0] * p[2], gnorm[0] * p[1] - gnorm[1] * p[0]};

        const double x = t[0] * e[0] + t[1] * e[1] + t[2] * e[2];              // east component
        const double y = t[0] * north[0] + t[1] * north[1] + t[2] * north[2];  // north component
        if (std::abs(x) < 1.0e-300 && std::abs(y) < 1.0e-300) {
            h_alpha(c) = 0.0;  // degenerate cell
            continue;
        }
        h_alpha(c) = std::atan2(y, x);  // CCW angle from east toward north
    }

    auto dev_alpha = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_alpha);
    return dev_alpha;
}

template Kokkos::View<double *, Kokkos::HostSpace> compute_rotation_angles<Kokkos::HostSpace>(const topology::UnstructuredMesh<Kokkos::HostSpace> &);

#ifdef KOKKOS_ENABLE_CUDA
template Kokkos::View<double *, Kokkos::CudaSpace> compute_rotation_angles<Kokkos::CudaSpace>(const topology::UnstructuredMesh<Kokkos::CudaSpace> &);
#endif

#ifdef KOKKOS_ENABLE_HIP
template Kokkos::View<double *, Kokkos::HIPSpace> compute_rotation_angles<Kokkos::HIPSpace>(const topology::UnstructuredMesh<Kokkos::HIPSpace> &);
#endif

}  // namespace axis::solver
