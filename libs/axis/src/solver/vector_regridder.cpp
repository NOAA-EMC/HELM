// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/solver/vector_regridder.cpp
/// @brief Vector (rotating) regridding weight generation and application.
///
/// Derives paired x/y weight matrices from the scalar spatial matrix and the
/// grid rotation fields so contravariant vector components transfer correctly.

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
    if (W_scalar.has_linear_reconstruction()) throw std::invalid_argument("vector weights do not yet support field-dependent linear reconstruction");
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

}  // namespace axis::solver
