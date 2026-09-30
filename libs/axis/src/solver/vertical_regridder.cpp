// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <axis/detail/tspack.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <stdexcept>
#include <string>

namespace axis::solver {

namespace {

KOKKOS_INLINE_FUNCTION double evaluate_column_spline_target(double target, const double *src_x, const double *src_y, const double *d,
                                                            std::size_t n_src, double tension) {
    std::size_t idx = 0;
    while (idx < n_src - 2 && src_x[idx + 1] < target) {
        idx++;
    }
    return axis::detail::tspack::evaluate_spline(target, src_x[idx], src_x[idx + 1], src_y[idx], src_y[idx + 1], d[idx], d[idx + 1], tension);
}

KOKKOS_INLINE_FUNCTION void get_column_scratch_pointers(double *scratch_data, std::size_t n_src, double *&src_x, double *&src_y, double *&d,
                                                        double *&scratch_arr) {
    src_x = scratch_data;
    src_y = src_x + n_src;
    d = src_y + n_src;
    scratch_arr = d + n_src;
}

inline void validate_vertical_regrid_params(std::size_t n_src, std::size_t n_dst, double tension) {
    if (tension < 0.0) {
        throw std::invalid_argument("VerticalRegridder: Tension parameter must be non-negative");
    }
    constexpr std::size_t MAX_LEVELS = 256;
    if (n_src > MAX_LEVELS || n_dst > MAX_LEVELS) {
        throw std::invalid_argument("VerticalRegridder: Vertical level dimensions exceed hard cap of 256 levels");
    }
}

}  // namespace

template <typename MemorySpace>
void VerticalRegridder<MemorySpace>::interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                                                 Kokkos::View<const double *, MemorySpace> src_levels,
                                                 Kokkos::View<const double *, MemorySpace> dst_levels, double tension) {
    const std::size_t n_col = src_field.extent(0);
    const std::size_t n_src = src_levels.extent(0);
    const std::size_t n_dst = dst_levels.extent(0);

    if (n_col != dst_field.extent(0)) {
        throw std::invalid_argument("VerticalRegridder: Source and destination column extents mismatch");
    }
    if (src_field.extent(1) != n_src) {
        throw std::invalid_argument("VerticalRegridder: Source field levels and source levels coordinates mismatch");
    }
    if (dst_field.extent(1) != n_dst) {
        throw std::invalid_argument("VerticalRegridder: Destination field levels and destination levels coordinates mismatch");
    }
    validate_vertical_regrid_params(n_src, n_dst, tension);

    using execution_space = typename MemorySpace::execution_space;
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename policy_type::member_type;

    std::size_t scratch_bytes = 4 * n_src * sizeof(double);
    policy_type policy(static_cast<int>(n_col), Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    constexpr std::size_t MAX_LEVELS = 256;

    Kokkos::parallel_for(
        "VerticalInterpolate1D", policy, KOKKOS_LAMBDA(const member_type &team) {
            const std::size_t c = team.league_rank();

            Kokkos::View<double *, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged> scratch_view(team.team_scratch(0),
                                                                                                                         4 * n_src);
            double *src_x, *src_y, *d, *scratch_arr;
            get_column_scratch_pointers(scratch_view.data(), n_src, src_x, src_y, d, scratch_arr);

            // Check if coordinates are descending
            bool is_descending = n_src > 1 && src_levels(0) > src_levels(n_src - 1);

            // Cooperatively fill src_x and src_y
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_src), [&](const std::size_t i) {
                std::size_t raw_i = is_descending ? n_src - 1 - i : i;
                src_y[i] = src_field(c, raw_i);
                src_x[i] = src_levels(raw_i);
            });
            team.team_barrier();

            // Thread 0 solves the column spline
            if (team.team_rank() == 0) {
                axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch_arr);
            }
            team.team_barrier();

            // Check if destination coordinates are descending
            bool dst_descending = n_dst > 1 && dst_levels(0) > dst_levels(n_dst - 1);

            // Cooperatively evaluate spline for each target destination level
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_dst), [&](const std::size_t j) {
                std::size_t raw_j = dst_descending ? n_dst - 1 - j : j;
                double target = dst_levels(raw_j);
                dst_field(c, raw_j) = evaluate_column_spline_target(target, src_x, src_y, d, n_src, tension);
            });
        });
}

template <typename MemorySpace>
void VerticalRegridder<MemorySpace>::interpolate(Kokkos::View<const double **, MemorySpace> src_field, Kokkos::View<double **, MemorySpace> dst_field,
                                                 Kokkos::View<const double **, MemorySpace> src_levels,
                                                 Kokkos::View<const double **, MemorySpace> dst_levels, double tension) {
    const std::size_t n_col = src_field.extent(0);
    const std::size_t n_src = src_levels.extent(1);
    const std::size_t n_dst = dst_levels.extent(1);

    if (n_col != dst_field.extent(0) || n_col != src_levels.extent(0) || n_col != dst_levels.extent(0)) {
        throw std::invalid_argument("VerticalRegridder: Grid column dimensions mismatch across fields and level views");
    }
    if (src_field.extent(1) != n_src) {
        throw std::invalid_argument("VerticalRegridder: Source field levels and source levels coordinates mismatch");
    }
    if (dst_field.extent(1) != n_dst) {
        throw std::invalid_argument("VerticalRegridder: Destination field levels and destination levels coordinates mismatch");
    }
    validate_vertical_regrid_params(n_src, n_dst, tension);

    using execution_space = typename MemorySpace::execution_space;
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename policy_type::member_type;

    std::size_t scratch_bytes = 4 * n_src * sizeof(double);
    policy_type policy(static_cast<int>(n_col), Kokkos::AUTO);
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));

    constexpr std::size_t MAX_LEVELS = 256;

    Kokkos::parallel_for(
        "VerticalInterpolate2D", policy, KOKKOS_LAMBDA(const member_type &team) {
            const std::size_t c = team.league_rank();

            Kokkos::View<double *, typename execution_space::scratch_memory_space, Kokkos::MemoryUnmanaged> scratch_view(team.team_scratch(0),
                                                                                                                         4 * n_src);
            double *src_x, *src_y, *d, *scratch_arr;
            get_column_scratch_pointers(scratch_view.data(), n_src, src_x, src_y, d, scratch_arr);

            // Check if coordinates are descending for this column
            bool is_descending = n_src > 1 && src_levels(c, 0) > src_levels(c, n_src - 1);

            // Cooperatively fill src_x and src_y
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_src), [&](const std::size_t i) {
                std::size_t raw_i = is_descending ? n_src - 1 - i : i;
                src_y[i] = src_field(c, raw_i);
                src_x[i] = src_levels(c, raw_i);
            });
            team.team_barrier();

            // Thread 0 solves the column spline
            if (team.team_rank() == 0) {
                axis::detail::tspack::solve_column_spline<MAX_LEVELS>(src_x, src_y, n_src, tension, d, scratch_arr);
            }
            team.team_barrier();

            // Check if destination coordinates are descending for this column
            bool dst_descending = n_dst > 1 && dst_levels(c, 0) > dst_levels(c, n_dst - 1);

            // Cooperatively evaluate spline for each target destination level
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, n_dst), [&](const std::size_t j) {
                std::size_t raw_j = dst_descending ? n_dst - 1 - j : j;
                double target = dst_levels(c, raw_j);
                dst_field(c, raw_j) = evaluate_column_spline_target(target, src_x, src_y, d, n_src, tension);
            });
        });
}

// Explicit template instantiations
template class VerticalRegridder<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class VerticalRegridder<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template class VerticalRegridder<Kokkos::HIPSpace>;
#endif

}  // namespace axis::solver
