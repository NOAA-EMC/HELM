// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/structured_grid.cpp
/// @brief StructuredGrid<MemorySpace> implementation.
///
/// Implements the constructor, accessors, set_corners, corner synthesis, and
/// the to_unstructured() conversion via Kokkos parallel kernel. Each logical
/// cell (i,j) maps to one quadrilateral element with 4 corner nodes.

#include <Kokkos_Core.hpp>
#include <axis/detail/memory_traits.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace axis::topology {

namespace {

std::size_t checked_product(std::size_t a, std::size_t b, const char *label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("StructuredGrid: size overflow computing ") + label);
    }
    return a * b;
}

std::size_t checked_increment(std::size_t value, const char *label) {
    if (value == std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error(std::string("StructuredGrid: size overflow computing ") + label);
    }
    return value + 1;
}

template <class MemorySpace>
void validate_rectilinear_centers(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> lon, Kokkos::View<double *, MemorySpace> lat,
                                  double period, bool wrap_longitude, bool longitude_increasing) {
    using exec_space = typename detail::exec_space_t<MemorySpace>;
    const double tolerance = 1e-10;
    std::size_t errors = 0;
    auto x = lon;
    auto y = lat;
    const double expected_direction = longitude_increasing ? 1.0 : -1.0;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    Kokkos::parallel_reduce(
        "validate_rectilinear_centers", Range(exec, 0, ni * nj),
        KOKKOS_LAMBDA(const std::size_t cell, std::size_t &count) {
            const std::size_t i = cell % ni;
            const std::size_t j = cell / ni;
            double lon_reference_step = ni > 1 ? x(1) - x(0) : 0.0;
            if (wrap_longitude) {
                if (lon_reference_step > 0.5 * period)
                    lon_reference_step -= period;
                else if (lon_reference_step < -0.5 * period)
                    lon_reference_step += period;
            }
            const double lat_reference_step = nj > 1 ? y(ni) - y(0) : 0.0;
            const double x_value = x(cell);
            const double y_value = y(cell);
            if (!Kokkos::isfinite(x_value) || !Kokkos::isfinite(y_value)) {
                ++count;
                return;
            }

            double x_difference = x_value - x(i);
            if (wrap_longitude) {
                if (x_difference > 0.5 * period)
                    x_difference -= period;
                else if (x_difference < -0.5 * period)
                    x_difference += period;
            }
            const double x_scale = Kokkos::fmax(1.0, Kokkos::fmax(Kokkos::abs(x_value), Kokkos::abs(x(i))));
            const double y_difference = y_value - y(j * ni);
            const double y_scale = Kokkos::fmax(1.0, Kokkos::fmax(Kokkos::abs(y_value), Kokkos::abs(y(j * ni))));
            bool invalid = Kokkos::abs(x_difference) > tolerance * x_scale || Kokkos::abs(y_difference) > tolerance * y_scale;

            if (i > 0 && ni > 1) {
                double step = x(i) - x(i - 1);
                if (wrap_longitude) {
                    if (step > 0.5 * period)
                        step -= period;
                    else if (step < -0.5 * period)
                        step += period;
                }
                invalid = invalid || Kokkos::abs(step) <= tolerance || step * lon_reference_step <= 0.0 || step * expected_direction <= 0.0;
            }
            if (j > 0 && nj > 1) {
                const double step = y(j * ni) - y((j - 1) * ni);
                invalid = invalid || Kokkos::abs(step) <= tolerance || step * lat_reference_step <= 0.0;
            }
            if (invalid) ++count;
        },
        errors);
    if (errors != 0) {
        throw std::invalid_argument("StructuredGrid: RectilinearMidpoint requires finite, separable, strictly monotone center axes");
    }
}

template <class MemorySpace>
void validate_axis_bounds(Kokkos::View<double *, MemorySpace> bounds, double period, bool wrap, bool require_period_span) {
    using exec_space = typename detail::exec_space_t<MemorySpace>;
    const std::size_t n = bounds.extent(0);
    auto values = bounds;
    std::size_t errors = 0;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    Kokkos::parallel_reduce(
        "validate_rectilinear_bounds", Range(exec, 0, n),
        KOKKOS_LAMBDA(const std::size_t k, std::size_t &count) {
            if (!Kokkos::isfinite(values(k))) {
                ++count;
                return;
            }
            if (k > 0) {
                double first_step = values(1) - values(0);
                if (wrap) {
                    if (first_step > 0.5 * period)
                        first_step -= period;
                    else if (first_step < -0.5 * period)
                        first_step += period;
                }
                double step = values(k) - values(k - 1);
                if (wrap) {
                    if (step > 0.5 * period)
                        step -= period;
                    else if (step < -0.5 * period)
                        step += period;
                }
                if (Kokkos::abs(step) <= 1e-12 || step * first_step <= 0.0) ++count;
            }
        },
        errors);
    if (errors != 0) throw std::invalid_argument("StructuredGrid: rectilinear bounds must be finite and strictly monotone");
    if (require_period_span) {
        double span = 0.0;
        Kokkos::parallel_reduce(
            "rectilinear_bounds_period_span", Range(exec, 0, n),
            KOKKOS_LAMBDA(const std::size_t k, double &sum) {
                if (k == 0) sum -= values(k);
                if (k + 1 == n) sum += values(k);
            },
            span);
        span = std::abs(span);
        if (Kokkos::abs(span - period) > 1e-10 * Kokkos::fmax(1.0, period)) {
            throw std::invalid_argument("StructuredGrid: periodic rectilinear longitude bounds must span the declared period");
        }
    }
}

template <class MemorySpace>
void validate_centers_inside_rectilinear_bounds(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> lon,
                                                Kokkos::View<double *, MemorySpace> lat, Kokkos::View<double *, MemorySpace> lon_bounds,
                                                Kokkos::View<double *, MemorySpace> lat_bounds, bool wrap_lon, double period) {
    using exec_space = typename detail::exec_space_t<MemorySpace>;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    auto x = lon;
    auto y = lat;
    auto xb = lon_bounds;
    auto yb = lat_bounds;
    std::size_t errors = 0;
    Kokkos::parallel_reduce(
        "validate_centers_inside_rectilinear_bounds", Range(exec, 0, ni * nj),
        KOKKOS_LAMBDA(const std::size_t cell, std::size_t &count) {
            const std::size_t i = cell % ni;
            const std::size_t j = cell / ni;
            double center_x = x(cell);
            const double west = xb(i);
            const double east = xb(i + 1);
            if (wrap_lon) {
                const double midpoint = 0.5 * (west + east);
                center_x += Kokkos::floor((midpoint - center_x) / period + 0.5) * period;
            }
            const double low_x = Kokkos::fmin(west, east);
            const double high_x = Kokkos::fmax(west, east);
            const double low_y = Kokkos::fmin(yb(j), yb(j + 1));
            const double high_y = Kokkos::fmax(yb(j), yb(j + 1));
            if (!Kokkos::isfinite(center_x) || !Kokkos::isfinite(y(cell)) || center_x < low_x - 1e-10 || center_x > high_x + 1e-10 ||
                y(cell) < low_y - 1e-10 || y(cell) > high_y + 1e-10)
                ++count;
        },
        errors);
    if (errors != 0) throw std::invalid_argument("StructuredGrid: every cell center must lie inside its declared rectilinear bounds");
}

template <class MemorySpace>
void validate_approximate_cells(std::size_t ni, std::size_t nj, std::size_t nip1, Kokkos::View<double *, MemorySpace> centers_lon,
                                Kokkos::View<double *, MemorySpace> centers_lat, Kokkos::View<double *, MemorySpace> corners_lon,
                                Kokkos::View<double *, MemorySpace> corners_lat, bool spherical, double period) {
    using exec_space = typename detail::exec_space_t<MemorySpace>;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    const std::size_t n_cells = ni * nj;
    std::size_t invalid_cells = 0;
    Kokkos::parallel_reduce(
        "validate_approximate_cell_geometry", Range(exec, 0, n_cells),
        KOKKOS_LAMBDA(const std::size_t cell, std::size_t &errors) {
            const std::size_t i = cell % ni;
            const std::size_t j = cell / ni;
            const std::size_t ids[4] = {i + j * nip1, i + 1 + j * nip1, i + 1 + (j + 1) * nip1, i + (j + 1) * nip1};
            double x[4], y[4];
            bool invalid = false;
            for (int k = 0; k < 4; ++k) {
                x[k] = corners_lon(ids[k]);
                y[k] = corners_lat(ids[k]);
                invalid = invalid || !Kokkos::isfinite(x[k]) || !Kokkos::isfinite(y[k]);
                if (spherical && k > 0) {
                    const double delta = x[k] - x[0];
                    if (delta > 0.5 * period)
                        x[k] -= period;
                    else if (delta < -0.5 * period)
                        x[k] += period;
                }
            }
            if (invalid) {
                ++errors;
                return;
            }
            double twice_area = 0.0;
            double scale = 1.0;
            for (int k = 0; k < 4; ++k) {
                const int next = (k + 1) % 4;
                twice_area += x[k] * y[next] - x[next] * y[k];
                scale = Kokkos::fmax(scale, Kokkos::fmax(Kokkos::abs(x[k]), Kokkos::abs(y[k])));
            }
            const double threshold = 1e-12 * scale * scale;
            if (Kokkos::abs(twice_area) <= threshold) {
                ++errors;
                return;
            }
            const double center_lon = centers_lon(cell);
            const double center_x =
                spherical && Kokkos::abs(center_lon - x[0]) > 0.5 * period ? center_lon + (center_lon < x[0] ? period : -period) : center_lon;
            const double center_y = centers_lat(cell);
            if (!Kokkos::isfinite(center_x) || !Kokkos::isfinite(center_y)) {
                ++errors;
                return;
            }
            int positive = 0;
            int negative = 0;
            int turn_positive = 0;
            int turn_negative = 0;
            for (int k = 0; k < 4; ++k) {
                const int next = (k + 1) % 4;
                const int after = (k + 2) % 4;
                const double cross = (x[next] - x[k]) * (center_y - y[k]) - (y[next] - y[k]) * (center_x - x[k]);
                if (cross > threshold)
                    ++positive;
                else if (cross < -threshold)
                    ++negative;
                const double turn = (x[next] - x[k]) * (y[after] - y[next]) - (y[next] - y[k]) * (x[after] - x[next]);
                if (turn > threshold)
                    ++turn_positive;
                else if (turn < -threshold)
                    ++turn_negative;
            }
            if ((positive != 0 && negative != 0) || (turn_positive != 0 && turn_negative != 0) || (turn_positive != 4 && turn_negative != 4))
                ++errors;
        },
        invalid_cells);
    if (invalid_cells != 0) {
        throw std::invalid_argument(
            "StructuredGrid::to_unstructured: curvilinear reconstruction produced non-finite, degenerate, folded, or center-excluding cells");
    }
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
StructuredGrid<MemorySpace>::StructuredGrid(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> center_lon,
                                            Kokkos::View<double *, MemorySpace> center_lat, CoordinateSystem coord_sys)
    : ni_(ni), nj_(nj), center_lon_(std::move(center_lon)), center_lat_(std::move(center_lat)), coord_sys_(coord_sys) {
    if (ni_ == 0 || nj_ == 0) {
        throw std::invalid_argument("StructuredGrid: ni and nj must be positive");
    }
    const auto n_cells = checked_product(ni_, nj_, "ni*nj");
    if (center_lon_.extent(0) != n_cells) {
        throw std::invalid_argument("StructuredGrid: center_lon extent (" + std::to_string(center_lon_.extent(0)) + ") does not match ni*nj (" +
                                    std::to_string(n_cells) + ")");
    }
    if (center_lat_.extent(0) != n_cells) {
        throw std::invalid_argument("StructuredGrid: center_lat extent (" + std::to_string(center_lat_.extent(0)) + ") does not match ni*nj (" +
                                    std::to_string(n_cells) + ")");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Coordinate accessors
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::center_lon() const noexcept {
    return field_view<const double, 1>{center_lon_.data(), center_lon_.extent(0)};
}

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::center_lat() const noexcept {
    return field_view<const double, 1>{center_lat_.data(), center_lat_.extent(0)};
}

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::corner_lon() const noexcept {
    return field_view<const double, 1>{corner_lon_.data(), corner_lon_.extent(0)};
}

template <class MemorySpace>
field_view<const double, 1> StructuredGrid<MemorySpace>::corner_lat() const noexcept {
    return field_view<const double, 1>{corner_lat_.data(), corner_lat_.extent(0)};
}

// ─────────────────────────────────────────────────────────────────────────────
// set_corners
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void StructuredGrid<MemorySpace>::set_corners(Kokkos::View<double *, MemorySpace> corner_lon, Kokkos::View<double *, MemorySpace> corner_lat) {
    const std::size_t nip1 = checked_increment(ni_, "ni+1");
    const std::size_t njp1 = checked_increment(nj_, "nj+1");
    const std::size_t expected = checked_product(nip1, njp1, "(ni+1)*(nj+1)");
    if (corner_lon.extent(0) != expected) {
        throw std::invalid_argument("StructuredGrid::set_corners: corner_lon extent (" + std::to_string(corner_lon.extent(0)) +
                                    ") does not match (ni+1)*(nj+1) (" + std::to_string(expected) + ")");
    }
    if (corner_lat.extent(0) != expected) {
        throw std::invalid_argument("StructuredGrid::set_corners: corner_lat extent (" + std::to_string(corner_lat.extent(0)) +
                                    ") does not match (ni+1)*(nj+1) (" + std::to_string(expected) + ")");
    }
    corner_lon_ = std::move(corner_lon);
    corner_lat_ = std::move(corner_lat);
    corners_explicit_ = true;
}

template <class MemorySpace>
void StructuredGrid<MemorySpace>::set_rectilinear_bounds(Kokkos::View<double *, MemorySpace> longitude_bounds,
                                                         Kokkos::View<double *, MemorySpace> latitude_bounds) {
    const std::size_t expected_lon = checked_increment(ni_, "ni+1");
    const std::size_t expected_lat = checked_increment(nj_, "nj+1");
    if (longitude_bounds.extent(0) != expected_lon || latitude_bounds.extent(0) != expected_lat) {
        throw std::invalid_argument("StructuredGrid::set_rectilinear_bounds: bounds extents must be ni+1 and nj+1");
    }
    rectilinear_lon_bounds_ = std::move(longitude_bounds);
    rectilinear_lat_bounds_ = std::move(latitude_bounds);
    rectilinear_bounds_explicit_ = true;
}

template <class MemorySpace>
void StructuredGrid<MemorySpace>::set_gaussian_latitude_weights(Kokkos::View<double *, MemorySpace> latitude_weights) {
    if (latitude_weights.extent(0) != nj_) {
        throw std::invalid_argument("StructuredGrid::set_gaussian_latitude_weights: weights extent must equal nj");
    }
    gaussian_latitude_weights_ = std::move(latitude_weights);
    gaussian_weights_explicit_ = true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared corner-synthesis kernel (single source of truth). Interior corners
// interpolate the four surrounding centers. Exterior corners use the tensor
// product of one-sided linear extrapolation weights, so the full-grid and
// halo-aware band paths produce identical geometry.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

/// Synthesize `nrows` rows of Cell_Corners for the global corner rows
/// [cj_lo, cj_lo + nrows) over a full grid of ni × nj centers. The output views
/// are (ni+1) * nrows, indexed so corner (ci, cj) lives at
/// ci + (cj - cj_lo) * (ni + 1). A linear one-sided stencil supplies missing
/// exterior neighbors; a singleton axis has no slope information and is held
/// constant along that axis.
template <class MemorySpace>
void synthesize_corner_rows(std::size_t ni, std::size_t nj, Kokkos::View<double *, MemorySpace> center_lon,
                            Kokkos::View<double *, MemorySpace> center_lat, std::size_t cj_lo, std::size_t nrows, bool is_periodic, double period,
                            bool increasing, bool unwrap_longitude, Kokkos::View<double *, MemorySpace> corner_lon,
                            Kokkos::View<double *, MemorySpace> corner_lat) {
    using exec_space = typename detail::exec_space_t<MemorySpace>;

    if (ni > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
        nj > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
        cj_lo > static_cast<std::size_t>(std::numeric_limits<long long>::max()) ||
        nrows > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("synthesize_corners: dimensions exceed supported kernel index ranges");
    }

    const std::size_t nip1 = ni + 1;

    auto clon = corner_lon;
    auto clat = corner_lat;

    exec_space exec{};
    const std::size_t n_output = checked_product(nip1, nrows, "(ni+1)*nrows");
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    Kokkos::parallel_for(
        "synthesize_corners", Range(exec, 0, n_output), KOKKOS_LAMBDA(const std::size_t idx) {
            const std::size_t ci = idx % nip1;
            const std::size_t cj = cj_lo + idx / nip1;

            std::size_t i0 = 0, i1 = 0;
            double wi0 = 1.0, wi1 = 0.0, shift0 = 0.0, shift1 = 0.0;
            if (ni > 1) {
                if (is_periodic) {
                    if (ci == 0 || ci == ni) {
                        i0 = ni - 1;
                        i1 = 0;
                        wi0 = wi1 = 0.5;
                        if (ci == 0)
                            shift0 = increasing ? -period : period;
                        else
                            shift1 = increasing ? period : -period;
                    } else {
                        i0 = ci - 1;
                        i1 = ci;
                        wi0 = wi1 = 0.5;
                    }
                } else if (ci == 0) {
                    i0 = 0;
                    i1 = 1;
                    wi0 = 1.5;
                    wi1 = -0.5;
                } else if (ci == ni) {
                    i0 = ni - 2;
                    i1 = ni - 1;
                    wi0 = -0.5;
                    wi1 = 1.5;
                } else {
                    i0 = ci - 1;
                    i1 = ci;
                    wi0 = wi1 = 0.5;
                }
            }

            std::size_t j0 = 0, j1 = 0;
            double wj0 = 1.0, wj1 = 0.0;
            if (nj > 1) {
                if (cj == 0) {
                    j0 = 0;
                    j1 = 1;
                    wj0 = 1.5;
                    wj1 = -0.5;
                } else if (cj == nj) {
                    j0 = nj - 2;
                    j1 = nj - 1;
                    wj0 = -0.5;
                    wj1 = 1.5;
                } else {
                    j0 = cj - 1;
                    j1 = cj;
                    wj0 = wj1 = 0.5;
                }
            }

            const std::size_t cells[4] = {i0 + j0 * ni, i1 + j0 * ni, i0 + j1 * ni, i1 + j1 * ni};
            const double weights[4] = {wi0 * wj0, wi1 * wj0, wi0 * wj1, wi1 * wj1};
            const double shifts[4] = {shift0, shift1, shift0, shift1};
            double lon_value = 0.0;
            double lat_value = 0.0;
            for (int sample = 0; sample < 4; ++sample) {
                double sample_lon = center_lon(cells[sample]) + shifts[sample];
                const bool declared_seam_vertex = is_periodic && (ci == 0 || ci == ni);
                if (unwrap_longitude && ni > 1 && (sample & 1) != 0 && !declared_seam_vertex) {
                    const double base_lon = center_lon(cells[sample - 1]) + shifts[sample - 1];
                    const double delta = sample_lon - base_lon;
                    if (Kokkos::abs(delta) > 0.5 * period) {
                        sample_lon += delta > 0.0 ? -period : period;
                    }
                }
                lon_value += weights[sample] * sample_lon;
                lat_value += weights[sample] * center_lat(cells[sample]);
            }
            clon(idx) = lon_value;
            clat(idx) = lat_value;
        });

    // Only wait for the execution-space instance used by this grid's kernels;
    // a global fence would also serialize unrelated Kokkos work on the rank.
    exec.fence("synthesize_corners_fence");
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// synthesize_corners — reconstruct vertices from center samples
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void StructuredGrid<MemorySpace>::synthesize_corners(bool is_periodic, double period, bool increasing,
                                                     Kokkos::View<double *, MemorySpace> &corner_lon,
                                                     Kokkos::View<double *, MemorySpace> &corner_lat) const {
    const std::size_t nip1 = checked_increment(ni_, "ni+1");
    const std::size_t njp1 = checked_increment(nj_, "nj+1");
    const std::size_t n_corners = checked_product(nip1, njp1, "(ni+1)*(nj+1)");

    corner_lon = Kokkos::View<double *, MemorySpace>("structured_grid_corner_lon", n_corners);
    corner_lat = Kokkos::View<double *, MemorySpace>("structured_grid_corner_lat", n_corners);

    // Whole grid: corner rows [0, nj_+1); band synthesis calls the same kernel
    // with global row coordinates and returns a selected subset.
    const bool spherical = coord_sys_ != CoordinateSystem::Cartesian3D;
    synthesize_corner_rows(ni_, nj_, center_lon_, center_lat_, /*cj_lo=*/0, njp1, is_periodic && spherical, period, increasing, spherical, corner_lon,
                           corner_lat);
}

// ─────────────────────────────────────────────────────────────────────────────
// to_unstructured — the core conversion (Requirement 18.1, 18.2, 18.3)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> StructuredGrid<MemorySpace>::to_unstructured() const {
    const double period = coord_sys_ == CoordinateSystem::SphericalRad ? 2.0 * std::numbers::pi : 360.0;
    const LongitudePeriodicity seam{false, period, true};
    CornerPolicy policy = gaussian_weights_explicit_ ? CornerPolicy::GaussianLatLon : CornerPolicy::RectilinearMidpoint;

    // The no-argument API is the convenience path for center-only inputs:
    // retain exact rectilinear midpoint construction when the axes are
    // separable, but use the explicit approximate policy for genuinely
    // curvilinear center fields (e.g., PROJ-transformed grids). Strict callers
    // can still request RequireExplicit through the policy overload.
    if (!corners_explicit_ && !rectilinear_bounds_explicit_ && !gaussian_weights_explicit_) {
        try {
            validate_rectilinear_centers(ni_, nj_, center_lon_, center_lat_, seam.period, false, seam.increasing);
        } catch (const std::invalid_argument &) {
            policy = CornerPolicy::CurvilinearApproximate;
        }
    }
    return to_unstructured(policy, seam);
}

template <class MemorySpace>
UnstructuredMesh<MemorySpace> StructuredGrid<MemorySpace>::to_unstructured(CornerPolicy corner_policy, LongitudePeriodicity seam) const {
    using exec_space = typename detail::exec_space_t<MemorySpace>;
    using Range = Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>;
    exec_space exec{};
    const bool spherical = coord_sys_ != CoordinateSystem::Cartesian3D;
    if (corner_policy != CornerPolicy::RequireExplicit && corner_policy != CornerPolicy::GaussianLatLon &&
        corner_policy != CornerPolicy::RectilinearMidpoint && corner_policy != CornerPolicy::CurvilinearApproximate) {
        throw std::invalid_argument("StructuredGrid::to_unstructured: unknown corner policy");
    }
    if ((seam.periodic || spherical) && (!std::isfinite(seam.period) || seam.period <= 0.0)) {
        throw std::invalid_argument("StructuredGrid::to_unstructured: longitude period must be finite and positive");
    }
    const bool explicit_geometry = corners_explicit_ || rectilinear_bounds_explicit_ || gaussian_weights_explicit_;

    // Reconstruct into local views to keep source geometry immutable and make
    // concurrent conversions with different seam declarations independent.
    Kokkos::View<double *, MemorySpace> clon;
    Kokkos::View<double *, MemorySpace> clat;
    if (!explicit_geometry) {
        if (corner_policy == CornerPolicy::RequireExplicit || corner_policy == CornerPolicy::GaussianLatLon) {
            throw std::invalid_argument("StructuredGrid::to_unstructured: selected corner policy requires explicit corners");
        }
    }
    if (corners_explicit_) {
        clon = corner_lon_;
        clat = corner_lat_;
    } else if (rectilinear_bounds_explicit_ && corner_policy != CornerPolicy::GaussianLatLon) {
        if (corner_policy != CornerPolicy::RectilinearMidpoint && corner_policy != CornerPolicy::RequireExplicit) {
            throw std::invalid_argument("StructuredGrid::to_unstructured: rectilinear bounds require a rectilinear policy");
        }
        validate_axis_bounds(rectilinear_lon_bounds_, seam.period, spherical, seam.periodic);
        validate_axis_bounds(rectilinear_lat_bounds_, 0.0, false, false);
        validate_rectilinear_centers(ni_, nj_, center_lon_, center_lat_, seam.period, spherical, seam.increasing);
        double bounds_span = 0.0;
        const auto longitude_bounds = rectilinear_lon_bounds_;
        const std::size_t ni_boundaries = ni_;
        Kokkos::parallel_reduce(
            "validate_rectilinear_bounds_direction", Range(exec, 0, 1),
            KOKKOS_LAMBDA(const std::size_t, double &span) { span += longitude_bounds(ni_boundaries) - longitude_bounds(0); }, bounds_span);
        if (bounds_span * (seam.increasing ? 1.0 : -1.0) <= 0.0) {
            throw std::invalid_argument("StructuredGrid::to_unstructured: longitude bounds direction disagrees with the declared seam orientation");
        }
        validate_centers_inside_rectilinear_bounds(ni_, nj_, center_lon_, center_lat_, rectilinear_lon_bounds_, rectilinear_lat_bounds_, spherical,
                                                   seam.period);
        const std::size_t nip1 = checked_increment(ni_, "ni+1");
        const std::size_t njp1 = checked_increment(nj_, "nj+1");
        const std::size_t n_boundaries = checked_product(nip1, njp1, "(ni+1)*(nj+1)");
        clon = Kokkos::View<double *, MemorySpace>("rectilinear_corner_lon", n_boundaries);
        clat = Kokkos::View<double *, MemorySpace>("rectilinear_corner_lat", n_boundaries);
        const auto lon_bounds = rectilinear_lon_bounds_;
        const auto lat_bounds = rectilinear_lat_bounds_;
        const auto out_lon = clon;
        const auto out_lat = clat;
        Kokkos::parallel_for(
            "fill_rectilinear_bounds", Range(exec, 0, n_boundaries), KOKKOS_LAMBDA(const std::size_t node) {
                const std::size_t ci = node % nip1;
                const std::size_t cj = node / nip1;
                out_lon(node) = lon_bounds(ci);
                out_lat(node) = lat_bounds(cj);
            });
    } else if (corner_policy == CornerPolicy::GaussianLatLon) {
        if (!gaussian_weights_explicit_ && !rectilinear_bounds_explicit_) {
            throw std::invalid_argument(
                "StructuredGrid::to_unstructured: GaussianLatLon requires authoritative quadrature weights, latitude bounds, or corners");
        }
        if (!spherical) {
            throw std::invalid_argument("StructuredGrid::to_unstructured: GaussianLatLon requires spherical coordinates");
        }
        validate_rectilinear_centers(ni_, nj_, center_lon_, center_lat_, seam.period, spherical, seam.increasing);
        const std::size_t nip1 = checked_increment(ni_, "ni+1");
        const std::size_t njp1 = checked_increment(nj_, "nj+1");
        const std::size_t n_boundaries = checked_product(nip1, njp1, "(ni+1)*(nj+1)");
        Kokkos::View<double *, MemorySpace> lon_bounds("gaussian_lon_bounds", nip1);
        Kokkos::View<double *, MemorySpace> lat_bounds("gaussian_lat_bounds", njp1);
        auto weights = gaussian_latitude_weights_;
        auto gaussian_lat_bounds = lat_bounds;
        const double latitude_scale = coord_sys_ == CoordinateSystem::SphericalDeg ? 180.0 / std::numbers::pi : 1.0;
        if (gaussian_weights_explicit_) {
            std::size_t invalid_weights = 0;
            double weight_sum = 0.0;
            Kokkos::parallel_reduce(
                "validate_gaussian_weights", Range(exec, 0, nj_),
                KOKKOS_LAMBDA(const std::size_t j, std::size_t &errors) {
                    const double weight = weights(j);
                    if (!Kokkos::isfinite(weight) || weight <= 0.0) ++errors;
                },
                invalid_weights);
            Kokkos::parallel_reduce(
                "sum_gaussian_weights", Range(exec, 0, nj_), KOKKOS_LAMBDA(const std::size_t j, double &sum) { sum += weights(j); }, weight_sum);
            if (invalid_weights != 0 || std::abs(weight_sum - 2.0) > 1e-10) {
                throw std::invalid_argument("StructuredGrid::to_unstructured: Gaussian weights must be finite, positive, and sum to 2");
            }
            Kokkos::parallel_for(
                "gaussian_polar_boundary", Range(exec, 0, 1),
                KOKKOS_LAMBDA(const std::size_t) { gaussian_lat_bounds(0) = 0.5 * std::numbers::pi * latitude_scale; });
            Kokkos::parallel_scan(
                "gaussian_mu_boundaries", Range(exec, 0, nj_), KOKKOS_LAMBDA(const std::size_t j, double &prefix, const bool final) {
                    prefix += weights(j);
                    if (final) {
                        const double mu = Kokkos::fmax(-1.0, Kokkos::fmin(1.0, 1.0 - prefix));
                        gaussian_lat_bounds(j + 1) = Kokkos::asin(mu) * latitude_scale;
                    }
                });
        } else {
            validate_axis_bounds(rectilinear_lat_bounds_, 0.0, false, false);
            Kokkos::deep_copy(gaussian_lat_bounds, rectilinear_lat_bounds_);
        }
        if (rectilinear_bounds_explicit_) {
            validate_axis_bounds(rectilinear_lon_bounds_, seam.period, true, seam.periodic);
            if (!gaussian_weights_explicit_) {
                validate_centers_inside_rectilinear_bounds(ni_, nj_, center_lon_, center_lat_, rectilinear_lon_bounds_, rectilinear_lat_bounds_, true,
                                                           seam.period);
            }
            Kokkos::deep_copy(lon_bounds, rectilinear_lon_bounds_);
        }
        auto lon = center_lon_;
        auto out_lon_bounds = lon_bounds;
        const std::size_t ni = ni_;
        const bool periodic = seam.periodic;
        const double period = seam.period;
        const bool increasing = seam.increasing;
        if (!rectilinear_bounds_explicit_)
            Kokkos::parallel_for(
                "gaussian_longitude_boundaries", Range(exec, 0, nip1), KOKKOS_LAMBDA(const std::size_t i) {
                    if (ni == 1) {
                        out_lon_bounds(i) = lon(0) + (i == 0 ? -0.5 : 0.5) * period;
                    } else if (periodic && (i == 0 || i == ni)) {
                        double first = lon(1) - lon(0);
                        if (first > 0.5 * period) first -= period;
                        if (first < -0.5 * period) first += period;
                        const double last = lon(ni - 1) - lon(ni - 2);
                        const double step = 0.5 * (first + last);
                        out_lon_bounds(i) = i == 0 ? lon(0) - 0.5 * step : lon(ni - 1) + 0.5 * step;
                    } else if (i == 0) {
                        out_lon_bounds(i) = lon(0) - 0.5 * (lon(1) - lon(0));
                    } else if (i == ni) {
                        out_lon_bounds(i) = lon(ni - 1) + 0.5 * (lon(ni - 1) - lon(ni - 2));
                    } else {
                        out_lon_bounds(i) = 0.5 * (lon(i - 1) + lon(i));
                    }
                });
        if (seam.periodic) {
            double longitude_span = 0.0;
            Kokkos::parallel_reduce(
                "validate_gaussian_longitude_period_span", Range(exec, 0, 1),
                KOKKOS_LAMBDA(const std::size_t, double &span) { span += out_lon_bounds(ni) - out_lon_bounds(0); }, longitude_span);
            const double expected_span = seam.increasing ? seam.period : -seam.period;
            if (std::abs(longitude_span - expected_span) > 1e-10 * std::max(1.0, seam.period)) {
                throw std::invalid_argument(
                    "StructuredGrid::to_unstructured: periodic Gaussian longitude boundaries must span the declared oriented period");
            }
        }
        auto center_lat = center_lat_;
        std::size_t centers_outside_bands = 0;
        Kokkos::parallel_reduce(
            "validate_gaussian_centers_in_bands", Range(exec, 0, nj_),
            KOKKOS_LAMBDA(const std::size_t j, std::size_t &errors) {
                const double latitude = center_lat(j * ni);
                const double low = Kokkos::fmin(gaussian_lat_bounds(j), gaussian_lat_bounds(j + 1));
                const double high = Kokkos::fmax(gaussian_lat_bounds(j), gaussian_lat_bounds(j + 1));
                if (!Kokkos::isfinite(latitude) || latitude < low - 1e-10 || latitude > high + 1e-10) ++errors;
            },
            centers_outside_bands);
        if (centers_outside_bands != 0) {
            throw std::invalid_argument("StructuredGrid::to_unstructured: Gaussian center latitudes must lie inside their quadrature bands");
        }
        auto out_lon = clon = Kokkos::View<double *, MemorySpace>("gaussian_corners_lon", n_boundaries);
        auto out_lat = clat = Kokkos::View<double *, MemorySpace>("gaussian_corners_lat", n_boundaries);
        Kokkos::parallel_for(
            "fill_gaussian_corners", Range(exec, 0, n_boundaries), KOKKOS_LAMBDA(const std::size_t node) {
                const std::size_t i = node % nip1;
                const std::size_t j = node / nip1;
                out_lon(node) = out_lon_bounds(i);
                out_lat(node) = gaussian_lat_bounds(j);
            });
    } else {
        if (corner_policy == CornerPolicy::RectilinearMidpoint) {
            validate_rectilinear_centers(ni_, nj_, center_lon_, center_lat_, seam.period, spherical, seam.increasing);
        }
        synthesize_corners(seam.periodic, seam.period, seam.increasing, clon, clat);
        if (corner_policy == CornerPolicy::CurvilinearApproximate) {
            const std::size_t nip1 = checked_increment(ni_, "ni+1");
            validate_approximate_cells(ni_, nj_, nip1, center_lon_, center_lat_, clon, clat, spherical, seam.period);
        }
    }

    const std::size_t ni = ni_;
    const std::size_t nj = nj_;
    const std::size_t nip1 = checked_increment(ni, "ni+1");
    const std::size_t njp1 = checked_increment(nj, "nj+1");
    const std::size_t n_cells = checked_product(ni, nj, "ni*nj");
    const std::size_t n_offsets = checked_increment(n_cells, "n_cells+1");
    const std::size_t n_nodes = checked_product(nip1, njp1, "(ni+1)*(nj+1)");
    const std::size_t n_indices = checked_product(n_cells, std::size_t{4}, "ni*nj*4");
    constexpr auto max_index = static_cast<std::uint64_t>(std::numeric_limits<index_t>::max());
    if (static_cast<std::uint64_t>(n_nodes) > max_index || static_cast<std::uint64_t>(n_indices) > max_index) {
        throw std::overflow_error("StructuredGrid::to_unstructured: mesh extents exceed index_t range");
    }

    // ── Allocate output arrays ───────────────────────────────────────────────

    // Spherical grids store (longitude, latitude); planar inputs tagged
    // Cartesian3D are embedded in z=0 so the mesh and area kernel agree on
    // the declared three-coordinate representation.
    const std::size_t coordinate_dimension = coord_sys_ == CoordinateSystem::Cartesian3D ? 3 : 2;
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("unstructured_node_coords", n_nodes, coordinate_dimension);

    // CSR offsets: [n_cells + 1]. For all-quad meshes: [0, 4, 8, 12, ...]
    Kokkos::View<index_t *, MemorySpace> cell_node_offsets("unstructured_cell_offsets", n_offsets);

    // CSR indices: [n_cells * 4] (4 nodes per quad cell)
    Kokkos::View<index_t *, MemorySpace> cell_node_indices("unstructured_cell_indices", n_indices);
    Kokkos::View<double *, MemorySpace> areas;
    const bool rectilinear_geometry =
        !corners_explicit_ && coord_sys_ != CoordinateSystem::Cartesian3D &&
        (corner_policy == CornerPolicy::RectilinearMidpoint || (rectilinear_bounds_explicit_ && corner_policy == CornerPolicy::RequireExplicit) ||
         corner_policy == CornerPolicy::GaussianLatLon);
    if (rectilinear_geometry) areas = Kokkos::View<double *, MemorySpace>("rectilinear_cell_areas", n_cells);

    Kokkos::parallel_for(
        "to_unstructured_fill_nodes", Range(exec, 0, n_nodes), KOKKOS_LAMBDA(const std::size_t node_idx) {
            node_coords(node_idx, 0) = clon(node_idx);
            node_coords(node_idx, 1) = clat(node_idx);
            if (coordinate_dimension == 3) node_coords(node_idx, 2) = 0.0;
        });

    auto offsets = cell_node_offsets;
    auto indices = cell_node_indices;
    Kokkos::parallel_for(
        "to_unstructured_fill_connectivity", Range(exec, 0, n_cells), KOKKOS_LAMBDA(const std::size_t cell_idx) {
            const std::size_t i = cell_idx % ni;
            const std::size_t j = cell_idx / ni;
            cell_node_offsets(cell_idx) = static_cast<index_t>(cell_idx * 4);

            const std::size_t base = cell_idx * 4;
            indices(base + 0) = static_cast<index_t>(i + j * nip1);              // bottom-left
            indices(base + 1) = static_cast<index_t>((i + 1) + j * nip1);        // bottom-right
            indices(base + 2) = static_cast<index_t>((i + 1) + (j + 1) * nip1);  // top-right
            indices(base + 3) = static_cast<index_t>(i + (j + 1) * nip1);        // top-left
        });

    if (rectilinear_geometry) {
        const double radians = coord_sys_ == CoordinateSystem::SphericalDeg ? std::numbers::pi / 180.0 : 1.0;
        auto cell_areas = areas;
        Kokkos::parallel_for(
            "rectilinear_strip_areas", Range(exec, 0, n_cells), KOKKOS_LAMBDA(const std::size_t cell) {
                const std::size_t i = cell % ni;
                const std::size_t j = cell / ni;
                const double west = clon(i + j * nip1);
                const double east = clon(i + 1 + j * nip1);
                const double south = clat(i + j * nip1);
                const double north = clat(i + (j + 1) * nip1);
                const double delta_lon = Kokkos::abs(east - west) * radians;
                const double south_rad = Kokkos::fmin(south, north) * radians;
                const double north_rad = Kokkos::fmax(south, north) * radians;
                cell_areas(cell) = delta_lon * (Kokkos::sin(north_rad) - Kokkos::sin(south_rad));
            });
    }

    // ── Fill the sentinel offset at the end ──────────────────────────────────
    // CSR requires offsets[n_cells] = total number of connectivity entries.
    Kokkos::parallel_for(
        "to_unstructured_sentinel_offset", Range(exec, 0, 1),
        KOKKOS_LAMBDA(const std::size_t /*unused*/) { offsets(n_cells) = static_cast<index_t>(n_indices); });

    // Preserve the synchronous conversion contract without fencing other
    // execution-space instances.
    exec.fence("to_unstructured_fence");

    const GeometryProvenance provenance =
        corners_explicit_
            ? GeometryProvenance::SourceAuthoritative
            : (rectilinear_bounds_explicit_ ? GeometryProvenance::SourceAuthoritative
                                            : (corner_policy == CornerPolicy::CurvilinearApproximate ? GeometryProvenance::Approximate
                                                                                                     : GeometryProvenance::DeclaredGridModel));
    const BoundaryModel boundary =
        rectilinear_geometry ? BoundaryModel::ConstantLatitude
                             : (coord_sys_ == CoordinateSystem::Cartesian3D ? BoundaryModel::CartesianStraight : BoundaryModel::GreatCircle);
    const AreaModel area_model = rectilinear_geometry
                                     ? AreaModel::ConstantLatitudeStrip
                                     : (coord_sys_ == CoordinateSystem::Cartesian3D ? AreaModel::PlanarPolygon : AreaModel::SphericalExcess);
    return make_unstructured(std::move(node_coords), std::move(cell_node_offsets), std::move(cell_node_indices), coord_sys_, std::move(areas), {},
                             GeometryMetadata{provenance, boundary, area_model, seam});
}

// ─────────────────────────────────────────────────────────────────────────────
// synthesize_band_corners — global-context (halo-aware) band corner synthesis
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void synthesize_band_corners(std::size_t ni, std::size_t nj_global, Kokkos::View<double *, MemorySpace> center_lon,
                             Kokkos::View<double *, MemorySpace> center_lat, std::size_t j0, std::size_t j1,
                             Kokkos::View<double *, MemorySpace> &corner_lon, Kokkos::View<double *, MemorySpace> &corner_lat) {
    if (ni == 0) {
        throw std::invalid_argument("synthesize_band_corners: ni must be positive");
    }
    if (j1 < j0) {
        throw std::invalid_argument("synthesize_band_corners: j1 must be >= j0");
    }
    if (j1 > nj_global) {
        throw std::invalid_argument("synthesize_band_corners: j1 (" + std::to_string(j1) + ") exceeds nj_global (" + std::to_string(nj_global) + ")");
    }
    const std::size_t n_centers = checked_product(ni, nj_global, "ni*nj_global");
    if (center_lon.extent(0) != n_centers || center_lat.extent(0) != n_centers) {
        throw std::invalid_argument("synthesize_band_corners: center array extent must equal ni * nj_global");
    }

    synthesize_band_corners(ni, nj_global, std::move(center_lon), std::move(center_lat), j0, j1, LongitudePeriodicity{false, 360.0, true}, corner_lon,
                            corner_lat);
}

template <class MemorySpace>
void synthesize_band_corners(std::size_t ni, std::size_t nj_global, Kokkos::View<double *, MemorySpace> center_lon,
                             Kokkos::View<double *, MemorySpace> center_lat, std::size_t j0, std::size_t j1, LongitudePeriodicity seam,
                             Kokkos::View<double *, MemorySpace> &corner_lon, Kokkos::View<double *, MemorySpace> &corner_lat) {
    if (ni == 0) throw std::invalid_argument("synthesize_band_corners: ni must be positive");
    if (j1 < j0) throw std::invalid_argument("synthesize_band_corners: j1 must be >= j0");
    if (j1 > nj_global) throw std::invalid_argument("synthesize_band_corners: j1 exceeds nj_global");
    const std::size_t n_centers = checked_product(ni, nj_global, "ni*nj_global");
    if (center_lon.extent(0) != n_centers || center_lat.extent(0) != n_centers) {
        throw std::invalid_argument("synthesize_band_corners: center array extent must equal ni * nj_global");
    }
    if (!std::isfinite(seam.period) || seam.period <= 0.0) {
        throw std::invalid_argument("synthesize_band_corners: longitude period must be finite and positive");
    }

    const std::size_t nip1 = checked_increment(ni, "ni+1");
    const std::size_t nrows = j1 - j0 + 1;  // corner rows [j0, j1]
    const std::size_t n_corners = checked_product(nip1, nrows, "(ni+1)*nrows");
    corner_lon = Kokkos::View<double *, MemorySpace>("band_corner_lon", n_corners);
    corner_lat = Kokkos::View<double *, MemorySpace>("band_corner_lat", n_corners);

    // Same kernel as the whole-grid path, restricted to global corner rows [j0, j1].
    synthesize_corner_rows(ni, nj_global, center_lon, center_lat, j0, nrows, seam.periodic, seam.period, seam.increasing, true, corner_lon,
                           corner_lat);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

template class StructuredGrid<Kokkos::HostSpace>;
template void synthesize_band_corners<Kokkos::HostSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::HostSpace>,
                                                         Kokkos::View<double *, Kokkos::HostSpace>, std::size_t, std::size_t,
                                                         Kokkos::View<double *, Kokkos::HostSpace> &, Kokkos::View<double *, Kokkos::HostSpace> &);
template void synthesize_band_corners<Kokkos::HostSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::HostSpace>,
                                                         Kokkos::View<double *, Kokkos::HostSpace>, std::size_t, std::size_t, LongitudePeriodicity,
                                                         Kokkos::View<double *, Kokkos::HostSpace> &, Kokkos::View<double *, Kokkos::HostSpace> &);

#ifdef KOKKOS_ENABLE_CUDA
template class StructuredGrid<Kokkos::CudaSpace>;
template void synthesize_band_corners<Kokkos::CudaSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::CudaSpace>,
                                                         Kokkos::View<double *, Kokkos::CudaSpace>, std::size_t, std::size_t,
                                                         Kokkos::View<double *, Kokkos::CudaSpace> &, Kokkos::View<double *, Kokkos::CudaSpace> &);
template void synthesize_band_corners<Kokkos::CudaSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::CudaSpace>,
                                                         Kokkos::View<double *, Kokkos::CudaSpace>, std::size_t, std::size_t, LongitudePeriodicity,
                                                         Kokkos::View<double *, Kokkos::CudaSpace> &, Kokkos::View<double *, Kokkos::CudaSpace> &);
#endif

#ifdef KOKKOS_ENABLE_HIP
template class StructuredGrid<Kokkos::HIPSpace>;
template void synthesize_band_corners<Kokkos::HIPSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::HIPSpace>,
                                                        Kokkos::View<double *, Kokkos::HIPSpace>, std::size_t, std::size_t,
                                                        Kokkos::View<double *, Kokkos::HIPSpace> &, Kokkos::View<double *, Kokkos::HIPSpace> &);
template void synthesize_band_corners<Kokkos::HIPSpace>(std::size_t, std::size_t, Kokkos::View<double *, Kokkos::HIPSpace>,
                                                        Kokkos::View<double *, Kokkos::HIPSpace>, std::size_t, std::size_t, LongitudePeriodicity,
                                                        Kokkos::View<double *, Kokkos::HIPSpace> &, Kokkos::View<double *, Kokkos::HIPSpace> &);
#endif

}  // namespace axis::topology
