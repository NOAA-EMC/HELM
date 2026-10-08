// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/rule_generator.cpp
/// @brief RuleGenerator implementation — builds meshes from GridRulesParams via
///        Kokkos parallel kernels. Zero file I/O, zero YAML/JSON parsing.

#include <Kokkos_Core.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/rule_generator.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef AXIS_ENABLE_PROJ
#include <axis/topology/projection_builder.hpp>
#endif

namespace axis::topology {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Validation helpers
// ─────────────────────────────────────────────────────────────────────────────

void validate_bbox(const ingest::GridRulesParams &rules) {
    if (rules.max_x <= rules.min_x) {
        throw std::invalid_argument("RuleGenerator: max_x (" + std::to_string(rules.max_x) + ") must be greater than min_x (" +
                                    std::to_string(rules.min_x) + ")");
    }
    if (rules.max_y <= rules.min_y) {
        throw std::invalid_argument("RuleGenerator: max_y (" + std::to_string(rules.max_y) + ") must be greater than min_y (" +
                                    std::to_string(rules.min_y) + ")");
    }
}

void validate_resolution(const ingest::GridRulesParams &rules) {
    if (rules.r_x <= 0.0) {
        throw std::invalid_argument("RuleGenerator: r_x must be positive, got " + std::to_string(rules.r_x));
    }
    if (rules.r_y <= 0.0) {
        throw std::invalid_argument("RuleGenerator: r_y must be positive, got " + std::to_string(rules.r_y));
    }
}

void validate_gaussian_n(const ingest::GridRulesParams &rules) {
    if (rules.gaussian_n <= 0) {
        throw std::invalid_argument("RuleGenerator: gaussian_n must be positive for Gaussian kinds, got " + std::to_string(rules.gaussian_n));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian latitude computation (Newton's method on Legendre polynomials)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Gaussian latitudes for N latitude circles in the northern hemisphere.
/// Returns 2*N latitudes (full globe, symmetric about equator) in degrees,
/// ordered from north to south.
std::vector<double> compute_gaussian_latitudes(int N) {
    const int n_total = 2 * N;
    std::vector<double> lats(static_cast<std::size_t>(n_total));

    // Compute northern hemisphere roots of the Legendre polynomial P_n(x)
    // using Newton's method, then mirror to southern hemisphere.
    for (int i = 0; i < N; ++i) {
        // Initial guess using Tricomi approximation
        const double theta = M_PI * (4.0 * (i + 1) - 1.0) / (4.0 * n_total + 2.0);
        double x = std::cos(theta);

        // Newton iterations to find root of P_n(x)
        for (int iter = 0; iter < 100; ++iter) {
            double p0 = 1.0;
            double p1 = x;
            for (int k = 2; k <= n_total; ++k) {
                double pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
                p0 = p1;
                p1 = pk;
            }
            // p1 = P_n(x), derivative: P_n'(x) = n*(x*P_n - P_{n-1}) / (x^2 - 1)
            double dp = n_total * (x * p1 - p0) / (x * x - 1.0);
            double dx = p1 / dp;
            x -= dx;
            if (std::abs(dx) < 1.0e-15) break;
        }

        // Convert from sin(lat) = x to latitude in degrees
        double lat_deg = std::asin(x) * (180.0 / M_PI);
        lats[static_cast<std::size_t>(i)] = lat_deg;
        lats[static_cast<std::size_t>(n_total - 1 - i)] = -lat_deg;
    }

    return lats;
}

/// Gauss-Legendre quadrature weights associated with the latitude roots above,
/// ordered north-to-south. These determine exact equal-area boundaries in mu.
std::vector<double> compute_gaussian_weights(int N, const std::vector<double> &latitudes) {
    const int degree = 2 * N;
    std::vector<double> weights(latitudes.size());
    for (std::size_t j = 0; j < latitudes.size(); ++j) {
        const double x = std::sin(latitudes[j] * (M_PI / 180.0));
        double p0 = 1.0;
        double p1 = x;
        for (int k = 2; k <= degree; ++k) {
            const double pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
            p0 = p1;
            p1 = pk;
        }
        const double derivative = degree * (x * p1 - p0) / (x * x - 1.0);
        weights[j] = 2.0 / ((1.0 - x * x) * derivative * derivative);
    }
    return weights;
}

// ─────────────────────────────────────────────────────────────────────────────
// RegularLatLon generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a regular lat-lon grid as an UnstructuredMesh.
/// ni = floor((max_x - min_x) / r_x), nj = floor((max_y - min_y) / r_y)
/// Each cell is a quadrilateral defined by 4 corner nodes.
template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_regular_latlon(const ingest::GridRulesParams &rules) {
    validate_bbox(rules);
    validate_resolution(rules);

    const auto ni = static_cast<std::size_t>(std::floor((rules.max_x - rules.min_x) / rules.r_x));
    const auto nj = static_cast<std::size_t>(std::floor((rules.max_y - rules.min_y) / rules.r_y));

    if (ni == 0 || nj == 0) {
        throw std::invalid_argument(
            "RuleGenerator: RegularLatLon resolution produces zero cells "
            "(ni=" +
            std::to_string(ni) + ", nj=" + std::to_string(nj) + ")");
    }

    const std::size_t n_nodes = (ni + 1) * (nj + 1);
    const std::size_t n_cells = ni * nj;

    // Allocate Kokkos views
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("rule_rll_coords", n_nodes, 2);
    Kokkos::View<index_t *, MemorySpace> conn_offsets("rule_rll_offsets", n_cells + 1);
    Kokkos::View<index_t *, MemorySpace> conn_indices("rule_rll_indices", n_cells * 4);

    const double min_x = rules.min_x;
    const double min_y = rules.min_y;
    const double r_x = rules.r_x;
    const double r_y = rules.r_y;
    const std::size_t ni_cap = ni;

    // Fill node coordinates via Kokkos parallel kernel
    Kokkos::parallel_for(
        "RuleGen_RLL_Nodes", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_nodes), KOKKOS_LAMBDA(const std::size_t idx) {
            const std::size_t i = idx % (ni_cap + 1);
            const std::size_t j = idx / (ni_cap + 1);
            node_coords(idx, 0) = min_x + static_cast<double>(i) * r_x;
            node_coords(idx, 1) = min_y + static_cast<double>(j) * r_y;
        });

    // Fill CSR connectivity via Kokkos parallel kernel
    // Each cell (i, j) -> quad with 4 corners: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
    Kokkos::parallel_for(
        "RuleGen_RLL_Cells", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_cells), KOKKOS_LAMBDA(const std::size_t c) {
            const std::size_t i = c % ni_cap;
            const std::size_t j = c / ni_cap;
            const std::size_t stride = ni_cap + 1;

            conn_offsets(c) = static_cast<index_t>(c * 4);

            const std::size_t base = c * 4;
            conn_indices(base + 0) = static_cast<index_t>(j * stride + i);
            conn_indices(base + 1) = static_cast<index_t>(j * stride + (i + 1));
            conn_indices(base + 2) = static_cast<index_t>((j + 1) * stride + (i + 1));
            conn_indices(base + 3) = static_cast<index_t>((j + 1) * stride + i);
        });

    // Set the last offset
    auto conn_offsets_h = Kokkos::create_mirror_view(conn_offsets);
    Kokkos::deep_copy(conn_offsets_h, conn_offsets);
    conn_offsets_h(n_cells) = static_cast<index_t>(n_cells * 4);
    Kokkos::deep_copy(conn_offsets, conn_offsets_h);

    Kokkos::fence();

    return make_unstructured<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), CoordinateSystem::SphericalDeg);
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianRegular generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a regular Gaussian grid as an UnstructuredMesh.
/// Uses gaussian_n to determine latitude circles (2*N latitudes), with
/// regular longitude spacing (4*N points per latitude circle).
template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_gaussian_regular(const ingest::GridRulesParams &rules) {
    validate_gaussian_n(rules);

    const int N = static_cast<int>(rules.gaussian_n);
    const std::size_t n_lat = static_cast<std::size_t>(2 * N);  // number of latitude circles
    const std::size_t n_lon = static_cast<std::size_t>(4 * N);  // points per circle

    const std::size_t n_nodes = (n_lon + 1) * (n_lat + 1);
    const std::size_t n_cells = n_lon * n_lat;

    // Compute Gaussian latitudes on host
    auto gauss_lats = compute_gaussian_latitudes(N);

    // Gaussian quadrature weights define equal-area strips in mu=sin(latitude).
    const auto gauss_weights = compute_gaussian_weights(N, gauss_lats);
    std::vector<double> lat_bounds(n_lat + 1);
    double mu_boundary = 1.0;
    lat_bounds[0] = 90.0;
    for (std::size_t j = 0; j < n_lat; ++j) {
        mu_boundary -= gauss_weights[j];
        lat_bounds[j + 1] = std::asin(std::max(-1.0, std::min(1.0, mu_boundary))) * (180.0 / M_PI);
    }
    lat_bounds[n_lat] = -90.0;  // close exactly at the southern pole despite roundoff

    // Longitude bounds: regular spacing from 0 to 360
    const double dlon = 360.0 / static_cast<double>(n_lon);

    // Allocate Kokkos views
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("rule_gr_coords", n_nodes, 2);
    Kokkos::View<index_t *, MemorySpace> conn_offsets("rule_gr_offsets", n_cells + 1);
    Kokkos::View<index_t *, MemorySpace> conn_indices("rule_gr_indices", n_cells * 4);
    Kokkos::View<double *, MemorySpace> areas("rule_gr_areas", n_cells);

    // Copy latitude bounds to device
    Kokkos::View<double *, MemorySpace> lat_bounds_d("rule_gr_lat_bounds", n_lat + 1);
    auto lat_bounds_h = Kokkos::create_mirror_view(lat_bounds_d);
    for (std::size_t j = 0; j <= n_lat; ++j) {
        lat_bounds_h(j) = lat_bounds[j];
    }
    Kokkos::deep_copy(lat_bounds_d, lat_bounds_h);

    const std::size_t n_lon_cap = n_lon;

    // Fill node coordinates
    Kokkos::parallel_for(
        "RuleGen_GR_Nodes", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_nodes), KOKKOS_LAMBDA(const std::size_t idx) {
            const std::size_t i = idx % (n_lon_cap + 1);
            const std::size_t j = idx / (n_lon_cap + 1);
            node_coords(idx, 0) = static_cast<double>(i) * dlon;
            node_coords(idx, 1) = lat_bounds_d(j);
        });

    // Fill CSR connectivity (quads)
    Kokkos::parallel_for(
        "RuleGen_GR_Cells", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_cells), KOKKOS_LAMBDA(const std::size_t c) {
            const std::size_t i = c % n_lon_cap;
            const std::size_t j = c / n_lon_cap;
            const std::size_t stride = n_lon_cap + 1;

            conn_offsets(c) = static_cast<index_t>(c * 4);

            const std::size_t base = c * 4;
            conn_indices(base + 0) = static_cast<index_t>(j * stride + i);
            conn_indices(base + 1) = static_cast<index_t>(j * stride + (i + 1));
            conn_indices(base + 2) = static_cast<index_t>((j + 1) * stride + (i + 1));
            conn_indices(base + 3) = static_cast<index_t>((j + 1) * stride + i);
        });

    // Set the last offset
    auto conn_offsets_h = Kokkos::create_mirror_view(conn_offsets);
    Kokkos::deep_copy(conn_offsets_h, conn_offsets);
    conn_offsets_h(n_cells) = static_cast<index_t>(n_cells * 4);
    Kokkos::deep_copy(conn_offsets, conn_offsets_h);

    auto area_bounds = lat_bounds_d;
    auto cell_areas = areas;
    Kokkos::parallel_for(
        "RuleGen_GR_Areas", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_cells), KOKKOS_LAMBDA(const std::size_t c) {
            const std::size_t j = c / n_lon_cap;
            const double south = area_bounds(j + 1) * (M_PI / 180.0);
            const double north = area_bounds(j) * (M_PI / 180.0);
            cell_areas(c) = dlon * (M_PI / 180.0) * (Kokkos::sin(north) - Kokkos::sin(south));
        });

    Kokkos::fence();

    return make_unstructured<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), CoordinateSystem::SphericalDeg,
                                          std::move(areas), {},
                                          GeometryMetadata{GeometryProvenance::DeclaredGridModel, BoundaryModel::ConstantLatitude,
                                                           AreaModel::ConstantLatitudeStrip, LongitudePeriodicity{true, 360.0, true}});
}

// ─────────────────────────────────────────────────────────────────────────────
// GaussianReduced (octahedral) generation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the number of longitude points per latitude circle for a reduced
/// (octahedral) Gaussian grid. Uses the octahedral formula: nlon(j) = 20 + 4*j
/// for j=0..N-1 in the northern hemisphere, mirrored to the south.
inline std::vector<std::size_t> octahedral_nlon_per_lat(int N) {
    const std::size_t n_lat = static_cast<std::size_t>(2 * N);
    std::vector<std::size_t> nlons(n_lat);
    for (int j = 0; j < N; ++j) {
        std::size_t nlon = static_cast<std::size_t>(20 + 4 * j);
        nlons[static_cast<std::size_t>(j)] = nlon;
        nlons[static_cast<std::size_t>(2 * N - 1 - j)] = nlon;
    }
    return nlons;
}

/// Generate a reduced (octahedral) Gaussian grid as an UnstructuredMesh.
/// Each latitude circle has a variable number of points (octahedral formula).
/// Cells are quadrilaterals connecting adjacent latitude circles.
template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_gaussian_reduced(const ingest::GridRulesParams &rules) {
    validate_gaussian_n(rules);

    const int N = static_cast<int>(rules.gaussian_n);
    const std::size_t n_lat = static_cast<std::size_t>(2 * N);

    // Compute Gaussian latitudes
    auto gauss_lats = compute_gaussian_latitudes(N);

    // Compute number of longitude points per latitude circle
    auto nlons = octahedral_nlon_per_lat(N);

    // Compute total number of nodes: sum of (nlon + 1) for each latitude boundary
    // For a reduced grid, nodes are placed at cell centers on each latitude circle.
    // We use a simpler representation: one node per grid point on each latitude.
    std::size_t total_nodes = 0;
    for (std::size_t j = 0; j < n_lat; ++j) {
        total_nodes += nlons[j];
    }

    // For the unstructured mesh, we store each grid point as a node (cell center).
    // Each grid point becomes a single cell (polygon) in the unstructured representation.
    // For a reduced Gaussian grid, cells are defined by connecting points between
    // adjacent latitudes. However, since adjacent latitudes have different numbers
    // of points, the topology is irregular.
    //
    // Simplest correct representation: each grid point is a cell center, and we
    // create a triangular/quad mesh using Voronoi-like cells. For AXIS's purposes,
    // we represent it as one quad per grid point using lat bounds and lon bounds.

    // For reduced Gaussian, we treat each point as having a "cell" defined by
    // its latitude extent (half-way to adjacent Gaussian latitudes) and longitude
    // extent (half-way to adjacent points on the same circle).
    //
    // Strategy: each cell is a quad with 4 corner nodes.
    const std::size_t n_cells = total_nodes;

    // Each cell has 4 corner nodes (its lon-lat bounding box corners)
    const std::size_t n_mesh_nodes = n_cells * 4;  // unique corners per cell

    // Compute latitude boundaries (midpoints between Gaussian latitudes + poles)
    std::vector<double> lat_bounds(n_lat + 1);
    lat_bounds[0] = 90.0;
    for (std::size_t j = 1; j < n_lat; ++j) {
        lat_bounds[j] = 0.5 * (gauss_lats[j - 1] + gauss_lats[j]);
    }
    lat_bounds[n_lat] = -90.0;

    // Allocate Kokkos views
    Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> node_coords("rule_gred_coords", n_mesh_nodes, 2);
    Kokkos::View<index_t *, MemorySpace> conn_offsets("rule_gred_offsets", n_cells + 1);
    Kokkos::View<index_t *, MemorySpace> conn_indices("rule_gred_indices", n_cells * 4);

    // Copy lat_bounds and nlons to device using index_t (int64_t) for portability
    Kokkos::View<double *, MemorySpace> lat_bounds_d("gred_lat_bounds", n_lat + 1);
    Kokkos::View<index_t *, MemorySpace> nlons_d("gred_nlons", n_lat);

    {
        auto lat_bounds_h = Kokkos::create_mirror_view(lat_bounds_d);
        auto nlons_h = Kokkos::create_mirror_view(nlons_d);

        for (std::size_t j = 0; j <= n_lat; ++j) {
            lat_bounds_h(j) = lat_bounds[j];
        }
        for (std::size_t j = 0; j < n_lat; ++j) {
            nlons_h(j) = static_cast<index_t>(nlons[j]);
        }

        Kokkos::deep_copy(lat_bounds_d, lat_bounds_h);
        Kokkos::deep_copy(nlons_d, nlons_h);
    }

    const std::size_t n_lat_cap = n_lat;

    // Fill nodes and connectivity via Kokkos parallel kernel.
    // Each cell c corresponds to grid point (lat_j, lon_i) where c is computed
    // from the prefix sum. Each cell gets 4 unique corner nodes.
    Kokkos::parallel_for(
        "RuleGen_GRed_Cells", Kokkos::RangePolicy<typename MemorySpace::execution_space>(0, n_cells), KOKKOS_LAMBDA(const std::size_t c) {
            // Find which latitude circle this cell belongs to via linear scan
            // (acceptable for generation — O(N) per cell, total O(N * sum(nlons)))
            std::size_t j = 0;
            std::size_t remaining = c;
            while (j < n_lat_cap && remaining >= static_cast<std::size_t>(nlons_d(j))) {
                remaining -= static_cast<std::size_t>(nlons_d(j));
                ++j;
            }
            const std::size_t i = remaining;  // longitude index within circle j
            const auto nlon_j = static_cast<std::size_t>(nlons_d(j));
            const double dlon = 360.0 / static_cast<double>(nlon_j);

            // Cell longitude bounds
            const double lon_lo = static_cast<double>(i) * dlon;
            const double lon_hi = static_cast<double>(i + 1) * dlon;

            // Cell latitude bounds
            const double lat_hi = lat_bounds_d(j);      // north edge
            const double lat_lo = lat_bounds_d(j + 1);  // south edge

            // 4 corner nodes for this cell (unique per cell)
            const std::size_t node_base = c * 4;
            node_coords(node_base + 0, 0) = lon_lo;
            node_coords(node_base + 0, 1) = lat_hi;
            node_coords(node_base + 1, 0) = lon_hi;
            node_coords(node_base + 1, 1) = lat_hi;
            node_coords(node_base + 2, 0) = lon_hi;
            node_coords(node_base + 2, 1) = lat_lo;
            node_coords(node_base + 3, 0) = lon_lo;
            node_coords(node_base + 3, 1) = lat_lo;

            // CSR connectivity
            conn_offsets(c) = static_cast<index_t>(c * 4);
            conn_indices(c * 4 + 0) = static_cast<index_t>(node_base + 0);
            conn_indices(c * 4 + 1) = static_cast<index_t>(node_base + 1);
            conn_indices(c * 4 + 2) = static_cast<index_t>(node_base + 2);
            conn_indices(c * 4 + 3) = static_cast<index_t>(node_base + 3);
        });

    // Set the last offset
    auto conn_offsets_h = Kokkos::create_mirror_view(conn_offsets);
    Kokkos::deep_copy(conn_offsets_h, conn_offsets);
    conn_offsets_h(n_cells) = static_cast<index_t>(n_cells * 4);
    Kokkos::deep_copy(conn_offsets, conn_offsets_h);

    Kokkos::fence();

    return make_unstructured<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), CoordinateSystem::SphericalDeg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Projected generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a regular grid in projection space, then transform to geographic
/// coordinates via ProjectionBuilder (requires AXIS_ENABLE_PROJ).
template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_projected(const ingest::GridRulesParams &rules) {
    validate_bbox(rules);
    validate_resolution(rules);

    if (rules.proj_string.empty()) {
        throw std::invalid_argument("RuleGenerator: Projected kind requires a non-empty proj_string");
    }

#ifdef AXIS_ENABLE_PROJ
    // Generate a regular grid of center coordinates in projection space
    const auto ni = static_cast<std::size_t>(std::floor((rules.max_x - rules.min_x) / rules.r_x));
    const auto nj = static_cast<std::size_t>(std::floor((rules.max_y - rules.min_y) / rules.r_y));

    if (ni == 0 || nj == 0) {
        throw std::invalid_argument(
            "RuleGenerator: Projected resolution produces zero cells "
            "(ni=" +
            std::to_string(ni) + ", nj=" + std::to_string(nj) + ")");
    }

    const std::size_t n_points = ni * nj;

    // Build center coordinates in projection space on host
    std::vector<double> center_x_buf(n_points);
    std::vector<double> center_y_buf(n_points);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            const std::size_t idx = j * ni + i;  // layout_left: i varies fastest
            center_x_buf[idx] = rules.min_x + (static_cast<double>(i) + 0.5) * rules.r_x;
            center_y_buf[idx] = rules.min_y + (static_cast<double>(j) + 0.5) * rules.r_y;
        }
    }

    // Build BufferViews with center coordinates for ProjectionBuilder
    ingest::BufferViews buffers{};
    buffers.center_x = field_view<const double, 1>{center_x_buf.data(), n_points};
    buffers.center_y = field_view<const double, 1>{center_y_buf.data(), n_points};
    buffers.ni = ni;
    buffers.nj = nj;

    ingest::ProjectedParams proj_params;
    proj_params.proj_string = rules.proj_string;

    // ProjectionBuilder transforms projection-space coordinates to geographic
    auto grid = ProjectionBuilder::build<MemorySpace>(proj_params, buffers);
    return grid.to_unstructured();
#else
    throw std::invalid_argument(
        "RuleGenerator: Projected kind requires PROJ (AXIS_ENABLE_PROJ), "
        "but AXIS was built without PROJ support");
#endif
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// RuleGenerator::generate — explicit template instantiation
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> RuleGenerator::generate(const ingest::GridRulesParams &rules) {
    const auto &kind = rules.kind;

    if (kind == "RegularLatLon") {
        return generate_regular_latlon<MemorySpace>(rules);
    } else if (kind == "GaussianRegular") {
        return generate_gaussian_regular<MemorySpace>(rules);
    } else if (kind == "GaussianReduced") {
        return generate_gaussian_reduced<MemorySpace>(rules);
    } else if (kind == "Projected") {
        return generate_projected<MemorySpace>(rules);
    } else {
        throw std::invalid_argument("RuleGenerator: unrecognized rule kind '" + kind +
                                    "'. "
                                    "Supported kinds: RegularLatLon, GaussianRegular, GaussianReduced, Projected");
    }
}

// Explicit instantiation for Kokkos::HostSpace
template UnstructuredMesh<Kokkos::HostSpace> RuleGenerator::generate<Kokkos::HostSpace>(const ingest::GridRulesParams &rules);

#ifdef KOKKOS_ENABLE_CUDA
template UnstructuredMesh<Kokkos::CudaSpace> RuleGenerator::generate<Kokkos::CudaSpace>(const ingest::GridRulesParams &rules);
#endif

#ifdef KOKKOS_ENABLE_HIP
template UnstructuredMesh<Kokkos::HIPSpace> RuleGenerator::generate<Kokkos::HIPSpace>(const ingest::GridRulesParams &rules);
#endif

}  // namespace axis::topology
