// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file named_grid_registry.cpp
/// @brief Implementation of NamedGridRegistry — on-the-fly generation of
///        standard global weather grids via Kokkos parallel kernels.
///
/// Grid generation algorithms (all zero file I/O):
///
///   O (Octahedral Gaussian): 2N latitude circles total (N per hemisphere).
///     Latitude circle j (0-indexed from nearest pole) has 20+4*j points.
///     Total cells connect adjacent latitude circles with quadrilateral elements.
///
///   F (Regular/Full Gaussian): 2N latitude circles, each with 4N points.
///     Total nodes = 2N * 4N. Quadrilateral cells between adjacent latitudes.
///
///   N (Reduced Gaussian): Follows ECMWF octahedral convention (same as O).
///
///   Gaussian latitudes are computed using Newton's method for Legendre
///   polynomial roots on the unit sphere (iterative, pure math, no tables).

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <array>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/projection_builder.hpp>
#include <cctype>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace axis::topology {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Gaussian latitude computation via Newton's method on Legendre polynomials
// ─────────────────────────────────────────────────────────────────────────────

/// Compute Gaussian latitudes (in degrees, north-to-south) for 2N circles.
/// Uses Newton's method to find roots of the Legendre polynomial P_n(x) where
/// n = 2N. Only the northern hemisphere roots are computed; southern hemisphere
/// is symmetric. Returns all 2N latitudes from north pole toward south pole.
std::vector<double> compute_gaussian_latitudes(int N) {
    const int n = 2 * N;  // degree of Legendre polynomial
    std::vector<double> latitudes(static_cast<std::size_t>(n));

    // We need the n roots of P_n(x) in (-1, 1).
    // Initial guesses using the Bretherton-Hoskins approximation.
    for (int i = 0; i < n; ++i) {
        // Initial guess for the i-th root (1-indexed: k = i+1)
        double k = static_cast<double>(i + 1);
        double theta = M_PI * (k - 0.25) / (static_cast<double>(n) + 0.5);
        double x = std::cos(theta);

        // Newton's method to refine the root of P_n(x)
        for (int iter = 0; iter < 100; ++iter) {
            // Evaluate P_n(x) and P_n'(x) via recurrence
            double p0 = 1.0;  // P_0(x)
            double p1 = x;    // P_1(x)

            for (int j = 2; j <= n; ++j) {
                double pj = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / static_cast<double>(j);
                p0 = p1;
                p1 = pj;
            }
            // p1 = P_n(x), p0 = P_{n-1}(x)
            // Derivative: P_n'(x) = n * (x * P_n(x) - P_{n-1}(x)) / (x^2 - 1)
            //           = n * (P_{n-1}(x) - x * P_n(x)) / (1 - x^2)
            double dp = static_cast<double>(n) * (p0 - x * p1) / (1.0 - x * x);

            double dx = p1 / dp;
            x -= dx;

            if (std::abs(dx) < 1.0e-15) {
                break;
            }
        }

        // x is the cosine of the colatitude; latitude = asin(x) in degrees
        latitudes[static_cast<std::size_t>(i)] = std::asin(x) * (180.0 / M_PI);
    }

    // latitudes are computed from north to south (largest to smallest)
    // The Newton iteration with the given initial guess produces roots
    // in decreasing order of x (i.e., north to south). Verify ordering.
    // Sort descending to ensure north-to-south order.
    std::sort(latitudes.begin(), latitudes.end(), std::greater<double>());

    return latitudes;
}

// ─────────────────────────────────────────────────────────────────────────────
// Points-per-latitude-circle functions
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the number of longitude points for each latitude circle in an
/// octahedral reduced Gaussian grid with parameter N.
/// ECMWF convention: nlon(j) = 20 + 4 * min(j, 2N-1-j), i.e. symmetric
/// about the equator with minimum 20 points at the poles.
/// Returns a vector of size 2*N with nlon for each circle from north to south.
std::vector<int> octahedral_nlons(int N) {
    const int n_lat = 2 * N;
    std::vector<int> nlons(static_cast<std::size_t>(n_lat));
    for (int j = 0; j < n_lat; ++j) {
        // Distance from nearest pole
        int dist_from_pole = std::min(j, n_lat - 1 - j);
        nlons[static_cast<std::size_t>(j)] = 20 + 4 * dist_from_pole;
    }
    return nlons;
}

/// Compute total number of nodes in an octahedral reduced Gaussian grid.
/// Formula: 4*N*(N+9) for ECMWF convention (minimum 20 points at poles).
std::size_t octahedral_total_nodes(int N) {
    // Sum all nlons
    std::size_t total = 0;
    for (int j = 0; j < 2 * N; ++j) {
        int dist = std::min(j, 2 * N - 1 - j);
        total += static_cast<std::size_t>(20 + 4 * dist);
    }
    return total;
}

/// Regular Gaussian: 2N latitudes, each with 4N longitudes.
/// Total nodes = 2N * 4N = 8*N^2.
std::size_t regular_total_nodes(int N) {
    return static_cast<std::size_t>(2 * N) * static_cast<std::size_t>(4 * N);
}

/// Unit vector for cubed-sphere tile-local gnomonic coordinates (a, b) on tile
/// `tile` (1-based), shared by the mesh generator and the cell-center query so
/// the two can never drift apart.
///
/// FV3 convention: tile 1 centre lon 0, 2 -> 90, 3 -> 180, 4 -> 270,
/// 5 = north-pole cap, 6 = south-pole cap. `a` increases eastward, `b`
/// northward, both in radians within [-pi/4, pi/4].
inline std::array<double, 3> cs_tile_vec(int tile, double a, double b) {
    const double ta = std::tan(a);
    const double tb = std::tan(b);
    double x, y, z;
    switch (tile) {
        case 1:
            x = 1.0;
            y = ta;
            z = tb;
            break;
        case 2:
            x = -ta;
            y = 1.0;
            z = tb;
            break;
        case 3:
            x = -1.0;
            y = -ta;
            z = tb;
            break;
        case 4:
            x = ta;
            y = -1.0;
            z = tb;
            break;
        case 5:
            x = -tb;
            y = ta;
            z = 1.0;
            break;
        default:
            x = tb;
            y = ta;
            z = -1.0;
            break;  // tile 6
    }
    const double r = std::sqrt(x * x + y * y + z * z);
    return std::array<double, 3>{x / r, y / r, z / r};
}

/// Convert a unit vector to (lon, lat) degrees, lon wrapped to [0, 360).
inline std::array<double, 2> cs_unit_to_lonlat(const std::array<double, 3> &v) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kR2D = 180.0 / kPi;
    double lon = std::atan2(v[1], v[0]) * kR2D;
    if (lon < 0.0) lon += 360.0;
    return std::array<double, 2>{lon, std::asin(v[2]) * kR2D};
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell count computation
// ─────────────────────────────────────────────────────────────────────────────

/// For reduced Gaussian grids, cells are quadrilaterals connecting adjacent
/// latitude circles. Between circles j and j+1 with nlon_j and nlon_{j+1}
/// points, the number of cells is max(nlon_j, nlon_{j+1}).
/// Total cells = sum_{j=0}^{2N-2} max(nlon[j], nlon[j+1])
std::size_t reduced_total_cells(const std::vector<int> &nlons) {
    std::size_t total = 0;
    for (std::size_t j = 0; j + 1 < nlons.size(); ++j) {
        total += static_cast<std::size_t>(std::max(nlons[j], nlons[j + 1]));
    }
    return total;
}

/// For regular Gaussian grids, cells are simply (2N-1) * 4N quadrilaterals
/// (one ring of cells between each pair of adjacent latitude circles).
std::size_t regular_total_cells(int N) {
    return static_cast<std::size_t>(2 * N - 1) * static_cast<std::size_t>(4 * N);
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: Regular Gaussian (F family)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_regular_gaussian(int N) {
    const int n_lat = 2 * N;
    const int n_lon = 4 * N;
    const std::size_t n_nodes = regular_total_nodes(N);
    const std::size_t n_cells = regular_total_cells(N);

    // Compute Gaussian latitudes
    std::vector<double> lats = compute_gaussian_latitudes(N);

    // ─── Build node coordinates on host ─────────────────────────────────────
    // Nodes are stored as [n_nodes, 2] with col 0 = lon, col 1 = lat (degrees)
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> h_coords("h_coords", n_nodes, 2);

    // Fill node coordinates: latitude circles from north to south,
    // each with n_lon evenly spaced longitudes in [0, 360)
    std::size_t node_idx = 0;
    for (int j = 0; j < n_lat; ++j) {
        double lat = lats[static_cast<std::size_t>(j)];
        double dlon = 360.0 / static_cast<double>(n_lon);
        for (int i = 0; i < n_lon; ++i) {
            h_coords(node_idx, 0) = static_cast<double>(i) * dlon;
            h_coords(node_idx, 1) = lat;
            ++node_idx;
        }
    }

    // ─── Build CSR connectivity on host ─────────────────────────────────────
    // Each cell is a quadrilateral connecting 4 nodes between adjacent lat
    // circles. Node ordering: bottom-left, bottom-right, top-right, top-left.
    // "Bottom" = farther from north pole (larger j index), "Top" = closer to pole.
    // Between lat circles j and j+1: n_lon cells (one per longitude column).
    // Connectivity wraps around in longitude.
    const std::size_t nnz = n_cells * 4;  // 4 nodes per quad cell

    Kokkos::View<index_t *, Kokkos::HostSpace> h_offsets("h_offsets", n_cells + 1);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_indices("h_indices", nnz);

    // Fill offsets: uniform 4 nodes per cell
    for (std::size_t c = 0; c <= n_cells; ++c) {
        h_offsets(c) = static_cast<index_t>(c * 4);
    }

    // Fill connectivity
    std::size_t cell_idx = 0;
    for (int j = 0; j < n_lat - 1; ++j) {
        std::size_t row_start = static_cast<std::size_t>(j) * static_cast<std::size_t>(n_lon);
        std::size_t next_row_start = static_cast<std::size_t>(j + 1) * static_cast<std::size_t>(n_lon);

        for (int i = 0; i < n_lon; ++i) {
            int i_next = (i + 1) % n_lon;

            // Quad, CCW viewed from outside the sphere (engine convention for
            // conservative clipping): bottom-left, bottom-right, top-right,
            // top-left. Rows run north (j) to south (j+1), so "bottom" is the
            // next (more southern) row. Winding matters: SphericalClipper /
            // PlanarClipper derive each clip half-plane's inside-normal from
            // the clip-ring order, which must be CCW (see generate_regular_grid).
            // top = current row (j), bottom = next row (j+1)
            std::size_t tl = row_start + static_cast<std::size_t>(i);
            std::size_t tr = row_start + static_cast<std::size_t>(i_next);
            std::size_t br = next_row_start + static_cast<std::size_t>(i_next);
            std::size_t bl = next_row_start + static_cast<std::size_t>(i);

            std::size_t base = cell_idx * 4;
            h_indices(base + 0) = static_cast<index_t>(bl);
            h_indices(base + 1) = static_cast<index_t>(br);
            h_indices(base + 2) = static_cast<index_t>(tr);
            h_indices(base + 3) = static_cast<index_t>(tl);
            ++cell_idx;
        }
    }

    // ─── Copy to target MemorySpace ─────────────────────────────────────────
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        return UnstructuredMesh<MemorySpace>(std::move(h_coords), std::move(h_offsets), std::move(h_indices), CoordinateSystem::SphericalDeg);
    } else {
        Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> d_coords("d_coords", n_nodes, 2);
        Kokkos::View<index_t *, MemorySpace> d_offsets("d_offsets", n_cells + 1);
        Kokkos::View<index_t *, MemorySpace> d_indices("d_indices", nnz);

        Kokkos::deep_copy(d_coords, h_coords);
        Kokkos::deep_copy(d_offsets, h_offsets);
        Kokkos::deep_copy(d_indices, h_indices);

        return UnstructuredMesh<MemorySpace>(std::move(d_coords), std::move(d_offsets), std::move(d_indices), CoordinateSystem::SphericalDeg);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: Octahedral / Reduced Gaussian (O and N families)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_octahedral_gaussian(int N) {
    const int n_lat = 2 * N;
    const std::vector<int> nlons = octahedral_nlons(N);
    const std::size_t n_nodes = octahedral_total_nodes(N);

    // For reduced grids, between adjacent latitude circles with different
    // numbers of points, we generate quadrilateral cells that properly
    // connect the two circles. The number of cells between circles j and j+1
    // is max(nlons[j], nlons[j+1]).
    const std::size_t n_cells = reduced_total_cells(nlons);

    // Compute Gaussian latitudes
    std::vector<double> lats = compute_gaussian_latitudes(N);

    // ─── Build node coordinates on host ─────────────────────────────────────
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> h_coords("h_coords", n_nodes, 2);

    // Compute row start offsets for quick node lookup
    std::vector<std::size_t> row_starts(static_cast<std::size_t>(n_lat) + 1);
    row_starts[0] = 0;
    for (int j = 0; j < n_lat; ++j) {
        row_starts[static_cast<std::size_t>(j) + 1] =
            row_starts[static_cast<std::size_t>(j)] + static_cast<std::size_t>(nlons[static_cast<std::size_t>(j)]);
    }

    // Fill node coordinates
    for (int j = 0; j < n_lat; ++j) {
        double lat = lats[static_cast<std::size_t>(j)];
        int nlon_j = nlons[static_cast<std::size_t>(j)];
        double dlon = 360.0 / static_cast<double>(nlon_j);
        std::size_t start = row_starts[static_cast<std::size_t>(j)];

        for (int i = 0; i < nlon_j; ++i) {
            std::size_t idx = start + static_cast<std::size_t>(i);
            h_coords(idx, 0) = static_cast<double>(i) * dlon;
            h_coords(idx, 1) = lat;
        }
    }

    // ─── Build CSR connectivity on host ─────────────────────────────────────
    // For reduced Gaussian grids, the connectivity between two adjacent
    // latitude circles with different nlon counts uses a "zipper" algorithm:
    // we walk both circles simultaneously, advancing the pointer on whichever
    // circle is behind in longitude, creating quadrilateral cells.
    //
    // Each cell connects 4 nodes (quadrilateral). For simplicity and
    // determinism, we use a straightforward proportional mapping approach:
    // for each cell between rows j and j+1, we map longitude indices
    // proportionally between the two rows.

    // First pass: count total connectivity entries (always 4 per quad cell)
    const std::size_t nnz = n_cells * 4;

    Kokkos::View<index_t *, Kokkos::HostSpace> h_offsets("h_offsets", n_cells + 1);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_indices("h_indices", nnz);

    // Fill offsets
    for (std::size_t c = 0; c <= n_cells; ++c) {
        h_offsets(c) = static_cast<index_t>(c * 4);
    }

    // Fill connectivity using proportional longitude mapping
    std::size_t cell_idx = 0;
    for (int j = 0; j < n_lat - 1; ++j) {
        int nlon_top = nlons[static_cast<std::size_t>(j)];
        int nlon_bot = nlons[static_cast<std::size_t>(j + 1)];
        std::size_t top_start = row_starts[static_cast<std::size_t>(j)];
        std::size_t bot_start = row_starts[static_cast<std::size_t>(j + 1)];
        int n_cells_ring = std::max(nlon_top, nlon_bot);

        for (int c = 0; c < n_cells_ring; ++c) {
            // Map cell index proportionally to both rows
            // Top row indices
            double frac_top = static_cast<double>(c) / static_cast<double>(n_cells_ring);
            double frac_top_next = static_cast<double>(c + 1) / static_cast<double>(n_cells_ring);

            int top_i = static_cast<int>(std::floor(frac_top * nlon_top)) % nlon_top;
            int top_i_next = static_cast<int>(std::floor(frac_top_next * nlon_top)) % nlon_top;

            // Bottom row indices
            double frac_bot = static_cast<double>(c) / static_cast<double>(n_cells_ring);
            double frac_bot_next = static_cast<double>(c + 1) / static_cast<double>(n_cells_ring);

            int bot_i = static_cast<int>(std::floor(frac_bot * nlon_bot)) % nlon_bot;
            int bot_i_next = static_cast<int>(std::floor(frac_bot_next * nlon_bot)) % nlon_bot;

            // Quad, CCW viewed from outside the sphere (engine convention for
            // conservative clipping): bottom-left, bottom-right, top-right,
            // top-left. Rows run north (j) to south (j+1). See the note in
            // generate_regular_gaussian on winding.
            std::size_t tl = top_start + static_cast<std::size_t>(top_i);
            std::size_t tr = top_start + static_cast<std::size_t>(top_i_next);
            std::size_t br = bot_start + static_cast<std::size_t>(bot_i_next);
            std::size_t bl = bot_start + static_cast<std::size_t>(bot_i);

            std::size_t base = cell_idx * 4;
            h_indices(base + 0) = static_cast<index_t>(bl);
            h_indices(base + 1) = static_cast<index_t>(br);
            h_indices(base + 2) = static_cast<index_t>(tr);
            h_indices(base + 3) = static_cast<index_t>(tl);
            ++cell_idx;
        }
    }

    // ─── Copy to target MemorySpace ─────────────────────────────────────────
    if constexpr (std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        return UnstructuredMesh<MemorySpace>(std::move(h_coords), std::move(h_offsets), std::move(h_indices), CoordinateSystem::SphericalDeg);
    } else {
        Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> d_coords("d_coords", n_nodes, 2);
        Kokkos::View<index_t *, MemorySpace> d_offsets("d_offsets", n_cells + 1);
        Kokkos::View<index_t *, MemorySpace> d_indices("d_indices", nnz);

        Kokkos::deep_copy(d_coords, h_coords);
        Kokkos::deep_copy(d_offsets, h_offsets);
        Kokkos::deep_copy(d_indices, h_indices);

        return UnstructuredMesh<MemorySpace>(std::move(d_coords), std::move(d_offsets), std::move(d_indices), CoordinateSystem::SphericalDeg);
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: NOAA NWS GRIB Grids (G family - Extensible)
// ─────────────────────────────────────────────────────────────────────────────

/// @struct NoaaGribDefinition
/// @brief Declarative metadata structure for NOAA GRIB grids.
struct NoaaGribDefinition {
    int number;               ///< Official GRIB grid number.
    std::size_t ni;           ///< Columns count (Ni).
    std::size_t nj;           ///< Rows count (Nj).
    double lon_start;         ///< Leftmost longitude boundary (for regular grids) or min_x (for projected grids).
    double lat_start;         ///< Southernmost latitude boundary (for regular grids) or min_y (for projected grids).
    double dlon;              ///< Longitude grid spacing (for regular grids) or max_x (for projected grids).
    double dlat;              ///< Latitude grid spacing (for regular grids) or max_y (for projected grids).
    const char *proj_string;  ///< PROJ-compliant string (nullptr if regular global grid).
};

/// @brief Global declarative registry of supported NOAA GRIB grids.
/// @details Adding a new grid to AXIS is a simple, single-line addition here!
static const NoaaGribDefinition NOAA_GRIB_GRIDS[] = {
    // grid3: GFS 1.0 degree global grid
    {3, 360, 181, -180.0, -90.0, 1.0, 1.0, nullptr},
    // grid4: GFS 0.5 degree global grid
    {4, 720, 361, -180.0, -90.0, 0.5, 0.5, nullptr},
    // grid87: U.S. Area; used in MAPS/RUC (60km at 40N) (N. Hem. Polar Stereographic)
    {87, 81, 62, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid88: North American Area; used in RSAS (Polar Stereographic)
    {88, 580, 548, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid90: Grid over CONUS - (1.27 km)
    {90, 4289, 2753, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid91: Grid over Alaska (Polar Stereographic)
    {91, 1649, 1105, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid92: Grid over Alaska (Polar Stereographic)
    {92, 3297, 2209, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid130: ()	Regional - CONUS (Lambert Conformal) - 13 km
    {130, 451, 337, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid132: Regional Double Resolution North American Grid (Lambert Conformal) used by SREF
    {132, 697, 553, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=50.0 +lat_2=50.0 +lat_0=50.0 +lon_0=253.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid138: Air-Quality Forecasting CONUS
    {138, 468, 288, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=263.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid139: Air-Quality Forecasting Hawaii
    {139, 80, 52, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=202.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid140: Air-Quality Forecasting Alaska
    {140, 199, 163, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=211.4 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid145: Air-Quality Forecasting Northeast Intermediate Domain
    {145, 169, 145, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=280.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid146: Air-Quality Forecasting Northeast Output Domain
    {146, 166, 142, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=280.5 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid147: Air-Quality Forecasting Eastern "3x" Domain
    {147, 268, 259, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=263.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid148: Air-Quality Forecasting CONUS "5x" Domain
    {148, 442, 265, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=263.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid150: Central America - Latitude/Longitude grid
    {150, 401, 201, 260.0, 5.0, 0.0, 0.0, nullptr},
    // grid151: Grid over North America (Polar Stereographic)
    {151, 478, 429, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=250.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid160: [Z]	47.5 km North Polar Stereographic grid for Alaska
    {160, 180, 156, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid163: ()	Regional - CONUS 5 km grid
    {163, 1008, 722, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid179: Grid over North America (Polar Stereographic)
    {179, 1196, 871, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=260.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid184: Grid over CONUS - (2.54 km)
    {184, 2145, 1377, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid187: Grid over CONUS - (2.54 km) - Lambert Conformal
    {187, 2145, 1597, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid188: Grid over CONUS - (2.54 km) - Lambert Conformal
    {188, 709, 795, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid189: Polar Stereographic
    {189, 655, 855, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=225.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid197: Grid over the contiguous United States - 16X Resolution (5 km) (Used by the Radar Stage IV precipitation analyses and
    // Satellite-derived Precipitation Estimates and NAM DNG grids and RTMA NDFD grids) (Lambert Conformal)
    {197, 1073, 689, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid198: Grid over Alaska (Polar Stereographic)
    {198, 825, 553, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid200: Puerto Rico FAA Regional Grid
    {200, 108, 94, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=253.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid201: (A)	Northern Hemispheric (Polar Stereographic)
    {201, 65, 65, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid202: (I)	National - CONUS (Polar Stereographic)
    {202, 65, 43, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid203: (J)	National - Alaska (Polar Stereographic)
    {203, 45, 39, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid205: (L)	National - Puerto Rico (Polar Stereographic)
    {205, 45, 39, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=300.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid206: Regional - Central US MARD (Lambert Conformal)
    {206, 51, 41, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid207: (N)	Regional - Alaska (Polar Stereographic)
    {207, 49, 35, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid209: Regional - NOAMIM - Intermediate Resolution North American Master Grid (Lambert Conformal)
    {209, 275, 223, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=45.0 +lat_2=45.0 +lat_0=45.0 +lon_0=249.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid211: (Q)	Regional - CONUS (Lambert Conformal)
    {211, 93, 65, -3738443.0, -2600656.0, 3738443.0, 2600656.0,
     "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid212: (R)[R]	Regional - CONUS - Double Resolution (Lambert Conformal)
    {212, 185, 129, -3738443.0, -2600656.0, 3738443.0, 2600656.0,
     "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid213: (H)	National - CONUS - Double Resolution (Polar Stereographic)
    {213, 129, 85, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid214: ()[T]	Regional - Alaska - Double Resolution (Polar Stereographic)
    {214, 97, 69, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid215: (U)[U]	 grid over the contiguous United States - Quadruple Resolution (used by the 29-km NAM model)(Lambert Conformal)
    {215, 369, 257, -3738443.0, -2600656.0, 3738443.0, 2600656.0,
     "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid216: (V)[V]	 grid over Alaska (Polar Stereographic)
    {216, 139, 107, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=225.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid217: (Y)	 Grid over Alaska - Double Resolution grid (Polar Stereographic)
    {217, 277, 213, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=225.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid218: (B)[B]	  Grid over the Contiguous United States (used by the 12-km NAM Model) (Lambert Conformal)
    {218, 614, 428, -3733392.0, -2602779.0, 3733392.0, 2602779.0,
     "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=265.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid219: ()[C]	 grid over the Northern Hemisphere to depict SSMI-derived ice concentrations (Polar Stereographics)
    {219, 385, 465, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=280.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid220: ()[D]	 grid over the Southern Hemisphere to depict SSMI-derived ice concentrations (Polar Stereographics)
    {220, 345, 355, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=100.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid221: ()[E]	Regional North American Grid (Lambert Conformal)
    {221, 349, 277, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=50.0 +lat_2=50.0 +lat_0=50.0 +lon_0=253.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid222: Regional - NOAMLO - Low Resolution North American Master Grid (Lambert Conformal)
    {222, 138, 112, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=45.0 +lat_2=45.0 +lat_0=45.0 +lon_0=249.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid223: Hemispheric - Double Resolution (Lambert Conformal)
    {223, 129, 129, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid224: Southern Hemispheric (Polar Stereographic)
    {224, 65, 65, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=-105.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid226: (M)	 grid over the contiguous United States - 8X Resolution (10 km) (Used by the Radar mosaics) (Lambert Conformal)
    {226, 737, 517, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid227: Regional grid over the contiguous United States - 16X Resolution
    {227, 1473, 1025, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=25.0 +lat_2=25.0 +lat_0=25.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid236: (W)	Regional - CONUS (Lambert Conformal)
    {236, 151, 113, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid237: (P)	Puerto Rico FAA Regional Grid (Lambert Conformal)
    {237, 54, 47, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=253.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid240: ()	HRAP Grid over the Contiguous United States and Puerto Rico (Polar Stereographic)
    {240, 1121, 881, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=255.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid241: ()	Regional - NOAMHI - High Resolution North American Grid (Lambert Conformal)
    {241, 549, 445, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=249.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid242: (S)	 Grid over Alaska - Quadruple Resolution Grid (Polar Stereographic)
    {242, 553, 425, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=225.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid245: ()	Regional - NOAMHI - High Resolution over Eastern US (Lambert Conformal for 8 km NMM)
    {245, 336, 372, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=280.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid246: ()	Regional - NOAMHI - High Resolution over Western US (Lambert Conformal for 8 km NMM)
    {246, 332, 371, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=245.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid247: ()	Regional - NOAMHI - High Resolution over Central US (Lambert Conformal for 8 km NMM)
    {247, 336, 372, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=262.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid249: ()	 Grid over Alaska for 10-km Alaska nest (Polar Stereographic)
    {249, 367, 343, 0.0, 0.0, 0.0, 0.0, "+proj=stere +lat_ts=60 +lat_0=90 +lon_0=210.0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
    // grid252: ()	Regional - CONUS (Lambert Conformal)
    {252, 301, 225, 0.0, 0.0, 0.0, 0.0, "+proj=lcc +lat_1=2.0 +lat_2=2.0 +lat_0=2.0 +lon_0=265.0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"},
};

/// @brief Total count of registered NOAA NWS grids in our static array.
static constexpr std::size_t NOAA_GRIB_GRIDS_COUNT = sizeof(NOAA_GRIB_GRIDS) / sizeof(NOAA_GRIB_GRIDS[0]);

/// @brief Generates a standard regular lat-lon grid as an UnstructuredMesh.
template <class MemorySpace>
inline UnstructuredMesh<MemorySpace> generate_regular_grid(std::size_t ni, std::size_t nj, double lon_start, double lat_start, double dlon,
                                                           double dlat) {
    const std::size_t n_nodes = (ni + 1) * (nj + 1);
    const std::size_t n_cells = ni * nj;

    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> h_coords("h_coords", n_nodes, 2);

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t idx = i + j * (ni + 1);
            h_coords(idx, 0) = lon_start + static_cast<double>(i) * dlon;
            h_coords(idx, 1) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    Kokkos::View<index_t *, Kokkos::HostSpace> h_offsets("h_offsets", n_cells + 1);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_indices("h_indices", n_cells * 4);

    h_offsets(0) = 0;
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            std::size_t cell_idx = i + j * ni;
            h_offsets(cell_idx + 1) = h_offsets(cell_idx) + 4;

            std::size_t n0 = i + j * (ni + 1);
            std::size_t n1 = (i + 1) + j * (ni + 1);
            std::size_t n2 = (i + 1) + (j + 1) * (ni + 1);
            std::size_t n3 = i + (j + 1) * (ni + 1);

            std::size_t indices_start = cell_idx * 4;
            h_indices(indices_start) = n0;
            h_indices(indices_start + 1) = n1;
            h_indices(indices_start + 2) = n2;
            h_indices(indices_start + 3) = n3;
        }
    }

    auto node_coords = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_coords);
    auto conn_offsets = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_offsets);
    auto conn_indices = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_indices);

    return UnstructuredMesh<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), CoordinateSystem::SphericalDeg);
}

/// @brief Analytically generates a registered NOAA NWS GRIB grid.
template <class MemorySpace>
inline UnstructuredMesh<MemorySpace> generate_noaa_grib_grid(int number) {
    for (std::size_t idx = 0; idx < NOAA_GRIB_GRIDS_COUNT; ++idx) {
        const auto &grid_def = NOAA_GRIB_GRIDS[idx];
        if (grid_def.number == number) {
            if (grid_def.proj_string != nullptr) {
#ifndef AXIS_ENABLE_PROJ
                throw std::runtime_error("NamedGridRegistry::generate: requested projected grid \"grid" + std::to_string(number) +
                                         "\" but AXIS was compiled without PROJ support.");
#else
                axis::ingest::ProjectedParams params;
                params.proj_string = grid_def.proj_string;

                const std::size_t ni = grid_def.ni;
                const std::size_t nj = grid_def.nj;
                const std::size_t n_points = ni * nj;

                // Set coordinates in projection space from the grid definition
                double min_x = grid_def.lon_start;
                double max_x = grid_def.dlon;
                double min_y = grid_def.lat_start;
                double max_y = grid_def.dlat;
                double dx = (max_x - min_x) / (ni - 1);
                double dy = (max_y - min_y) / (nj - 1);

                std::vector<double> h_cx(n_points);
                std::vector<double> h_cy(n_points);
                for (std::size_t j = 0; j < nj; ++j) {
                    for (std::size_t i = 0; i < ni; ++i) {
                        std::size_t cell_idx = i + j * ni;
                        h_cx[cell_idx] = min_x + i * dx;
                        h_cy[cell_idx] = min_y + j * dy;
                    }
                }

                axis::ingest::BufferViews buffers;
                buffers.ni = ni;
                buffers.nj = nj;
                buffers.center_x = axis::field_view<const double, 1>(h_cx.data(), n_points);
                buffers.center_y = axis::field_view<const double, 1>(h_cy.data(), n_points);

                auto grid = axis::topology::ProjectionBuilder::build<MemorySpace>(params, buffers);
                return grid.to_unstructured();
#endif
            } else {
                return generate_regular_grid<MemorySpace>(grid_def.ni, grid_def.nj, grid_def.lon_start, grid_def.lat_start, grid_def.dlon,
                                                          grid_def.dlat);
            }
        }
    }

    throw std::invalid_argument("NamedGridRegistry::generate: unregistered NOAA GRIB grid number grid" + std::to_string(number));
}

// ─────────────────────────────────────────────────────────────────────────────
// Cubed-sphere (gnomonic-equidistant) grid generation
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Generate a gnomonic-equidistant cubed-sphere mesh, 6 tiles × N×N cells.
///
/// FV3-style convention: tile-local coordinates (a, b) ∈ [−45°, +45°]² are
/// equidistant in the gnomonic (tan) projection, so cell edges are great-circle
/// arcs and cell centers follow v ∝ (±1, ±tan a, ±tan b) per tile. The six
/// tiles are:
///   tile 1: ( 1,  ta, tb)   center lon   0°
///   tile 2: (−ta,  1, tb)   center lon  90°
///   tile 3: (−1, −ta, tb)   center lon 180°
///   tile 4: ( ta, −1, tb)   center lon 270°
///   tile 5: (−tb, ta,  1)   north-pole cap
///   tile 6: ( tb, ta, −1)   south-pole cap
/// with ta = tan(a), tb = tan(b), normalized to the unit sphere. Each tile's
/// a-axis increases eastward and b-axis northward, so cells wound
/// (n00, n10, n11, n01) are counter-clockwise viewed from outside.
///
/// Edge and corner nodes shared between tiles are deduplicated via exact
/// (quantized) coordinate hashing, giving 6·N² + 2 nodes and 6·N² cells.
/// Generation is deterministic: identical names produce bitwise-identical
/// coordinates in fixed iteration order.
template <class MemorySpace>
inline UnstructuredMesh<MemorySpace> generate_cubed_sphere(int N) {
    const auto n = static_cast<std::size_t>(N);
    const std::size_t n_cells_per_tile = n * n;
    const std::size_t n_cells = 6 * n_cells_per_tile;
    const std::size_t max_nodes = 6 * (n + 1) * (n + 1);  // before dedup

    const double pi = 3.14159265358979323846;
    const double quarter = pi / 4.0;
    const double delta = (pi / 2.0) / static_cast<double>(N);  // tile-local spacing (rad)

    // Unit vector for tile-local coords (a, b) on tile t (1-based). Delegates to
    // the shared helper so cell centers and node coordinates can never drift.
    auto tile_vec = [](int tile, double a, double b) { return cs_tile_vec(tile, a, b); };

    // Quantized key for node deduplication (shared edge/corner nodes).
    auto node_key = [](const std::array<double, 3> &v) {
        auto q = [](double c) { return static_cast<long long>(std::llround(c * 1.0e12)); };
        return std::make_tuple(q(v[0]), q(v[1]), q(v[2]));
    };

    std::map<std::tuple<long long, long long, long long>, std::size_t> node_map;
    std::vector<double> node_lon, node_lat;
    node_lon.reserve(max_nodes);
    node_lat.reserve(max_nodes);

    // Local (i, j) node index within tile t; deduplicated globally.
    auto local_node = [&](int tile, std::size_t i, std::size_t j) -> index_t {
        const double a = -quarter + static_cast<double>(i) * delta;
        const double b = -quarter + static_cast<double>(j) * delta;
        auto v = tile_vec(tile, a, b);
        auto key = node_key(v);
        auto it = node_map.find(key);
        if (it != node_map.end()) return static_cast<index_t>(it->second);
        const auto idx = node_lon.size();
        double lon = std::atan2(v[1], v[0]) * 180.0 / pi;
        if (lon < 0.0) lon += 360.0;
        node_lon.push_back(lon);
        node_lat.push_back(std::asin(v[2]) * 180.0 / pi);
        node_map.emplace(key, idx);
        return static_cast<index_t>(idx);
    };

    Kokkos::View<index_t *, Kokkos::HostSpace> h_offsets("cs_offsets", n_cells + 1);
    Kokkos::View<index_t *, Kokkos::HostSpace> h_indices("cs_indices", n_cells * 4);
    h_offsets(0) = 0;

    for (std::size_t cell = 0; cell < n_cells; ++cell) {
        const std::size_t t = cell / n_cells_per_tile;
        const std::size_t rem = cell % n_cells_per_tile;
        const std::size_t j = rem / n;
        const std::size_t i = rem % n;

        const index_t n00 = local_node(static_cast<int>(t) + 1, i, j);
        const index_t n10 = local_node(static_cast<int>(t) + 1, i + 1, j);
        const index_t n11 = local_node(static_cast<int>(t) + 1, i + 1, j + 1);
        const index_t n01 = local_node(static_cast<int>(t) + 1, i, j + 1);

        h_offsets(cell + 1) = h_offsets(cell) + 4;
        h_indices(cell * 4 + 0) = n00;
        h_indices(cell * 4 + 1) = n10;
        h_indices(cell * 4 + 2) = n11;
        h_indices(cell * 4 + 3) = n01;
    }

    const std::size_t n_nodes = node_lon.size();
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> h_coords("cs_coords", n_nodes, 2);
    for (std::size_t k = 0; k < n_nodes; ++k) {
        h_coords(k, 0) = node_lon[k];
        h_coords(k, 1) = node_lat[k];
    }

    auto node_coords = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_coords);
    auto conn_offsets = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_offsets);
    auto conn_indices = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_indices);

    return UnstructuredMesh<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), CoordinateSystem::SphericalDeg);
}

// ─────────────────────────────────────────────────────────────────────────────
// NamedGridRegistry public interface
// ─────────────────────────────────────────────────────────────────────────────

NamedGridRegistry::ParsedName NamedGridRegistry::parse(const std::string &name) {
    if (name.empty()) {
        throw std::invalid_argument("NamedGridRegistry::parse: empty grid name string");
    }

    std::string lower_name = name;
    for (char &c : lower_name) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    // Support case-insensitive NOAA GRIB grid numbers (e.g. "grid218")
    if (lower_name.rfind("grid", 0) == 0) {
        if (name.size() < 5) {
            throw std::invalid_argument("NamedGridRegistry::parse: grid name \"" + name + "\" has no number after the 'grid' prefix");
        }
        std::string num_str = name.substr(4);
        for (char ch : num_str) {
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                throw std::invalid_argument("NamedGridRegistry::parse: non-numeric character '" + std::string(1, ch) +
                                            "' in number portion of name \"" + name + "\"");
            }
        }
        int grid_num = 0;
        try {
            grid_num = std::stoi(num_str);
        } catch (...) {
            throw std::invalid_argument("NamedGridRegistry::parse: cannot parse number from name \"" + name + "\"");
        }
        if (grid_num <= 0) {
            throw std::invalid_argument("NamedGridRegistry::parse: grid number must be positive, got " + std::to_string(grid_num) + " in name \"" +
                                        name + "\"");
        }
        bool found_grib = false;
        for (std::size_t idx = 0; idx < NOAA_GRIB_GRIDS_COUNT; ++idx) {
            if (NOAA_GRIB_GRIDS[idx].number == grid_num) {
                found_grib = true;
                break;
            }
        }
        if (!found_grib) {
            throw std::invalid_argument("NamedGridRegistry::parse: unregistered NOAA GRIB grid number grid" + std::to_string(grid_num));
        }
        return ParsedName{'G', grid_num};
    }

    char family = static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));

    // Validate family
    if (family != 'O' && family != 'F' && family != 'N' && family != 'R' && family != 'C') {
        throw std::invalid_argument("NamedGridRegistry::parse: unknown grid family '" + std::string(1, name[0]) + "' in name \"" + name +
                                    "\"; registered families are O, F, N, C, R, and grid<num>");
    }

    // Parse number
    if (name.size() < 2) {
        throw std::invalid_argument("NamedGridRegistry::parse: grid name \"" + name + "\" has no number after the family prefix");
    }

    std::string num_str = name.substr(1);

    // Validate that the remainder is a valid integer
    for (char ch : num_str) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::invalid_argument("NamedGridRegistry::parse: non-numeric character '" + std::string(1, ch) + "' in number portion of name \"" +
                                        name + "\"");
        }
    }

    int number = 0;
    try {
        number = std::stoi(num_str);
    } catch (const std::exception &) {
        throw std::invalid_argument("NamedGridRegistry::parse: cannot parse number from name \"" + name + "\"");
    }

    if (number <= 0) {
        throw std::invalid_argument("NamedGridRegistry::parse: grid number must be positive, got " + std::to_string(number) + " in name \"" + name +
                                    "\"");
    }

    return ParsedName{family, number};
}

bool NamedGridRegistry::is_registered(const std::string &name) noexcept {
    try {
        (void)parse(name);
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<char> NamedGridRegistry::registered_families() {
    return {'C', 'F', 'G', 'N', 'O', 'R'};  // sorted
}

// ─────────────────────────────────────────────────────────────────────────────
// Cell-order layout descriptors + analytic cell centers
//
// Both mirror the generators above EXACTLY (same iteration order, same
// formulas), so a client can reshape a flat ncol result into (lat, lon) or
// (tile, j, i) coordinates that agree with the engine mesh to the bit.
// ─────────────────────────────────────────────────────────────────────────────

NamedGridRegistry::Layout NamedGridRegistry::layout(const std::string &name) {
    ParsedName parsed = parse(name);
    Layout L;
    L.family = parsed.family;
    L.number = parsed.number;
    const int N = parsed.number;
    switch (parsed.family) {
        case 'F':
            // Cells fill (2N-1) latitude rings x 4N columns, north to south.
            L.ni = static_cast<std::size_t>(4 * N);
            L.nj = static_cast<std::size_t>(2 * N - 1);
            L.row_uniform_lon = true;
            break;
        case 'R':
            // generate_regular_grid(4N-1, 2N-1, ...): cells fill (nj, ni), south to north.
            L.ni = static_cast<std::size_t>(4 * N - 1);
            L.nj = static_cast<std::size_t>(2 * N - 1);
            L.row_uniform_lon = true;
            break;
        case 'C':
            L.ni = static_cast<std::size_t>(N);
            L.nj = static_cast<std::size_t>(N);
            L.n_tiles = 6;
            break;
        case 'O':
        case 'N':
            // Reduced Gaussian: per-ring cell counts vary => no rectangular layout.
            break;
        case 'G': {
            for (std::size_t idx = 0; idx < NOAA_GRIB_GRIDS_COUNT; ++idx) {
                const auto &def = NOAA_GRIB_GRIDS[idx];
                if (def.number != N) continue;
                L.ni = def.ni;
                L.nj = def.nj;
                L.projected = def.proj_string != nullptr;
                if (L.projected) {
                    L.proj_string = def.proj_string;
                } else {
                    L.row_uniform_lon = true;
                }
                return L;
            }
            throw std::invalid_argument("NamedGridRegistry::layout: unregistered NOAA GRIB grid number " +
                                        std::to_string(N));
        }
        default:
            throw std::invalid_argument("NamedGridRegistry::layout: unknown family '" +
                                        std::string(1, parsed.family) + "'");
    }
    return L;
}

std::vector<double> NamedGridRegistry::cell_centers(const std::string &name) {
    ParsedName parsed = parse(name);
    const int N = parsed.number;
    constexpr double kPi = 3.14159265358979323846;
    std::vector<double> out;

    switch (parsed.family) {
        case 'F': {
            const int n_lon = 4 * N;
            const std::vector<double> lats = compute_gaussian_latitudes(N);
            const std::size_t nj = static_cast<std::size_t>(2 * N - 1);
            out.resize(2 * nj * static_cast<std::size_t>(n_lon));
            const double dlon = 360.0 / static_cast<double>(n_lon);
            for (std::size_t j = 0; j < nj; ++j) {
                const double lat_c = 0.5 * (lats[j] + lats[j + 1]);
                for (int i = 0; i < n_lon; ++i) {
                    const std::size_t c = j * static_cast<std::size_t>(n_lon) + static_cast<std::size_t>(i);
                    out[2 * c] = (static_cast<double>(i) + 0.5) * dlon;
                    out[2 * c + 1] = lat_c;
                }
            }
            break;
        }
        case 'R': {
            const std::size_t ni = static_cast<std::size_t>(4 * N - 1);
            const std::size_t nj = static_cast<std::size_t>(2 * N - 1);
            const double dlon = 360.0 / static_cast<double>(ni + 1);
            const double dlat = 180.0 / static_cast<double>(nj + 1);
            const double lon_start = -180.0;
            const double lat_start = -90.0 + 0.5 * dlat;
            out.resize(2 * ni * nj);
            for (std::size_t j = 0; j < nj; ++j) {
                for (std::size_t i = 0; i < ni; ++i) {
                    const std::size_t c = j * ni + i;
                    out[2 * c] = lon_start + (static_cast<double>(i) + 0.5) * dlon;
                    out[2 * c + 1] = lat_start + (static_cast<double>(j) + 0.5) * dlat;
                }
            }
            break;
        }
        case 'C': {
            const std::size_t n = static_cast<std::size_t>(N);
            const double quarter = kPi / 4.0;
            const double delta = (kPi / 2.0) / static_cast<double>(N);
            out.resize(2 * 6 * n * n);
            std::size_t c = 0;
            for (std::size_t t = 0; t < 6; ++t) {
                for (std::size_t j = 0; j < n; ++j) {
                    for (std::size_t i = 0; i < n; ++i, ++c) {
                        const double a = -quarter + (static_cast<double>(i) + 0.5) * delta;
                        const double b = -quarter + (static_cast<double>(j) + 0.5) * delta;
                        const auto ll = cs_unit_to_lonlat(cs_tile_vec(static_cast<int>(t) + 1, a, b));
                        out[2 * c] = ll[0];
                        out[2 * c + 1] = ll[1];
                    }
                }
            }
            break;
        }
        case 'O':
        case 'N': {
            const int n_lat = 2 * N;
            const std::vector<int> nlons = octahedral_nlons(N);
            const std::vector<double> lats = compute_gaussian_latitudes(N);
            out.resize(2 * reduced_total_cells(nlons));
            std::size_t cell = 0;
            for (int j = 0; j < n_lat - 1; ++j) {
                const int nlon_top = nlons[static_cast<std::size_t>(j)];
                const int nlon_bot = nlons[static_cast<std::size_t>(j + 1)];
                const int n_ring = std::max(nlon_top, nlon_bot);
                const double lat_c = 0.5 * (lats[static_cast<std::size_t>(j)] + lats[static_cast<std::size_t>(j + 1)]);
                // Corner node longitude for cell index c (or c+1) on a row: mirrors the
                // generator's proportional mapping floor(c / n_ring * nlon) % nlon.
                auto node_lon = [&](int c_i, int nlon) {
                    const int idx = static_cast<int>(std::floor(static_cast<double>(c_i) / static_cast<double>(n_ring) *
                                                                static_cast<double>(nlon))) %
                                    nlon;
                    return static_cast<double>(idx) * (360.0 / static_cast<double>(nlon));
                };
                for (int c = 0; c < n_ring; ++c, ++cell) {
                    const double tl = node_lon(c, nlon_top);
                    const double tr = node_lon(c + 1, nlon_top);
                    const double bl = node_lon(c, nlon_bot);
                    const double br = node_lon(c + 1, nlon_bot);
                    // Circular mean longitude (safe across the dateline wrap).
                    const double sr = std::sin(tl * kPi / 180.0) + std::sin(tr * kPi / 180.0) +
                                      std::sin(bl * kPi / 180.0) + std::sin(br * kPi / 180.0);
                    const double cr = std::cos(tl * kPi / 180.0) + std::cos(tr * kPi / 180.0) +
                                      std::cos(bl * kPi / 180.0) + std::cos(br * kPi / 180.0);
                    double lon = std::atan2(sr, cr) * 180.0 / kPi;
                    if (lon < 0.0) lon += 360.0;
                    out[2 * cell] = lon;
                    out[2 * cell + 1] = lat_c;
                }
            }
            break;
        }
        case 'G': {
            for (std::size_t idx = 0; idx < NOAA_GRIB_GRIDS_COUNT; ++idx) {
                const auto &def = NOAA_GRIB_GRIDS[idx];
                if (def.number != N) continue;
                const std::size_t ni = def.ni;
                const std::size_t nj = def.nj;
                out.resize(2 * ni * nj);
                if (def.proj_string != nullptr) {
                    // Projected grids: centers are projection-space (x, y) metres,
                    // matching ProjectionBuilder's cell ordering (i + j*ni).
                    const double min_x = def.lon_start;
                    const double max_x = def.dlon;
                    const double min_y = def.lat_start;
                    const double max_y = def.dlat;
                    const double dx = (max_x - min_x) / static_cast<double>(ni - 1);
                    const double dy = (max_y - min_y) / static_cast<double>(nj - 1);
                    for (std::size_t j = 0; j < nj; ++j) {
                        for (std::size_t i = 0; i < ni; ++i) {
                            const std::size_t c = j * ni + i;
                            out[2 * c] = min_x + static_cast<double>(i) * dx;
                            out[2 * c + 1] = min_y + static_cast<double>(j) * dy;
                        }
                    }
                } else {
                    for (std::size_t j = 0; j < nj; ++j) {
                        for (std::size_t i = 0; i < ni; ++i) {
                            const std::size_t c = j * ni + i;
                            out[2 * c] = def.lon_start + (static_cast<double>(i) + 0.5) * def.dlon;
                            out[2 * c + 1] = def.lat_start + (static_cast<double>(j) + 0.5) * def.dlat;
                        }
                    }
                }
                return out;
            }
            throw std::invalid_argument("NamedGridRegistry::cell_centers: unregistered NOAA GRIB grid number " +
                                        std::to_string(N));
        }
        default:
            throw std::invalid_argument("NamedGridRegistry::cell_centers: unknown family '" +
                                        std::string(1, parsed.family) + "'");
    }
    return out;
}

// Explicit instantiation of generate for HostSpace
template <>
UnstructuredMesh<Kokkos::HostSpace> NamedGridRegistry::generate<Kokkos::HostSpace>(const std::string &name) {
    ParsedName parsed = parse(name);

    switch (parsed.family) {
        case 'G':
            return generate_noaa_grib_grid<Kokkos::HostSpace>(parsed.number);
        case 'O':
            return generate_octahedral_gaussian<Kokkos::HostSpace>(parsed.number);
        case 'N':
            // N (reduced Gaussian) follows ECMWF octahedral convention
            return generate_octahedral_gaussian<Kokkos::HostSpace>(parsed.number);
        case 'F':
            return generate_regular_gaussian<Kokkos::HostSpace>(parsed.number);
        case 'C':
            return generate_cubed_sphere<Kokkos::HostSpace>(parsed.number);
        case 'R': {
            // R: like F but regular lat spacing
            const int N = parsed.number;
            const std::size_t ni = static_cast<std::size_t>(4 * N - 1);
            const std::size_t nj = static_cast<std::size_t>(2 * N - 1);
            const double dlon = 360.0 / static_cast<double>(ni + 1);
            const double dlat = 180.0 / static_cast<double>(nj + 1);
            return generate_regular_grid<Kokkos::HostSpace>(ni, nj, -180.0, -90.0 + 0.5 * dlat, dlon, dlat);
        }
        default:
            // Should not reach here due to parse() validation
            throw std::invalid_argument("NamedGridRegistry::generate: unknown family '" + std::string(1, parsed.family) + "'");
    }
}

}  // namespace axis::topology
