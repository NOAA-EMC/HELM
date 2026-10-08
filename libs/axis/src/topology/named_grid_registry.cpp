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
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/topology/reduced_gaussian_grid.hpp>
#include <axis/topology/rule_generator.hpp>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: Regular Gaussian (F family)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_regular_gaussian(int N) {
    ingest::GridRulesParams rules;
    rules.kind = "GaussianRegular";
    rules.gaussian_n = N;
    return RuleGenerator::generate<MemorySpace>(rules);
}

// ─────────────────────────────────────────────────────────────────────────────
// Grid generation: Octahedral / Reduced Gaussian (O and N families)
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> generate_octahedral_gaussian(int N) {
    if (N <= 0 || N > std::numeric_limits<int>::max() / 2) {
        throw std::invalid_argument("NamedGridRegistry: reduced Gaussian N must be positive and safely representable");
    }
    const int n_lat = 2 * N;
    const std::vector<int> nlons = octahedral_nlons(N);
    const std::vector<double> north_to_south = compute_gaussian_latitudes(N);

    // Legendre weights partition mu = sin(latitude) into latitude strips.
    // These are the Gaussian quadrature boundaries, not midpoint estimates
    // between center latitudes. The ragged provider builds one polygon per
    // longitude cell and explicitly splits hanging edges between rows.
    const int degree = n_lat;
    std::vector<double> weights(static_cast<std::size_t>(n_lat));
    for (int j = 0; j < n_lat; ++j) {
        const double x = std::sin(north_to_south[static_cast<std::size_t>(j)] * (M_PI / 180.0));
        double p0 = 1.0;
        double p1 = x;
        for (int k = 2; k <= degree; ++k) {
            const double pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / static_cast<double>(k);
            p0 = p1;
            p1 = pk;
        }
        const double derivative = degree * (x * p1 - p0) / (x * x - 1.0);
        weights[static_cast<std::size_t>(j)] = 2.0 / ((1.0 - x * x) * derivative * derivative);
    }

    std::vector<double> north_boundaries(static_cast<std::size_t>(n_lat) + 1);
    north_boundaries.front() = 90.0;
    double mu = 1.0;
    for (int j = 0; j < n_lat; ++j) {
        mu -= weights[static_cast<std::size_t>(j)];
        north_boundaries[static_cast<std::size_t>(j) + 1] = (j + 1 == n_lat) ? -90.0 : std::asin(std::clamp(mu, -1.0, 1.0)) * (180.0 / M_PI);
    }

    std::vector<double> latitude_boundaries(north_boundaries.rbegin(), north_boundaries.rend());
    std::vector<ReducedGaussianRow> rows;
    rows.reserve(static_cast<std::size_t>(n_lat));
    for (int j = n_lat - 1; j >= 0; --j) {
        const int nlon = nlons[static_cast<std::size_t>(j)];
        ReducedGaussianRow row;
        row.longitude_boundaries.resize(static_cast<std::size_t>(nlon) + 1);
        for (int i = 0; i <= nlon; ++i) {
            row.longitude_boundaries[static_cast<std::size_t>(i)] = 360.0 * static_cast<double>(i) / static_cast<double>(nlon);
        }
        rows.push_back(std::move(row));
    }

    ReducedGaussianGrid grid(std::move(latitude_boundaries), std::move(rows), CoordinateSystem::SphericalDeg,
                             LongitudePeriodicity{true, 360.0, true});
    return grid.to_unstructured<MemorySpace>(true);
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
    Kokkos::View<double *, Kokkos::HostSpace> h_areas("h_areas", n_cells);

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

            const double radians = std::acos(-1.0) / 180.0;
            const double south = lat_start + static_cast<double>(j) * dlat;
            const double north = south + dlat;
            h_areas(cell_idx) = dlon * radians * (std::sin(north * radians) - std::sin(south * radians));
        }
    }

    auto node_coords = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_coords);
    auto conn_offsets = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_offsets);
    auto conn_indices = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_indices);
    auto areas = Kokkos::create_mirror_view_and_copy(MemorySpace(), h_areas);

    GeometryMetadata geometry;
    geometry.provenance = GeometryProvenance::DeclaredGridModel;
    geometry.boundary_model = BoundaryModel::ConstantLatitude;
    geometry.area_model = AreaModel::ConstantLatitudeStrip;
    geometry.longitude_periodicity = LongitudePeriodicity{false, 360.0, true};
    return make_unstructured<MemorySpace>(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), CoordinateSystem::SphericalDeg,
                                          std::move(areas), {}, geometry);
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
    if (family != 'O' && family != 'F' && family != 'N' && family != 'R') {
        throw std::invalid_argument("NamedGridRegistry::parse: unknown grid family '" + std::string(1, name[0]) + "' in name \"" + name +
                                    "\"; registered families are O, F, N, R, and grid<num>");
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
    return {'F', 'G', 'N', 'O', 'R'};  // sorted
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
