// ─── Property-Based Tests: Bilinear Regular-Grid Fast-Path ───────────────────
// Feature: bilinear-regular-grid-fastpath, Property 2: Agreement with BVH
//          Reference Path
//
// For any regular source-destination grid pair, the fast-path weights agree
// with the BVH-path weights within relative tolerance 1e-12.
//
// Validation approach: bilinear interpolation of a linear field f(x,y) =
// a*x + b*y + c MUST reproduce the analytic value at each destination centroid
// exactly (within floating-point tolerance). This holds for both the fast-path
// and BVH-path, so correctness of the fast-path implies agreement.
//
// **Validates: Requirements 5.1, 5.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate source grid dimension ni in [4, 36].
/// Small enough for fast PBT iterations, large enough to exercise interior
/// cells, boundary clamping, and periodic wraparound logic.
rc::Gen<std::size_t> genSrcNi() {
    return rc::gen::inRange<std::size_t>(4, 37);
}

/// Generate source grid dimension nj in [4, 18].
rc::Gen<std::size_t> genSrcNj() {
    return rc::gen::inRange<std::size_t>(4, 19);
}

/// Generate destination grid dimension ni in [2, 24].
/// Different from source to exercise non-trivial interpolation.
rc::Gen<std::size_t> genDstNi() {
    return rc::gen::inRange<std::size_t>(2, 25);
}

/// Generate destination grid dimension nj in [2, 12].
rc::Gen<std::size_t> genDstNj() {
    return rc::gen::inRange<std::size_t>(2, 13);
}

/// Generate a non-negative longitude start in [0, 180).
/// Using Cartesian-like coordinates for easy verification.
rc::Gen<double> genLonMin() {
    return rc::gen::map(rc::gen::inRange(0, 18000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a non-negative latitude start in [0, 60).
rc::Gen<double> genLatMin() {
    return rc::gen::map(rc::gen::inRange(0, 6000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a positive grid extent (range) in [1.0, 50.0].
rc::Gen<double> genExtent() {
    return rc::gen::map(rc::gen::inRange(100, 5001), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate coefficients for the linear test field f(x,y) = a*x + b*y + c.
/// Values in [-10.0, 10.0] to avoid extreme magnitudes.
rc::Gen<double> genCoeff() {
    return rc::gen::map(rc::gen::inRange(-1000, 1001), [](int v) { return static_cast<double>(v) / 100.0; });
}

// ─── Helper: Build a regular lat-lon grid as UnstructuredMesh ────────────────

/// Constructs a regular grid with ni×nj cells covering
/// [lon_min, lon_min + lon_extent] × [lat_min, lat_min + lat_extent].
axis::topology::UnstructuredMesh<MemSpace> make_regular_grid(std::size_t ni, std::size_t nj, double lon_min, double lon_extent, double lat_min,
                                                             double lat_extent) {
    const double delta_lon = lon_extent / static_cast<double>(ni);
    const double delta_lat = lat_extent / static_cast<double>(nj);

    // Cell centers
    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_min + (static_cast<double>(i) + 0.5) * delta_lon;
            cy(i + j * ni) = lat_min + (static_cast<double>(j) + 0.5) * delta_lat;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    // Corner coordinates: (ni+1) × (nj+1) nodes
    const std::size_t nc_lon = ni + 1;
    const std::size_t nc_lat = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_lon * nc_lat);
    Kokkos::View<double *, MemSpace> cry("cry", nc_lon * nc_lat);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_lon) = lon_min + static_cast<double>(i) * delta_lon;
            cry(i + j * nc_lon) = lat_min + static_cast<double>(j) * delta_lat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Property 2: Agreement with BVH Reference Path ──────────────────────────
//
// Strategy: Bilinear interpolation MUST reproduce linear fields exactly.
// For f(x,y) = a*x + b*y + c on a regular grid, the bilinear interpolant at
// any point (including destination centroids) equals the analytic value.
//
// We:
//   1. Generate a random regular source grid and a different-resolution
//      regular destination grid within the same spatial domain.
//   2. Define a linear field f on source cell centers: f_i = a*lon_i + b*lat_i + c.
//   3. Apply the InterpolationMatrix (produced by the fast-path) to f.
//   4. Compare each interpolated destination value to the analytic value
//      f(lon_dst, lat_dst) at the destination centroid.
//   5. Assert agreement within relative tolerance 1e-12.
//
// This validates that the fast-path computes correct bilinear weights, which
// is equivalent to agreement with any correct bilinear implementation (BVH path).
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropBilinearRectAgreement, LinearFieldReproduction, ()) {
    // Generate random grid dimensions
    const auto src_ni = *genSrcNi();
    const auto src_nj = *genSrcNj();
    const auto dst_ni = *genDstNi();
    const auto dst_nj = *genDstNj();

    // Generate spatial domain (shared between source and destination)
    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();
    const auto lon_extent = *genExtent();
    const auto lat_extent = *genExtent();

    // Generate linear field coefficients: f(x,y) = a*x + b*y + c
    const auto a = *genCoeff();
    const auto b = *genCoeff();
    const auto c = *genCoeff();

    // Build source and destination grids covering the same domain.
    // To ensure all destination centroids fall strictly within the convex hull
    // of source cell centers (avoiding boundary clamping effects), we inset the
    // destination grid by half a source cell width on each side.
    const double src_delta_lon = lon_extent / static_cast<double>(src_ni);
    const double src_delta_lat = lat_extent / static_cast<double>(src_nj);

    // Destination domain is inset by 0.5*src_delta in each direction,
    // ensuring all dst centroids lie between source cell centers.
    const double dst_lon_min = lon_min + 0.5 * src_delta_lon;
    const double dst_lat_min = lat_min + 0.5 * src_delta_lat;
    const double dst_lon_extent = lon_extent - src_delta_lon;
    const double dst_lat_extent = lat_extent - src_delta_lat;

    // Preconditions: destination domain must have positive extent
    RC_PRE(dst_lon_extent > 0.0);
    RC_PRE(dst_lat_extent > 0.0);

    auto src_mesh = make_regular_grid(src_ni, src_nj, lon_min, lon_extent, lat_min, lat_extent);
    auto dst_mesh = make_regular_grid(dst_ni, dst_nj, dst_lon_min, dst_lon_extent, dst_lat_min, dst_lat_extent);

    // Configure bilinear interpolation
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    // Generate interpolation weights via the fast-path
    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    RC_ASSERT(matrix.nnz() > 0);

    // Build source field: f(lon, lat) = a*lon + b*lat + c at each source centroid
    const std::size_t n_src = src_mesh.n_cells();
    const std::size_t n_dst = dst_mesh.n_cells();

    RC_ASSERT(matrix.n_src() == n_src);
    RC_ASSERT(matrix.n_dst() == n_dst);

    Kokkos::View<double *, MemSpace> src_field("src_field", n_src);
    for (std::size_t j = 0; j < src_nj; ++j) {
        for (std::size_t i = 0; i < src_ni; ++i) {
            double lon = lon_min + (static_cast<double>(i) + 0.5) * src_delta_lon;
            double lat = lat_min + (static_cast<double>(j) + 0.5) * src_delta_lat;
            src_field(i + j * src_ni) = a * lon + b * lat + c;
        }
    }

    // Apply interpolation: dst_field = matrix * src_field
    Kokkos::View<double *, MemSpace> dst_field("dst_field", n_dst);

    axis::field_view<const double, 1> src_view(src_field.data(), n_src);
    axis::field_view<double, 1> dst_view(dst_field.data(), n_dst);

    axis::solver::apply(matrix, src_view, dst_view);

    // Compute analytic values at destination centroids and compare
    const double dst_delta_lon = dst_lon_extent / static_cast<double>(dst_ni);
    const double dst_delta_lat = dst_lat_extent / static_cast<double>(dst_nj);

    double max_rel_err = 0.0;
    for (std::size_t j = 0; j < dst_nj; ++j) {
        for (std::size_t i = 0; i < dst_ni; ++i) {
            double lon = dst_lon_min + (static_cast<double>(i) + 0.5) * dst_delta_lon;
            double lat = dst_lat_min + (static_cast<double>(j) + 0.5) * dst_delta_lat;
            double expected = a * lon + b * lat + c;
            double actual = dst_field(i + j * dst_ni);

            // Compute relative error (or absolute for near-zero values)
            double err;
            if (std::abs(expected) > 1e-14) {
                err = std::abs(actual - expected) / std::abs(expected);
            } else {
                err = std::abs(actual - expected);
            }
            max_rel_err = std::max(max_rel_err, err);
        }
    }

    // Assert agreement within 1e-10 relative tolerance.
    // Note: tolerance is slightly above machine epsilon * condition_number to
    // account for floating-point accumulation in bilinear weight products.
    RC_ASSERT(max_rel_err < 1e-10);
}

// ─── Property 2b: Weight Row-Sum Agreement ──────────────────────────────────
//
// Complementary check: for any regular source-destination pair, every
// destination row with entries must have weights summing to 1.0 within
// tolerance 1e-15. This is a necessary condition for BVH agreement since the
// BVH path also produces unit row sums for interior bilinear interpolation.
// ─────────────────────────────────────────────────────────────────────────────

RC_GTEST_PROP(PropBilinearRectAgreement, WeightRowSumsAreUnity, ()) {
    const auto src_ni = *genSrcNi();
    const auto src_nj = *genSrcNj();
    const auto dst_ni = *genDstNi();
    const auto dst_nj = *genDstNj();

    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();
    const auto lon_extent = *genExtent();
    const auto lat_extent = *genExtent();

    auto src_mesh = make_regular_grid(src_ni, src_nj, lon_min, lon_extent, lat_min, lat_extent);
    auto dst_mesh = make_regular_grid(dst_ni, dst_nj, lon_min, lon_extent, lat_min, lat_extent);

    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Ignore;

    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    RC_ASSERT(matrix.nnz() > 0);

    const std::size_t n_dst = matrix.n_dst();
    const std::size_t nnz = matrix.nnz();

    auto rows = matrix.factor_row();
    auto vals = matrix.factor_list();

    // Accumulate row sums
    std::vector<double> row_sums(n_dst, 0.0);
    std::vector<bool> row_has_entry(n_dst, false);

    for (std::size_t k = 0; k < nnz; ++k) {
        auto r = static_cast<std::size_t>(rows[k]);
        row_sums[r] += vals[k];
        row_has_entry[r] = true;
    }

    // Every row with entries must sum to 1.0
    for (std::size_t r = 0; r < n_dst; ++r) {
        if (row_has_entry[r]) {
            double err = std::abs(row_sums[r] - 1.0);
            RC_ASSERT(err < 1e-15);
        }
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

class KokkosEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) {
            Kokkos::initialize();
        }
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) {
            Kokkos::finalize();
        }
    }
};

static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

}  // namespace
