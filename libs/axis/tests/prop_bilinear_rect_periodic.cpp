// ─── Property-Based Tests: Bilinear Regular-Grid Fast-Path ───────────────────
// Feature: bilinear-regular-grid-fastpath, Property 4: Periodic Longitude Wraparound Invariance
//
// For any periodic source grid (spanning 360° longitude), and for any
// destination longitude value offset by ±k*360°, the produced bilinear weights
// SHALL be identical to those produced for the canonical (normalized) longitude.
// Furthermore, no destination point SHALL be treated as unmapped in the
// longitude direction, regardless of the UnmappedAction policy.
//
// **Validates: Requirements 4.2, 4.3, 7.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/test_kokkos_env.hpp"

namespace {

using MemSpace = Kokkos::HostSpace;

// ─── Kokkos Initialization ───────────────────────────────────────────────────

// ─── Generators ──────────────────────────────────────────────────────────────

/// Generate periodic source grid longitude cell count in [4, 72].
rc::Gen<std::size_t> genPeriodicNi() {
    return rc::gen::inRange<std::size_t>(4, 73);
}

/// Generate source grid latitude cell count in [4, 36].
rc::Gen<std::size_t> genNj() {
    return rc::gen::inRange<std::size_t>(4, 37);
}

/// Generate a starting longitude in [-180, 180) for the periodic grid.
rc::Gen<double> genLonMin() {
    return rc::gen::map(rc::gen::inRange(-18000, 18000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a starting latitude in [-90, 60] (leaves room for nj cells).
rc::Gen<double> genLatMin() {
    return rc::gen::map(rc::gen::inRange(-9000, 6000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a latitude extent in [10, 180] degrees (total lat range).
rc::Gen<double> genLatExtent() {
    return rc::gen::map(rc::gen::inRange(1000, 18000), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a random destination longitude in [-540, 900] — wide range to
/// exercise normalization across multiple 360° wraps.
rc::Gen<double> genDstLon() {
    return rc::gen::map(rc::gen::inRange(-54000, 90001), [](int v) { return static_cast<double>(v) / 100.0; });
}

/// Generate a random destination latitude within a given [lat_min, lat_max].
rc::Gen<double> genDstLat(double lat_min, double lat_max) {
    const double lo = (lat_min <= lat_max) ? lat_min : lat_max;
    const double hi = (lat_min <= lat_max) ? lat_max : lat_min;
    return rc::gen::map(rc::gen::inRange(0, 10001), [lo, hi](int t) {
        const double alpha = static_cast<double>(t) / 10000.0;
        return lo + alpha * (hi - lo);
    });
}

/// Generate a wraparound offset multiplier k in [-3, 3] (excluding 0).
rc::Gen<int> genNonZeroK() {
    return rc::gen::suchThat(rc::gen::inRange(-3, 4), [](int k) { return k != 0; });
}

// ─── Helper: Build a periodic regular-grid UnstructuredMesh ──────────────────

/// Constructs a periodic (360° longitude) regular lat-lon grid.
/// Grid covers [lon_min, lon_min + 360°] × [lat_min, lat_min + lat_extent].
axis::topology::UnstructuredMesh<MemSpace> make_periodic_grid(std::size_t ni, std::size_t nj, double lon_min, double lat_min, double lat_extent) {
    const double lon_max = lon_min + 360.0;
    const double lat_max = lat_min + lat_extent;
    const double delta_lon = 360.0 / static_cast<double>(ni);
    const double delta_lat = lat_extent / static_cast<double>(nj);

    const std::size_t n_centers = ni * nj;
    Kokkos::View<double *, MemSpace> cx("cx", n_centers);
    Kokkos::View<double *, MemSpace> cy("cy", n_centers);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_min + (static_cast<double>(i) + 0.5) * delta_lon;
            cy(i + j * ni) = lat_min + (static_cast<double>(j) + 0.5) * delta_lat;
        }
    }

    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    const std::size_t nc_i = ni + 1;
    const std::size_t nc_j = nj + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc_i * nc_j);
    Kokkos::View<double *, MemSpace> cry("cry", nc_i * nc_j);
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * nc_i) = lon_min + static_cast<double>(i) * delta_lon;
            cry(i + j * nc_i) = lat_min + static_cast<double>(j) * delta_lat;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Helper: Build a single-cell destination mesh at a given point ───────────

/// Creates a 1×1 destination mesh with its center at (lon, lat).
/// Uses a small cell extent (±0.1°) around the center.
axis::topology::UnstructuredMesh<MemSpace> make_single_cell_dst(double lon, double lat) {
    constexpr double half_dx = 0.1;

    Kokkos::View<double *, MemSpace> cx("dst_cx", 1);
    Kokkos::View<double *, MemSpace> cy("dst_cy", 1);
    cx(0) = lon;
    cy(0) = lat;

    axis::topology::StructuredGrid<MemSpace> grid(1, 1, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);

    // Corners: 2×2 nodes around the center
    Kokkos::View<double *, MemSpace> crx("dst_crx", 4);
    Kokkos::View<double *, MemSpace> cry("dst_cry", 4);
    crx(0) = lon - half_dx;
    cry(0) = lat - half_dx;
    crx(1) = lon + half_dx;
    cry(1) = lat - half_dx;
    crx(2) = lon - half_dx;
    cry(2) = lat + half_dx;
    crx(3) = lon + half_dx;
    cry(3) = lat + half_dx;
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Helper: Extract weight entries for a given row ──────────────────────────

struct WeightEntry {
    axis::index_t col;
    double weight;
};

std::vector<WeightEntry> get_row_entries(const axis::solver::InterpolationMatrix<MemSpace> &matrix, axis::index_t row_idx) {
    std::vector<WeightEntry> entries;
    const auto nnz = matrix.nnz();
    auto rows = matrix.factor_row();
    auto cols = matrix.factor_col();
    auto vals = matrix.factor_list();
    for (std::size_t k = 0; k < nnz; ++k) {
        if (rows[k] == row_idx) {
            entries.push_back({cols[k], vals[k]});
        }
    }
    // Sort by column index for stable comparison
    std::sort(entries.begin(), entries.end(), [](const WeightEntry &a, const WeightEntry &b) { return a.col < b.col; });
    return entries;
}

// ─── Property 4a: Weights are identical for shifted vs canonical longitude ───
// For any periodic source grid and any destination longitude, running the
// interpolation with the longitude offset by k*360° (k ≠ 0) produces the
// same weights as the canonical longitude.
//
// **Validates: Requirements 4.2, 4.3, 7.4**

RC_GTEST_PROP(PropBilinearRectPeriodic, ShiftedLongitudeProducesSameWeights, ()) {
    const auto ni = *genPeriodicNi();
    const auto nj = *genNj();
    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();
    const auto lat_extent = *genLatExtent();

    // Ensure lat_max stays within reasonable bounds
    const double lat_max = lat_min + lat_extent;
    RC_PRE(lat_max <= 90.0);
    RC_PRE(lat_min >= -90.0);

    // Build the periodic source grid (360° longitude span)
    auto src_mesh = make_periodic_grid(ni, nj, lon_min, lat_min, lat_extent);

    // Generate a random destination longitude and latitude within source domain
    const auto dst_lon = *genDstLon();
    const auto dst_lat = *genDstLat(lat_min, lat_max);

    // Generate a non-zero wraparound offset
    const auto k = *genNonZeroK();
    const double shifted_lon = dst_lon + static_cast<double>(k) * 360.0;

    // Build two single-cell destination meshes: canonical and shifted
    auto dst_canonical = make_single_cell_dst(dst_lon, dst_lat);
    auto dst_shifted = make_single_cell_dst(shifted_lon, dst_lat);

    // Configure interpolation with Error policy — should NOT throw for periodic
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Error;

    // Run interpolation for canonical longitude — must not throw
    auto matrix_canonical = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_canonical, cfg);

    // Run interpolation for shifted longitude — must not throw
    auto matrix_shifted = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_shifted, cfg);

    // Extract weight entries for the single destination cell (row 0)
    auto entries_canonical = get_row_entries(matrix_canonical, 0);
    auto entries_shifted = get_row_entries(matrix_shifted, 0);

    // Both must have the same number of entries
    RC_ASSERT(entries_canonical.size() == entries_shifted.size());
    RC_ASSERT(!entries_canonical.empty());

    // Column indices and weights must match exactly
    for (std::size_t i = 0; i < entries_canonical.size(); ++i) {
        RC_ASSERT(entries_canonical[i].col == entries_shifted[i].col);
        RC_ASSERT(std::abs(entries_canonical[i].weight - entries_shifted[i].weight) < 1e-13);
    }
}

// ─── Property 4b: No unmapped errors for any longitude on periodic grids ─────
// For any periodic source grid, no destination point is treated as unmapped in
// the longitude direction, regardless of the UnmappedAction policy. The
// interpolation must produce valid weights without throwing.
//
// **Validates: Requirements 4.2, 4.3, 7.4**

RC_GTEST_PROP(PropBilinearRectPeriodic, NoUnmappedForAnyLongitude, ()) {
    const auto ni = *genPeriodicNi();
    const auto nj = *genNj();
    const auto lon_min = *genLonMin();
    const auto lat_min = *genLatMin();
    const auto lat_extent = *genLatExtent();

    const double lat_max = lat_min + lat_extent;
    RC_PRE(lat_max <= 90.0);
    RC_PRE(lat_min >= -90.0);

    auto src_mesh = make_periodic_grid(ni, nj, lon_min, lat_min, lat_extent);

    // Generate an arbitrary destination longitude (may be far outside [lon_min, lon_min+360])
    const auto dst_lon = *genDstLon();
    // Keep latitude within the source grid's latitude range to avoid lat-unmapped
    const auto dst_lat = *genDstLat(lat_min, lat_max);

    auto dst_mesh = make_single_cell_dst(dst_lon, dst_lat);

    // Use Error policy — if the point were unmapped, this would throw
    axis::solver::RegridConfig cfg;
    cfg.method = axis::solver::InterpolationMethod::Bilinear;
    cfg.line_type = axis::solver::LineType::GreatCircle;
    cfg.unmapped = axis::solver::UnmappedAction::Error;

    // Must not throw — periodic grids should never have unmapped longitude
    auto matrix = axis::solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // Must produce valid weight entries (non-empty)
    auto entries = get_row_entries(matrix, 0);
    RC_ASSERT(!entries.empty());

    // Verify partition of unity
    double wsum = 0.0;
    for (const auto &e : entries) {
        RC_ASSERT(e.weight >= 0.0);
        RC_ASSERT(e.weight <= 1.0);
        wsum += e.weight;
    }
    RC_ASSERT(std::abs(wsum - 1.0) < 1e-15);
}

}  // namespace
