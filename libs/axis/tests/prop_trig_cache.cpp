// ─── Property-Based Tests: Trig Cache Equivalence ────────────────────────────
// Feature: axis-performance-optimizations
//
// Property 6: Trig Cache Equivalence
//   For any regular lat-lon grid coordinate (lon_i, lat_j), the XYZ vector
//   produced by trig-cache lookup (cos_lat[j]*cos_lon[i], cos_lat[j]*sin_lon[i],
//   sin_lat[j]) SHALL agree with the direct computation (cos(lat)*cos(lon),
//   cos(lat)*sin(lon), sin(lat)) within relative tolerance 1e-15 per component.
//
// **Validates: Requirements 4.4**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/detail/trig_cache.hpp>
#include <cmath>
#include <cstddef>

namespace {

constexpr double deg2rad = 3.14159265358979323846 / 180.0;

/// Compare two doubles with relative tolerance per component.
/// When both are near zero, use absolute tolerance.
bool components_agree(double cached, double direct, double rel_tol = 1.0e-15) {
    double max_abs = std::fmax(std::fabs(cached), std::fabs(direct));
    if (max_abs < 1.0e-30) {
        return std::fabs(cached - direct) < 1.0e-30;
    }
    double rel_error = std::fabs(cached - direct) / max_abs;
    return rel_error <= rel_tol;
}

// ─── Property 6: Cell-Center TrigCache Equivalence ───────────────────────────
// Generate random RegularGridInfo structs; build the TrigCache; for each (i, j),
// compare lonlat_to_xyz_cached() output vs direct sin/cos computation.
//
// **Validates: Requirements 4.4**

RC_GTEST_PROP(PropTrigCache, CellCenterEquivalence, ()) {
    // Generate grid dimensions: ni in [1, 200], nj in [1, 100]
    std::size_t ni = static_cast<std::size_t>(*rc::gen::inRange(1, 201));
    std::size_t nj = static_cast<std::size_t>(*rc::gen::inRange(1, 101));

    // Generate lon_min in [-180, 180], delta_lon in [0.1, 10.0]
    double lon_min = *rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return v * 0.01; });
    double delta_lon = *rc::gen::map(rc::gen::inRange(10, 1000), [](int v) { return v * 0.01; });

    // Generate lat_min in [-90, 90], delta_lat in [0.1, 5.0]
    double lat_min = *rc::gen::map(rc::gen::inRange(-9000, 9001), [](int v) { return v * 0.01; });
    double delta_lat = *rc::gen::map(rc::gen::inRange(10, 500), [](int v) { return v * 0.01; });

    // Construct RegularGridInfo
    axis::detail::RegularGridInfo info;
    info.is_regular = true;
    info.ni = ni;
    info.nj = nj;
    info.lon_min = lon_min;
    info.delta_lon = delta_lon;
    info.lat_min = lat_min;
    info.delta_lat = delta_lat;
    info.lon_max = lon_min + ni * delta_lon;
    info.lat_max = lat_min + nj * delta_lat;

    // Build trig cache on host
    auto cache = axis::detail::build_trig_cache<Kokkos::HostSpace>(info);
    RC_ASSERT(cache.valid);

    // Check a sample of indices (up to 50 lon x 50 lat to keep runtime bounded)
    std::size_t lon_step = std::max(ni / 50, std::size_t{1});
    std::size_t lat_step = std::max(nj / 50, std::size_t{1});

    for (std::size_t i = 0; i < ni; i += lon_step) {
        for (std::size_t j = 0; j < nj; j += lat_step) {
            // Cached XYZ
            auto cached_xyz = axis::detail::lonlat_to_xyz_cached(cache, i, j);

            // Direct computation
            double lon_rad = (lon_min + (static_cast<double>(i) + 0.5) * delta_lon) * deg2rad;
            double lat_rad = (lat_min + (static_cast<double>(j) + 0.5) * delta_lat) * deg2rad;

            double direct_x = std::cos(lat_rad) * std::cos(lon_rad);
            double direct_y = std::cos(lat_rad) * std::sin(lon_rad);
            double direct_z = std::sin(lat_rad);

            RC_ASSERT(components_agree(cached_xyz.x, direct_x, 1.0e-15));
            RC_ASSERT(components_agree(cached_xyz.y, direct_y, 1.0e-15));
            RC_ASSERT(components_agree(cached_xyz.z, direct_z, 1.0e-15));
        }
    }
}

// ─── Property 6 (continued): Node TrigCache Equivalence ─────────────────────
// Same property for NodeTrigCache — uses node positions at (lon_min + i * delta_lon)
// without the +0.5 cell-center offset.
//
// **Validates: Requirements 4.4**

RC_GTEST_PROP(PropTrigCache, NodeCacheEquivalence, ()) {
    // Generate grid dimensions: ni in [1, 150], nj in [1, 75]
    std::size_t ni = static_cast<std::size_t>(*rc::gen::inRange(1, 151));
    std::size_t nj = static_cast<std::size_t>(*rc::gen::inRange(1, 76));

    // Generate lon_min in [-180, 180], delta_lon in [0.1, 10.0]
    double lon_min = *rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return v * 0.01; });
    double delta_lon = *rc::gen::map(rc::gen::inRange(10, 1000), [](int v) { return v * 0.01; });

    // Generate lat_min in [-90, 90], delta_lat in [0.1, 5.0]
    double lat_min = *rc::gen::map(rc::gen::inRange(-9000, 9001), [](int v) { return v * 0.01; });
    double delta_lat = *rc::gen::map(rc::gen::inRange(10, 500), [](int v) { return v * 0.01; });

    // Construct RegularGridInfo
    axis::detail::RegularGridInfo info;
    info.is_regular = true;
    info.ni = ni;
    info.nj = nj;
    info.lon_min = lon_min;
    info.delta_lon = delta_lon;
    info.lat_min = lat_min;
    info.delta_lat = delta_lat;
    info.lon_max = lon_min + ni * delta_lon;
    info.lat_max = lat_min + nj * delta_lat;

    // Build node trig cache on host
    auto node_cache = axis::detail::build_node_trig_cache<Kokkos::HostSpace>(info);
    RC_ASSERT(node_cache.valid);

    // Node indices go from 0 to ni (inclusive) for lon, 0 to nj (inclusive) for lat
    std::size_t n_lon_nodes = ni + 1;
    std::size_t n_lat_nodes = nj + 1;

    std::size_t lon_step = std::max(n_lon_nodes / 50, std::size_t{1});
    std::size_t lat_step = std::max(n_lat_nodes / 50, std::size_t{1});

    for (std::size_t i = 0; i < n_lon_nodes; i += lon_step) {
        for (std::size_t j = 0; j < n_lat_nodes; j += lat_step) {
            // Cached XYZ via node cache
            auto cached_xyz = axis::detail::lonlat_to_xyz_node_cached(node_cache, i, j);

            // Direct computation: node positions at (lon_min + i * delta_lon), no +0.5 offset
            double lon_rad = (lon_min + static_cast<double>(i) * delta_lon) * deg2rad;
            double lat_rad = (lat_min + static_cast<double>(j) * delta_lat) * deg2rad;

            double direct_x = std::cos(lat_rad) * std::cos(lon_rad);
            double direct_y = std::cos(lat_rad) * std::sin(lon_rad);
            double direct_z = std::sin(lat_rad);

            RC_ASSERT(components_agree(cached_xyz.x, direct_x, 1.0e-15));
            RC_ASSERT(components_agree(cached_xyz.y, direct_y, 1.0e-15));
            RC_ASSERT(components_agree(cached_xyz.z, direct_z, 1.0e-15));
        }
    }
}

// ─── Property 6 (continued): Boundary Index Correctness ─────────────────────
// Verify that first and last indices (boundary cells) are also exact.
//
// **Validates: Requirements 4.4**

RC_GTEST_PROP(PropTrigCache, BoundaryIndexCorrectness, ()) {
    // Generate grid dimensions
    std::size_t ni = static_cast<std::size_t>(*rc::gen::inRange(2, 100));
    std::size_t nj = static_cast<std::size_t>(*rc::gen::inRange(2, 100));

    double lon_min = *rc::gen::map(rc::gen::inRange(-18000, 18001), [](int v) { return v * 0.01; });
    double delta_lon = *rc::gen::map(rc::gen::inRange(10, 1000), [](int v) { return v * 0.01; });
    double lat_min = *rc::gen::map(rc::gen::inRange(-9000, 9001), [](int v) { return v * 0.01; });
    double delta_lat = *rc::gen::map(rc::gen::inRange(10, 500), [](int v) { return v * 0.01; });

    axis::detail::RegularGridInfo info;
    info.is_regular = true;
    info.ni = ni;
    info.nj = nj;
    info.lon_min = lon_min;
    info.delta_lon = delta_lon;
    info.lat_min = lat_min;
    info.delta_lat = delta_lat;
    info.lon_max = lon_min + ni * delta_lon;
    info.lat_max = lat_min + nj * delta_lat;

    auto cache = axis::detail::build_trig_cache<Kokkos::HostSpace>(info);
    RC_ASSERT(cache.valid);

    // Test all four corners: (0,0), (0,nj-1), (ni-1,0), (ni-1,nj-1)
    std::size_t corners_i[] = {0, 0, ni - 1, ni - 1};
    std::size_t corners_j[] = {0, nj - 1, 0, nj - 1};

    for (int c = 0; c < 4; ++c) {
        std::size_t i = corners_i[c];
        std::size_t j = corners_j[c];

        auto cached_xyz = axis::detail::lonlat_to_xyz_cached(cache, i, j);

        double lon_rad = (lon_min + (static_cast<double>(i) + 0.5) * delta_lon) * deg2rad;
        double lat_rad = (lat_min + (static_cast<double>(j) + 0.5) * delta_lat) * deg2rad;

        double direct_x = std::cos(lat_rad) * std::cos(lon_rad);
        double direct_y = std::cos(lat_rad) * std::sin(lon_rad);
        double direct_z = std::sin(lat_rad);

        RC_ASSERT(components_agree(cached_xyz.x, direct_x, 1.0e-15));
        RC_ASSERT(components_agree(cached_xyz.y, direct_y, 1.0e-15));
        RC_ASSERT(components_agree(cached_xyz.z, direct_z, 1.0e-15));
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
