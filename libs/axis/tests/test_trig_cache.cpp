// AXIS unit test: TrigCache and build_trig_cache()
// Verifies that cached trig values match direct sin/cos computation,
// and that lonlat_to_xyz_cached() produces correct unit-sphere coordinates.

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"

#include <Kokkos_Core.hpp>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/detail/trig_cache.hpp>
#include <cmath>
#include <cstddef>

namespace {

}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

static constexpr double pi = 3.14159265358979323846;
static constexpr double deg2rad = pi / 180.0;

// Helper: create a valid RegularGridInfo for testing
static detail::RegularGridInfo make_grid_info(std::size_t ni, std::size_t nj, double lon_min, double delta_lon, double lat_min, double delta_lat) {
    detail::RegularGridInfo info;
    info.is_regular = true;
    info.ni = ni;
    info.nj = nj;
    info.lon_min = lon_min;
    info.lon_max = lon_min + static_cast<double>(ni) * delta_lon;
    info.delta_lon = delta_lon;
    info.lat_min = lat_min;
    info.lat_max = lat_min + static_cast<double>(nj) * delta_lat;
    info.delta_lat = delta_lat;
    return info;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: build_trig_cache with invalid info returns invalid cache
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, InvalidInfoReturnsInvalidCache) {
    detail::RegularGridInfo info;  // is_regular=false by default
    auto cache = detail::build_trig_cache<MemSpace>(info);
    EXPECT_FALSE(cache.valid);
}

TEST(TrigCache, ZeroSizeReturnsInvalidCache) {
    detail::RegularGridInfo info;
    info.is_regular = true;
    info.ni = 0;
    info.nj = 5;
    auto cache = detail::build_trig_cache<MemSpace>(info);
    EXPECT_FALSE(cache.valid);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: build_trig_cache produces correct dimensions
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, CorrectDimensions) {
    auto info = make_grid_info(10, 5, 0.0, 1.0, -2.5, 1.0);
    auto cache = detail::build_trig_cache<MemSpace>(info);

    EXPECT_TRUE(cache.valid);
    EXPECT_EQ(cache.ni, 10u);
    EXPECT_EQ(cache.nj, 5u);
    EXPECT_EQ(cache.sin_lon.extent(0), 10u);
    EXPECT_EQ(cache.cos_lon.extent(0), 10u);
    EXPECT_EQ(cache.sin_lat.extent(0), 5u);
    EXPECT_EQ(cache.cos_lat.extent(0), 5u);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Cached sin/cos values match direct computation
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, CachedValuesMatchDirectComputation) {
    const std::size_t ni = 36;
    const std::size_t nj = 18;
    const double lon_min = 0.0;
    const double delta_lon = 10.0;
    const double lat_min = -90.0;
    const double delta_lat = 10.0;

    auto info = make_grid_info(ni, nj, lon_min, delta_lon, lat_min, delta_lat);
    auto cache = detail::build_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);

    // Verify longitude sin/cos values
    for (std::size_t i = 0; i < ni; ++i) {
        double lon_rad = (lon_min + (static_cast<double>(i) + 0.5) * delta_lon) * deg2rad;
        double expected_sin = std::sin(lon_rad);
        double expected_cos = std::cos(lon_rad);

        EXPECT_NEAR(cache.sin_lon(i), expected_sin, 1e-15) << "sin_lon mismatch at i=" << i;
        EXPECT_NEAR(cache.cos_lon(i), expected_cos, 1e-15) << "cos_lon mismatch at i=" << i;
    }

    // Verify latitude sin/cos values
    for (std::size_t j = 0; j < nj; ++j) {
        double lat_rad = (lat_min + (static_cast<double>(j) + 0.5) * delta_lat) * deg2rad;
        double expected_sin = std::sin(lat_rad);
        double expected_cos = std::cos(lat_rad);

        EXPECT_NEAR(cache.sin_lat(j), expected_sin, 1e-15) << "sin_lat mismatch at j=" << j;
        EXPECT_NEAR(cache.cos_lat(j), expected_cos, 1e-15) << "cos_lat mismatch at j=" << j;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: lonlat_to_xyz_cached matches direct sin/cos computation
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, XyzCachedMatchesDirectComputation) {
    const std::size_t ni = 72;
    const std::size_t nj = 36;
    const double lon_min = -180.0;
    const double delta_lon = 5.0;
    const double lat_min = -90.0;
    const double delta_lat = 5.0;

    auto info = make_grid_info(ni, nj, lon_min, delta_lon, lat_min, delta_lat);
    auto cache = detail::build_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);

    // Check a sample of (i, j) pairs
    for (std::size_t j = 0; j < nj; j += 4) {
        for (std::size_t i = 0; i < ni; i += 4) {
            auto xyz = detail::lonlat_to_xyz_cached(cache, i, j);

            double lon_rad = (lon_min + (static_cast<double>(i) + 0.5) * delta_lon) * deg2rad;
            double lat_rad = (lat_min + (static_cast<double>(j) + 0.5) * delta_lat) * deg2rad;

            double expected_x = std::cos(lat_rad) * std::cos(lon_rad);
            double expected_y = std::cos(lat_rad) * std::sin(lon_rad);
            double expected_z = std::sin(lat_rad);

            EXPECT_NEAR(xyz.x, expected_x, 1e-15) << "x mismatch at i=" << i << ", j=" << j;
            EXPECT_NEAR(xyz.y, expected_y, 1e-15) << "y mismatch at i=" << i << ", j=" << j;
            EXPECT_NEAR(xyz.z, expected_z, 1e-15) << "z mismatch at i=" << i << ", j=" << j;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: XYZ output is on the unit sphere
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, XyzOutputIsUnitSphere) {
    const std::size_t ni = 20;
    const std::size_t nj = 10;
    const double lon_min = 0.0;
    const double delta_lon = 18.0;
    const double lat_min = -90.0;
    const double delta_lat = 18.0;

    auto info = make_grid_info(ni, nj, lon_min, delta_lon, lat_min, delta_lat);
    auto cache = detail::build_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            auto xyz = detail::lonlat_to_xyz_cached(cache, i, j);
            double norm = std::sqrt(xyz.x * xyz.x + xyz.y * xyz.y + xyz.z * xyz.z);
            EXPECT_NEAR(norm, 1.0, 1e-14) << "Not unit sphere at i=" << i << ", j=" << j;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: Grid metadata is stored correctly in cache
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, GridMetadataStoredCorrectly) {
    auto info = make_grid_info(100, 50, -180.0, 3.6, -90.0, 3.6);
    auto cache = detail::build_trig_cache<MemSpace>(info);

    EXPECT_TRUE(cache.valid);
    EXPECT_DOUBLE_EQ(cache.lon_min, -180.0);
    EXPECT_DOUBLE_EQ(cache.delta_lon, 3.6);
    EXPECT_DOUBLE_EQ(cache.lat_min, -90.0);
    EXPECT_DOUBLE_EQ(cache.delta_lat, 3.6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test: 1x1 grid cache works correctly (minimal case)
// ─────────────────────────────────────────────────────────────────────────────

TEST(TrigCache, MinimalGrid1x1) {
    auto info = make_grid_info(1, 1, 0.0, 360.0, 0.0, 180.0);
    auto cache = detail::build_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);
    EXPECT_EQ(cache.sin_lon.extent(0), 1u);
    EXPECT_EQ(cache.cos_lon.extent(0), 1u);
    EXPECT_EQ(cache.sin_lat.extent(0), 1u);
    EXPECT_EQ(cache.cos_lat.extent(0), 1u);

    // Cell center is at (180 deg, 90 deg) in radians
    double lon_rad = 180.0 * deg2rad;
    double lat_rad = 90.0 * deg2rad;

    EXPECT_NEAR(cache.sin_lon(0), std::sin(lon_rad), 1e-15);
    EXPECT_NEAR(cache.cos_lon(0), std::cos(lon_rad), 1e-15);
    EXPECT_NEAR(cache.sin_lat(0), std::sin(lat_rad), 1e-15);
    EXPECT_NEAR(cache.cos_lat(0), std::cos(lat_rad), 1e-15);
}

// ─────────────────────────────────────────────────────────────────────────────
// NodeTrigCache tests — vertex-level trig caching for regular grids
// ─────────────────────────────────────────────────────────────────────────────

TEST(NodeTrigCache, InvalidInfoReturnsInvalidCache) {
    detail::RegularGridInfo info;  // is_regular=false by default
    auto cache = detail::build_node_trig_cache<MemSpace>(info);
    EXPECT_FALSE(cache.valid);
}

TEST(NodeTrigCache, CorrectDimensions) {
    auto info = make_grid_info(10, 5, 0.0, 1.0, -2.5, 1.0);
    auto cache = detail::build_node_trig_cache<MemSpace>(info);

    EXPECT_TRUE(cache.valid);
    EXPECT_EQ(cache.ni, 10u);
    EXPECT_EQ(cache.nj, 5u);
    // Node cache has ni+1 lon entries and nj+1 lat entries
    EXPECT_EQ(cache.sin_lon.extent(0), 11u);
    EXPECT_EQ(cache.cos_lon.extent(0), 11u);
    EXPECT_EQ(cache.sin_lat.extent(0), 6u);
    EXPECT_EQ(cache.cos_lat.extent(0), 6u);
}

TEST(NodeTrigCache, NodeValuesMatchDirectComputation) {
    const std::size_t ni = 36;
    const std::size_t nj = 18;
    const double lon_min = 0.0;
    const double delta_lon = 10.0;
    const double lat_min = -90.0;
    const double delta_lat = 10.0;

    auto info = make_grid_info(ni, nj, lon_min, delta_lon, lat_min, delta_lat);
    auto cache = detail::build_node_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);

    // Verify longitude node sin/cos (ni+1 entries at lon_min + i * delta_lon)
    for (std::size_t i = 0; i <= ni; ++i) {
        double lon_rad = (lon_min + static_cast<double>(i) * delta_lon) * deg2rad;
        double expected_sin = std::sin(lon_rad);
        double expected_cos = std::cos(lon_rad);

        EXPECT_NEAR(cache.sin_lon(i), expected_sin, 1e-15) << "node sin_lon mismatch at i=" << i;
        EXPECT_NEAR(cache.cos_lon(i), expected_cos, 1e-15) << "node cos_lon mismatch at i=" << i;
    }

    // Verify latitude node sin/cos (nj+1 entries at lat_min + j * delta_lat)
    for (std::size_t j = 0; j <= nj; ++j) {
        double lat_rad = (lat_min + static_cast<double>(j) * delta_lat) * deg2rad;
        double expected_sin = std::sin(lat_rad);
        double expected_cos = std::cos(lat_rad);

        EXPECT_NEAR(cache.sin_lat(j), expected_sin, 1e-15) << "node sin_lat mismatch at j=" << j;
        EXPECT_NEAR(cache.cos_lat(j), expected_cos, 1e-15) << "node cos_lat mismatch at j=" << j;
    }
}

TEST(NodeTrigCache, XyzNodeCachedMatchesDirectComputation) {
    const std::size_t ni = 72;
    const std::size_t nj = 36;
    const double lon_min = -180.0;
    const double delta_lon = 5.0;
    const double lat_min = -90.0;
    const double delta_lat = 5.0;

    auto info = make_grid_info(ni, nj, lon_min, delta_lon, lat_min, delta_lat);
    auto cache = detail::build_node_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);

    // Check all node positions (i in [0, ni], j in [0, nj])
    for (std::size_t j = 0; j <= nj; j += 3) {
        for (std::size_t i = 0; i <= ni; i += 3) {
            auto xyz = detail::lonlat_to_xyz_node_cached(cache, i, j);

            double lon_rad = (lon_min + static_cast<double>(i) * delta_lon) * deg2rad;
            double lat_rad = (lat_min + static_cast<double>(j) * delta_lat) * deg2rad;

            double expected_x = std::cos(lat_rad) * std::cos(lon_rad);
            double expected_y = std::cos(lat_rad) * std::sin(lon_rad);
            double expected_z = std::sin(lat_rad);

            EXPECT_NEAR(xyz.x, expected_x, 1e-15) << "node x mismatch at i=" << i << ", j=" << j;
            EXPECT_NEAR(xyz.y, expected_y, 1e-15) << "node y mismatch at i=" << i << ", j=" << j;
            EXPECT_NEAR(xyz.z, expected_z, 1e-15) << "node z mismatch at i=" << i << ", j=" << j;
        }
    }
}

TEST(NodeTrigCache, XyzNodeOutputIsUnitSphere) {
    const std::size_t ni = 20;
    const std::size_t nj = 10;
    const double lon_min = 0.0;
    const double delta_lon = 18.0;
    const double lat_min = -90.0;
    const double delta_lat = 18.0;

    auto info = make_grid_info(ni, nj, lon_min, delta_lon, lat_min, delta_lat);
    auto cache = detail::build_node_trig_cache<MemSpace>(info);

    ASSERT_TRUE(cache.valid);

    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            auto xyz = detail::lonlat_to_xyz_node_cached(cache, i, j);
            double norm = std::sqrt(xyz.x * xyz.x + xyz.y * xyz.y + xyz.z * xyz.z);
            EXPECT_NEAR(norm, 1.0, 1e-14) << "Node not on unit sphere at i=" << i << ", j=" << j;
        }
    }
}

}  // namespace axis::test
