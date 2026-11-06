// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/rule_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/types.hpp>
#include <limits>
#include <stdexcept>

namespace {

class KokkosEnv : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnv);

using MemSpace = Kokkos::HostSpace;
using namespace axis::topology;

StructuredGrid<MemSpace> make_grid() {
    Kokkos::View<double *, MemSpace> lon("lon", 4);
    Kokkos::View<double *, MemSpace> lat("lat", 4);
    lon(0) = 45.0;
    lon(1) = 135.0;
    lon(2) = 45.0;
    lon(3) = 135.0;
    lat(0) = -10.0;
    lat(1) = -10.0;
    lat(2) = 10.0;
    lat(3) = 10.0;
    return StructuredGrid<MemSpace>(2, 2, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);
}

TEST(StructuredGridPolicies, RequireExplicitRejectsMissingCorners) {
    auto grid = make_grid();
    EXPECT_THROW(grid.to_unstructured(CornerPolicy::RequireExplicit, {}), std::invalid_argument);
    EXPECT_THROW(grid.to_unstructured(CornerPolicy::GaussianLatLon, {}), std::invalid_argument);
}

TEST(StructuredGridPolicies, ExplicitPeriodicityControlsTheReconstructedSeam) {
    Kokkos::View<double *, MemSpace> lon("lon", 2);
    Kokkos::View<double *, MemSpace> lat("lat", 2);
    lon(0) = 45.0;
    lon(1) = 135.0;
    lat(0) = 0.0;
    lat(1) = 0.0;
    StructuredGrid<MemSpace> grid(2, 1, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    auto mesh = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{true, 360.0, true});
    auto coords = mesh.node_coords();
    EXPECT_NEAR(coords(0, 0), -90.0, 1e-12);
    EXPECT_NEAR(coords(2, 0), 270.0, 1e-12);
    EXPECT_NEAR(coords(2, 0) - coords(0, 0), 360.0, 1e-12);
}

TEST(StructuredGridPolicies, GeneratedCornersDoNotBecomeAuthoritativeOrSurviveSeamChanges) {
    Kokkos::View<double *, MemSpace> lon("lon", 2);
    Kokkos::View<double *, MemSpace> lat("lat", 2);
    lon(0) = 45.0;
    lon(1) = 135.0;
    lat(0) = 0.0;
    lat(1) = 0.0;
    StructuredGrid<MemSpace> grid(2, 1, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    const auto periodic = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{true, 360.0, true});
    EXPECT_FALSE(grid.has_explicit_corners());
    EXPECT_EQ(grid.corner_lon().extent(0), 0u);
    EXPECT_NEAR(periodic.node_coords()(0, 0), -90.0, 1e-12);
    EXPECT_NEAR(periodic.node_coords()(2, 0), 270.0, 1e-12);
    EXPECT_THROW(grid.to_unstructured(CornerPolicy::RequireExplicit, {}), std::invalid_argument);

    const auto nonperiodic = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{false, 360.0, true});
    EXPECT_NEAR(nonperiodic.node_coords()(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(nonperiodic.node_coords()(1, 0), 90.0, 1e-12);
    EXPECT_NEAR(nonperiodic.node_coords()(2, 0), 180.0, 1e-12);
}

TEST(StructuredGridPolicies, NoArgumentConversionDoesNotInferPeriodicityFromSpacing) {
    Kokkos::View<double *, MemSpace> lon("lon", 2);
    Kokkos::View<double *, MemSpace> lat("lat", 2);
    lon(0) = 45.0;
    lon(1) = 135.0;
    lat(0) = 0.0;
    lat(1) = 0.0;
    StructuredGrid<MemSpace> grid(2, 1, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    const auto mesh = grid.to_unstructured();
    EXPECT_NEAR(mesh.node_coords()(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(mesh.node_coords()(2, 0), 180.0, 1e-12);
}

TEST(StructuredGridPolicies, NoArgumentConversionAcceptsCurvilinearCenterCoordinates) {
    Kokkos::View<double *, MemSpace> lon("lon", 4);
    Kokkos::View<double *, MemSpace> lat("lat", 4);
    lon(0) = 0.0;
    lon(1) = 2.0;
    lon(2) = 0.2;
    lon(3) = 2.2;
    lat(0) = 0.0;
    lat(1) = 0.1;
    lat(2) = 2.0;
    lat(3) = 2.1;
    StructuredGrid<MemSpace> grid(2, 2, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    const auto mesh = grid.to_unstructured();
    EXPECT_EQ(mesh.n_cells(), 4u);
    EXPECT_EQ(mesh.geometry_metadata().provenance, GeometryProvenance::Approximate);
    EXPECT_EQ(mesh.geometry_metadata().boundary_model, BoundaryModel::GreatCircle);
}

TEST(StructuredGridPolicies, NonperiodicPerimeterUsesOneSidedExtrapolation) {
    Kokkos::View<double *, MemSpace> lon("lon", 4);
    Kokkos::View<double *, MemSpace> lat("lat", 4);
    lon(0) = 0.0;
    lon(1) = 2.0;
    lon(2) = 0.0;
    lon(3) = 2.0;
    lat(0) = 10.0;
    lat(1) = 10.0;
    lat(2) = 14.0;
    lat(3) = 14.0;
    StructuredGrid<MemSpace> grid(2, 2, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    auto mesh = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{false, 360.0, true});
    const auto coords = mesh.node_coords();
    constexpr double expected_lon[] = {-1.0, 1.0, 3.0, -1.0, 1.0, 3.0, -1.0, 1.0, 3.0};
    constexpr double expected_lat[] = {8.0, 8.0, 8.0, 12.0, 12.0, 12.0, 16.0, 16.0, 16.0};
    for (std::size_t node = 0; node < 9; ++node) {
        EXPECT_NEAR(coords(node, 0), expected_lon[node], 1e-12);
        EXPECT_NEAR(coords(node, 1), expected_lat[node], 1e-12);
    }

    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();
    EXPECT_EQ(offsets(0), 0);
    EXPECT_EQ(indices(0), 0);
    EXPECT_EQ(indices(1), 1);
    EXPECT_EQ(indices(2), 4);
    EXPECT_EQ(indices(3), 3);
    mesh.compute_areas();
    for (std::size_t cell = 0; cell < mesh.n_cells(); ++cell) EXPECT_GT(mesh.cell_areas()(cell), 0.0);
}

TEST(StructuredGridPolicies, NonperiodicDatelineCrossingIsUnwrappedLocally) {
    Kokkos::View<double *, MemSpace> lon("lon", 4);
    Kokkos::View<double *, MemSpace> lat("lat", 4);
    lon(0) = 179.0;
    lon(1) = -179.0;
    lon(2) = 179.0;
    lon(3) = -179.0;
    lat(0) = -1.0;
    lat(1) = -1.0;
    lat(2) = 1.0;
    lat(3) = 1.0;
    StructuredGrid<MemSpace> grid(2, 2, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    const auto mesh = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{false, 360.0, true});
    const auto coords = mesh.node_coords();
    constexpr double expected_lon[] = {178.0, 180.0, 182.0, 178.0, 180.0, 182.0, 178.0, 180.0, 182.0};
    for (std::size_t node = 0; node < 9; ++node) EXPECT_NEAR(coords(node, 0), expected_lon[node], 1e-12);
}

TEST(StructuredGridPolicies, NonuniformRectilinearCentersUseMidpointsAndOneSidedBounds) {
    constexpr std::size_t ni = 3;
    constexpr std::size_t nj = 2;
    Kokkos::View<double *, MemSpace> lon("lon", ni * nj);
    Kokkos::View<double *, MemSpace> lat("lat", ni * nj);
    const double x[] = {1.0, 3.0, 8.0};
    const double y[] = {-2.0, 4.0};
    for (std::size_t j = 0; j < nj; ++j)
        for (std::size_t i = 0; i < ni; ++i) {
            lon(i + j * ni) = x[i];
            lat(i + j * ni) = y[j];
        }
    StructuredGrid<MemSpace> grid(ni, nj, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);
    const auto mesh = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{false, 360.0, true});
    const double expected_x[] = {0.0, 2.0, 5.5, 10.5};
    const double expected_y[] = {-5.0, 1.0, 7.0};
    const auto coords = mesh.node_coords();
    for (std::size_t j = 0; j <= nj; ++j)
        for (std::size_t i = 0; i <= ni; ++i) {
            const std::size_t node = i + j * (ni + 1);
            EXPECT_NEAR(coords(node, 0), expected_x[i], 1e-12);
            EXPECT_NEAR(coords(node, 1), expected_y[j], 1e-12);
        }
    EXPECT_EQ(mesh.geometry_metadata().provenance, GeometryProvenance::DeclaredGridModel);
    EXPECT_EQ(mesh.geometry_metadata().boundary_model, BoundaryModel::ConstantLatitude);
    EXPECT_EQ(mesh.geometry_metadata().area_model, AreaModel::ConstantLatitudeStrip);
    EXPECT_NEAR(
        mesh.cell_areas()(0),
        (2.0 * 3.14159265358979323846 / 180.0) * (std::sin(1.0 * 3.14159265358979323846 / 180.0) - std::sin(-5.0 * 3.14159265358979323846 / 180.0)),
        1e-12);
}

TEST(StructuredGridPolicies, ExplicitRectilinearBoundsOverrideCenterReconstruction) {
    constexpr std::size_t ni = 2;
    constexpr std::size_t nj = 2;
    Kokkos::View<double *, MemSpace> lon("lon", 4);
    Kokkos::View<double *, MemSpace> lat("lat", 4);
    for (std::size_t j = 0; j < nj; ++j)
        for (std::size_t i = 0; i < ni; ++i) {
            lon(i + j * ni) = 10.0 + 20.0 * static_cast<double>(i);
            lat(i + j * ni) = -10.0 + 20.0 * static_cast<double>(j);
        }
    Kokkos::View<double *, MemSpace> lon_bounds("lon_bounds", 3);
    Kokkos::View<double *, MemSpace> lat_bounds("lat_bounds", 3);
    lon_bounds(0) = 0.0;
    lon_bounds(1) = 15.0;
    lon_bounds(2) = 50.0;
    lat_bounds(0) = -20.0;
    lat_bounds(1) = 0.0;
    lat_bounds(2) = 20.0;
    StructuredGrid<MemSpace> grid(ni, nj, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);
    grid.set_rectilinear_bounds(std::move(lon_bounds), std::move(lat_bounds));
    const auto mesh = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{false, 360.0, true});
    const double expected_x[] = {0.0, 15.0, 50.0};
    const double expected_y[] = {-20.0, 0.0, 20.0};
    for (std::size_t j = 0; j <= nj; ++j)
        for (std::size_t i = 0; i <= ni; ++i) {
            const std::size_t node = i + j * (ni + 1);
            EXPECT_DOUBLE_EQ(mesh.node_coords()(node, 0), expected_x[i]);
            EXPECT_DOUBLE_EQ(mesh.node_coords()(node, 1), expected_y[j]);
        }
}

TEST(StructuredGridPolicies, GaussianWeightsDefinePolarClosedConstantLatitudeBands) {
    constexpr std::size_t ni = 2;
    constexpr std::size_t nj = 2;
    Kokkos::View<double *, MemSpace> lon("lon", ni * nj);
    Kokkos::View<double *, MemSpace> lat("lat", ni * nj);
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            lon(i + j * ni) = 90.0 + 180.0 * static_cast<double>(i);
            lat(i + j * ni) = j == 0 ? 35.264389682754654 : -35.264389682754654;
        }
    }
    Kokkos::View<double *, MemSpace> weights("weights", nj);
    weights(0) = 1.0;
    weights(1) = 1.0;
    StructuredGrid<MemSpace> grid(ni, nj, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);
    grid.set_gaussian_latitude_weights(std::move(weights));

    const auto mesh = grid.to_unstructured(CornerPolicy::GaussianLatLon, LongitudePeriodicity{true, 360.0, true});
    const auto coords = mesh.node_coords();
    EXPECT_DOUBLE_EQ(coords(0, 1), 90.0);
    EXPECT_DOUBLE_EQ(coords(3, 1), 0.0);
    EXPECT_DOUBLE_EQ(coords(6, 1), -90.0);
    EXPECT_DOUBLE_EQ(coords(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(coords(2, 0), 360.0);
    EXPECT_EQ(mesh.geometry_metadata().boundary_model, BoundaryModel::ConstantLatitude);
    EXPECT_EQ(mesh.geometry_metadata().area_model, AreaModel::ConstantLatitudeStrip);
    double area = 0.0;
    for (std::size_t c = 0; c < mesh.n_cells(); ++c) {
        EXPECT_GT(mesh.cell_areas()(c), 0.0);
        area += mesh.cell_areas()(c);
    }
    EXPECT_NEAR(area, 4.0 * 3.14159265358979323846, 1e-12);
}

TEST(StructuredGridPolicies, GaussianWeightsMustBeAuthoritativeAndValid) {
    auto grid = make_grid();
    EXPECT_THROW(grid.to_unstructured(CornerPolicy::GaussianLatLon, LongitudePeriodicity{true, 360.0, true}), std::invalid_argument);
    Kokkos::View<double *, MemSpace> weights("weights", 2);
    weights(0) = 0.5;
    weights(1) = 0.5;
    grid.set_gaussian_latitude_weights(std::move(weights));
    EXPECT_THROW(grid.to_unstructured(CornerPolicy::GaussianLatLon, LongitudePeriodicity{true, 360.0, true}), std::invalid_argument);
}

TEST(GaussianGridGeometry, RuleGeneratorUsesQuadratureStripBoundariesAndAreas) {
    axis::ingest::GridRulesParams rules;
    rules.kind = "GaussianRegular";
    rules.gaussian_n = 1;
    const auto mesh = RuleGenerator::generate<MemSpace>(rules);
    ASSERT_EQ(mesh.n_cells(), 8u);
    EXPECT_EQ(mesh.geometry_metadata().boundary_model, BoundaryModel::ConstantLatitude);
    EXPECT_EQ(mesh.geometry_metadata().area_model, AreaModel::ConstantLatitudeStrip);
    EXPECT_DOUBLE_EQ(mesh.node_coords()(0, 1), 90.0);
    EXPECT_NEAR(mesh.node_coords()(5, 1), 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(mesh.node_coords()(10, 1), -90.0);
    double area = 0.0;
    for (std::size_t cell = 0; cell < mesh.n_cells(); ++cell) {
        EXPECT_NEAR(mesh.cell_areas()(cell), 0.5 * 3.14159265358979323846, 1e-12);
        area += mesh.cell_areas()(cell);
    }
    EXPECT_NEAR(area, 4.0 * 3.14159265358979323846, 1e-12);
}

TEST(StructuredGridPolicies, CurvilinearApproximationIsLabeledAndRejectsFoldedCells) {
    Kokkos::View<double *, MemSpace> lon("lon", 4);
    Kokkos::View<double *, MemSpace> lat("lat", 4);
    lon(0) = 0.0;
    lon(1) = 2.0;
    lon(2) = 0.2;
    lon(3) = 2.2;
    lat(0) = 0.0;
    lat(1) = 0.1;
    lat(2) = 2.0;
    lat(3) = 2.1;
    StructuredGrid<MemSpace> warped(2, 2, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);
    EXPECT_THROW(warped.to_unstructured(CornerPolicy::RequireExplicit, {}), std::invalid_argument);
    const auto approximate = warped.to_unstructured(CornerPolicy::CurvilinearApproximate, LongitudePeriodicity{false, 360.0, true});
    EXPECT_EQ(approximate.geometry_metadata().provenance, GeometryProvenance::Approximate);
    EXPECT_FALSE(warped.has_explicit_corners());

    Kokkos::View<double *, MemSpace> folded_lon("folded_lon", 4);
    Kokkos::View<double *, MemSpace> folded_lat("folded_lat", 4);
    folded_lon(0) = 0.0;
    folded_lon(1) = 1.0;
    folded_lon(2) = 1.0;
    folded_lon(3) = 0.0;
    folded_lat(0) = 0.0;
    folded_lat(1) = 0.0;
    folded_lat(2) = 1.0;
    folded_lat(3) = 1.0;
    StructuredGrid<MemSpace> folded(2, 2, std::move(folded_lon), std::move(folded_lat), CoordinateSystem::SphericalDeg);
    EXPECT_THROW(folded.to_unstructured(CornerPolicy::CurvilinearApproximate, LongitudePeriodicity{false, 360.0, true}), std::invalid_argument);
}

TEST(StructuredGridPolicies, SingletonAxisUsesConstantExtrapolationWithoutUninitializedCorners) {
    Kokkos::View<double *, MemSpace> lon("lon", 2);
    Kokkos::View<double *, MemSpace> lat("lat", 2);
    lon(0) = 1.0;
    lon(1) = 3.0;
    lat(0) = 5.0;
    lat(1) = 5.0;
    StructuredGrid<MemSpace> grid(2, 1, std::move(lon), std::move(lat), CoordinateSystem::SphericalDeg);

    const auto mesh = grid.to_unstructured(CornerPolicy::RectilinearMidpoint, LongitudePeriodicity{false, 360.0, true});
    const auto coords = mesh.node_coords();
    EXPECT_DOUBLE_EQ(coords(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(coords(1, 1), 5.0);
    EXPECT_DOUBLE_EQ(coords(2, 1), 5.0);
    EXPECT_DOUBLE_EQ(coords(3, 1), 5.0);
    EXPECT_DOUBLE_EQ(coords(4, 1), 5.0);
    EXPECT_DOUBLE_EQ(coords(5, 1), 5.0);
}

TEST(StructuredGridPolicies, RejectsInvalidPeriodicPeriod) {
    auto grid = make_grid();
    EXPECT_THROW(grid.to_unstructured(CornerPolicy::CurvilinearApproximate, LongitudePeriodicity{true, 0.0, true}), std::invalid_argument);
}

TEST(StructuredGridPolicies, RejectsDimensionProductOverflow) {
    Kokkos::View<double *, MemSpace> empty_lon;
    Kokkos::View<double *, MemSpace> empty_lat;
    EXPECT_THROW(StructuredGrid<MemSpace>(std::numeric_limits<std::size_t>::max(), 2, std::move(empty_lon), std::move(empty_lat),
                                          CoordinateSystem::SphericalDeg),
                 std::overflow_error);
}

TEST(StructuredGridPolicies, EmbedsCartesianCornersInThreeDimensions) {
    Kokkos::View<double *, MemSpace> x("x", 1);
    Kokkos::View<double *, MemSpace> y("y", 1);
    x(0) = 0.5;
    y(0) = 0.5;
    StructuredGrid<MemSpace> grid(1, 1, std::move(x), std::move(y), CoordinateSystem::Cartesian3D);
    Kokkos::View<double *, MemSpace> corner_x("corner_x", 4);
    Kokkos::View<double *, MemSpace> corner_y("corner_y", 4);
    corner_x(0) = 0.0;
    corner_x(1) = 1.0;
    corner_x(2) = 0.0;
    corner_x(3) = 1.0;
    corner_y(0) = 0.0;
    corner_y(1) = 0.0;
    corner_y(2) = 1.0;
    corner_y(3) = 1.0;
    grid.set_corners(std::move(corner_x), std::move(corner_y));
    EXPECT_TRUE(grid.has_explicit_corners());

    auto mesh = grid.to_unstructured();
    ASSERT_EQ(mesh.node_coords().extent(1), 3);
    for (std::size_t node = 0; node < mesh.n_nodes(); ++node) EXPECT_DOUBLE_EQ(mesh.node_coords()(node, 2), 0.0);
    mesh.compute_areas();
    EXPECT_NEAR(mesh.cell_areas()(0), 1.0, 1e-12);
}

}  // namespace
