// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/weight_generator.hpp>
#include <cmath>
#include <stdexcept>

namespace {

class SamplePositionWeightsEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
static auto *const sample_position_environment = ::testing::AddGlobalTestEnvironment(new SamplePositionWeightsEnvironment);

class SamplePositionWeightsTest : public ::testing::Test {};

using HostPositions = Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace>;

TEST_F(SamplePositionWeightsTest, NearestUsesSuppliedNonuniformCurvilinearPositions) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 4, 2);
    src.positions(0, 0) = -2.0;
    src.positions(0, 1) = -10.0;
    src.positions(1, 0) = 3.0;
    src.positions(1, 1) = 1.0;
    src.positions(2, 0) = 0.0;
    src.positions(2, 1) = 4.0;
    src.positions(3, 0) = 1.0;
    src.positions(3, 1) = -3.0;
    dst.positions = HostPositions("dst_positions", 1, 2);
    dst.positions(0, 0) = 2.8;
    dst.positions(0, 1) = 0.1;

    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 1u);
    EXPECT_EQ(matrix.factor_col()(0), 1);
    EXPECT_DOUBLE_EQ(matrix.factor_list()(0), 1.0);
}

TEST_F(SamplePositionWeightsTest, CurvilinearNearestUsesCenterCoordinates) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("curvilinear_centers", 4, 2);
    src.positions(0, 0) = -2.0;
    src.positions(0, 1) = -1.0;
    src.positions(1, 0) = 1.0;
    src.positions(1, 1) = -0.5;
    src.positions(2, 0) = -1.0;
    src.positions(2, 1) = 2.0;
    src.positions(3, 0) = 2.2;
    src.positions(3, 1) = 1.8;
    src.nx = 2;
    src.ny = 2;
    dst.positions = HostPositions("destination_center", 1, 2);
    dst.positions(0, 0) = 2.0;
    dst.positions(0, 1) = 1.7;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 1u);
    EXPECT_EQ(matrix.factor_col()(0), 3);
}

TEST_F(SamplePositionWeightsTest, SphericalNearestWrapsAtDatelineWithoutCells) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 2, 2);
    src.positions(0, 0) = 359.0;
    src.positions(0, 1) = 0.0;
    src.positions(1, 0) = 180.0;
    src.positions(1, 1) = 0.0;
    dst.positions = HostPositions("dst_positions", 1, 2);
    dst.positions(0, 0) = 1.0;
    dst.positions(0, 1) = 0.0;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 1u);
    EXPECT_EQ(matrix.factor_col()(0), 0);
}

TEST_F(SamplePositionWeightsTest, ValidatesCoordinatesMaskAndMethod) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 1, 2);
    dst.positions = HostPositions("dst_positions", 1, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 91.0;
    dst.positions(0, 0) = 0.0;
    dst.positions(0, 1) = 0.0;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    EXPECT_THROW(axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config),
                 std::invalid_argument);
    src.positions(0, 1) = 0.0;
    src.mask = Kokkos::View<int *, Kokkos::HostSpace>("mask", 2);
    EXPECT_THROW(axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config),
                 std::invalid_argument);
    src.mask = {};
    config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
    EXPECT_THROW(axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config),
                 std::invalid_argument);
}

TEST_F(SamplePositionWeightsTest, EmptyDestinationReturnsEmptyMatrix) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 1, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 0.0;
    dst.positions = HostPositions("dst_positions", 0, 2);
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    EXPECT_EQ(matrix.n_src(), 1u);
    EXPECT_EQ(matrix.n_dst(), 0u);
    EXPECT_EQ(matrix.nnz(), 0u);
}

TEST_F(SamplePositionWeightsTest, StructuredBilinearUsesFourSurroundingCenters) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 4, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 0.0;
    src.positions(1, 0) = 1.0;
    src.positions(1, 1) = 0.0;
    src.positions(2, 0) = 0.0;
    src.positions(2, 1) = 1.0;
    src.positions(3, 0) = 1.0;
    src.positions(3, 1) = 1.0;
    src.nx = 2;
    src.ny = 2;
    dst.positions = HostPositions("dst_positions", 1, 2);
    dst.positions(0, 0) = 0.5;
    dst.positions(0, 1) = 0.5;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Bilinear;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::Cartesian3D, config);
    EXPECT_EQ(matrix.nnz(), 4u);
    for (std::size_t k = 0; k < matrix.nnz(); ++k) EXPECT_NEAR(matrix.factor_list()(k), 0.25, 1e-12);
}

TEST_F(SamplePositionWeightsTest, NearestSkipsMaskedSourceSamples) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 2, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 0.0;
    src.positions(1, 0) = 10.0;
    src.positions(1, 1) = 0.0;
    src.mask = Kokkos::View<int *, Kokkos::HostSpace>("src_mask", 2);
    src.mask(0) = 0;
    src.mask(1) = 1;
    dst.positions = HostPositions("dst_positions", 1, 2);
    dst.positions(0, 0) = 0.0;
    dst.positions(0, 1) = 0.0;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 1u);
    EXPECT_EQ(matrix.factor_col()(0), 1);
}

TEST_F(SamplePositionWeightsTest, NearestOmitsMaskedDestinationRows) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 1, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 0.0;
    dst.positions = HostPositions("dst_positions", 2, 2);
    dst.positions(0, 0) = 0.0;
    dst.positions(0, 1) = 0.0;
    dst.positions(1, 0) = 1.0;
    dst.positions(1, 1) = 0.0;
    dst.mask = Kokkos::View<int *, Kokkos::HostSpace>("dst_mask", 2);
    dst.mask(0) = 1;
    dst.mask(1) = 0;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 1u);
    EXPECT_EQ(matrix.factor_row()(0), 0);
    EXPECT_EQ(matrix.n_dst(), 2u);
}

TEST_F(SamplePositionWeightsTest, PreservesStableGlobalSourceColumns) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 3, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 0.0;
    src.positions(1, 0) = 10.0;
    src.positions(1, 1) = 0.0;
    src.positions(2, 0) = 20.0;
    src.positions(2, 1) = 0.0;
    src.global_indices = Kokkos::View<axis::index_t *, Kokkos::HostSpace>("global_ids", 3);
    src.global_indices(0) = 7;
    src.global_indices(1) = 2;
    src.global_indices(2) = 5;
    src.global_source_count = 8;
    dst.positions = HostPositions("dst_positions", 1, 2);
    dst.positions(0, 0) = 10.0;
    dst.positions(0, 1) = 0.0;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 1u);
    EXPECT_EQ(matrix.n_src(), 8u);
    EXPECT_EQ(matrix.factor_col()(0), 2);
}

TEST_F(SamplePositionWeightsTest, PreservesStableGlobalDestinationRows) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 1, 2);
    src.positions(0, 0) = 0.0;
    src.positions(0, 1) = 0.0;
    dst.positions = HostPositions("dst_positions", 2, 2);
    dst.positions(0, 0) = 0.0;
    dst.positions(0, 1) = 0.0;
    dst.positions(1, 0) = 10.0;
    dst.positions(1, 1) = 0.0;
    dst.global_indices = Kokkos::View<axis::index_t *, Kokkos::HostSpace>("global_destination_ids", 2);
    dst.global_indices(0) = 11;
    dst.global_indices(1) = 3;
    dst.global_destination_count = 12;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::NearestNeighbor;
    const auto matrix = axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config);
    ASSERT_EQ(matrix.nnz(), 2u);
    EXPECT_EQ(matrix.n_dst(), 12u);
    EXPECT_EQ(matrix.factor_row()(0), 11);
    EXPECT_EQ(matrix.factor_row()(1), 3);
}

TEST_F(SamplePositionWeightsTest, RejectsMethodsWithoutRequiredStructuredStencil) {
    axis::solver::SampleGrid src, dst;
    src.positions = HostPositions("src_positions", 4, 2);
    dst.positions = HostPositions("dst_positions", 1, 2);
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Bilinear;
    EXPECT_THROW(axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config),
                 std::invalid_argument);
    src.nx = 2;
    src.ny = 2;
    config.method = axis::solver::InterpolationMethod::Bicubic;
    EXPECT_THROW(axis::solver::WeightGenerator::generate_from_samples(src, dst, axis::topology::CoordinateSystem::SphericalDeg, config),
                 std::invalid_argument);
}

}  // namespace
