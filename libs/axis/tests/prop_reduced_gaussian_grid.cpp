// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/reduced_gaussian_grid.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using Space = Kokkos::HostSpace;
using axis::topology::CoordinateSystem;
using axis::topology::LongitudePeriodicity;
using axis::topology::ReducedGaussianGrid;
using axis::topology::ReducedGaussianRow;

class KokkosEnvironment : public ::testing::Environment {
   public:
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
static auto *const kokkos_env = ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);

RC_GTEST_PROP(PropReducedGaussianGrid, RaggedRowsPreserveOrderAndConserveArea, ()) {
    const std::size_t n_rows = *rc::gen::inRange<std::size_t>(2, 7);
    std::vector<double> latitude_bounds(n_rows + 1);
    for (std::size_t j = 0; j <= n_rows; ++j) {
        latitude_bounds[j] = -70.0 + 140.0 * static_cast<double>(j) / static_cast<double>(n_rows);
    }

    std::vector<std::size_t> row_counts(n_rows);
    std::vector<ReducedGaussianRow> rows(n_rows);
    std::vector<axis::index_t> expected_offsets(n_rows + 1, 0);
    for (std::size_t j = 0; j < n_rows; ++j) {
        row_counts[j] = *rc::gen::inRange<std::size_t>(2, 10);
        const double origin = static_cast<double>(*rc::gen::inRange(-30, 31));
        auto &bounds = rows[j].longitude_boundaries;
        bounds.resize(row_counts[j] + 1);
        for (std::size_t i = 0; i <= row_counts[j]; ++i) {
            bounds[i] = origin + 360.0 * static_cast<double>(i) / static_cast<double>(row_counts[j]);
        }
        expected_offsets[j + 1] = expected_offsets[j] + static_cast<axis::index_t>(row_counts[j]);
    }

    const ReducedGaussianGrid grid(latitude_bounds, rows, CoordinateSystem::SphericalDeg, LongitudePeriodicity{true, 360.0, true});
    RC_ASSERT(grid.row_offsets() == expected_offsets);
    const auto mesh = grid.to_unstructured<Space>(true);
    const auto repeated = grid.to_unstructured<Space>(true);
    const std::size_t expected_cells = static_cast<std::size_t>(expected_offsets.back());
    RC_ASSERT(mesh.n_cells() == expected_cells);
    RC_ASSERT(mesh.conn_offsets().extent(0) == expected_cells + 1);
    RC_ASSERT(mesh.conn_offsets()(0) == 0);
    RC_ASSERT(mesh.conn_offsets()(expected_cells) == static_cast<axis::index_t>(mesh.conn_indices().extent(0)));

    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();
    const auto repeated_offsets = repeated.conn_offsets();
    const auto repeated_indices = repeated.conn_indices();
    for (std::size_t cell = 0; cell < expected_cells; ++cell) {
        RC_ASSERT(offsets(cell) == repeated_offsets(cell));
        RC_ASSERT(offsets(cell + 1) == repeated_offsets(cell + 1));
        RC_ASSERT(offsets(cell + 1) - offsets(cell) >= 4);
        const auto begin = static_cast<std::size_t>(offsets(cell));
        const auto end = static_cast<std::size_t>(offsets(cell + 1));
        for (std::size_t k = begin; k < end; ++k) {
            RC_ASSERT(indices(k) >= 0);
            RC_ASSERT(static_cast<std::size_t>(indices(k)) < mesh.n_nodes());
            RC_ASSERT(indices(k) == repeated_indices(k));
        }
    }

    const auto areas = mesh.cell_areas();
    double total_area = 0.0;
    for (std::size_t cell = 0; cell < expected_cells; ++cell) {
        RC_ASSERT(std::isfinite(areas(cell)));
        RC_ASSERT(areas(cell) > 0.0);
        total_area += areas(cell);
    }
    const double expected_area =
        2.0 * std::acos(-1.0) *
        (std::sin(latitude_bounds.back() * std::acos(-1.0) / 180.0) - std::sin(latitude_bounds.front() * std::acos(-1.0) / 180.0));
    RC_ASSERT(std::abs(total_area - expected_area) < 1e-10);
}

}  // namespace
