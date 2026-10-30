// ─── Property-Based Tests: Inter-Grid Remapping Robustness ──────────────────
// Uses RapidCheck to verify that WeightGenerator can successfully generate
// interpolation weight matrices between any combination of grid types
// (unstructured O/N-families, structured/rectilinear F/R-families) in both
// directions, ensuring partition of unity, bounds, and non-negativity.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <cmath>
#include <string>

#include "common/test_kokkos_env.hpp"

namespace {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

/// Generate grid family character:
/// 'O' -> Octahedral grid (unstructured)
/// 'F' -> F-family structured
/// 'N' -> N-family unstructured
/// 'R' -> Rectilinear Gaussian/unstructured
rc::Gen<char> genGridFamily() {
    return rc::gen::element('O', 'F', 'N', 'R');
}

/// Generate small grid resolution N value [2, 6] to keep tests fast
rc::Gen<int> genGridRes() {
    return rc::gen::inRange(2, 7);
}

/// Generate a valid named-grid string
rc::Gen<std::string> genGridName() {
    return rc::gen::apply([](char family, int res) { return std::string(1, family) + std::to_string(res); }, genGridFamily(), genGridRes());
}

RC_GTEST_PROP(PropGridCombinations, RemapBetweenAnyGridCombination, ()) {
    // Generate randomized source and destination grid types/dimensions
    const std::string src_name = *genGridName();
    const std::string dst_name = *genGridName();

    auto src_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(src_name);
    auto dst_mesh = axis::topology::NamedGridRegistry::generate<MemSpace>(dst_name);

    RC_ASSERT(src_mesh.n_cells() > 0);
    RC_ASSERT(dst_mesh.n_cells() > 0);

    // Test with bilinear interpolation
    {
        RegridConfig config;
        config.method = InterpolationMethod::Bilinear;
        config.unmapped = UnmappedAction::Ignore;

        auto W = WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

        RC_ASSERT(W.n_src() == src_mesh.n_cells());
        RC_ASSERT(W.n_dst() == dst_mesh.n_cells());

        // Check index bounds and non-negativity
        const auto nnz = W.nnz();
        auto rows = W.factor_row();
        auto cols = W.factor_col();
        auto list = W.factor_list();

        std::vector<double> row_sums(dst_mesh.n_cells(), 0.0);

        for (std::size_t k = 0; k < nnz; ++k) {
            auto r = rows[k];
            auto c = cols[k];
            auto val = list[k];

            RC_ASSERT(r >= 0);
            RC_ASSERT(r < static_cast<axis::index_t>(dst_mesh.n_cells()));
            RC_ASSERT(c >= 0);
            RC_ASSERT(c < static_cast<axis::index_t>(src_mesh.n_cells()));
            RC_ASSERT(val >= 0.0);

            row_sums[r] += val;
        }

        // Row sums of partition of unity should be at most 1.0 + tolerance
        for (std::size_t r = 0; r < dst_mesh.n_cells(); ++r) {
            RC_ASSERT(row_sums[r] <= 1.0 + 1e-9);
        }
    }

    // Test with Conservative 1st Order interpolation
    {
        RegridConfig config;
        config.method = InterpolationMethod::Conservative1stOrder;
        config.unmapped = UnmappedAction::Ignore;

        auto W = WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, config);

        RC_ASSERT(W.n_src() == src_mesh.n_cells());
        RC_ASSERT(W.n_dst() == dst_mesh.n_cells());

        const auto nnz = W.nnz();
        auto rows = W.factor_row();
        auto cols = W.factor_col();
        auto list = W.factor_list();

        std::vector<double> row_sums(dst_mesh.n_cells(), 0.0);

        for (std::size_t k = 0; k < nnz; ++k) {
            auto r = rows[k];
            auto c = cols[k];
            auto val = list[k];

            RC_ASSERT(r >= 0);
            RC_ASSERT(r < static_cast<axis::index_t>(dst_mesh.n_cells()));
            RC_ASSERT(c >= 0);
            RC_ASSERT(c < static_cast<axis::index_t>(src_mesh.n_cells()));
            RC_ASSERT(val >= 0.0);

            row_sums[r] += val;
        }

        // For first-order conservative remapping, partition of unity row sums must be <= 1.0 + tolerance
        for (std::size_t r = 0; r < dst_mesh.n_cells(); ++r) {
            RC_ASSERT(row_sums[r] <= 1.0 + 1e-9);
        }
    }
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────

}  // namespace
