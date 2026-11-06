// ─── Property-Based Tests: Vertical Regridder Spline Correctness ──────────────────
// Feature: helm-axis-microlibrary, Property: Vertical Regridder Spline Correctness
//
// Uses RapidCheck to verify that VerticalRegridder preserves exact linear
// profiles and performs consistent 1D vs 2D level coordinate interpolation.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <cmath>
#include <vector>

namespace {

using MemSpace = Kokkos::HostSpace;
using namespace axis::solver;

/// Generate valid number of columns [1, 20]
rc::Gen<std::size_t> genColumns() {
    return rc::gen::inRange<std::size_t>(1, 21);
}

/// Generate valid number of source levels [3, 50]
rc::Gen<std::size_t> genSourceLevels() {
    return rc::gen::inRange<std::size_t>(3, 51);
}

/// Generate valid number of destination levels [2, 50]
rc::Gen<std::size_t> genDestLevels() {
    return rc::gen::inRange<std::size_t>(2, 51);
}

/// Generate non-negative tension parameter by scaling an integer
rc::Gen<double> genTension() {
    return rc::gen::map(rc::gen::inRange(0, 100), [](int v) { return static_cast<double>(v) / 10.0; });
}

/// Helper to generate a double in range by scaling an integer
rc::Gen<double> genDoubleInRange(int min_val, int max_val, double scale = 1.0) {
    return rc::gen::map(rc::gen::inRange(min_val, max_val), [scale](int v) { return static_cast<double>(v) * scale; });
}

// ─── Property: Linear Field Reproduction ────────────────────────────────────
// A cubic spline (tension = 0.0) must reproduce any exact linear profile
// perfectly (to high precision) over its levels.
RC_GTEST_PROP(PropVerticalRegridder, LinearFieldReproduction, ()) {
    const std::size_t n_col = *genColumns();
    const std::size_t n_src = *genSourceLevels();
    const std::size_t n_dst = *genDestLevels();
    const double tension = *genTension();

    // Generate linear profile parameters: y = m * x + c
    const double slope = *genDoubleInRange(-10, 10, 1.0);
    const double intercept = *genDoubleInRange(-100, 100, 1.0);

    Kokkos::View<double **, MemSpace> src_field("src_field", n_col, n_src);
    Kokkos::View<double **, MemSpace> dst_field("dst_field", n_col, n_dst);
    Kokkos::View<double *, MemSpace> src_levels("src_levels", n_src);
    Kokkos::View<double *, MemSpace> dst_levels("dst_levels", n_dst);

    for (std::size_t i = 0; i < n_src; ++i) {
        src_levels(i) = static_cast<double>(i);
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        dst_levels(j) = (static_cast<double>(j) + 0.37) * (n_src - 1) / n_dst;
    }

    // Fill source field with exact linear profile
    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t i = 0; i < n_src; ++i) {
            src_field(c, i) = slope * src_levels(i) + intercept;
        }
    }

    // Sample between knots, including positive tension.
    VerticalRegridder<MemSpace>::interpolate(src_field, dst_field, src_levels, dst_levels, tension);

    // Verify destination field holds exact linear profile
    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t j = 0; j < n_dst; ++j) {
            double expected = slope * dst_levels(j) + intercept;
            RC_ASSERT(std::abs(dst_field(c, j) - expected) < 1e-12);
        }
    }
}

// ─── Property: 1D vs 2D Consistency ──────────────────────────────────────────
// Uniform 1D levels and 2D level views with identical values must produce identical interpolation results.
RC_GTEST_PROP(PropVerticalRegridder, OneDAndTwoDConsistency, ()) {
    const std::size_t n_col = *genColumns();
    const std::size_t n_src = *genSourceLevels();
    const std::size_t n_dst = *genDestLevels();
    const double tension = *genTension();

    Kokkos::View<double **, MemSpace> src_field("src_field", n_col, n_src);
    Kokkos::View<double **, MemSpace> dst_field_1d("dst_field_1d", n_col, n_dst);
    Kokkos::View<double **, MemSpace> dst_field_2d("dst_field_2d", n_col, n_dst);

    Kokkos::View<double *, MemSpace> src_levels_1d("src_levels_1d", n_src);
    Kokkos::View<double *, MemSpace> dst_levels_1d("dst_levels_1d", n_dst);

    Kokkos::View<double **, MemSpace> src_levels_2d("src_levels_2d", n_col, n_src);
    Kokkos::View<double **, MemSpace> dst_levels_2d("dst_levels_2d", n_col, n_dst);

    // Populate strictly increasing source levels
    double current_level = 0.0;
    for (std::size_t i = 0; i < n_src; ++i) {
        src_levels_1d(i) = current_level;
        current_level += *genDoubleInRange(10, 50, 0.1);
    }

    // Populate destination levels within bounds
    for (std::size_t j = 0; j < n_dst; ++j) {
        dst_levels_1d(j) =
            src_levels_1d(0) + (src_levels_1d(n_src - 1) - src_levels_1d(0)) * (static_cast<double>(j) / static_cast<double>(n_dst - 1));
    }

    // Copy to 2D levels and initialize random source field
    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t i = 0; i < n_src; ++i) {
            src_levels_2d(c, i) = src_levels_1d(i);
            src_field(c, i) = *genDoubleInRange(-1000, 1000, 0.1);
        }
        for (std::size_t j = 0; j < n_dst; ++j) {
            dst_levels_2d(c, j) = dst_levels_1d(j);
        }
    }

    // Run 1D uniform levels interpolation
    VerticalRegridder<MemSpace>::interpolate(src_field, dst_field_1d, src_levels_1d, dst_levels_1d, tension);

    // Run 2D varying levels interpolation
    VerticalRegridder<MemSpace>::interpolate(src_field, dst_field_2d, src_levels_2d, dst_levels_2d, tension);

    // Verify outputs are identical bitwise
    for (std::size_t c = 0; c < n_col; ++c) {
        for (std::size_t j = 0; j < n_dst; ++j) {
            RC_ASSERT(dst_field_1d(c, j) == dst_field_2d(c, j));
        }
    }
}

// ─── Property: Mismatched Input Validation ───────────────────────────────────
// Verify that standard exception handling (std::invalid_argument) occurs on mismatched input sizes.
TEST(PropVerticalRegridder, ExceptionValidation) {
    // 1. Column extent mismatch
    {
        Kokkos::View<double **, MemSpace> src_field("src", 5, 10);
        Kokkos::View<double **, MemSpace> dst_field("dst", 4, 10);
        Kokkos::View<double *, MemSpace> src_levels("src_l", 10);
        Kokkos::View<double *, MemSpace> dst_levels("dst_l", 10);
        EXPECT_THROW(VerticalRegridder<MemSpace>::interpolate(src_field, dst_field, src_levels, dst_levels), std::invalid_argument);
    }

    // 2. Source levels size mismatch
    {
        Kokkos::View<double **, MemSpace> src_field("src", 5, 10);
        Kokkos::View<double **, MemSpace> dst_field("dst", 5, 10);
        Kokkos::View<double *, MemSpace> src_levels("src_l", 9);
        Kokkos::View<double *, MemSpace> dst_levels("dst_l", 10);
        EXPECT_THROW(VerticalRegridder<MemSpace>::interpolate(src_field, dst_field, src_levels, dst_levels), std::invalid_argument);
    }

    // 3. Negative tension mismatch
    {
        Kokkos::View<double **, MemSpace> src_field("src", 5, 10);
        Kokkos::View<double **, MemSpace> dst_field("dst", 5, 10);
        Kokkos::View<double *, MemSpace> src_levels("src_l", 10);
        Kokkos::View<double *, MemSpace> dst_levels("dst_l", 10);
        EXPECT_THROW(VerticalRegridder<MemSpace>::interpolate(src_field, dst_field, src_levels, dst_levels, -1.0), std::invalid_argument);
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
