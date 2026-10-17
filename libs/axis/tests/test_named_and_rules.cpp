// AXIS unit test: named-grid and GridRules generation
// Verifies NamedGridRegistry::generate produces non-empty meshes and
// RuleGenerator produces meshes with the correct cell count.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/rule_generator.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cmath>

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
static auto *const kenv = ::testing::AddGlobalTestEnvironment(new KokkosEnv);
}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Test: generate("O4") produces a non-empty mesh with expected structure
TEST(NamedAndRules, O4GeneratesNonEmptyMesh) {
    auto mesh = topology::NamedGridRegistry::generate<MemSpace>("O4");

    EXPECT_GT(mesh.n_nodes(), std::size_t(0));
    EXPECT_GT(mesh.n_cells(), std::size_t(0));

    // CSR offsets should have n_cells + 1 entries
    auto offsets = mesh.conn_offsets();
    EXPECT_EQ(offsets.extent(0), mesh.n_cells() + 1);
}

TEST(NamedAndRules, O4UsesGaussianLatitudeStripsAndRaggedCellRows) {
    auto mesh = topology::NamedGridRegistry::generate<MemSpace>("O4");

    // The octahedral rows contain 20, 24, 28, 32, 32, 28, 24, and 20 cells.
    EXPECT_EQ(mesh.n_cells(), std::size_t(208));
    const auto offsets = mesh.conn_offsets();
    const auto indices = mesh.conn_indices();
    bool has_non_quad = false;
    for (std::size_t cell = 0; cell < mesh.n_cells(); ++cell) {
        const auto arity = offsets(cell + 1) - offsets(cell);
        EXPECT_GE(arity, axis::index_t(4));
        has_non_quad = has_non_quad || arity != axis::index_t(4);
        for (axis::index_t entry = offsets(cell); entry < offsets(cell + 1); ++entry) {
            EXPECT_GE(indices(static_cast<std::size_t>(entry)), axis::index_t(0));
            EXPECT_LT(indices(static_cast<std::size_t>(entry)), static_cast<axis::index_t>(mesh.n_nodes()));
        }
    }
    EXPECT_TRUE(has_non_quad);  // conforming edge splits at row transitions

    const auto areas = mesh.cell_areas();
    double total_area = 0.0;
    for (std::size_t cell = 0; cell < mesh.n_cells(); ++cell) {
        EXPECT_GT(areas(cell), 0.0);
        total_area += areas(cell);
    }
    EXPECT_NEAR(total_area, 4.0 * M_PI, 1e-10);
}

// Test: generate("F4") produces a non-empty mesh
TEST(NamedAndRules, F4GeneratesNonEmptyMesh) {
    auto mesh = topology::NamedGridRegistry::generate<MemSpace>("F4");

    constexpr std::size_t N = 4;
    EXPECT_EQ(mesh.n_nodes(), std::size_t((4 * N + 1) * (2 * N + 1)));  // shared corner lattice
    EXPECT_GT(mesh.n_cells(), std::size_t(0));

    // Verify expected range
    auto coords = mesh.node_coords();
    double min_x = 999.0, max_x = -999.0;
    double min_y = 999.0, max_y = -999.0;
    for (std::size_t i = 0; i < mesh.n_nodes(); ++i) {
        double x = coords(i, 0);
        double y = coords(i, 1);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    EXPECT_NEAR(min_x, 0.0, 1e-11);
    EXPECT_NEAR(max_x, 360.0, 1e-11);
    EXPECT_NEAR(min_y, -90.0, 1e-11);
    EXPECT_NEAR(max_y, 90.0, 1e-11);
}

// Test: parse("O4") returns family='O', number=4
TEST(NamedAndRules, ParseO4ReturnsCorrectParsedName) {
    auto parsed = topology::NamedGridRegistry::parse("O4");
    EXPECT_EQ(parsed.family, 'O');
    EXPECT_EQ(parsed.number, 4);
}

// Test: parse with invalid family throws
TEST(NamedAndRules, ParseInvalidFamilyThrows) {
    EXPECT_THROW((void)topology::NamedGridRegistry::parse("Z100"), std::invalid_argument);
}

// Test: RuleGenerator RegularLatLon produces correct cell count
TEST(NamedAndRules, RegularLatLonCellCount) {
    ingest::GridRulesParams rules;
    rules.kind = "RegularLatLon";
    rules.min_x = 0.0;
    rules.max_x = 10.0;
    rules.min_y = 0.0;
    rules.max_y = 6.0;
    rules.r_x = 2.0;
    rules.r_y = 3.0;

    auto mesh = topology::RuleGenerator::generate<MemSpace>(rules);

    // Expected: floor((10-0)/2) * floor((6-0)/3) = 5 * 2 = 10 cells
    const std::size_t expected_cells = static_cast<std::size_t>(std::floor((rules.max_x - rules.min_x) / rules.r_x)) *
                                       static_cast<std::size_t>(std::floor((rules.max_y - rules.min_y) / rules.r_y));

    EXPECT_EQ(mesh.n_cells(), expected_cells);
    EXPECT_EQ(mesh.n_cells(), std::size_t(10));
}

// Test: RuleGenerator with zero resolution throws
TEST(NamedAndRules, ZeroResolutionThrows) {
    ingest::GridRulesParams rules;
    rules.kind = "RegularLatLon";
    rules.min_x = 0.0;
    rules.max_x = 10.0;
    rules.min_y = 0.0;
    rules.max_y = 10.0;
    rules.r_x = 0.0;  // invalid
    rules.r_y = 1.0;

    EXPECT_THROW(topology::RuleGenerator::generate<MemSpace>(rules), std::invalid_argument);
}

// Test: generate("R4") produces a rectilinear normal lat-lon grid with 16x8 cells
TEST(NamedAndRules, R4GeneratesCorrectMesh) {
    auto mesh = topology::NamedGridRegistry::generate<MemSpace>("R4");

    constexpr std::size_t N = 4;
    EXPECT_EQ(mesh.n_nodes(), std::size_t(4 * N * 2 * N));  // used as cell centers
    EXPECT_GT(mesh.n_cells(), std::size_t(0));

    // Verify coordinate range spans [-180, 180] in x and [-90, 90] in y
    // but note these are the cell centers, not edges
    auto coords = mesh.node_coords();
    double min_x = 999.0, max_x = -999.0;
    double min_y = 999.0, max_y = -999.0;
    for (std::size_t i = 0; i < mesh.n_nodes(); ++i) {
        double x = coords(i, 0);
        double y = coords(i, 1);
        if (x < min_x) min_x = x;
        if (x > max_x) max_x = x;
        if (y < min_y) min_y = y;
        if (y > max_y) max_y = y;
    }
    EXPECT_NEAR(min_x, -180.0, 1e-11);
    EXPECT_NEAR(max_x, 180.0 - 22.5, 1e-11);
    EXPECT_NEAR(min_y, -90.0 + 11.25, 1e-11);
    EXPECT_NEAR(max_y, 90.0 - 11.25, 1e-11);
}

}  // namespace axis::test
