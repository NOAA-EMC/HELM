// AXIS unit test: Cell masking for weight generation and apply
// Tests Requirements 8.1-8.6 for source/destination masking.

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>

namespace {

}  // namespace

namespace axis::test {

using MemSpace = Kokkos::HostSpace;

// Helper: build a simple NxN structured grid with optional mask
static topology::UnstructuredMesh<MemSpace> make_uniform_mesh(std::size_t n, double size, Kokkos::View<int *, MemSpace> mask = {}) {
    const double dx = size / static_cast<double>(n);

    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = (static_cast<double>(j) + 0.5) * dx;
        }
    }

    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = static_cast<double>(i) * dx;
            cry(i + j * nc) = static_cast<double>(j) * dx;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    auto mesh = grid.to_unstructured();

    // If a mask was provided, rebuild with mask
    if (mask.extent(0) > 0) {
        return topology::UnstructuredMesh<MemSpace>(mesh.node_coords_view(), mesh.conn_offsets_view(), mesh.conn_indices_view(), mesh.coord_system(),
                                                    mesh.cell_areas_view(), std::move(mask));
    }

    return mesh;
}

// ─────────────────────────────────────────────────────────────────────────────
// Req 8.1: Masked source cells excluded from weight generation
// ─────────────────────────────────────────────────────────────────────────────

TEST(Masking, MaskedSourceCellsExcludedFromWeights) {
    // 3x3 src with cell 4 (center) masked
    const std::size_t n = 3;
    Kokkos::View<int *, MemSpace> src_mask("src_mask", n * n);
    for (std::size_t i = 0; i < n * n; ++i) src_mask(i) = 1;
    src_mask(4) = 0;  // mask center cell

    auto src_mesh = make_uniform_mesh(n, 3.0, src_mask);
    auto dst_mesh = make_uniform_mesh(n, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // Verify: no weight entry references the masked source cell (index 4)
    auto factor_col = matrix.factor_col();
    auto nnz = matrix.nnz();
    for (std::size_t k = 0; k < nnz; ++k) {
        EXPECT_NE(factor_col[k], 4) << "Weight entry k=" << k << " references masked source cell 4";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Req 8.2: Masked destination cells produce no entries
// ─────────────────────────────────────────────────────────────────────────────

TEST(Masking, MaskedDestinationCellsProduceNoEntries) {
    const std::size_t n = 3;
    Kokkos::View<int *, MemSpace> dst_mask("dst_mask", n * n);
    for (std::size_t i = 0; i < n * n; ++i) dst_mask(i) = 1;
    dst_mask(0) = 0;  // mask first destination cell
    dst_mask(8) = 0;  // mask last destination cell

    auto src_mesh = make_uniform_mesh(n, 3.0);
    auto dst_mesh = make_uniform_mesh(n, 3.0, dst_mask);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // Verify: no entry with row == 0 or row == 8
    auto factor_row = matrix.factor_row();
    auto nnz = matrix.nnz();
    for (std::size_t k = 0; k < nnz; ++k) {
        EXPECT_NE(factor_row[k], 0) << "Weight entry k=" << k << " targets masked dst cell 0";
        EXPECT_NE(factor_row[k], 8) << "Weight entry k=" << k << " targets masked dst cell 8";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Req 8.3: Coverage fraction reflects partial masking
// ─────────────────────────────────────────────────────────────────────────────

TEST(Masking, CoverageFractionReflectsPartialMasking) {
    // 2x2 src mesh, mask one of the cells; 1x1 dst mesh covering the whole domain.
    // With one of 4 src cells masked, frac_b for the single dst should < 1.
    const std::size_t n_src = 2;
    const std::size_t n_dst = 1;

    Kokkos::View<int *, MemSpace> src_mask("src_mask", n_src * n_src);
    for (std::size_t i = 0; i < n_src * n_src; ++i) src_mask(i) = 1;
    src_mask(0) = 0;  // mask one corner cell

    auto src_mesh = make_uniform_mesh(n_src, 2.0, src_mask);
    auto dst_mesh = make_uniform_mesh(n_dst, 2.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    // frac_b for the single destination cell should be ~0.75 (3 of 4 src cells)
    auto frac_b = matrix.frac_b();
    ASSERT_GE(frac_b.extent(0), static_cast<std::size_t>(1));
    EXPECT_NEAR(frac_b[0], 0.75, 0.01) << "frac_b should reflect partial coverage (~0.75 with 1 of 4 cells masked)";
}

// ─────────────────────────────────────────────────────────────────────────────
// Req 8.4: apply skips masked destination cells
// ─────────────────────────────────────────────────────────────────────────────

TEST(Masking, ApplySkipsMaskedDestinationCells) {
    auto src_mesh = make_uniform_mesh(3, 3.0);
    auto dst_mesh = make_uniform_mesh(3, 3.0);

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Ignore;

    auto matrix = solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg);

    const std::size_t n_src = matrix.n_src();
    const std::size_t n_dst = matrix.n_dst();

    // Source field = 5.0 everywhere
    std::vector<double> src_data(n_src, 5.0);
    // Destination field initialized to sentinel -999.0
    std::vector<double> dst_data(n_dst, -999.0);

    // Mask: cells 0 and 4 are masked (inactive)
    std::vector<int> mask_data(n_dst, 1);
    mask_data[0] = 0;
    mask_data[4] = 0;

    field_view<const double, 1> src_view(src_data.data(), n_src);
    field_view<double, 1> dst_view(dst_data.data(), n_dst);
    field_view<const int, 1> mask_view(mask_data.data(), n_dst);

    solver::apply<MemSpace>(matrix, src_view, dst_view, mask_view);

    // Masked cells should retain sentinel value
    EXPECT_DOUBLE_EQ(dst_data[0], -999.0);
    EXPECT_DOUBLE_EQ(dst_data[4], -999.0);

    // Unmasked cells should have interpolated values (approx 5.0 for identity grids)
    for (std::size_t j = 0; j < n_dst; ++j) {
        if (mask_data[j] != 0) {
            EXPECT_NE(dst_data[j], -999.0) << "Unmasked cell " << j << " was not written";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Req 8.5: Mask stored as View<int*, MemorySpace> with 0=masked, nonzero=active
// ─────────────────────────────────────────────────────────────────────────────

TEST(Masking, MaskAccessorsWork) {
    const std::size_t n = 3;
    Kokkos::View<int *, MemSpace> mask("mask", n * n);
    for (std::size_t i = 0; i < n * n; ++i) mask(i) = 1;
    mask(0) = 0;

    auto mesh = make_uniform_mesh(n, 3.0, mask);

    auto mask_view = mesh.cell_mask();
    ASSERT_EQ(mask_view.extent(0), n * n);
    EXPECT_EQ(mask_view[0], 0);
    EXPECT_EQ(mask_view[1], 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Req 8.6: Throw for unmasked dst cells with zero coverage when unmapped==Error
// ─────────────────────────────────────────────────────────────────────────────

TEST(Masking, ThrowForUnmaskedDstWithZeroCoverageWhenError) {
    // Scenario: src mesh does NOT overlap dst mesh at all, but both have masks.
    // The masked dst cells should NOT trigger an error — only unmasked ones.
    // We create non-overlapping meshes to ensure zero coverage.

    const std::size_t n = 2;
    const double dx = 1.0;

    // Source mesh at [0,2]x[0,2]
    auto src_mesh = make_uniform_mesh(n, 2.0);

    // Destination mesh at [10,12]x[10,12] (no overlap with source)
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = 10.0 + (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = 10.0 + (static_cast<double>(j) + 0.5) * dx;
        }
    }

    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = 10.0 + static_cast<double>(i) * dx;
            cry(i + j * nc) = 10.0 + static_cast<double>(j) * dx;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    // Destination mask: mask cells 0 and 1, leave 2 and 3 unmasked
    Kokkos::View<int *, MemSpace> dst_mask("dst_mask", n * n);
    dst_mask(0) = 0;  // masked — should NOT trigger error
    dst_mask(1) = 0;  // masked — should NOT trigger error
    dst_mask(2) = 1;  // unmasked — WILL have zero coverage
    dst_mask(3) = 1;  // unmasked — WILL have zero coverage

    auto dst_umesh = grid.to_unstructured();
    auto dst_mesh = topology::UnstructuredMesh<MemSpace>(dst_umesh.node_coords_view(), dst_umesh.conn_offsets_view(), dst_umesh.conn_indices_view(),
                                                         dst_umesh.coord_system(), dst_umesh.cell_areas_view(), std::move(dst_mask));

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Error;

    // Should throw because unmasked dst cells (2, 3) have zero coverage
    EXPECT_THROW(solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg), std::runtime_error);
}

TEST(Masking, NoThrowForMaskedDstWithZeroCoverageWhenError) {
    // All dst cells are masked — no error should be thrown even with unmapped==Error
    const std::size_t n = 2;
    const double dx = 1.0;

    auto src_mesh = make_uniform_mesh(n, 2.0);

    // Non-overlapping dst mesh
    Kokkos::View<double *, MemSpace> cx("cx", n * n);
    Kokkos::View<double *, MemSpace> cy("cy", n * n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            cx(i + j * n) = 10.0 + (static_cast<double>(i) + 0.5) * dx;
            cy(i + j * n) = 10.0 + (static_cast<double>(j) + 0.5) * dx;
        }
    }

    topology::StructuredGrid<MemSpace> grid(n, n, std::move(cx), std::move(cy), topology::CoordinateSystem::Cartesian3D);

    const std::size_t nc = n + 1;
    Kokkos::View<double *, MemSpace> crx("crx", nc * nc);
    Kokkos::View<double *, MemSpace> cry("cry", nc * nc);
    for (std::size_t j = 0; j <= n; ++j) {
        for (std::size_t i = 0; i <= n; ++i) {
            crx(i + j * nc) = 10.0 + static_cast<double>(i) * dx;
            cry(i + j * nc) = 10.0 + static_cast<double>(j) * dx;
        }
    }
    grid.set_corners(std::move(crx), std::move(cry));

    // All dst cells masked
    Kokkos::View<int *, MemSpace> dst_mask("dst_mask", n * n);
    Kokkos::deep_copy(dst_mask, 0);  // all masked

    auto dst_umesh = grid.to_unstructured();
    auto dst_mesh = topology::UnstructuredMesh<MemSpace>(dst_umesh.node_coords_view(), dst_umesh.conn_offsets_view(), dst_umesh.conn_indices_view(),
                                                         dst_umesh.coord_system(), dst_umesh.cell_areas_view(), std::move(dst_mask));

    solver::RegridConfig cfg;
    cfg.method = solver::InterpolationMethod::Conservative1stOrder;
    cfg.norm_type = solver::NormType::DstArea;
    cfg.line_type = solver::LineType::Cartesian;
    cfg.unmapped = solver::UnmappedAction::Error;

    // Should NOT throw because all dst cells are masked
    EXPECT_NO_THROW(solver::WeightGenerator::generate<MemSpace>(src_mesh, dst_mesh, cfg));
}

}  // namespace axis::test
