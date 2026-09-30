// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors
//
// Unit tests for the Phase-2 engine entry points added for the Python API
// redesign: explicit periodicity override, src/dst masking with the
// unmapped-mask post-pass, C<N> cubed-sphere named grids, float32 apply,
// and per-cell vector rotation angles.

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/solver/weight_cache.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <random>
#include <vector>

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
using namespace axis::solver;

namespace {

// Helper: global-ish regular lat-lon mesh (cell-centered), mirroring the
// Python binding's make_regular_mesh layout.
topology::UnstructuredMesh<MemSpace> make_latlon_mesh(std::size_t ni, std::size_t nj, double lon_start, double lat_start, double dlon,
                                                     double dlat) {
    Kokkos::View<double *, MemSpace> cx("cx", ni * nj);
    Kokkos::View<double *, MemSpace> cy("cy", ni * nj);
    Kokkos::View<double *, MemSpace> crx("crx", (ni + 1) * (nj + 1));
    Kokkos::View<double *, MemSpace> cry("cry", (ni + 1) * (nj + 1));
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_start + (static_cast<double>(i) + 0.5) * dlon;
            cy(i + j * ni) = lat_start + (static_cast<double>(j) + 0.5) * dlat;
        }
    }
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * (ni + 1)) = lon_start + static_cast<double>(i) * dlon;
            cry(i + j * (ni + 1)) = lat_start + static_cast<double>(j) * dlat;
        }
    }
    topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(cx), std::move(cy), topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(std::move(crx), std::move(cry));
    return grid.to_unstructured();
}

// Helper: single-point "mesh" of n cells placed at explicit lon/lat.
// (Nearest-neighbor style probing of coverage via bilinear on tiny quads is
// avoided; instead we build a regular fine grid and read selected rows.)

// Sum of weights in matrix row r.
double row_weight_sum(const InterpolationMatrix<MemSpace> &m, std::size_t r) {
    auto rows = Kokkos::create_mirror_view_and_copy(MemSpace{}, m.factor_row_view());
    auto vals = Kokkos::create_mirror_view_and_copy(MemSpace{}, m.factor_list_view());
    double sum = 0.0;
    for (std::size_t k = 0; k < m.nnz(); ++k) {
        if (static_cast<std::size_t>(rows(k)) == r) sum += vals(k);
    }
    return sum;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Periodicity override (RegridConfig::periodic, tri-state)
// ─────────────────────────────────────────────────────────────────────────────

TEST(PeriodicityOverride, AutoDetectWrapsSeamOnFullGrid) {
    // Source spans exactly 360° → auto-detected periodic. A destination just
    // west of the dateline must interpolate across the seam (partition of unity).
    auto src = make_latlon_mesh(20, 10, 0.0, -85.0, 18.0, 20.0);
    auto dst = make_latlon_mesh(4, 2, 3.0, -80.0, 90.0, 80.0);

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    auto W = WeightGenerator::generate<MemSpace>(src, dst, cfg);

    // Every dst row of a global→global regrid must be fully covered.
    for (std::size_t r = 0; r < W.n_dst(); ++r) {
        EXPECT_NEAR(row_weight_sum(W, r), 1.0, 1e-12) << "row " << r;
    }
}

TEST(PeriodicityOverride, ForceNonPeriodicBreaksSeamCoverage) {
    // A destination cell just east of lon 0 needs a source stencil point just
    // west of lon 0. With wrapping on it resolves across the seam (the last
    // source column i = ni-1); with wrapping off the stencil clamps to the
    // first column i = 0 instead.
    auto src = make_latlon_mesh(20, 10, 0.0, -85.0, 18.0, 20.0);
    auto dst = make_latlon_mesh(4, 2, -40.0, -80.0, 90.0, 80.0);  // centers at lon 5, 95, 185, 275

    RegridConfig cfg_on;
    cfg_on.method = InterpolationMethod::Bilinear;
    auto W_on = WeightGenerator::generate<MemSpace>(src, dst, cfg_on);

    RegridConfig cfg_off;
    cfg_off.method = InterpolationMethod::Bilinear;
    cfg_off.periodic = -1;  // force off
    auto W_off = WeightGenerator::generate<MemSpace>(src, dst, cfg_off);

    const std::size_t ni = 20;
    const std::size_t row1 = ni;  // j = 1
    // Delta at the wrapped (last-column) source cell must reach dst row 0 only
    // when periodicity is active.
    auto probe = [](const InterpolationMatrix<MemSpace> &W, std::size_t cell) {
        Kokkos::View<double *, MemSpace> f("f", W.n_src());
        auto hf = Kokkos::create_mirror_view(f);
        Kokkos::deep_copy(hf, 0.0);
        hf(cell) = 1.0;
        Kokkos::deep_copy(f, hf);
        Kokkos::View<double *, MemSpace> o("o", W.n_dst());
        apply(W, field_view<const double, 1>(f.data(), f.extent(0)), field_view<double, 1>(o.data(), o.extent(0)));
        return Kokkos::create_mirror_view_and_copy(MemSpace{}, o)(0);
    };

    EXPECT_GT(probe(W_on, row1 + ni - 1), 0.0) << "periodic: seam row should draw on last source column";
    EXPECT_DOUBLE_EQ(probe(W_off, row1 + ni - 1), 0.0) << "non-periodic: last column must not reach row 0";
    EXPECT_GT(probe(W_off, row1), 0.0) << "non-periodic: clamps to first source column instead";
    EXPECT_DOUBLE_EQ(probe(W_on, row1), 0.0) << "periodic: first column is not the seam neighbor";
}

TEST(PeriodicityOverride, ForcePeriodicMatchesAutoOnFullGrid) {
    auto src = make_latlon_mesh(20, 10, 0.0, -85.0, 18.0, 20.0);
    auto dst = make_latlon_mesh(4, 2, 3.0, -80.0, 90.0, 80.0);

    RegridConfig auto_cfg;
    auto_cfg.method = InterpolationMethod::Bilinear;
    auto W_auto = WeightGenerator::generate<MemSpace>(src, dst, auto_cfg);

    RegridConfig force_cfg = auto_cfg;
    force_cfg.periodic = 1;  // force on (same as auto here)
    auto W_force = WeightGenerator::generate<MemSpace>(src, dst, force_cfg);

    ASSERT_EQ(W_auto.nnz(), W_force.nnz());
    auto ra = Kokkos::create_mirror_view_and_copy(MemSpace{}, W_auto.factor_list_view());
    auto rb = Kokkos::create_mirror_view_and_copy(MemSpace{}, W_force.factor_list_view());
    for (std::size_t k = 0; k < W_auto.nnz(); ++k) {
        EXPECT_DOUBLE_EQ(ra(k), rb(k));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Source / destination masking + unmapped mask
// ─────────────────────────────────────────────────────────────────────────────

TEST(MaskPostPass, SourceMaskProducesWetRenormalizedRows) {
    // 6x4 src with a dry block; dst bilinear over it. Wet-renormalized rows
    // adjacent to dry cells must still sum to 1 (partition of unity restored).
    auto src = make_latlon_mesh(6, 4, 5.0, -75.0, 20.0, 25.0);
    auto dst = make_latlon_mesh(3, 2, 15.0, -62.5, 40.0, 50.0);

    Kokkos::View<int *, MemSpace> cell_mask("mask", src.n_cells());
    auto hm = Kokkos::create_mirror_view(cell_mask);
    for (std::size_t c = 0; c < src.n_cells(); ++c) hm(c) = 1;
    hm(8) = 0;  // one dry interior cell
    Kokkos::deep_copy(cell_mask, hm);

    auto src_masked = topology::UnstructuredMesh<MemSpace>(src.node_coords_view(), src.conn_offsets_view(), src.conn_indices_view(),
                                                           src.coord_system(), src.cell_areas_view(), std::move(cell_mask));

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    auto W = WeightGenerator::generate<MemSpace>(src_masked, dst, cfg);

    for (std::size_t r = 0; r < W.n_dst(); ++r) {
        EXPECT_NEAR(row_weight_sum(W, r), 1.0, 1e-10) << "dst row " << r << " not renormalized over wet sources";
    }
    // The dry source column must carry zero weight everywhere.
    auto rows = Kokkos::create_mirror_view_and_copy(MemSpace{}, W.factor_row_view());
    auto cols = Kokkos::create_mirror_view_and_copy(MemSpace{}, W.factor_col_view());
    auto vals = Kokkos::create_mirror_view_and_copy(MemSpace{}, W.factor_list_view());
    for (std::size_t k = 0; k < W.nnz(); ++k) {
        if (static_cast<std::size_t>(cols(k)) == 8) EXPECT_DOUBLE_EQ(vals(k), 0.0);
    }
}

TEST(MaskPostPass, DestinationMaskDropsRowsAndMarksUnmapped) {
    auto src = make_latlon_mesh(6, 4, 5.0, -75.0, 20.0, 25.0);
    auto dst = make_latlon_mesh(3, 2, 15.0, -62.5, 40.0, 50.0);

    std::vector<int> h_dst_mask(dst.n_cells(), 1);
    h_dst_mask[0] = 0;
    h_dst_mask[4] = 0;
    Kokkos::View<int *, MemSpace> dst_mask("dm", dst.n_cells());
    auto hdm = Kokkos::create_mirror_view(dst_mask);
    for (std::size_t c = 0; c < dst.n_cells(); ++c) hdm(c) = h_dst_mask[c];
    Kokkos::deep_copy(dst_mask, hdm);

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    cfg.unmapped = UnmappedAction::Mask;
    cfg.dst_mask = field_view<const int, 1>{dst_mask.data(), dst_mask.extent(0)};
    auto W = WeightGenerator::generate<MemSpace>(src, dst, cfg);

    ASSERT_TRUE(W.has_unmapped_mask());
    auto um = Kokkos::create_mirror_view_and_copy(MemSpace{}, W.unmapped_mask_view());
    EXPECT_EQ(um(0), 1);
    EXPECT_EQ(um(4), 1);
    EXPECT_EQ(um(1), 0);
    // Masked-out rows carry no weights.
    EXPECT_NEAR(row_weight_sum(W, 0), 0.0, 1e-15);
    EXPECT_NEAR(row_weight_sum(W, 4), 0.0, 1e-15);
    // Unmasked rows still form a partition of unity.
    EXPECT_NEAR(row_weight_sum(W, 1), 1.0, 1e-12);
}

TEST(MaskPostPass, UnmappedMaskRoundTripsThroughWeightCache) {
    auto src = make_latlon_mesh(6, 4, 5.0, -75.0, 20.0, 25.0);
    auto dst = make_latlon_mesh(3, 2, 15.0, -62.5, 40.0, 50.0);

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    cfg.unmapped = UnmappedAction::Mask;
    auto W = WeightGenerator::generate<MemSpace>(src, dst, cfg);
    ASSERT_TRUE(W.has_unmapped_mask());

    const std::size_t size = WeightCache::serialize(W, nullptr, 0);
    std::vector<std::uint8_t> blob(size);
    ASSERT_EQ(WeightCache::serialize(W, blob.data(), blob.size()), size);
    auto restored = WeightCache::deserialize<MemSpace>(blob.data(), blob.size());

    ASSERT_TRUE(restored.has_unmapped_mask());
    auto a = Kokkos::create_mirror_view_and_copy(MemSpace{}, W.unmapped_mask_view());
    auto b = Kokkos::create_mirror_view_and_copy(MemSpace{}, restored.unmapped_mask_view());
    ASSERT_EQ(a.extent(0), b.extent(0));
    bool any_difference = false;
    for (std::size_t r = 0; r < a.extent(0); ++r) {
        if (a(r) != b(r)) any_difference = true;
    }
    EXPECT_FALSE(any_difference);
}

// ─────────────────────────────────────────────────────────────────────────────
// C<N> cubed-sphere named grids
// ─────────────────────────────────────────────────────────────────────────────

TEST(CubedSphereNamedGrid, RegistrationAndParsing) {
    EXPECT_TRUE(topology::NamedGridRegistry::is_registered("C6"));
    EXPECT_TRUE(topology::NamedGridRegistry::is_registered("c12"));  // case-insensitive
    EXPECT_TRUE(topology::NamedGridRegistry::is_registered("C96"));
    EXPECT_FALSE(topology::NamedGridRegistry::is_registered("C0"));
    EXPECT_FALSE(topology::NamedGridRegistry::is_registered("C"));
    EXPECT_FALSE(topology::NamedGridRegistry::is_registered("C-3"));
}

TEST(CubedSphereNamedGrid, NodeAndCellCounts) {
    for (int N : {2, 3, 6, 12}) {
        auto mesh = topology::NamedGridRegistry::generate<MemSpace>("C" + std::to_string(N));
        EXPECT_EQ(mesh.n_cells(), 6u * N * N) << "N=" << N;
        EXPECT_EQ(mesh.n_nodes(), 6u * N * N + 2u) << "N=" << N;
    }
}

TEST(CubedSphereNamedGrid, Deterministic) {
    auto a = topology::NamedGridRegistry::generate<MemSpace>("C4");
    auto b = topology::NamedGridRegistry::generate<MemSpace>("C4");
    auto ca = Kokkos::create_mirror_view_and_copy(MemSpace{}, a.node_coords_view());
    auto cb = Kokkos::create_mirror_view_and_copy(MemSpace{}, b.node_coords_view());
    ASSERT_EQ(ca.extent(0), cb.extent(0));
    for (std::size_t n = 0; n < ca.extent(0); ++n) {
        for (std::size_t d = 0; d < ca.extent(1); ++d) {
            EXPECT_DOUBLE_EQ(ca(n, d), cb(n, d));
        }
    }
}

TEST(CubedSphereNamedGrid, CoversSphere) {
    // Node latitudes must reach both poles; longitudes must span the globe.
    auto mesh = topology::NamedGridRegistry::generate<MemSpace>("C8");
    auto coords = Kokkos::create_mirror_view_and_copy(MemSpace{}, mesh.node_coords_view());
    double min_lat = 90.0, max_lat = -90.0;
    for (std::size_t n = 0; n < coords.extent(0); ++n) {
        const double lat = coords(n, 1);
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
    }
    EXPECT_NEAR(max_lat, 90.0, 1e-10);
    EXPECT_NEAR(min_lat, -90.0, 1e-10);
}

// ─────────────────────────────────────────────────────────────────────────────
// float32 apply (FR-018)
// ─────────────────────────────────────────────────────────────────────────────

TEST(Float32Apply, MatchesDoubleApply) {
    auto src = make_latlon_mesh(12, 6, 0.0, -80.0, 30.0, 30.0);
    auto dst = make_latlon_mesh(5, 3, 15.0, -65.0, 72.0, 50.0);

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    auto W = WeightGenerator::generate<MemSpace>(src, dst, cfg);
    W.to_csr();  // float32 path requires CSR

    const std::size_t n_src = W.n_src();
    const std::size_t n_dst = W.n_dst();

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    Kokkos::View<double *, MemSpace> src64("s64", n_src);
    Kokkos::View<double *, MemSpace> dst64("d64", n_dst);
    Kokkos::View<float *, MemSpace> src32("s32", n_src);
    Kokkos::View<float *, MemSpace> dst32("d32", n_dst);
    auto hs = Kokkos::create_mirror_view(src64);
    for (std::size_t i = 0; i < n_src; ++i) hs(i) = dist(rng);
    Kokkos::deep_copy(src64, hs);
    auto hf = Kokkos::create_mirror_view(src32);
    for (std::size_t i = 0; i < n_src; ++i) hf(i) = static_cast<float>(hs(i));
    Kokkos::deep_copy(src32, hf);

    apply(W, field_view<const double, 1>{src64.data(), n_src}, field_view<double, 1>{dst64.data(), n_dst});
    apply(W, field_view<const float, 1>{src32.data(), n_src}, field_view<float, 1>{dst32.data(), n_dst});

    auto hd = Kokkos::create_mirror_view_and_copy(MemSpace{}, dst64);
    auto hfd = Kokkos::create_mirror_view_and_copy(MemSpace{}, dst32);
    for (std::size_t r = 0; r < n_dst; ++r) {
        EXPECT_NEAR(static_cast<double>(hfd(r)), hd(r), 1e-4) << "row " << r;
    }
}

TEST(Float32Apply, BatchApplyMatchesDouble) {
    auto src = make_latlon_mesh(12, 6, 0.0, -80.0, 30.0, 30.0);
    auto dst = make_latlon_mesh(5, 3, 15.0, -65.0, 72.0, 50.0);

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    auto W = WeightGenerator::generate<MemSpace>(src, dst, cfg);
    W.to_csr();

    const std::size_t n_src = W.n_src();
    const std::size_t n_dst = W.n_dst();
    const std::size_t n_vars = 3;

    Kokkos::View<double **, MemSpace> src64("s64", n_src, n_vars);
    Kokkos::View<double **, MemSpace> dst64("d64", n_dst, n_vars);
    Kokkos::View<float **, MemSpace> src32("s32", n_src, n_vars);
    Kokkos::View<float **, MemSpace> dst32("d32", n_dst, n_vars);
    auto hs = Kokkos::create_mirror_view(src64);
    for (std::size_t i = 0; i < n_src; ++i)
        for (std::size_t v = 0; v < n_vars; ++v) hs(i, v) = std::sin(0.1 * i) + 0.5 * static_cast<double>(v);
    Kokkos::deep_copy(src64, hs);
    auto hf = Kokkos::create_mirror_view(src32);
    for (std::size_t i = 0; i < n_src; ++i)
        for (std::size_t v = 0; v < n_vars; ++v) hf(i, v) = static_cast<float>(hs(i, v));
    Kokkos::deep_copy(src32, hf);

    batch_apply(W, field_view<const double, 2>(src64.data(), n_src, n_vars), field_view<double, 2>(dst64.data(), n_dst, n_vars));
    batch_apply(W, field_view<const float, 2>(src32.data(), n_src, n_vars), field_view<float, 2>(dst32.data(), n_dst, n_vars));

    auto hd = Kokkos::create_mirror_view_and_copy(MemSpace{}, dst64);
    auto hfd = Kokkos::create_mirror_view_and_copy(MemSpace{}, dst32);
    for (std::size_t r = 0; r < n_dst; ++r) {
        for (std::size_t v = 0; v < n_vars; ++v) {
            EXPECT_NEAR(static_cast<double>(hfd(r, v)), hd(r, v), 1e-4) << "row " << r << " var " << v;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector rotation angles (R8)
// ─────────────────────────────────────────────────────────────────────────────

TEST(RotationAngles, LatLonGridIsIdentity) {
    // Regular lat-lon: rows are east-west great circles → α = 0 everywhere.
    auto mesh = make_latlon_mesh(18, 9, 0.0, -85.0, 20.0, 20.0);
    auto alpha = compute_rotation_angles<MemSpace>(mesh);
    auto h = Kokkos::create_mirror_view_and_copy(MemSpace{}, alpha);
    ASSERT_EQ(h.extent(0), mesh.n_cells());
    for (std::size_t c = 0; c < h.extent(0); ++c) {
        EXPECT_NEAR(h(c), 0.0, 1e-12) << "cell " << c;
    }
}

TEST(RotationAngles, CubedSphereBeltTilesAntisymmetric) {
    // Tile 1 of C<N>: the a-axis runs due east along the tile equator (b = 0),
    // so α(b=0 row) = 0 and α is antisymmetric in b.
    auto mesh = topology::NamedGridRegistry::generate<MemSpace>("C12");
    auto alpha = compute_rotation_angles<MemSpace>(mesh);
    auto h = Kokkos::create_mirror_view_and_copy(MemSpace{}, alpha);
    const int N = 12;
    const auto tile = [&](int t, int i, int j) { return h(static_cast<std::size_t>(t * N * N + j * N + i)); };
    // Middle row of tile 0 (b ≈ 0): near-zero.
    const double mid = tile(0, N / 2, N / 2);  // straddles b=0
    EXPECT_LT(std::abs(mid), 0.02);
    // Antisymmetry in j about the tile equator.
    for (int j = 0; j < N / 2; ++j) {
        EXPECT_NEAR(tile(0, 3, j), -tile(0, 3, N - 1 - j), 1e-10) << "j=" << j;
    }
    // Magnitudes bounded by the tile corner deflection (< 45°).
    double max_abs = 0.0;
    for (std::size_t c = 0; c < 4u * N * N; ++c) max_abs = std::max(max_abs, std::abs(h(c)));  // equatorial-belt tiles only
    EXPECT_LE(max_abs, M_PI / 4.0);
}

TEST(RotationAngles, NonQuadrilateralCellsAreZero) {
    // MPAS-style: hex cells → geographic basis (α = 0). Build a mesh whose
    // cells are pentagons via the hex-ish C-grid? Simplest deterministic
    // check: the C<N> generator emits quads only, so instead craft a tiny
    // manual CGNS-style pentagon mesh.
    // 5 nodes on the equator ring + 1 pole node, one pentagonal cell.
    std::vector<double> lon = {0.0, 72.0, 144.0, 216.0, 288.0};
    std::vector<double> lat(5, 0.0);
    Kokkos::View<double **, Kokkos::LayoutLeft, MemSpace> coords("coords", 5, 2);
    auto hc = Kokkos::create_mirror_view(coords);
    for (int n = 0; n < 5; ++n) {
        hc(n, 0) = lon[n];
        hc(n, 1) = lat[n];
    }
    Kokkos::deep_copy(coords, hc);
    Kokkos::View<index_t *, MemSpace> offsets("offs", 2);
    Kokkos::View<index_t *, MemSpace> indices("inds", 5);
    auto ho = Kokkos::create_mirror_view(offsets);
    ho(0) = 0;
    ho(1) = 5;
    Kokkos::deep_copy(offsets, ho);
    auto hi = Kokkos::create_mirror_view(indices);
    for (int n = 0; n < 5; ++n) hi(n) = n;
    Kokkos::deep_copy(indices, hi);

    topology::UnstructuredMesh<MemSpace> pent(coords, offsets, indices, topology::CoordinateSystem::SphericalDeg);
    auto alpha = compute_rotation_angles<MemSpace>(pent);
    auto h = Kokkos::create_mirror_view_and_copy(MemSpace{}, alpha);
    ASSERT_EQ(h.extent(0), 1u);
    EXPECT_DOUBLE_EQ(h(0), 0.0);
}

TEST(RotationAngles, VectorRegridOfConstantEastWindOnLatLon) {
    // A constant geographic east wind (u=cos α, v=sin α with α=0 → (1,0))
    // regridded lat-lon → lat-lon must stay (1,0) everywhere.
    auto src = make_latlon_mesh(18, 9, 0.0, -85.0, 20.0, 20.0);
    auto dst = make_latlon_mesh(9, 5, 10.0, -80.0, 40.0, 34.0);

    auto sa = compute_rotation_angles<MemSpace>(src);
    auto da = compute_rotation_angles<MemSpace>(dst);
    GridRotation<MemSpace> srot{sa};
    GridRotation<MemSpace> drot{da};

    RegridConfig cfg;
    cfg.method = InterpolationMethod::Bilinear;
    auto [Wu, Wv] = VectorWeightGenerator<MemSpace>::generate(src, dst, srot, drot, cfg);

    Kokkos::View<double *, MemSpace> suv("suv", 2 * src.n_cells());
    auto hs = Kokkos::create_mirror_view(suv);
    for (std::size_t c = 0; c < src.n_cells(); ++c) {
        hs(c) = 1.0;                // u = 1
        hs(c + src.n_cells()) = 0.;  // v = 0
    }
    Kokkos::deep_copy(suv, hs);

    Kokkos::View<double *, MemSpace> du("du", dst.n_cells());
    Kokkos::View<double *, MemSpace> dv("dv", dst.n_cells());
    apply(Wu, field_view<const double, 1>{suv.data(), suv.extent(0)}, field_view<double, 1>{du.data(), du.extent(0)});
    apply(Wv, field_view<const double, 1>{suv.data(), suv.extent(0)}, field_view<double, 1>{dv.data(), dv.extent(0)});

    auto hud = Kokkos::create_mirror_view_and_copy(MemSpace{}, du);
    auto hvd = Kokkos::create_mirror_view_and_copy(MemSpace{}, dv);
    for (std::size_t c = 0; c < dst.n_cells(); ++c) {
        // Interior rows fully covered: (1, 0). Seam/partial rows may scale,
        // so only assert direction where the row sums to unity.
        const double sum = hud(c) * hud(c) + hvd(c) * hvd(c);
        if (sum > 0.25) {
            EXPECT_NEAR(hud(c) / std::sqrt(sum), 1.0, 1e-9) << "cell " << c;
            EXPECT_NEAR(hvd(c) / std::sqrt(sum), 0.0, 1e-9) << "cell " << c;
        }
    }
}

}  // namespace axis::test
