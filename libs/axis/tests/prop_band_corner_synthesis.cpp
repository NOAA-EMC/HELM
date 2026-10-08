// ─── Property-Based Tests: Global-Context Band Corner Synthesis ──────────────
// Feature: distributed-domain-decomposition (AXIS-backed), Property 10/11 support
//
// axis::topology::synthesize_band_corners() is the single source of truth for
// how a latitude band's Cell_Corners are derived. The defining guarantee is that
// it must return EXACTLY the corresponding corner rows of the whole-grid
// synthesis that StructuredGrid::synthesize_corners() performs (via
// to_unstructured()) on the same global centers — no matter whether the grid is
// rectilinear or curvilinear, uniform or non-uniform, periodic or not. This is
// what makes a distributed band regrid match the global regrid at the seams and
// a single-rank (whole-grid) band byte-for-byte the global mesh.
//
// The tests:
//   1. band-subset-of-global: for a random global center layout and a random
//      band [j0, j1), synthesize_band_corners equals rows [j0, j1] of the whole-
//      grid synthesized corners, element for element.
//   2. whole-grid-equals-global: the whole-grid band [0, nj) equals the global
//      synthesized corners exactly (Req 9.6/9.7 boundary convention).
//   3. adjacent-bands-share-seam: two contiguous bands [j0,j1) and [j1,j2) agree
//      on their shared corner row j1 (no seam gap/overlap).
//
// All tests run on Kokkos::HostSpace.
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using MemSpace = Kokkos::HostSpace;

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

// Build whole-grid synthesized corners by running the real StructuredGrid path
// with the same explicit seam declaration supplied to the band helper.
void global_synthesized_corners(std::size_t ni, std::size_t nj, const std::vector<double> &clon, const std::vector<double> &clat,
                                axis::topology::LongitudePeriodicity seam, std::vector<double> &out_lon, std::vector<double> &out_lat) {
    Kokkos::View<double *, MemSpace> lon("lon", clon.size());
    Kokkos::View<double *, MemSpace> lat("lat", clat.size());
    for (std::size_t k = 0; k < clon.size(); ++k) {
        lon(k) = clon[k];
        lat(k) = clat[k];
    }
    axis::topology::StructuredGrid<MemSpace> grid(ni, nj, std::move(lon), std::move(lat), axis::topology::CoordinateSystem::SphericalDeg);
    auto mesh = grid.to_unstructured(axis::topology::CornerPolicy::CurvilinearApproximate, seam);
    auto nc = mesh.node_coords();
    const std::size_t nip1 = ni + 1;
    const std::size_t njp1 = nj + 1;
    out_lon.assign(nip1 * njp1, 0.0);
    out_lat.assign(nip1 * njp1, 0.0);
    for (std::size_t cj = 0; cj < njp1; ++cj)
        for (std::size_t ci = 0; ci < nip1; ++ci) {
            out_lon[ci + cj * nip1] = nc(ci + cj * nip1, 0);
            out_lat[ci + cj * nip1] = nc(ci + cj * nip1, 1);
        }
}

// Generate a global center layout: rectilinear (lon depends on i only, lat on j
// only) or curvilinear (both vary in i and j). Uniform or non-uniform spacing.
struct CenterLayout {
    std::size_t ni, nj;
    std::vector<double> lon, lat;  // size ni*nj, index i + j*ni
};

CenterLayout gen_layout(bool &periodic) {
    const std::size_t ni = *rc::gen::inRange<std::size_t>(2, 9);
    const std::size_t nj = *rc::gen::inRange<std::size_t>(2, 9);
    periodic = *rc::gen::inRange(0, 2) == 0;
    const bool curvilinear = *rc::gen::inRange(0, 2) == 0;
    const bool nonuniform = *rc::gen::inRange(0, 2) == 0;

    auto jitter = [&](std::size_t i, std::size_t j) {
        if (!curvilinear) return 0.0;
        return 2.0 * std::sin(0.7 * static_cast<double>(i) + 0.5 * static_cast<double>(j));
    };

    std::vector<double> lon(ni * nj), lat(ni * nj);
    const double dlon = periodic ? 360.0 / static_cast<double>(ni) : 20.0;
    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            double base_lon = periodic ? (static_cast<double>(i) + 0.5) * dlon : -180.0 + (static_cast<double>(i) + 0.5) * dlon;
            double base_lat = -80.0 + (static_cast<double>(j) + 0.5) * (160.0 / static_cast<double>(nj));
            if (nonuniform) {
                base_lon += 3.0 * std::cos(1.3 * static_cast<double>(i) + 0.4 * static_cast<double>(j));
                base_lat += 2.0 * std::sin(0.9 * static_cast<double>(j));
            }
            lon[i + j * ni] = base_lon + jitter(i, j);
            lat[i + j * ni] = base_lat + 0.5 * jitter(i, j);
        }
    }
    return {ni, nj, std::move(lon), std::move(lat)};
}

// ─── Property A: band corners == corresponding rows of the global corners ────
RC_GTEST_PROP(PropBandCornerSynthesis, BandSubsetOfGlobal, ()) {
    bool periodic = false;
    CenterLayout L = gen_layout(periodic);
    // Two antipodal longitude centers do not determine which periodic arc is
    // intended; the local unwrapping direction is ambiguous at 180 degrees.
    RC_PRE(!periodic || L.ni > 2);
    const std::size_t ni = L.ni, nj = L.nj;
    const std::size_t j0 = *rc::gen::inRange<std::size_t>(0, nj + 1);
    const std::size_t j1 = *rc::gen::inRange<std::size_t>(j0, nj + 1);

    std::vector<double> g_lon, g_lat;
    const axis::topology::LongitudePeriodicity seam{periodic, 360.0, true};
    global_synthesized_corners(ni, nj, L.lon, L.lat, seam, g_lon, g_lat);

    Kokkos::View<double *, MemSpace> clon("clon", L.lon.size());
    Kokkos::View<double *, MemSpace> clat("clat", L.lat.size());
    for (std::size_t k = 0; k < L.lon.size(); ++k) {
        clon(k) = L.lon[k];
        clat(k) = L.lat[k];
    }
    Kokkos::View<double *, MemSpace> bclon, bclat;
    axis::topology::synthesize_band_corners<MemSpace>(ni, nj, clon, clat, j0, j1, seam, bclon, bclat);

    const std::size_t nip1 = ni + 1;
    const std::size_t nrows = j1 - j0 + 1;
    RC_ASSERT(bclon.extent(0) == nip1 * nrows);
    RC_ASSERT(bclat.extent(0) == nip1 * nrows);
    for (std::size_t cj = 0; cj < nrows; ++cj) {
        for (std::size_t ci = 0; ci < nip1; ++ci) {
            const std::size_t band_idx = ci + cj * nip1;
            const std::size_t global_idx = ci + (j0 + cj) * nip1;
            RC_ASSERT(std::fabs(bclon(band_idx) - g_lon[global_idx]) <= 1e-12);
            RC_ASSERT(std::fabs(bclat(band_idx) - g_lat[global_idx]) <= 1e-12);
        }
    }
}

// ─── Property B: whole-grid band == global (boundary convention) ─────────────
RC_GTEST_PROP(PropBandCornerSynthesis, WholeGridEqualsGlobal, ()) {
    bool periodic = false;
    CenterLayout L = gen_layout(periodic);
    RC_PRE(!periodic || L.ni > 2);
    const std::size_t ni = L.ni, nj = L.nj;

    std::vector<double> g_lon, g_lat;
    const axis::topology::LongitudePeriodicity seam{periodic, 360.0, true};
    global_synthesized_corners(ni, nj, L.lon, L.lat, seam, g_lon, g_lat);

    Kokkos::View<double *, MemSpace> clon("clon", L.lon.size());
    Kokkos::View<double *, MemSpace> clat("clat", L.lat.size());
    for (std::size_t k = 0; k < L.lon.size(); ++k) {
        clon(k) = L.lon[k];
        clat(k) = L.lat[k];
    }
    Kokkos::View<double *, MemSpace> bclon, bclat;
    axis::topology::synthesize_band_corners<MemSpace>(ni, nj, clon, clat, 0, nj, seam, bclon, bclat);

    const std::size_t nip1 = ni + 1;
    RC_ASSERT(bclon.extent(0) == nip1 * (nj + 1));
    for (std::size_t k = 0; k < nip1 * (nj + 1); ++k) {
        RC_ASSERT(std::fabs(bclon(k) - g_lon[k]) <= 1e-12);
        RC_ASSERT(std::fabs(bclat(k) - g_lat[k]) <= 1e-12);
    }
}

// ─── Property C: adjacent bands agree on their shared seam row ───────────────
RC_GTEST_PROP(PropBandCornerSynthesis, AdjacentBandsShareSeam, ()) {
    bool periodic = false;
    CenterLayout L = gen_layout(periodic);
    RC_PRE(!periodic || L.ni > 2);
    const std::size_t ni = L.ni, nj = L.nj;
    const std::size_t j0 = *rc::gen::inRange<std::size_t>(0, nj);
    const std::size_t jm = *rc::gen::inRange<std::size_t>(j0, nj + 1);
    const std::size_t j1 = *rc::gen::inRange<std::size_t>(jm, nj + 1);

    Kokkos::View<double *, MemSpace> clon("clon", L.lon.size());
    Kokkos::View<double *, MemSpace> clat("clat", L.lat.size());
    for (std::size_t k = 0; k < L.lon.size(); ++k) {
        clon(k) = L.lon[k];
        clat(k) = L.lat[k];
    }
    Kokkos::View<double *, MemSpace> alon, alat, blon, blat;
    const axis::topology::LongitudePeriodicity seam{periodic, 360.0, true};
    axis::topology::synthesize_band_corners<MemSpace>(ni, nj, clon, clat, j0, jm, seam, alon, alat);
    axis::topology::synthesize_band_corners<MemSpace>(ni, nj, clon, clat, jm, j1, seam, blon, blat);

    const std::size_t nip1 = ni + 1;
    // The bottom corner row (index 0) of the upper band [jm, j1) is global corner
    // row jm; the top corner row of the lower band [j0, jm) is also row jm.
    const std::size_t a_top = (jm - j0) * nip1;
    for (std::size_t ci = 0; ci < nip1; ++ci) {
        RC_ASSERT(std::fabs(alon(a_top + ci) - blon(ci)) <= 1e-12);
        RC_ASSERT(std::fabs(alat(a_top + ci) - blat(ci)) <= 1e-12);
    }
}

}  // namespace
