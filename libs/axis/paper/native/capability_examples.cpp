// SPDX-License-Identifier: Apache-2.0
// Dependency-free analytic capability examples; output is CSV on stdout.
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/solver/apply.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>
using Space = Kokkos::HostSpace;
using namespace axis::solver;
static double vertical(bool varying, double tension) {
    constexpr int nc = 8, ns = 17, nd = 29;
    Kokkos::View<double **, Space> input("input", nc, ns), output("output", nc, nd);
    Kokkos::View<double *, Space> zs("zs", ns), zd("zd", nd);
    Kokkos::View<double **, Space> cs("cs", nc, ns), cd("cd", nc, nd);
    for (int i = 0; i < ns; ++i)
        zs(i) = double(i) / (ns - 1);
    for (int j = 0; j < nd; ++j)
        zd(j) = double(j) / (nd - 1);
    for (int c = 0; c < nc; ++c) {
        for (int i = 0; i < ns; ++i) {
            cs(c, i) = zs(i) * (1 + .05 * c) + .01 * c;
            input(c, i) = 2 * (varying ? cs(c, i) : zs(i)) + 5 + .1 * c;
        }
        for (int j = 0; j < nd; ++j)
            cd(c, j) = zd(j) * (1 + .05 * c) + .01 * c;
    }
    if (varying)
        VerticalRegridder<Space>::interpolate(input, output, cs, cd, tension);
    else
        VerticalRegridder<Space>::interpolate(input, output, zs, zd, tension);
    Kokkos::fence();
    double error = 0;
    for (int c = 0; c < nc; ++c)
        for (int j = 0; j < nd; ++j)
            error = std::max(
                error, std::abs(output(c, j) - (2 * (varying ? cd(c, j) : zd(j)) + 5 + .1 * c)));
    return error;
}
// Known Earth-relative vector represented in spatially varying grid frames.
static double vector_rotation() {
    constexpr int n = 4, nc = n * n;
    Kokkos::View<double *, Space> x("x", nc), y("y", nc), sa("sa", nc), da("da", nc);
    for (int i = 0; i < nc; ++i) {
        x(i) = i % n;
        y(i) = i / n;
        sa(i) = .01 * i;
        da(i) = .4 - .02 * i;
    }
    axis::topology::StructuredGrid<Space> grid(n, n, x, y,
                                               axis::topology::CoordinateSystem::SphericalDeg);
    auto mesh = grid.to_unstructured();
    RegridConfig cfg;
    cfg.method = InterpolationMethod::NearestNeighbor;
    cfg.extrap_method = ExtrapolationAction::None;
    cfg.unmapped = UnmappedAction::Ignore;
    auto [wu, wv] = VectorWeightGenerator<Space>::generate(mesh, mesh, {sa}, {da}, cfg);
    std::vector<double> uv(2 * nc), u(nc), v(nc);
    for (int i = 0; i < nc; ++i) {
        uv[i] = 2 * std::cos(sa(i)) + std::sin(sa(i));
        uv[nc + i] = -2 * std::sin(sa(i)) + std::cos(sa(i));
    }
    axis::field_view<const double, 1> src(uv.data(), uv.size());
    axis::field_view<double, 1> ou(u.data(), u.size()), ov(v.data(), v.size());
    apply(wu, src, ou);
    apply(wv, src, ov);
    Kokkos::fence();
    double error = 0;
    for (int i = 0; i < nc; ++i) {
        error = std::max(error, std::abs(u[i] - (2 * std::cos(da(i)) + std::sin(da(i)))));
        error = std::max(error, std::abs(v[i] - (-2 * std::sin(da(i)) + std::cos(da(i)))));
    }
    return error;
}
int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        std::cout << "example,max_absolute_error,tolerance,status\n" << std::setprecision(17);
        auto report = [&](const char *name, double e) {
            bool ok = std::isfinite(e) && e <= 1e-12;
            std::cout << name << ',' << e << ",1e-12," << (ok ? "pass" : "fail") << '\n';
            if (!ok)
                status = 1;
        };
        report("vertical_uniform_linear_tension0", vertical(false, 0));
        report("vertical_uniform_linear_tension2", vertical(false, 2));
        report("vertical_varying_linear_tension0", vertical(true, 0));
        report("vertical_varying_linear_tension2", vertical(true, 2));
        report("vector_varying_grid_rotation", vector_rotation());
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
