// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <axis/detail/linear_geometry.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <chrono>
#include <cmath>
#include <random>
namespace {
struct Init : testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
auto *init = testing::AddGlobalTestEnvironment(new Init);
}  // namespace
namespace {
using Space = Kokkos::HostSpace;
using Mesh = axis::topology::UnstructuredMesh<Space>;
Mesh triangles(int n, bool spherical = false) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> coords("nodes", (n + 1) * (n + 1), (spherical ? 2 : 3));
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            int k = i + (n + 1) * j;
            double x = double(i) / n, y = double(j) / n;
            if (spherical) {
                double u = 0.18 * (x - 0.5), v = 0.18 * (y - 0.5), r = std::sqrt(1 + u * u + v * v);
                coords(k, 0) = std::atan2(u, 1.0) * 180 / 3.14159265358979323846;
                coords(k, 1) = std::asin(v / r) * 180 / 3.14159265358979323846;
            } else {
                coords(k, 0) = x;
                coords(k, 1) = y;
                coords(k, 2) = 0;
            }
        }
    Kokkos::View<axis::index_t *, Space> offsets("offsets", 2 * n * n + 1), indices("indices", 6 * n * n);
    int c = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int a = i + (n + 1) * j, b = a + 1, d = a + n + 1, e = d + 1;
            for (auto tri : {std::array<int, 3>{a, b, e}, std::array<int, 3>{a, e, d}}) {
                offsets(c) = 3 * c;
                for (int k = 0; k < 3; ++k) indices(3 * c + k) = tri[k];
                ++c;
            }
        }
    offsets(c) = 3 * c;
    return Mesh(coords, offsets, indices, spherical ? axis::topology::CoordinateSystem::SphericalDeg : axis::topology::CoordinateSystem::Cartesian3D);
}
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}
}  // namespace

Mesh rectangles(int n) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> co("nodes", (n + 1) * (n + 1), 2);
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            co(i + (n + 1) * j, 0) = -5 + 10.0 * i / n;
            co(i + (n + 1) * j, 1) = -5 + 10.0 * j / n;
        }
    Kokkos::View<axis::index_t *, Space> off("offsets", n * n + 1), idx("indices", 4 * n * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int c = i + n * j, a = i + (n + 1) * j;
            off(c) = 4 * c;
            idx(4 * c) = a;
            idx(4 * c + 1) = a + 1;
            idx(4 * c + 2) = a + n + 2;
            idx(4 * c + 3) = a + n + 1;
        }
    off(n * n) = 4 * n * n;
    axis::topology::GeometryMetadata meta;
    meta.boundary_model = axis::topology::BoundaryModel::ConstantLatitude;
    return Mesh(co, off, idx, axis::topology::CoordinateSystem::SphericalDeg, {}, {}, meta);
}

auto make_matrix(const Mesh &a, const Mesh &b, bool limited = false) {
    axis::solver::RegridConfig c;
    c.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
    c.line_type = axis::solver::LineType::Cartesian;
    c.extrap_method = axis::solver::ExtrapolationAction::None;
    c.use_limiter = limited;
    return axis::solver::WeightGenerator::generate(a, b, c);
}
TEST(PerformanceChanges, PruningPreservesAffineBounds) {
    std::mt19937 gen(4811);
    std::uniform_real_distribution<double> u(-1, 1);
    for (int trial = 0; trial < 100; ++trial) {
        std::vector<std::array<double, 2>> p{{-1, -1}, {1, -1}, {1, 1}, {-1, 1}, {0, 0}, {0, 0}};
        for (int i = 0; i < 50; ++i) p.push_back({u(gen), u(gen)});
        if (trial % 2 == 0) p.push_back({0, 1 + 1e-15});
        auto hull = p;
        axis::detail::linear::prune_linear_controls(hull);
        for (int k = 0; k < 20; ++k) {
            double gx = u(gen) * 4, gy = u(gen) * 4;
            auto phi = [&](const auto &points) {
                double result = 1;
                for (auto q : points) {
                    double d = gx * q[0] + gy * q[1];
                    if (d != 0) result = std::min(result, 1 / std::abs(d));
                }
                return result;
            };
            EXPECT_NEAR(phi(p), phi(hull), 2e-14);
        }
    }
    std::vector<std::array<double, 2>> line{{-1, 0}, {0, 0}, {1, 0}, {2, 0}};
    axis::detail::linear::prune_linear_controls(line);
    EXPECT_GE(line.size(), 2);
}
template <class Layout>
void batch_check() {
    auto a = triangles(5), b = triangles(11);
    for (bool limited : {false, true})
        for (bool csr : {false, true})
            for (std::size_t vars : {0, 1, 3, 8, 13}) {
                auto m = make_matrix(a, b, limited);
                if (csr) m.to_csr();
                Kokkos::View<double **, Layout, Space> q("q", a.n_cells(), vars), out("out", b.n_cells(), vars);
                for (std::size_t v = 0; v < vars; ++v)
                    for (std::size_t i = 0; i < a.n_cells(); ++i) q(i, v) = v % 2 ? double(i % 3 == 0) : 1 + .2 * std::sin(i * .31 + v * .19);
                Kokkos::View<const double **, Layout, Space> cq = q;
                axis::solver::batch_apply<Space, Layout>(m, cq, out);
                Kokkos::View<double *[2], Space> gradients(Kokkos::view_alloc(Kokkos::WithoutInitializing, "workspace"), a.n_cells());
                for (std::size_t v = 0; v < vars; ++v) {
                    std::vector<double> x(a.n_cells()), y(b.n_cells());
                    for (std::size_t i = 0; i < x.size(); ++i) x[i] = q(i, v);
                    using In = Kokkos::View<const double *, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                    using Out = Kokkos::View<double *, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
                    axis::solver::apply_linear_with_workspace<Space>(m, In(x.data(), x.size()), Out(y.data(), y.size()), gradients);
                    for (std::size_t j = 0; j < y.size(); ++j) EXPECT_NEAR(y[j], out(j, v), 3e-13);
                }
            }
}
TEST(PerformanceChanges, BatchWidthsAndLayouts) {
    batch_check<Kokkos::LayoutLeft>();
    batch_check<Kokkos::LayoutRight>();
}
TEST(PerformanceChanges, WorkspaceShapeRejected) {
    auto a = triangles(3), b = triangles(5);
    auto m = make_matrix(a, b);
    Kokkos::View<double *, Space> q("q", a.n_cells()), out("out", b.n_cells());
    Kokkos::View<double *[2], Space> bad("bad", a.n_cells() + 1);
    EXPECT_THROW(axis::solver::apply_linear_with_workspace<Space>(m, q, out, bad), std::invalid_argument);
}
TEST(PerformanceChanges, CartesianTwoAndThreeColumnsAgree) {
    auto xyz = triangles(8);
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> c("xy", xyz.n_nodes(), 2);
    for (std::size_t i = 0; i < xyz.n_nodes(); ++i)
        for (int d = 0; d < 2; ++d) c(i, d) = xyz.node_coords()(i, d);
    Mesh xy(c, xyz.conn_offsets_view(), xyz.conn_indices_view(), xyz.coord_system());
    auto dst = triangles(11);
    for (auto method : {axis::solver::InterpolationMethod::Conservative1stOrder, axis::solver::InterpolationMethod::NearestNeighbor,
                        axis::solver::InterpolationMethod::Bilinear}) {
        axis::solver::RegridConfig cfg;
        cfg.method = method;
        cfg.line_type = axis::solver::LineType::Cartesian;
        cfg.extrap_method = axis::solver::ExtrapolationAction::None;
        auto a = axis::solver::WeightGenerator::generate(xy, dst, cfg), b = axis::solver::WeightGenerator::generate(xyz, dst, cfg);
        a.to_csr();
        b.to_csr();
        ASSERT_EQ(a.nnz(), b.nnz());
        for (std::size_t i = 0; i < a.nnz(); ++i) {
            EXPECT_EQ(a.col_idx()(i), b.col_idx()(i));
            EXPECT_NEAR(a.csr_values()(i), b.csr_values()(i), 3e-14);
        }
    }
}
TEST(PerformanceChanges, ParallelUnmappedErrorPropagates) {
    auto a = triangles(4), b = triangles(12);
    for (std::size_t i = 0; i < b.n_nodes(); ++i) b.node_coords_view()(i, 0) += 2;
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
    config.line_type = axis::solver::LineType::Cartesian;
    config.extrap_method = axis::solver::ExtrapolationAction::None;
    config.unmapped = axis::solver::UnmappedAction::Error;
    EXPECT_THROW(axis::solver::WeightGenerator::generate(a, b, config), std::runtime_error);
}
TEST(PerformanceChanges, MalformedCoordinatesRejected) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> c("bad", 4, 1);
    Kokkos::View<axis::index_t *, Space> off("off", 2), idx("idx", 3);
    EXPECT_THROW(Mesh(c, off, idx, axis::topology::CoordinateSystem::Cartesian3D), std::invalid_argument);
}
TEST(PerformanceChanges, StridedBatchViews) {
    auto a = triangles(5), b = triangles(11);
    constexpr std::size_t vars = 13;
    const std::size_t sx = 2 * vars + 3, sy = 2 * vars + 7;
    std::vector<double> q(a.n_cells() * sx, -99), out(b.n_cells() * sy, -99);
    using In = Kokkos::View<const double **, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Out = Kokkos::View<double **, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    for (std::size_t i = 0; i < a.n_cells(); ++i)
        for (std::size_t v = 0; v < vars; ++v) q[i * sx + 2 * v] = 1 + .2 * std::sin(i * .17 + v * .19);
    for (bool limited : {false, true})
        for (bool csr : {false, true}) {
            auto m = make_matrix(a, b, limited);
            if (csr) m.to_csr();
            axis::solver::batch_apply<Space, Kokkos::LayoutStride>(m, In(q.data(), Kokkos::LayoutStride(a.n_cells(), sx, vars, 2)),
                                                                   Out(out.data(), Kokkos::LayoutStride(b.n_cells(), sy, vars, 2)));
            for (std::size_t v = 0; v < vars; ++v) {
                std::vector<double> x(a.n_cells()), y(b.n_cells());
                for (std::size_t i = 0; i < x.size(); ++i) x[i] = q[i * sx + 2 * v];
                axis::solver::apply(m, axis::field_view<const double, 1>(x.data(), x.size()), axis::field_view<double, 1>(y.data(), y.size()));
                for (std::size_t j = 0; j < y.size(); ++j) EXPECT_NEAR(y[j], out[j * sy + 2 * v], 3e-13);
            }
        }
}

TEST(PerformanceChanges, StridedScalarOperators) {
    auto a = triangles(5), b = triangles(11);
    const std::size_t nv = 3, sx = 9, sy = 11;
    std::vector<double> q(a.n_cells() * sx), out(b.n_cells() * sy);
    for (std::size_t i = 0; i < a.n_cells(); ++i)
        for (std::size_t v = 0; v < nv; ++v) q[i * sx + v * 2] = 1 + .2 * std::sin(i * .13 + v * .19);
    using In = Kokkos::View<const double **, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using Out = Kokkos::View<double **, Kokkos::LayoutStride, Space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    for (auto method : {axis::solver::InterpolationMethod::NearestNeighbor, axis::solver::InterpolationMethod::Bilinear,
                        axis::solver::InterpolationMethod::Conservative1stOrder}) {
        axis::solver::RegridConfig c;
        c.method = method;
        c.line_type = axis::solver::LineType::Cartesian;
        c.extrap_method = axis::solver::ExtrapolationAction::None;
        auto m = axis::solver::WeightGenerator::generate(a, b, c);
        m.to_csr();
        axis::solver::batch_apply<Space, Kokkos::LayoutStride>(m, In(q.data(), Kokkos::LayoutStride(a.n_cells(), sx, nv, 2)),
                                                               Out(out.data(), Kokkos::LayoutStride(b.n_cells(), sy, nv, 2)));
        for (std::size_t v = 0; v < nv; ++v) {
            std::vector<double> x(a.n_cells()), y(b.n_cells());
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = q[i * sx + 2 * v];
            axis::solver::apply(m, axis::field_view<const double, 1>(x.data(), x.size()), axis::field_view<double, 1>(y.data(), y.size()));
            for (std::size_t j = 0; j < y.size(); ++j) EXPECT_NEAR(y[j], out[j * sy + 2 * v], 3e-13);
        }
    }
}
