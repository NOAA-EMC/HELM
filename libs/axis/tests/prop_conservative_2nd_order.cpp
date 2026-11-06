// SPDX-License-Identifier: Apache-2.0
// Randomized tests use a common planar domain and genuine cell averages.
// Spherical/seam/pole coverage lives in test_linear_conservative.cpp.
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <algorithm>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <cmath>
#include <vector>

namespace {
using Space = Kokkos::HostSpace;
using Mesh = axis::topology::UnstructuredMesh<Space>;
Mesh rectangular_mesh(int ni, int nj) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> coords("nodes", (ni + 1) * (nj + 1), 2);
    for (int j = 0; j <= nj; ++j)
        for (int i = 0; i <= ni; ++i) {
            int k = i + (ni + 1) * j;
            coords(k, 0) = double(i) / ni;
            coords(k, 1) = double(j) / nj;
        }
    Kokkos::View<axis::index_t *, Space> offsets("offsets", ni * nj + 1), indices("indices", 4 * ni * nj);
    for (int j = 0; j < nj; ++j)
        for (int i = 0; i < ni; ++i) {
            int k = i + ni * j, a = i + (ni + 1) * j;
            offsets(k) = 4 * k;
            indices(4 * k) = a;
            indices(4 * k + 1) = a + 1;
            indices(4 * k + 2) = a + ni + 2;
            indices(4 * k + 3) = a + ni + 1;
        }
    offsets(ni * nj) = 4 * ni * nj;
    return Mesh(coords, offsets, indices, axis::topology::CoordinateSystem::Cartesian3D);
}
auto matrix(int ni, int nj, int mi, int mj, bool limited, axis::solver::NormType norm) {
    axis::solver::RegridConfig config;
    config.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
    config.line_type = axis::solver::LineType::Cartesian;
    config.extrap_method = axis::solver::ExtrapolationAction::None;
    config.norm_type = norm;
    config.use_limiter = limited;
    return axis::solver::WeightGenerator::generate(rectangular_mesh(ni, nj), rectangular_mesh(mi, mj), config);
}
std::vector<double> remap_values(const auto &m, const std::vector<double> &q) {
    std::vector<double> out(m.n_dst());
    axis::solver::apply(m, axis::field_view<const double, 1>{q.data(), q.size()}, axis::field_view<double, 1>{out.data(), out.size()});
    return out;
}
RC_GTEST_PROP(PropConservative2ndOrder, LinearExactness, ()) {
    int ni = *rc::gen::inRange(3, 8), nj = *rc::gen::inRange(3, 8), mi = *rc::gen::inRange(3, 9), mj = *rc::gen::inRange(3, 9);
    double a = *rc::gen::inRange(-100, 101) / 10.0, b = *rc::gen::inRange(-100, 101) / 10.0, c = *rc::gen::inRange(-100, 101) / 10.0;
    auto m = matrix(ni, nj, mi, mj, false, axis::solver::NormType::DstArea);
    m.to_csr();
    std::vector<double> q(ni * nj);
    for (int j = 0; j < nj; ++j)
        for (int i = 0; i < ni; ++i) q[i + ni * j] = a * (i + 0.5) / ni + b * (j + 0.5) / nj + c;
    auto out = remap_values(m, q);
    for (int j = 0; j < mj; ++j)
        for (int i = 0; i < mi; ++i) RC_ASSERT(std::abs(out[i + mi * j] - (a * (i + 0.5) / mi + b * (j + 0.5) / mj + c)) < 2e-12);
}
RC_GTEST_PROP(PropConservative2ndOrder, IntegralPreservation, ()) {
    int ni = *rc::gen::inRange(3, 8), nj = *rc::gen::inRange(3, 8), mi = *rc::gen::inRange(3, 9), mj = *rc::gen::inRange(3, 9);
    bool limited = *rc::gen::arbitrary<bool>(), frac = *rc::gen::arbitrary<bool>();
    auto m = matrix(ni, nj, mi, mj, limited, frac ? axis::solver::NormType::FracArea : axis::solver::NormType::DstArea);
    std::vector<double> q(ni * nj);
    for (auto &value : q) value = *rc::gen::inRange(-100, 101) / 10.0;
    auto out = remap_values(m, q);
    double source = 0, dest = 0;
    for (std::size_t i = 0; i < q.size(); ++i) source += q[i] * m.area_a()[i];
    for (std::size_t j = 0; j < out.size(); ++j) dest += out[j] * m.area_b()[j] * (frac ? m.frac_b()[j] : 1);
    RC_ASSERT(std::abs(source - dest) < 2e-12);
}
RC_GTEST_PROP(PropConservative2ndOrder, MonotonicityLimiterPreventsNewExtrema, ()) {
    int ni = *rc::gen::inRange(3, 8), nj = *rc::gen::inRange(3, 8), mi = *rc::gen::inRange(3, 9), mj = *rc::gen::inRange(3, 9);
    auto m = matrix(ni, nj, mi, mj, true, axis::solver::NormType::FracArea);
    m.to_csr();
    std::vector<double> q(ni * nj);
    for (auto &value : q) value = *rc::gen::inRange(-100, 101) / 10.0;
    double lo = *std::min_element(q.begin(), q.end()), hi = *std::max_element(q.begin(), q.end());
    for (double value : remap_values(m, q)) {
        RC_ASSERT(value >= lo - 2e-12);
        RC_ASSERT(value <= hi + 2e-12);
    }
}
class LinearPropertyEnv : public testing::Environment {
    void SetUp() override {
        if (!Kokkos::is_initialized()) Kokkos::initialize();
    }
    void TearDown() override {
        if (Kokkos::is_initialized()) Kokkos::finalize();
    }
};
auto *const env = testing::AddGlobalTestEnvironment(new LinearPropertyEnv);
}  // namespace
