// SPDX-License-Identifier: Apache-2.0
// Focused CPU diagnostic, separate from the paper acceptance timing protocol.
#include <algorithm>
#include <array>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
namespace {
using Space = Kokkos::HostSpace;
using Mesh = axis::topology::UnstructuredMesh<Space>;
Mesh triangles(int n) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> coords("nodes", (n + 1) * (n + 1), 3);
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            int k = i + (n + 1) * j;
            coords(k, 0) = double(i) / n;
            coords(k, 1) = double(j) / n;
            coords(k, 2) = 0;
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
    return Mesh(coords, offsets, indices, axis::topology::CoordinateSystem::Cartesian3D);
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
int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    {
        const int n = argc > 1 ? std::stoi(argv[1]) : 32;
        auto src = triangles(n), dst = triangles(2 * n + 1);
        std::vector<double> field(src.n_cells()), out(dst.n_cells());
        for (std::size_t i = 0; i < field.size(); ++i) field[i] = 1 + 0.2 * std::sin(double(i) * 0.13);
        std::cout << "method,source_cells,destination_cells,nnz,setup_csr_ms_median,apply_ms_median,checksum\n";
        for (int mode = 0; mode < 3; ++mode) {
            axis::solver::RegridConfig config;
            config.method =
                mode == 0 ? axis::solver::InterpolationMethod::Conservative1stOrder : axis::solver::InterpolationMethod::Conservative2ndOrder;
            config.line_type = axis::solver::LineType::Cartesian;
            config.use_limiter = mode == 2;
            config.extrap_method = axis::solver::ExtrapolationAction::None;
            axis::solver::InterpolationMatrix<Space> matrix;
            std::vector<double> setups, applies;
            for (int k = 0; k < 7; ++k) {
                auto t = Clock::now();
                matrix = axis::solver::WeightGenerator::generate(src, dst, config);
                matrix.to_csr();
                setups.push_back(milliseconds(t));
            }
            auto apply = [&]() {
                axis::solver::apply(matrix, axis::field_view<const double, 1>{field.data(), field.size()},
                                    axis::field_view<double, 1>{out.data(), out.size()});
            };
            for (int k = 0; k < 5; ++k) apply();
            for (int k = 0; k < 101; ++k) {
                auto t = Clock::now();
                apply();
                applies.push_back(milliseconds(t));
            }
            double checksum = 0;
            for (double value : out) checksum += value;
            std::cout << std::setprecision(12)
                      << (mode == 0   ? "first_order"
                          : mode == 1 ? "linear_unlimited"
                                      : "linear_limited")
                      << ',' << src.n_cells() << ',' << dst.n_cells() << ',' << matrix.nnz() << ',' << median(setups) << ',' << median(applies) << ','
                      << checksum << '\n';
        }
    }
    Kokkos::finalize();
}
