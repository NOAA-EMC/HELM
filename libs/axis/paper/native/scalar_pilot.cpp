// SPDX-License-Identifier: Apache-2.0
// Small in-memory spherical pilot: independent cell-mean truth, no NetCDF/ESMC.
#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <iomanip>
#include <iostream>

#include "axis_methods.hpp"
#include "reference_geometry.hpp"
using namespace paper;
using Space = Kokkos::HostSpace;
static Mesh mesh(int n, bool masked) {
    Mesh m;
    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            double u = .15 * (double(i) / n - .5), v = .15 * (double(j) / n - .5);
            m.lon.push_back(std::atan2(u, 1.) * 180 / std::numbers::pi);
            m.lat.push_back(std::atan2(v, std::sqrt(1 + u * u)) * 180 / std::numbers::pi);
        }
    m.offsets.push_back(0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            int a = i + (n + 1) * j, b = a + 1, d = a + n + 1, e = d + 1;
            for (auto t : {std::array<int, 3>{a, b, e}, std::array<int, 3>{a, e, d}}) {
                m.indices.insert(m.indices.end(), t.begin(), t.end());
                m.offsets.push_back(m.indices.size());
                int k = m.mask.size();
                m.ids.push_back(k + 1);
                m.parent.push_back(k);
                m.mask.push_back(!masked || i < n / 2);
            }
        }
    return m;
}
static auto axis_mesh(const Mesh &m) {
    Kokkos::View<double **, Kokkos::LayoutLeft, Space> nodes("nodes", m.lon.size(), 2);
    Kokkos::View<axis::index_t *, Space> offsets("offsets", m.offsets.size()),
        indices("indices", m.indices.size());
    Kokkos::View<int *, Space> mask("mask", m.mask.size());
    for (size_t i = 0; i < m.lon.size(); ++i) {
        nodes(i, 0) = m.lon[i];
        nodes(i, 1) = m.lat[i];
    }
    for (size_t i = 0; i < m.offsets.size(); ++i)
        offsets(i) = m.offsets[i];
    for (size_t i = 0; i < m.indices.size(); ++i)
        indices(i) = m.indices[i];
    for (size_t i = 0; i < m.mask.size(); ++i)
        mask(i) = m.mask[i];
    return axis::topology::make_unstructured<Space>(
        nodes, offsets, indices, axis::topology::CoordinateSystem::SphericalDeg, {}, mask);
}
int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    int status = 0;
    try {
        std::cout << "method,field,source_mask,rms_cell_mean_error,constant_response_error,mapped,"
                     "coverage_eligible,status\n"
                  << std::setprecision(17);
        for (bool masked : {false, true})
            for (const auto *field : {"constant", "smooth"}) {
                Case c;
                c.src = mesh(4, masked);
                c.dst = mesh(7, false);
                c.geometry = "great_circle";
                c.norm = "dstarea";
                c.field = field;
                auto reference = ReferenceGeometry<long double>::compute_reference(c);
                auto src = axis_mesh(c.src), dst = axis_mesh(c.dst);
                for (const auto *method :
                     {"nearest", "bilinear", "bicubic", "patch", "conservative_first_order",
                      "conservative_second_order", "conservative_second_order_limited"}) {
                    try {
                        axis::solver::RegridConfig cfg;
                        select_axis_method(cfg, method);
                        cfg.extrap_method = axis::solver::ExtrapolationAction::None;
                        cfg.unmapped = axis::solver::UnmappedAction::Ignore;
                        auto w = axis::solver::WeightGenerator::generate(src, dst, cfg);
                        std::vector<double> values(reference.src), out(c.dst.size()),
                            ones(c.src.size(), 1), response(c.dst.size());
                        axis::field_view<const double, 1> input(values.data(), values.size()),
                            one(ones.data(), ones.size());
                        axis::field_view<double, 1> output(out.data(), out.size()),
                            sum(response.data(), response.size());
                        axis::solver::apply(w, input, output);
                        axis::solver::apply(w, one, sum);
                        Kokkos::fence();
                        double squared = 0, area = 0, constant_error = 0;
                        int mapped = 0, eligible = 0;
                        for (size_t j = 0; j < out.size(); ++j) {
                            if (std::abs(response[j]) < 1e-12)
                                continue;
                            ++mapped;
                            bool cons = conservative_method(method);
                            if (reference.dst_covered[j] <= 0 ||
                                (!cons &&
                                 reference.dst_covered[j] < reference.dst_area[j] * (1 - 1e-10)))
                                continue;
                            ++eligible;
                            double expected =
                                cons ? reference.dst_covered[j] / reference.dst_area[j] : 1.;
                            constant_error =
                                std::max(constant_error, std::abs(response[j] - expected));
                            double error = out[j] - reference.dst[j];
                            squared += reference.dst_covered[j] * error * error;
                            area += reference.dst_covered[j];
                        }
                        bool ok = eligible > 0 && std::isfinite(squared) && constant_error < 1e-9;
                        std::cout << method << ',' << field << ',' << masked << ','
                                  << (area > 0 ? std::sqrt(squared / area) : NAN) << ','
                                  << constant_error << ',' << mapped << ',' << eligible << ','
                                  << (ok ? "pass" : "fail") << '\n';
                        if (!ok)
                            status = 1;
                    } catch (const std::exception &e) {
                        std::cerr << method << ": " << e.what() << '\n';
                        status = 1;
                    }
                }
            }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        status = 1;
    }
    Kokkos::finalize();
    return status;
}
