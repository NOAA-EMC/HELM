// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <axis/solver/vertical_conservative.hpp>
// Included by axis_py.cpp after nanobind aliases and ensure_kokkos().
namespace {
using HostVerticalConservative = axis::solver::VerticalConservativeRegridder<Kokkos::HostSpace>;
using VerticalArray = nb::ndarray<const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using VerticalMaskArray = nb::ndarray<const int, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
inline auto vertical_order(const std::string &method) {
    using O = axis::solver::VerticalConservativeOrder;
    if (method == "first") return O::First;
    if (method == "limited_linear") return O::LimitedLinear;
    throw std::invalid_argument("method must be first or limited_linear");
}
inline auto vertical_norm(const std::string &norm) {
    using N = axis::solver::VerticalConservativeNorm;
    if (norm == "dstarea") return N::DestinationMeasure;
    if (norm == "fracarea") return N::CoveredMeasure;
    throw std::invalid_argument("norm must be dstarea or fracarea");
}
template <class View>
auto vertical_numpy(View view) {
    auto data = std::make_unique<double[]>(view.size());
    std::size_t shape[2] = {view.extent(0), view.extent(1)};
    if constexpr (View::rank == 1)
        for (std::size_t i = 0; i < shape[0]; ++i) data[i] = view(i);
    else
        for (std::size_t c = 0; c < shape[0]; ++c)
            for (std::size_t i = 0; i < shape[1]; ++i) data[c * shape[1] + i] = view(c, i);
    auto ptr = data.release();
    nb::capsule owner(ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return nb::ndarray<nb::numpy, double>(ptr, View::rank, shape, std::move(owner));
}
inline HostVerticalConservative::Field vertical_workspace(nb::object work) {
    if (work.is_none()) return {};
    auto a = nb::cast<nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu>>(work);
    return HostVerticalConservative::Field(a.data(), a.shape(0), a.shape(1));
}
inline void bind_vertical_conservative(nb::module_ &m) {
    nb::class_<HostVerticalConservative>(m, "VerticalConservativeOperator")
        .def(
            "apply",
            [](const HostVerticalConservative &op, VerticalArray input, const std::string &method, const std::string &norm, nb::object workspace) {
                HostVerticalConservative::ConstField src(input.data(), input.shape(0), input.shape(1));
                HostVerticalConservative::Field dst("vertical_python_output", op.columns(), op.destination_layers());
                op.apply(src, dst, vertical_order(method), vertical_norm(norm), vertical_workspace(workspace));
                return vertical_numpy(dst);
            },
            "source"_a, "method"_a = "first", "norm"_a = "dstarea", "workspace"_a = nb::none())
        .def(
            "covered_source_integral",
            [](const HostVerticalConservative &op, VerticalArray input, const std::string &method, nb::object workspace) {
                HostVerticalConservative::ConstField src(input.data(), input.shape(0), input.shape(1));
                return vertical_numpy(op.covered_source_integral(src, vertical_order(method), vertical_workspace(workspace)));
            },
            "source"_a, "method"_a = "first", "workspace"_a = nb::none())
        .def_prop_ro("source_coverage", [](const HostVerticalConservative &op) { return vertical_numpy(op.source_coverage()); })
        .def_prop_ro("destination_coverage", [](const HostVerticalConservative &op) { return vertical_numpy(op.destination_coverage()); })
        .def_prop_ro("destination_covered_measure",
                     [](const HostVerticalConservative &op) { return vertical_numpy(op.destination_covered_measure()); });
    m.def(
        "make_vertical_conservative_operator",
        [](VerticalArray source, VerticalArray destination, std::size_t columns, VerticalMaskArray source_mask, VerticalMaskArray destination_mask) {
            ensure_kokkos();
            HostVerticalConservative::ConstField src(source.data(), source.shape(0), source.shape(1)),
                dst(destination.data(), destination.shape(0), destination.shape(1));
            HostVerticalConservative::Mask sm(source_mask.data(), source_mask.shape(0), source_mask.shape(1)),
                dm(destination_mask.data(), destination_mask.shape(0), destination_mask.shape(1));
            return HostVerticalConservative(src, dst, columns, sm, dm);
        },
        "source_interfaces"_a, "destination_interfaces"_a, "columns"_a, "source_mask"_a, "destination_mask"_a);
}
}  // namespace
