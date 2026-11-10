// SPDX-License-Identifier: Apache-2.0
#include <KokkosKernels_config.h>

#include <ArborX_Version.hpp>
#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/paper_diagnostics.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <numeric>

#include "case_io.hpp"
#include "provenance.hpp"
#include "timing.hpp"
using namespace paper;
static auto make_axis_mesh(const Mesh &m, const std::string &geometry) {
    using namespace axis::topology;
    Kokkos::View<double **, Kokkos::LayoutLeft, Kokkos::HostSpace> nodes("paper_nodes",
                                                                         m.lon.size(), 2);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> offsets("paper_offsets", m.offsets.size()),
        indices("paper_indices", m.indices.size());
    Kokkos::View<int *, Kokkos::HostSpace> mask("paper_mask", m.mask.size());
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
    // Constant-latitude metadata selects the strip solver, whose public mesh
    // contract expects supplied spherical areas. Without them its generic
    // fallback computes planar degree-squared areas.
    Kokkos::View<double *, Kokkos::HostSpace> areas;
    if (geometry == "constant_latitude") {
        areas = decltype(areas)("paper_strip_areas", m.area.size());
        for (size_t cell = 0; cell < m.area.size(); ++cell)
            areas(cell) = m.area[cell];
    }
    GeometryMetadata g;
    g.provenance = GeometryProvenance::SourceAuthoritative;
    g.boundary_model = geometry == "constant_latitude" ? BoundaryModel::ConstantLatitude
                                                       : BoundaryModel::GreatCircle;
    g.area_model = geometry == "constant_latitude" ? AreaModel::ConstantLatitudeStrip
                                                   : AreaModel::SphericalExcess;
    return make_unstructured<Kokkos::HostSpace>(nodes, offsets, indices,
                                                CoordinateSystem::SphericalDeg, areas, mask, g);
}
template <class T>
static std::vector<double> copy_host_view(T x) {
    std::vector<double> v(x.extent(0));
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = x(i);
    return v;
}
int main(int argc, char **argv) {
    return guarded([&] {
        Args arguments(argc, argv);
        auto case_data = read_case(arguments.require("case"));
        auto reference = read_reference(arguments.require("reference"));
        if (arguments.get("mode", "quality") != "quality" && arguments.get("mode") != "timing")
            throw Error(2, "Invalid --mode");
        bool timing = arguments.get("mode", "quality") == "timing";
        int repetitions = arguments.count("repetitions", 3),
            warmups = arguments.count("warmups", 1);
        auto reuse_sizes = reuse_counts(arguments);
        auto output_directory = run_directory(arguments);
        Kokkos::initialize();
        try {
            auto fence = []() { Kokkos::fence(); };
            decltype(make_axis_mesh(case_data.src, case_data.geometry)) src, dst;
            std::vector<Observation> timing_observations;
            ObservationJournal journal(output_directory / "observations.csv", timing_observations);
            double preparation_seconds = elapsed(
                [&] {
                    src = make_axis_mesh(case_data.src, case_data.geometry);
                    dst = make_axis_mesh(case_data.dst, case_data.geometry);
                },
                fence);
            if (timing)
                timing_observations.push_back({0, "preparation", 1, preparation_seconds});
            axis::solver::RegridConfig config;
            config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
            config.norm_type = case_data.norm == "fracarea" ? axis::solver::NormType::FracArea
                                                            : axis::solver::NormType::DstArea;
            config.extrap_method = axis::solver::ExtrapolationAction::None;
            config.unmapped = axis::solver::UnmappedAction::Ignore;
            config.line_type = axis::solver::LineType::GreatCircle;
            axis::solver::PaperDiagnostics diagnostics;
            axis::solver::paper_diagnostics = timing ? nullptr : &diagnostics;
            Output out;
            out.src = reference.src;
            out.dst.resize(case_data.dst.size());
            axis::field_view<const double, 1> source_values(out.src.data(), out.src.size());
            axis::field_view<double, 1> destination_values(out.dst.data(), out.dst.size());
            axis::solver::InterpolationMatrix<Kokkos::HostSpace> weights;
            for (int r = -warmups; r < (timing ? repetitions : 1); ++r) {
                // Release the previous operator before starting the next setup
                // interval, matching ESMC route.release() cleanup semantics.
                weights = decltype(weights){};
                double t = elapsed(
                    [&] { weights = axis::solver::WeightGenerator::generate(src, dst, config); },
                    fence);
                double final = elapsed(
                    [&] {
                        weights.to_csr();
                        axis::solver::apply(weights, source_values, destination_values);
                    },
                    fence);
                if (timing && r >= 0) {
                    timing_observations.push_back({r, "generate", 1, t});
                    timing_observations.push_back({r, "operator_finalize", 1, final});
                    // Sum native intervals for this freshly constructed operator.
                    // Cleanup, reference work and I/O remain outside the total.
                    for (int applications : reuse_sizes) {
                        double reuse_seconds = elapsed(
                            [&] {
                                for (int iteration = 0; iteration < applications; ++iteration)
                                    axis::solver::apply(weights, source_values, destination_values);
                            },
                            fence);
                        timing_observations.push_back(
                            {r, "setup_and_reuse", applications, t + final + reuse_seconds});
                    }
                }
            }
            auto apply = [&] { axis::solver::apply(weights, source_values, destination_values); };
            measure_reused_operator(apply, fence, timing, warmups, repetitions, reuse_sizes,
                                    timing_observations);
            out.src_area = copy_host_view(weights.area_a());
            out.dst_area = copy_host_view(weights.area_b());
            out.src_frac = copy_host_view(weights.frac_a());
            out.dst_frac = copy_host_view(weights.frac_b());
            out.row_sum.assign(case_data.dst.size(), 0);
            auto rows = weights.factor_row();
            auto values = weights.factor_list();
            for (size_t k = 0; k < weights.nnz(); ++k)
                out.row_sum[rows(k)] += values(k);
            write_output(output_directory / "fields.nc", out);
            observations(output_directory / "observations.csv", timing_observations);
            auto count = [](std::int64_t n) { return n < 0 ? "null" : std::to_string(n); };
            atomic_text(
                output_directory / "native.json",
                "{\"schema_version\":1,\"status\":\"complete\",\"engine\":\"axis\",\"build\":" +
                    build_provenance() + ",\"backend\":\"HostSpace\",\"host_concurrency\":" +
                    std::to_string(Kokkos::DefaultHostExecutionSpace().concurrency()) +
                    ",\"kokkos_version\":" + std::to_string(KOKKOS_VERSION) +
                    ",\"kokkoskernels_version\":" + std::to_string(KOKKOSKERNELS_VERSION) +
                    ",\"arborx_version\":" + quote(ArborX::version()) +
                    ",\"arborx_commit\":" + quote(ArborX::gitCommitHash()) +
                    ",\"mode\":" + quote(timing ? "timing" : "quality") + ",\"instrumentation\":" +
                    quote(timing ? "disabled" : "host_optional_counters") + ",\"algorithm\":" +
                    quote(diagnostics.path) + ",\"nnz\":" + std::to_string(weights.nnz()) +
                    ",\"checksum\":" + number(std::accumulate(out.dst.begin(), out.dst.end(), 0.)) +
                    ",\"diagnostics\":{\"candidates\":" + count(diagnostics.candidates) +
                    ",\"invalid\":" + count(diagnostics.invalid) +
                    ",\"zero_overlap\":" + count(diagnostics.zero_overlap) +
                    ",\"discarded\":" + count(diagnostics.discarded) +
                    ",\"availability_reason\":\"Serial spherical, rectangle and strip overlap "
                    "counters are exposed; "
                    "parallel/device paths remain null\"}}");
            axis::solver::paper_diagnostics = nullptr;
        } catch (...) {
            axis::solver::paper_diagnostics = nullptr;
            Kokkos::finalize();
            throw;
        }
        Kokkos::finalize();
    });
}
