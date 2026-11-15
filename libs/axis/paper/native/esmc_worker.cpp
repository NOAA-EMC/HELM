// SPDX-License-Identifier: Apache-2.0
// Interfaces verified against the OMD ESMF 8.9.1 C reference manual.
#include <ESMC.h>
#include <mpi.h>

#include <algorithm>
#include <numeric>

#include "case_io.hpp"
#include "provenance.hpp"
#include "timing.hpp"
using namespace paper;
static void check_esmc(int rc) {
    if (rc != ESMF_SUCCESS)
        throw Error(5, "ESMC error " + std::to_string(rc));
}
struct NativeMesh {
    ESMC_Mesh value{};
    bool live = false;
    NativeMesh(const Mesh &m) {
        int rc;
        auto coord = ESMC_COORDSYS_SPH_DEG;
        value = ESMC_MeshCreate(2, 2, &coord, &rc);
        check_esmc(rc);
        live = true;
        try {
            std::vector<int> nodes(m.lon.size()), owners(m.lon.size(), 0), types(m.size()),
                conn = m.indices, ids(m.size());
            std::vector<double> xy(2 * m.lon.size()), centers(2 * m.size());
            for (size_t i = 0; i < nodes.size(); ++i) {
                nodes[i] = int(i + 1);
                xy[2 * i] = m.lon[i];
                xy[2 * i + 1] = m.lat[i];
            }
            for (size_t i = 0; i < m.size(); ++i) {
                int n = m.offsets[i + 1] - m.offsets[i];
                if (n != 3 && n != 4)
                    throw Error(
                        4,
                        "ESMC adapter accepts triangles/quads; prepare a shared triangulated case");
                types[i] = n;
                ids[i] = int(i + 1);
                centers[2 * i] = m.center_lon[i];
                centers[2 * i + 1] = m.center_lat[i];
            }
            for (auto &v : conn)
                ++v;
            check_esmc(ESMC_MeshAddNodes(value, nodes.size(), nodes.data(), xy.data(),
                                         owners.data(), nullptr));
            check_esmc(ESMC_MeshAddElements(value, m.size(), ids.data(), types.data(), conn.data(),
                                            const_cast<int *>(m.mask.data()), nullptr,
                                            centers.data()));
        } catch (...) {
            ESMC_MeshDestroy(&value);
            live = false;
            throw;
        }
    }
    ~NativeMesh() {
        if (live)
            ESMC_MeshDestroy(&value);
    }
    NativeMesh(const NativeMesh &) = delete;
};
struct NativeField {
    ESMC_Field value{};
    bool live = false;
    NativeField(ESMC_Mesh m) {
        int rc;
        value = ESMC_FieldCreateMeshTypeKind(m, ESMC_TYPEKIND_R8, ESMC_MESHLOC_ELEMENT, nullptr,
                                             nullptr, nullptr, nullptr, &rc);
        check_esmc(rc);
        live = true;
    }
    ~NativeField() {
        if (live)
            ESMC_FieldDestroy(&value);
    }
    double *data() {
        int rc;
        auto p = static_cast<double *>(ESMC_FieldGetPtr(value, 0, &rc));
        check_esmc(rc);
        return p;
    }
    NativeField(const NativeField &) = delete;
};
struct Route {
    ESMC_RouteHandle value{};
    bool live = false;
    void release() {
        if (live) {
            check_esmc(ESMC_FieldRegridRelease(&value));
            live = false;
        }
    }
    ~Route() {
        if (live)
            ESMC_FieldRegridRelease(&value);
    }
};
int main(int argc, char **argv) {
    return guarded([&] {
        Args arguments(argc, argv);
        auto output_directory = run_directory(arguments);
        int rc;
        check_esmc(ESMC_Initialize(&rc, ESMC_InitArgLogKindFlag(ESMC_LOGKIND_NONE), ESMC_ArgLast));
        check_esmc(rc);
        try {
            int process_count = 0;
            if (MPI_Comm_size(MPI_COMM_WORLD, &process_count) != MPI_SUCCESS || process_count != 1)
                throw Error(4, "Paper ESMC worker requires exactly one MPI process");
            char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING]{};
            int mpi_version_length = 0;
            if (MPI_Get_library_version(mpi_version, &mpi_version_length) != MPI_SUCCESS)
                throw Error(5, "Could not query native MPI library version");
            auto case_data = read_case(arguments.require("case"));
            auto reference = read_reference(arguments.require("reference"));
            if (case_data.geometry != "great_circle")
                throw Error(4, "ESMC great-circle edges do not match constant-latitude rectangles");
            if (arguments.get("mode", "quality") != "quality" && arguments.get("mode") != "timing")
                throw Error(2, "Invalid --mode");
            bool timing = arguments.get("mode", "quality") == "timing";
            int repetitions = arguments.count("repetitions", 3),
                warmups = arguments.count("warmups", 1);
            auto reuse_sizes = reuse_counts(arguments);
            auto fence = []() {};
            std::vector<Observation> timing_observations;
            ObservationJournal journal(output_directory / "observations.csv", timing_observations);
            // Single PET, synchronous ESMC operations. MPI ownership belongs to ESMF.
            auto prep_start = Clock::now();
            NativeMesh source_mesh(case_data.src), destination_mesh(case_data.dst);
            NativeField src(source_mesh.value), dst(destination_mesh.value),
                source_fraction(source_mesh.value), destination_fraction(destination_mesh.value),
                source_area(source_mesh.value), destination_area(destination_mesh.value);
            std::copy(reference.src.begin(), reference.src.end(), src.data());
            check_esmc(ESMC_FieldRegridGetArea(source_area.value));
            check_esmc(ESMC_FieldRegridGetArea(destination_area.value));
            if (timing)
                timing_observations.push_back(
                    {0, "preparation", 1,
                     std::chrono::duration<double>(Clock::now() - prep_start).count()});
            int masked = 0;
            ESMC_InterArrayInt mask;
            check_esmc(ESMC_InterArrayIntSet(&mask, &masked, 1));
            auto method = ESMC_REGRIDMETHOD_CONSERVE;
            auto pole = ESMC_POLEMETHOD_NONE;
            auto line = ESMC_LINETYPE_GREAT_CIRCLE;
            auto norm =
                case_data.norm == "fracarea" ? ESMC_NORMTYPE_FRACAREA : ESMC_NORMTYPE_DSTAREA;
            auto vector = ESMF_FALSE;
            auto extrap = ESMC_EXTRAPMETHOD_NONE;
            auto unmapped = ESMC_UNMAPPEDACTION_IGNORE;
            auto degenerate = ESMF_FALSE;
            auto zero = ESMC_REGION_TOTAL;
            Route route;
            auto apply = [&] {
                check_esmc(ESMC_FieldRegrid(src.value, dst.value, route.value, &zero));
            };
            for (int r = -warmups; r < (timing ? repetitions : 1); ++r) {
                route.release();
                double t = elapsed(
                    [&] {
                        check_esmc(ESMC_FieldRegridStore(
                            src.value, dst.value, &mask, &mask, &route.value, &method, &pole,
                            nullptr, &line, &norm, &vector, &extrap, nullptr, nullptr, nullptr,
                            &unmapped, &degenerate, nullptr, nullptr, nullptr,
                            &source_fraction.value, &destination_fraction.value));
                        route.live = true;
                    },
                    fence);
                double final = elapsed(apply, fence);
                if (timing && r >= 0) {
                    timing_observations.push_back({r, "generate", 1, t});
                    timing_observations.push_back({r, "operator_finalize", 1, final});
                    // Sum native intervals for this freshly constructed operator.
                    // Cleanup, reference work and I/O remain outside the total.
                    for (int applications : reuse_sizes) {
                        double reuse_seconds = elapsed(
                            [&] {
                                for (int iteration = 0; iteration < applications; ++iteration)
                                    apply();
                            },
                            fence);
                        timing_observations.push_back(
                            {r, "setup_and_reuse", applications, t + final + reuse_seconds});
                    }
                }
            }
            measure_reused_operator(apply, fence, timing, warmups, repetitions, reuse_sizes,
                                    timing_observations);
            Output out;
            out.src = reference.src;
            auto get = [](NativeField &f, size_t n) {
                auto p = f.data();
                return std::vector<double>(p, p + n);
            };
            out.dst = get(dst, case_data.dst.size());
            out.src_area = get(source_area, case_data.src.size());
            out.dst_area = get(destination_area, case_data.dst.size());
            out.src_frac = get(source_fraction, case_data.src.size());
            out.dst_frac = get(destination_fraction, case_data.dst.size());
            std::fill(src.data(), src.data() + case_data.src.size(), 1.);
            apply();
            out.row_sum = get(dst, case_data.dst.size());
            write_output(output_directory / "fields.nc", out);
            observations(output_directory / "observations.csv", timing_observations);
            atomic_text(
                output_directory / "native.json",
                "{\"schema_version\":1,\"status\":\"complete\",\"engine\":\"esmc\",\"build\":" +
                    build_provenance() + ",\"esmf_version\":" + quote(PAPER_ESMF_VERSION) +
                    ",\"mpi_library_version\":" +
                    quote(std::string(mpi_version, mpi_version_length)) +
                    ",\"backend\":\"ESMF native single PET\",\"mode\":" +
                    quote(timing ? "timing" : "quality") +
                    ",\"instrumentation\":\"disabled\",\"algorithm\":\"ESMC_CONSERVE_GREAT_"
                    "CIRCLE\",\"checksum\":" +
                    number(std::accumulate(out.dst.begin(), out.dst.end(), 0.)) +
                    ",\"diagnostics\":{\"candidates\":null,\"invalid\":null,\"discarded\":null,"
                    "\"availability_reason\":\"ESMC does not expose overlap rejection "
                    "counters\"}}");
        } catch (...) {
            ESMC_Finalize();
            throw;
        }
        check_esmc(ESMC_Finalize());
    });
}
