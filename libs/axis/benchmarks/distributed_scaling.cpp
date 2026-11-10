// SPDX-License-Identifier: Apache-2.0
#include <sys/resource.h>
#include <unistd.h>

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <axis/distributed/regridder.hpp>
#include <chrono>
#include <cmath>
#include <halo/collectives.hpp>
#include <halo/environment.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <string>

namespace {
using Mem = Kokkos::HostSpace;
struct OwnedMesh {
    axis::topology::UnstructuredMesh<Mem> mesh;
    std::vector<long long> nodes;
    std::vector<std::size_t> cells;
};
OwnedMesh partition(int nx, int ny, int rank, int size, double shift, int x_tile = 0) {
    OwnedMesh p;
    const std::size_t count = static_cast<std::size_t>(nx) * ny;
    const std::size_t begin = count * rank / size, end = count * (rank + 1) / size;
    std::map<long long, axis::index_t> nodes;
    std::vector<axis::index_t> offsets{0}, indices;
    if (x_tile) {
        for (int j = 0; j < ny; ++j)
            for (int i = rank * x_tile; i < (rank + 1) * x_tile; ++i) p.cells.push_back(static_cast<std::size_t>(j) * nx + i);
    } else
        for (std::size_t c = begin; c < end; ++c) p.cells.push_back(c);
    for (std::size_t c : p.cells) {
        long long i = c % nx, j = c / nx;
        for (long long id : {j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i}) {
            auto [it, added] = nodes.emplace(id, static_cast<axis::index_t>(nodes.size()));
            if (added) p.nodes.push_back(id);
            indices.push_back(it->second);
        }
        offsets.push_back(indices.size());
    }
    Kokkos::View<double **, Kokkos::LayoutLeft, Mem> coords("owned_nodes", nodes.size(), 2);
    Kokkos::View<axis::index_t *, Mem> off("owned_offsets", offsets.size()), ind("owned_indices", indices.size());
    for (auto [id, k] : nodes) {
        coords(k, 0) = id % (nx + 1) + shift;
        coords(k, 1) = id / (nx + 1) + shift;
    }
    for (std::size_t k = 0; k < offsets.size(); ++k) off(k) = offsets[k];
    for (std::size_t k = 0; k < indices.size(); ++k) ind(k) = indices[k];
    axis::topology::GeometryMetadata g;
    g.boundary_model = axis::topology::BoundaryModel::CartesianStraight;
    p.mesh = decltype(p.mesh)(coords, off, ind, axis::topology::CoordinateSystem::Cartesian3D, {}, {}, g);
    return p;
}
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
double quantile(std::vector<double> x, double p) {
    std::sort(x.begin(), x.end());
    double k = p * (x.size() - 1);
    auto lo = static_cast<std::size_t>(k), hi = std::min(lo + 1, x.size() - 1);
    return x[lo] + (k - lo) * (x[hi] - x[lo]);
}
}  // namespace

// Positional arguments: nx ny repetitions method fields strong|weak.
// Weak mode treats nx as local x cells, and extends global nx by MPI size.
int distributed_scaling_main(int argc, char **argv) {
    halo::Environment::initialize();
    halo::Communicator comm(MPI_COMM_WORLD);
    int nx = argc > 1 ? std::stoi(argv[1]) : 64, ny = argc > 2 ? std::stoi(argv[2]) : 64, reps = argc > 3 ? std::stoi(argv[3]) : 7;
    std::string name = argc > 4 ? argv[4] : "conservative", mode = argc > 6 ? argv[6] : "strong";
    int fields = argc > 5 ? std::stoi(argv[5]) : 8;
    halo::collective_require(comm, nx > 0 && ny > 0 && reps > 0 && fields > 0 && (mode == "strong" || mode == "weak"),
                             "invalid scaling benchmark arguments");
    halo::collective_require(comm, mode != "weak" || nx <= INT_MAX / comm.size(), "weak-scaling dimensions overflow");
    if (mode == "weak") nx *= comm.size();
    axis::solver::RegridConfig config;
    config.line_type = axis::solver::LineType::Cartesian;
    config.norm_type = axis::solver::NormType::FracArea;
    config.extrap_method = axis::solver::ExtrapolationAction::None;
    using M = axis::solver::InterpolationMethod;
    if (name == "nearest")
        config.method = M::NearestNeighbor;
    else if (name == "bilinear")
        config.method = M::Bilinear;
    else if (name == "bicubic")
        config.method = M::Bicubic;
    else if (name == "patch")
        config.method = M::Patch;
    else if (name == "conservative")
        config.method = M::Conservative1stOrder;
    else if (name == "linear" || name == "limited") {
        config.method = M::Conservative2ndOrder;
        config.use_limiter = name == "limited";
    } else
        throw std::invalid_argument("method must be nearest, bilinear, bicubic, patch, conservative, linear, or limited");
    int x_tile = mode == "weak" ? nx / comm.size() : 0;
    auto source = partition(nx, ny, comm.rank(), comm.size(), 0, x_tile), destination = partition(nx, ny, comm.rank(), comm.size(), .19, x_tile);
    std::vector<double> src(source.cells.size() * fields), dst(destination.cells.size() * fields);
    for (int f = 0; f < fields; ++f)
        for (std::size_t k = 0; k < source.cells.size(); ++k) {
            auto c = source.cells[k];
            src[k + f * source.cells.size()] = f == 0 ? 1 : 1 + std::sin(.01 * (c % nx) + .005 * (c / nx) + f * .1);
        }
    (void)halo::allreduce(comm, std::vector<int>{0}, MPI_SUM);
    double t = now();
    axis::distributed::DistributedRegridder op(source.mesh, destination.mesh, config, comm.handle(), source.nodes);
    double setup = halo::allreduce(comm, std::vector<double>{now() - t}, MPI_MAX)[0];
    auto run = [&] {
        op.apply(axis::field_view<const double, 2>(src.data(), source.cells.size(), fields),
                 axis::field_view<double, 2>(dst.data(), destination.cells.size(), fields));
    };
    run();
    run();  // two untimed warmups
    std::vector<double> samples;
    for (int r = 0; r < reps; ++r) {
        (void)halo::allreduce(comm, std::vector<int>{0}, MPI_SUM);
        t = now();
        run();
        samples.push_back(halo::allreduce(comm, std::vector<double>{now() - t}, MPI_MAX)[0]);
    }
    double constant_error = 0;
    for (std::size_t j = 0; j < destination.cells.size(); ++j) constant_error = std::max(constant_error, std::abs(dst[j] - 1));
    constant_error = halo::allreduce(comm, std::vector<double>{constant_error}, MPI_MAX)[0];
    halo::collective_require(comm, constant_error < 1e-9, "scaling fixture failed constant preservation");
    double entries = halo::allreduce(comm, std::vector<double>{static_cast<double>(op.local_matrix().nnz())}, MPI_SUM)[0];
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
    double peak = halo::allreduce(comm, std::vector<double>{usage.ru_maxrss * 1024.0}, MPI_MAX)[0];
    int threads = Kokkos::DefaultHostExecutionSpace().concurrency();
    int thread_min = halo::allreduce(comm, std::vector<int>{threads}, MPI_MIN)[0],
        thread_max = halo::allreduce(comm, std::vector<int>{threads}, MPI_MAX)[0];
    char host[128]{};
    gethostname(host, sizeof(host) - 1);
    auto hosts = halo::allgather(comm, host, 128);
    std::set<std::string> unique;
    for (int r = 0; r < comm.size(); ++r) unique.insert(std::string(hosts.data() + 128 * r));
    auto &geometry = op.geometry_statistics();
    auto geometry_max = halo::allreduce<double>(comm,
                                                {static_cast<double>(geometry.imported_cells), static_cast<double>(geometry.estimated_import_bytes),
                                                 static_cast<double>(geometry.peak_received_message_bytes)},
                                                MPI_MAX);
    if (comm.rank() == 0)
        std::cout << std::setprecision(12)
                  << "mode,method,geometry_strategy,nodes,ranks,threads_min,threads_max,nx,ny,fields,repetitions,global_nnz,setup_max_rank_s,apply_"
                     "max_rank_median_s,apply_iqr_s,peak_rank_rss_bytes,constant_error,max_rank_geometry_cells,max_rank_estimated_import_bytes,max_"
                     "geometry_message_bytes\n"
                  << mode << ',' << name << ",selective_halo_geometry," << unique.size() << ',' << comm.size() << ',' << thread_min << ','
                  << thread_max << ',' << nx << ',' << ny << ',' << fields << ',' << reps << ',' << entries << ',' << setup << ','
                  << quantile(samples, .5) << ',' << quantile(samples, .75) - quantile(samples, .25) << ',' << peak << ',' << constant_error << ','
                  << geometry_max[0] << ',' << geometry_max[1] << ',' << geometry_max[2] << std::endl;
    return 0;
}
#ifndef AXIS_VIRTUAL_MPI
int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "MPI_THREAD_FUNNELED support required" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    Kokkos::initialize(argc, argv);
    int rc = 0;
    try {
        rc = distributed_scaling_main(argc, argv);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        rc = 1;
    }
    Kokkos::finalize();
    MPI_Finalize();
    return rc;
}
#endif
