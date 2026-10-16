// SPDX-License-Identifier: Apache-2.0
// An MPI integration executable: run with 1, 2, 4, or 8 ranks.
#include <Kokkos_Core.hpp>
#include <axis/distributed/regridder.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/vertical_conservative.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <axis/solver/weight_generator.hpp>
#include <cmath>
#include <cstdlib>
#include <halo/indexed_gather_plan.hpp>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {
using Mem = Kokkos::HostSpace;
using Mesh = axis::topology::UnstructuredMesh<Mem>;
using namespace axis::solver;
struct Partition {
    Mesh mesh;
    std::vector<long long> nodes;
    std::vector<int> cells;
};

Partition grid(int nx, int ny, int rank, int size, bool spherical, bool masked, bool source, bool partial = false, bool contiguous = false) {
    Partition p;
    // Interleaved ownership puts geometric neighbors on different ranks.
    // Serial reference uses the same rank-concatenated cell ordering as setup.
    for (int owner = 0; owner < size; ++owner)
        if (rank < 0 || owner == rank)
            for (int c = 0; c < nx * ny; ++c)
                if ((contiguous ? static_cast<int>(static_cast<long long>(c) * size / (nx * ny)) : c % size) == owner) p.cells.push_back(c);
    std::map<long long, axis::index_t> nodes;
    std::vector<axis::index_t> offsets{0}, indices;
    for (int c : p.cells) {
        int i = c % nx, j = c / nx;
        for (long long id : {static_cast<long long>(j * (nx + 1) + i), static_cast<long long>(j * (nx + 1) + i + 1),
                             static_cast<long long>((j + 1) * (nx + 1) + i + 1), static_cast<long long>((j + 1) * (nx + 1) + i)}) {
            auto [it, added] = nodes.emplace(id, static_cast<axis::index_t>(nodes.size()));
            if (added) p.nodes.push_back(id);
            indices.push_back(it->second);
        }
        offsets.push_back(indices.size());
    }
    Kokkos::View<double **, Kokkos::LayoutLeft, Mem> coords("grid_nodes", nodes.size(), 2);
    for (auto [id, local] : nodes) {
        double x = 4.0 * (id % (nx + 1)) / nx;
        double y = 3.0 * (id / (nx + 1)) / ny;
        if (partial && !source) {
            x += 0.2;
            y += 0.13;
        }
        coords(local, 0) = x;
        coords(local, 1) = spherical ? y + 65 : y;
    }
    Kokkos::View<axis::index_t *, Mem> off("offsets", offsets.size()), ind("indices", indices.size());
    Kokkos::View<int *, Mem> mask("mask", p.cells.size());
    for (std::size_t k = 0; k < offsets.size(); ++k) off(k) = offsets[k];
    for (std::size_t k = 0; k < indices.size(); ++k) ind(k) = indices[k];
    for (std::size_t k = 0; k < p.cells.size(); ++k) mask(k) = masked && p.cells[k] % (source ? 13 : 17) == 0 ? 0 : 1;
    axis::topology::GeometryMetadata g;
    g.boundary_model = spherical ? axis::topology::BoundaryModel::GreatCircle : axis::topology::BoundaryModel::CartesianStraight;
    p.mesh = Mesh(coords, off, ind, spherical ? axis::topology::CoordinateSystem::SphericalDeg : axis::topology::CoordinateSystem::Cartesian3D, {},
                  mask, g);
    return p;
}
std::vector<double> field(const Partition &p, int nx, int ny, int which) {
    std::vector<double> q;
    for (int c : p.cells) {
        double x = 4.0 * (c % nx + 0.5) / nx, y = 3.0 * (c / nx + 0.5) / ny;
        q.push_back(which == 0 ? 0.3 + std::sin(0.61 * x) + 0.07 * y * y : which == 1 ? (x + 0.3 * y < 2 ? 0.1 : 1.0) : 2.0);
    }
    return q;
}
void check(bool x, const char *what) {
    if (!x) throw std::runtime_error(what);
}
void compare(const std::vector<double> &a, const std::vector<double> &b, const std::vector<int> &local_cells, const std::vector<int> &global_cells,
             double tolerance = 5e-11) {
    std::map<int, std::size_t> pos;
    for (std::size_t k = 0; k < global_cells.size(); ++k) pos[global_cells[k]] = k;
    for (std::size_t k = 0; k < local_cells.size(); ++k) {
        if (std::abs(a[k] - b[pos.at(local_cells[k])]) > tolerance * (1 + std::abs(b[pos.at(local_cells[k])])))
            std::cerr << "cell " << local_cells[k] << " got " << a[k] << " expected " << b[pos.at(local_cells[k])] << std::endl;
        check(std::isfinite(a[k]) && std::abs(a[k] - b[pos.at(local_cells[k])]) <= tolerance * (1 + std::abs(b[pos.at(local_cells[k])])),
              "distributed field differs from serial reference");
    }
}
void indexed_plan_test(const halo::Communicator &comm) {
    int rank = comm.rank(), size = comm.size();
    std::size_t count = rank == size - 1 && size > 2 ? 0 : 3;
    long long n = count;
    auto counts = halo::allgather(comm, &n, 1);
    long long total = std::accumulate(counts.begin(), counts.end(), 0LL), start = std::accumulate(counts.begin(), counts.begin() + rank, 0LL);
    std::vector<long long> request;
    if (total && rank % 2 == 0) request = {total - 1, 0, total - 1};  // duplicate, remote, and self requests
    halo::Indexed_Gather_Plan plan(comm, count, request);
    std::vector<double> local(count);
    for (std::size_t i = 0; i < count; ++i) local[i] = start + i + 0.125;
    auto result = plan.gather<double>(local);
    for (std::size_t i = 0; i < request.size(); ++i) check(result[i] == request[i] + 0.125, "HALO indexed gather ordering");
    auto sum = plan.reduce(std::vector<double>(request.size(), 1), MPI_SUM, 0);
    int requestors = (size + 1) / 2;
    for (std::size_t i = 0; i < count; ++i) {
        long long id = start + i;
        double expected = requestors * ((id == 0 ? 1 : 0) + (id == total - 1 ? 2 : 0));
        check(sum[i] == expected, "HALO reverse reduction lost duplicate/requestor");
    }
    bool threw = false;
    try {
        std::vector<long long> bad = rank == 0 ? std::vector<long long>{total} : std::vector<long long>{};
        halo::Indexed_Gather_Plan invalid(comm, count, bad);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "invalid owner ID did not fail collectively");
}

void scalar_test(const halo::Communicator &comm, InterpolationMethod method, bool limited, bool spherical, bool masked, NormType norm,
                 bool tiny = false) {
    int rank = comm.rank(), size = comm.size();
    int sx = tiny ? 2 : 7, sy = tiny ? 1 : 6, dx = tiny ? 3 : 9, dy = tiny ? 1 : 8;
    auto src = grid(sx, sy, rank, size, spherical, masked, true);
    auto dst = grid(dx, dy, rank, size, spherical, masked, false, !tiny);
    if (tiny)
        for (auto *p : {&src, &dst})
            if (!p->mesh.n_cells()) p->mesh = Mesh({}, {}, {}, p->mesh.coord_system(), {}, {}, p->mesh.geometry_metadata());
    auto global_src = grid(sx, sy, -1, size, spherical, masked, true);
    auto global_dst = grid(dx, dy, -1, size, spherical, masked, false, !tiny);
    RegridConfig config;
    config.method = method;
    config.use_structured_fast_paths = false;
    config.use_limiter = limited;
    config.norm_type = norm;
    config.extrap_method = ExtrapolationAction::None;
    config.line_type = spherical ? LineType::GreatCircle : LineType::Cartesian;
    auto serial = WeightGenerator::generate(global_src.mesh, global_dst.mesh, config);
    serial.to_csr();
    axis::distributed::DistributedRegridder parallel(src.mesh, dst.mesh, config, comm.handle(), src.nodes);
    check(parallel.local_destination_size() == dst.cells.size(), "matrix rows are not local");
    check(parallel.local_matrix().n_src() <= global_src.cells.size(), "source dependency closure too large");
    for (int which = 0; which < 3; ++which) {
        auto q = field(src, sx, sy, which), gq = field(global_src, sx, sy, which);
        halo::Indexed_Gather_Plan verify(comm, q.size(), parallel.requested_source_ids());
        auto gathered = verify.gather<double>(q);
        for (std::size_t i = 0; i < gathered.size(); ++i)
            check(gathered[i] == gq[parallel.requested_source_ids()[i]], "gather field permutation incorrect");
        std::vector<double> actual(dst.cells.size(), -99), expected(global_dst.cells.size());
        parallel.apply(axis::field_view<const double, 1>(q.data(), q.size()), axis::field_view<double, 1>(actual.data(), actual.size()));
        axis::solver::apply(serial, axis::field_view<const double, 1>(gq.data(), gq.size()),
                            axis::field_view<double, 1>(expected.data(), expected.size()));
        compare(actual, expected, dst.cells, global_dst.cells);
    }
    // Batched fields are compared to independent scalar serial evaluations.
    std::vector<double> batch(src.cells.size() * 2), out(dst.cells.size() * 2);
    for (int f = 0; f < 2; ++f) {
        auto q = field(src, sx, sy, f);
        std::copy(q.begin(), q.end(), batch.begin() + f * q.size());
    }
    parallel.apply(axis::field_view<const double, 2>(batch.data(), src.cells.size(), 2),
                   axis::field_view<double, 2>(out.data(), dst.cells.size(), 2));
    for (int f = 0; f < 2; ++f) {
        auto q = field(global_src, sx, sy, f);
        std::vector<double> expected(global_dst.cells.size());
        axis::solver::apply(serial, axis::field_view<const double, 1>(q.data(), q.size()),
                            axis::field_view<double, 1>(expected.data(), expected.size()));
        compare(std::vector<double>(out.begin() + f * dst.cells.size(), out.begin() + (f + 1) * dst.cells.size()), expected, dst.cells,
                global_dst.cells);
    }
    // A common-frame constant vector must rotate identically across partitions.
    std::vector<double> u(src.cells.size()), v(u.size()), sa(u.size()), da(dst.cells.size()), du(da.size()), dv(da.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
        sa[i] = 0.01 * src.cells[i];
        u[i] = std::cos(sa[i]) * 2 + std::sin(sa[i]) * 3;
        v[i] = -std::sin(sa[i]) * 2 + std::cos(sa[i]) * 3;
    }
    for (std::size_t j = 0; j < da.size(); ++j) da[j] = -0.02 * dst.cells[j];
    parallel.apply_vector(axis::field_view<const double, 1>(u.data(), u.size()), axis::field_view<const double, 1>(v.data(), v.size()),
                          axis::field_view<const double, 1>(sa.data(), sa.size()), axis::field_view<const double, 1>(da.data(), da.size()),
                          axis::field_view<double, 1>(du.data(), du.size()), axis::field_view<double, 1>(dv.data(), dv.size()));
    std::vector<double> two(global_src.cells.size(), 2), three(two.size(), 3), e(global_dst.cells.size()), n(e.size());
    axis::solver::apply(serial, axis::field_view<const double, 1>(two.data(), two.size()), axis::field_view<double, 1>(e.data(), e.size()));
    axis::solver::apply(serial, axis::field_view<const double, 1>(three.data(), three.size()), axis::field_view<double, 1>(n.data(), n.size()));
    std::map<int, std::size_t> pos;
    for (std::size_t j = 0; j < global_dst.cells.size(); ++j) pos[global_dst.cells[j]] = j;
    for (std::size_t j = 0; j < du.size(); ++j) {
        auto k = pos.at(dst.cells[j]);
        check(std::abs(du[j] - (std::cos(da[j]) * e[k] + std::sin(da[j]) * n[k])) < 1e-10, "vector u frame exchange");
        check(std::abs(dv[j] - (-std::sin(da[j]) * e[k] + std::cos(da[j]) * n[k])) < 1e-10, "vector v frame exchange");
    }
    // Distributed covered fractions must include contributions on other ranks.
    auto owned_fracs = parallel.owned_source_fractions();
    std::map<int, std::size_t> source_pos;
    for (std::size_t i = 0; i < global_src.cells.size(); ++i) source_pos[global_src.cells[i]] = i;
    if (method == InterpolationMethod::Conservative1stOrder || method == InterpolationMethod::Conservative2ndOrder)
        for (std::size_t i = 0; i < src.cells.size(); ++i)
            check(std::abs(owned_fracs[i] - serial.frac_a_view()(source_pos[src.cells[i]])) < 1e-10, "source coverage is only rank-local");
    bool threw = false;
    try {
        auto q = field(src, sx, sy, 0);
        std::vector<double> d(dst.cells.size());
        parallel.apply(axis::field_view<const double, 1>(q.data(), q.size() + (rank == 0 ? 1 : 0)), axis::field_view<double, 1>(d.data(), d.size()));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "bad extent did not fail collectively");
    // One invalid destination-angle/output alias must fail on every rank.
    threw = false;
    try {
        const double *angles = rank == 0 && !du.empty() ? du.data() : da.data();
        parallel.apply_vector(axis::field_view<const double, 1>(u.data(), u.size()), axis::field_view<const double, 1>(v.data(), v.size()),
                              axis::field_view<const double, 1>(sa.data(), sa.size()), axis::field_view<const double, 1>(angles, da.size()),
                              axis::field_view<double, 1>(du.data(), du.size()), axis::field_view<double, 1>(dv.data(), dv.size()));
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "vector output alias did not fail collectively");
}
void vertical_test(const halo::Communicator &comm) {
    constexpr int total = 5, ns = 6, nd = 8;
    std::vector<int> columns;
    for (int c = 0; c < total; ++c)
        if (c % comm.size() == comm.rank()) columns.push_back(c);
    double source_stock = 0, destination_stock = 0;
    // Whole columns need no halo. Empty partitions skip local vertical setup.
    if (!columns.empty()) {
        using V = VerticalConservativeRegridder<Mem>;
        V::Field si("si", columns.size(), ns + 1), di("di", columns.size(), nd + 1), q("q", columns.size(), ns), out("out", columns.size(), nd);
        Kokkos::View<int **, Mem> mask("vertical_mask", columns.size(), ns);
        for (std::size_t c = 0; c < columns.size(); ++c) {
            double scale = 1 + .1 * columns[c];
            for (int i = 0; i <= ns; ++i) si(c, i) = scale * 3.0 * (ns - i) / ns;
            for (int j = 0; j <= nd; ++j) di(c, j) = scale * 3.0 * j / nd;
            for (int i = 0; i < ns; ++i) {
                double a = si(c, i + 1), b = si(c, i);
                q(c, i) = 1 + .2 * (a + b) + .13 * (a * a + a * b + b * b) / 3;
                mask(c, i) = columns[c] == 2 && i == 3 ? 0 : 1;
            }
        }
        V op(si, di, columns.size(), mask);
        for (auto order : {VerticalConservativeOrder::First, VerticalConservativeOrder::LimitedLinear}) {
            op.apply(q, out, order, VerticalConservativeNorm::CoveredMeasure);
            auto stock = op.covered_source_integral(q, order);
            for (std::size_t c = 0; c < columns.size(); ++c) {
                source_stock += stock(c);
                for (int j = 0; j < nd; ++j) destination_stock += out(c, j) * op.destination_covered_measure()(c, j);
            }
        }
        // Point-profile interpolation remains distinct from layer-mean remapping.
        Kokkos::View<double **, Mem> profile("profile", columns.size(), 4), result("profile_result", columns.size(), 4);
        Kokkos::View<double *, Mem> z("z", 4), target("target", 4);
        for (int i = 0; i < 4; ++i) z(i) = 3 - i;
        target(0) = 2.75;
        target(1) = 2.1;
        target(2) = 1.2;
        target(3) = .25;
        for (std::size_t c = 0; c < columns.size(); ++c)
            for (int i = 0; i < 4; ++i) profile(c, i) = 2 * z(i) + columns[c];
        VerticalRegridder<Mem>::interpolate(profile, result, z, target);
        for (std::size_t c = 0; c < columns.size(); ++c)
            for (int i = 0; i < 4; ++i)
                check(std::abs(result(c, i) - (2 * target(i) + columns[c])) < 1e-12, "distributed whole-column point profile");
    }
    auto stock = halo::allreduce(comm, std::vector<double>{source_stock, destination_stock}, MPI_SUM);
    check(std::abs(stock[0] - stock[1]) < 1e-11, "distributed vertical tracer inventory");
}

}  // namespace

namespace {
void seam_and_extrapolation_test(const halo::Communicator &comm) {
    using axis::distributed::DistributedRegridder;
    auto source = grid(7, 6, comm.rank(), comm.size(), true, true, true);
    auto target = grid(9, 8, comm.rank(), comm.size(), true, false, false, true);
    auto all_source = grid(7, 6, -1, comm.size(), true, true, true);
    auto all_target = grid(9, 8, -1, comm.size(), true, false, false, true);
    auto seam = [](Partition &p) {
        auto xy = p.mesh.node_coords_view();
        for (std::size_t i = 0; i < p.mesh.n_nodes(); ++i) xy(i, 0) = std::fmod(xy(i, 0) + 358.0, 360.0);
        auto g = p.mesh.geometry_metadata();
        g.boundary_model = axis::topology::BoundaryModel::ConstantLatitude;
        p.mesh = Mesh(xy, p.mesh.conn_offsets_view(), p.mesh.conn_indices_view(), p.mesh.coord_system(), p.mesh.cell_areas_view(),
                      p.mesh.cell_mask_view(), g);
    };
    seam(source);
    seam(target);
    seam(all_source);
    seam(all_target);
    RegridConfig config;
    config.method = InterpolationMethod::Conservative1stOrder;
    config.line_type = LineType::GreatCircle;
    config.norm_type = NormType::FracArea;
    config.extrap_method = ExtrapolationAction::None;
    config.use_structured_fast_paths = false;
    auto serial = WeightGenerator::generate(all_source.mesh, all_target.mesh, config);
    axis::distributed::DistributedOptions small_packets;
    small_packets.max_message_bytes = 4096;
    DistributedRegridder op(source.mesh, target.mesh, config, comm.handle(), source.nodes, small_packets);
    check(op.geometry_statistics().peak_received_message_bytes <= 4096, "fragmented geometry message exceeded budget");
    auto q = field(source, 7, 6, 0), global = field(all_source, 7, 6, 0);
    std::vector<double> d(target.cells.size()), reference(all_target.cells.size());
    op.apply(axis::field_view<const double, 1>(q.data(), q.size()), axis::field_view<double, 1>(d.data(), d.size()));
    axis::solver::apply(serial, axis::field_view<const double, 1>(global.data(), global.size()),
                        axis::field_view<double, 1>(reference.data(), reference.size()));
    compare(d, reference, target.cells, all_target.cells);
    // A displaced destination needs a remote wet nearest search, not just
    // intersections with nearby partitions. Use the existing raw-XY convention.
    source = grid(7, 6, comm.rank(), comm.size(), false, true, true);
    target = grid(9, 8, comm.rank(), comm.size(), false, false, false, true);
    all_source = grid(7, 6, -1, comm.size(), false, true, true);
    all_target = grid(9, 8, -1, comm.size(), false, false, false, true);
    for (auto *p : {&target, &all_target}) {
        auto xy = p->mesh.node_coords_view();
        for (std::size_t i = 0; i < p->mesh.n_nodes(); ++i) xy(i, 0) += 20;
    }
    config.line_type = LineType::Cartesian;
    config.extrap_method = ExtrapolationAction::NearestWet;
    serial = WeightGenerator::generate(all_source.mesh, all_target.mesh, config);
    DistributedRegridder extrap(source.mesh, target.mesh, config, comm.handle(), source.nodes);
    q = field(source, 7, 6, 0);
    global = field(all_source, 7, 6, 0);
    extrap.apply(axis::field_view<const double, 1>(q.data(), q.size()), axis::field_view<double, 1>(d.data(), d.size()));
    axis::solver::apply(serial, axis::field_view<const double, 1>(global.data(), global.size()),
                        axis::field_view<double, 1>(reference.data(), reference.size()));
    compare(d, reference, target.cells, all_target.cells);
}
void memory_and_reference_test(const halo::Communicator &comm) {
    using axis::distributed::DistributedOptions;
    using axis::distributed::DistributedRegridder;
    constexpr int nx = 128, ny = 64;
    auto source = grid(nx, ny, comm.rank(), comm.size(), false, false, true, false, true);
    auto destination = grid(nx, ny, comm.rank(), comm.size(), false, false, false, true, true);
    RegridConfig config;
    config.method = InterpolationMethod::Conservative2ndOrder;
    config.line_type = LineType::Cartesian;
    config.norm_type = NormType::FracArea;
    config.extrap_method = ExtrapolationAction::None;
    config.use_limiter = true;
    DistributedOptions bounded;
    bounded.max_message_bytes = 64 * 1024;
    DistributedRegridder op(source.mesh, destination.mesh, config, comm.handle(), source.nodes, bounded);
    auto &stats = op.geometry_statistics();
    check(stats.peak_received_message_bytes <= 8ULL * 1024 * 1024, "geometry payload exceeds limit");
    if (comm.size() >= 4) check(stats.imported_cells < static_cast<std::size_t>(nx * ny) / 2, "selective setup retained global geometry");
    std::vector<double> q(source.cells.size(), 1), d(destination.cells.size());
    op.apply(axis::field_view<const double, 1>(q.data(), q.size()), axis::field_view<double, 1>(d.data(), d.size()));
    for (std::size_t j = 0; j < d.size(); ++j)
        check(std::abs(d[j] - (op.local_matrix().frac_b_view()(j) > 0 ? 1.0 : 0.0)) < 1e-10, "large partition constant preservation");
    // Explicit small replicated-reference strategy must agree with selective.
    auto small_source = grid(7, 6, comm.rank(), comm.size(), false, true, true);
    auto small_destination = grid(9, 8, comm.rank(), comm.size(), false, true, false, true);
    DistributedOptions reference;
    reference.geometry = axis::distributed::GeometryStrategy::ReplicatedReference;
    DistributedRegridder a(small_source.mesh, small_destination.mesh, config, comm.handle(), small_source.nodes);
    DistributedRegridder b(small_source.mesh, small_destination.mesh, config, comm.handle(), small_source.nodes, reference);
    auto input = field(small_source, 7, 6, 1);
    std::vector<double> da(small_destination.cells.size()), db(da.size());
    a.apply(axis::field_view<const double, 1>(input.data(), input.size()), axis::field_view<double, 1>(da.data(), da.size()));
    b.apply(axis::field_view<const double, 1>(input.data(), input.size()), axis::field_view<double, 1>(db.data(), db.size()));
    for (std::size_t j = 0; j < da.size(); ++j) check(std::abs(da[j] - db[j]) < 1e-10, "selective/reference strategy mismatch");
    DistributedOptions tiny;
    tiny.max_import_bytes = 512;
    bool threw = false;
    try {
        DistributedRegridder bad(small_source.mesh, small_destination.mesh, config, comm.handle(), small_source.nodes, tiny);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    check(threw, "geometry budget failed to stop setup collectively");
    auto maxima =
        halo::allreduce<double>(comm, {static_cast<double>(stats.imported_cells), static_cast<double>(stats.estimated_import_bytes)}, MPI_MAX);
    if (comm.rank() == 0)
        std::cout << "Selective memory check: " << nx * ny << " global cells, " << maxima[0] << " maximum rank closure cells, " << maxima[1]
                  << " estimated import bytes" << std::endl;
}
}  // namespace
int distributed_test_suite() {
    halo::Communicator comm(MPI_COMM_WORLD);
    int threads = Kokkos::DefaultHostExecutionSpace().concurrency();
    auto thread_min = halo::allreduce<int>(comm, {threads}, MPI_MIN)[0];
    auto thread_max = halo::allreduce<int>(comm, {threads}, MPI_MAX)[0];
    if (const char *requested = std::getenv("OMP_NUM_THREADS"))
        check(thread_min == std::stoi(requested) && thread_max == std::stoi(requested), "requested OpenMP thread count was not used");
    if (comm.rank() == 0) std::cout << "Actual OpenMP/Kokkos threads per process: " << thread_min << ".." << thread_max << std::endl;
    indexed_plan_test(comm);
    vertical_test(comm);
    memory_and_reference_test(comm);
    seam_and_extrapolation_test(comm);
    for (bool sphere : {false, true})
        for (bool mask : {false, true})
            for (auto method : {InterpolationMethod::NearestNeighbor, InterpolationMethod::Bilinear, InterpolationMethod::Bicubic,
                                InterpolationMethod::Patch, InterpolationMethod::Conservative1stOrder, InterpolationMethod::Conservative2ndOrder}) {
                scalar_test(comm, method, false, sphere, mask, NormType::DstArea);
                if (method == InterpolationMethod::Conservative1stOrder || method == InterpolationMethod::Conservative2ndOrder)
                    scalar_test(comm, method, false, sphere, mask, NormType::FracArea);
                if (method == InterpolationMethod::Conservative2ndOrder) {
                    scalar_test(comm, method, true, sphere, mask, NormType::DstArea);
                    scalar_test(comm, method, true, sphere, mask, NormType::FracArea);
                }
            }
    scalar_test(comm, InterpolationMethod::Conservative1stOrder, false, false, false, NormType::DstArea, true);
    scalar_test(comm, InterpolationMethod::Conservative2ndOrder, true, false, false, NormType::DstArea, true);
    if (comm.rank() == 0)
        std::cout << "PASS: HALO indexed exchanges and 42 distributed configurations; scalar/batch/vector/reuse/coverage/collective errors"
                  << std::endl;
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
    halo::Environment::initialize();
    int result = 0;
    try {
        result = distributed_test_suite();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
        result = 1;
    }
    Kokkos::finalize();
    MPI_Finalize();
    return result;
}
#endif
