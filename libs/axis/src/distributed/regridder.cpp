// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <axis/distributed/regridder.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/weight_generator.hpp>
#include <cmath>
#include <halo/indexed_gather_plan.hpp>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

#include "selective_geometry.hpp"

namespace axis::distributed {
namespace {
using Mem = Kokkos::HostSpace;
using Mesh = topology::UnstructuredMesh<Mem>;
using Matrix = solver::InterpolationMatrix<Mem>;
halo::Communicator isolated(MPI_Comm raw) {
    halo::Environment::initialize();
    return halo::Communicator::duplicate_of(raw);
}
bool overlaps(const double *a, std::size_t an, const double *b, std::size_t bn) {
    if (!an || !bn) return false;
    auto x = reinterpret_cast<std::uintptr_t>(a), y = reinterpret_cast<std::uintptr_t>(b);
    return x < y + bn * sizeof(double) && y < x + an * sizeof(double);
}
template <typename T>
Kokkos::View<T *, Mem> view(const char *name, const std::vector<T> &x) {
    Kokkos::View<T *, Mem> v(name, x.size());
    for (std::size_t k = 0; k < x.size(); ++k) v(k) = x[k];
    return v;
}
template <typename T>
std::vector<T> replicated(const halo::Communicator &comm, const std::vector<T> &x) {
    halo::collective_require(comm, x.size() <= INT_MAX, "AXIS distributed geometry exceeds MPI count limit");
    int n = static_cast<int>(x.size());
    auto counts = halo::allgather(comm, &n, 1);
    long long total = std::accumulate(counts.begin(), counts.end(), 0LL);
    halo::collective_require(comm, total <= INT_MAX, "AXIS replicated geometry exceeds MPI displacement limit");
    return halo::allgatherv(comm, x, counts).buffer;
}

Mesh gather_source(const Mesh &mesh, std::span<const long long> node_ids, const halo::Communicator &comm, bool linear) {
    const auto &g = mesh.geometry_metadata();
    std::vector<int> meta{static_cast<int>(mesh.coord_system()), static_cast<int>(g.boundary_model), static_cast<int>(g.area_model),
                          static_cast<int>(g.provenance),        g.longitude_periodicity.periodic,   g.longitude_periodicity.increasing};
    auto all_meta = halo::allgather(comm, meta);
    bool valid = true;
    for (int r = 0; r < comm.size(); ++r)
        for (std::size_t k = 0; k < meta.size(); ++k) valid &= all_meta[r * meta.size() + k] == meta[k];
    auto periods = halo::allgather(comm, &g.longitude_periodicity.period, 1);
    for (double p : periods) valid &= p == g.longitude_periodicity.period;
    valid &= node_ids.empty() || node_ids.size() == mesh.n_nodes();
    valid &= !linear || comm.size() == 1 || node_ids.size() == mesh.n_nodes();
    int supplied = !node_ids.empty();
    auto supplied_flags = halo::allgather(comm, &supplied, 1);
    // Empty ranks may omit IDs. All nonempty ranks must agree.
    long long local_nodes = mesh.n_nodes();
    auto node_counts = halo::allgather(comm, &local_nodes, 1);
    int id_mode = 0;
    for (int r = 0; r < comm.size(); ++r)
        if (node_counts[r]) id_mode = std::max(id_mode, supplied_flags[r]);
    for (int r = 0; r < comm.size(); ++r)
        if (node_counts[r]) valid &= supplied_flags[r] == id_mode;
    auto coords = mesh.node_coords();
    valid &= !mesh.n_nodes() || coords.extent(1) == 2 || coords.extent(1) == 3;
    auto offsets = mesh.conn_offsets();
    auto indices = mesh.conn_indices();
    valid &= mesh.n_cells() == 0 ||
             (offsets.extent(0) == mesh.n_cells() + 1 && offsets(0) == 0 && offsets(mesh.n_cells()) == static_cast<index_t>(indices.extent(0)));
    halo::collective_require(comm, valid, "AXIS distributed source: invalid CSR shape or metadata");
    for (std::size_t i = 0; i < mesh.n_cells(); ++i) valid &= offsets(i) >= 0 && offsets(i + 1) >= offsets(i);
    for (std::size_t k = 0; k < indices.extent(0); ++k) valid &= indices(k) >= 0 && static_cast<std::size_t>(indices(k)) < mesh.n_nodes();
    for (std::size_t i = 0; i < mesh.n_nodes(); ++i) {
        for (std::size_t k = 0; k < coords.extent(1); ++k) valid &= std::isfinite(coords(i, k));
        if (!node_ids.empty()) valid &= node_ids[i] >= 0;
    }
    auto areas = mesh.cell_areas();
    auto mask = mesh.cell_mask();
    valid &= areas.extent(0) == 0 || areas.extent(0) == mesh.n_cells();
    valid &= mask.extent(0) == 0 || mask.extent(0) == mesh.n_cells();
    halo::collective_require(comm, valid, "AXIS distributed source: inconsistent metadata, connectivity, or global node IDs");
    long long node_start = std::accumulate(node_counts.begin(), node_counts.begin() + comm.rank(), 0LL);
    std::vector<double> xyz;
    std::vector<long long> ids, connectivity;
    std::vector<index_t> lengths;
    std::vector<int> masks;
    std::vector<double> cell_areas;
    int ndim = mesh.n_nodes() ? static_cast<int>(coords.extent(1)) : 0;
    ndim = halo::allreduce<int>(comm, {ndim}, MPI_MAX)[0];
    halo::collective_require(comm, !mesh.n_nodes() || coords.extent(1) == static_cast<std::size_t>(ndim), "AXIS coordinate dimensions differ");
    for (std::size_t i = 0; i < mesh.n_nodes(); ++i) {
        ids.push_back(id_mode ? node_ids[i] : node_start + static_cast<long long>(i));
        for (int k = 0; k < ndim; ++k) xyz.push_back(coords(i, k));
    }
    for (std::size_t i = 0; i < mesh.n_cells(); ++i) {
        lengths.push_back(offsets(i + 1) - offsets(i));
        masks.push_back(mask.extent(0) ? mask(i) : 1);
        for (index_t k = offsets(i); k < offsets(i + 1); ++k) connectivity.push_back(node_start + indices(k));
    }
    int area_mode = areas.extent(0) != 0;
    auto area_flags = halo::allgather(comm, &area_mode, 1);
    long long local_cells = mesh.n_cells();
    auto cell_counts = halo::allgather(comm, &local_cells, 1);
    int any_area = 0;
    for (int r = 0; r < comm.size(); ++r)
        if (cell_counts[r]) any_area = std::max(any_area, area_flags[r]);
    valid = true;
    for (int r = 0; r < comm.size(); ++r)
        if (cell_counts[r]) valid &= area_flags[r] == any_area;
    halo::collective_require(comm, valid, "AXIS distributed source: inconsistent supplied-area availability");
    for (std::size_t i = 0; i < areas.extent(0); ++i) cell_areas.push_back(areas(i));
    ids = replicated(comm, ids);
    xyz = replicated(comm, xyz);
    connectivity = replicated(comm, connectivity);
    lengths = replicated(comm, lengths);
    masks = replicated(comm, masks);
    cell_areas = replicated(comm, cell_areas);
    std::map<long long, index_t> node_map;
    std::vector<index_t> mapped(ids.size());
    std::vector<std::array<double, 3>> nodes;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        std::array<double, 3> point{};
        for (int k = 0; k < ndim; ++k) point[k] = xyz[i * ndim + k];
        auto [it, added] = node_map.emplace(ids[i], static_cast<index_t>(nodes.size()));
        if (added)
            nodes.push_back(point);
        else if (nodes[it->second] != point)
            throw std::invalid_argument("shared source node ID has inconsistent coordinates");
        mapped[i] = it->second;
    }
    Kokkos::View<double **, Kokkos::LayoutLeft, Mem> node_view("distributed_source_nodes", nodes.size(), ndim ? ndim : 2);
    for (std::size_t i = 0; i < nodes.size(); ++i)
        for (int k = 0; k < ndim; ++k) node_view(i, k) = nodes[i][k];
    std::vector<index_t> out_offsets{0}, out_indices;
    for (auto n : lengths) out_offsets.push_back(out_offsets.back() + n);
    for (auto n : connectivity) out_indices.push_back(mapped.at(static_cast<std::size_t>(n)));
    return Mesh(node_view, view("distributed_source_offsets", out_offsets), view("distributed_source_indices", out_indices), mesh.coord_system(),
                view("distributed_source_areas", cell_areas), view("distributed_source_mask", masks), g);
}

Matrix compact(Matrix &m, std::vector<long long> &ids) {
    std::set<index_t> active, needed;
    auto cols = m.factor_col_view();
    for (std::size_t k = 0; k < m.nnz(); ++k) active.insert(cols(k));
    needed = active;
    if (m.has_linear_reconstruction()) {
        auto &l = m.linear_reconstruction();
        for (auto i : active)
            for (index_t k = l.neighbor_offsets(i); k < l.neighbor_offsets(i + 1); ++k) needed.insert(l.neighbors(k));
    }
    std::unordered_map<index_t, index_t> map;
    for (auto i : needed) {
        map[i] = static_cast<index_t>(ids.size());
        ids.push_back(i);
    }
    std::vector<index_t> rows, new_cols;
    std::vector<double> values, fractions, areas;
    for (std::size_t k = 0; k < m.nnz(); ++k) {
        rows.push_back(m.factor_row_view()(k));
        new_cols.push_back(map.at(cols(k)));
        values.push_back(m.factor_list_view()(k));
    }
    for (auto i : ids) {
        fractions.push_back(m.frac_a_view()(i));
        areas.push_back(m.area_a_view()(i));
    }
    Matrix out(view("distributed_weights", values), view("distributed_rows", rows), view("distributed_cols", new_cols),
               view("distributed_source_fractions", fractions), m.frac_b_view(), view("distributed_source_areas", areas), m.area_b_view(), ids.size(),
               m.n_dst());
    if (m.has_linear_reconstruction()) {
        auto &l = m.linear_reconstruction();
        solver::LinearReconstruction<Mem> data;
        data.enabled = true;
        data.limited = l.limited;
        data.moments = l.moments;
        std::vector<index_t> no{0}, ns, co{0};
        std::vector<std::array<double, 2>> coeff, controls;
        for (auto i : ids) {
            if (active.contains(static_cast<index_t>(i))) {
                for (index_t k = l.neighbor_offsets(i); k < l.neighbor_offsets(i + 1); ++k) {
                    ns.push_back(map.at(l.neighbors(k)));
                    coeff.push_back({l.gradient_coefficients(k, 0), l.gradient_coefficients(k, 1)});
                }
                for (index_t k = l.control_offsets(i); k < l.control_offsets(i + 1); ++k)
                    controls.push_back({l.control_points(k, 0), l.control_points(k, 1)});
            }
            no.push_back(static_cast<index_t>(ns.size()));
            co.push_back(static_cast<index_t>(controls.size()));
        }
        data.neighbor_offsets = view("distributed_neighbor_offsets", no);
        data.neighbors = view("distributed_neighbors", ns);
        data.control_offsets = view("distributed_control_offsets", co);
        data.gradient_coefficients = decltype(data.gradient_coefficients)("distributed_fit", coeff.size());
        data.control_points = decltype(data.control_points)("distributed_controls", controls.size());
        for (std::size_t k = 0; k < coeff.size(); ++k)
            for (int a = 0; a < 2; ++a) data.gradient_coefficients(k, a) = coeff[k][a];
        for (std::size_t k = 0; k < controls.size(); ++k)
            for (int a = 0; a < 2; ++a) data.control_points(k, a) = controls[k][a];
        out.set_linear_reconstruction(std::move(data));
    }
    out.to_csr();
    return out;
}
}  // namespace

struct DistributedRegridder::Impl {
    halo::Communicator comm;
    std::size_t owned{}, destinations{};
    Matrix matrix;
    std::vector<long long> ids;
    std::unique_ptr<halo::Indexed_Gather_Plan> gather;
    std::vector<double> source_fractions;
    GeometryStatistics statistics;
    Kokkos::View<double *[2], Mem> gradients;
    Kokkos::View<double *, Mem> phi;
    Impl(const Mesh &src, const Mesh &dst, const solver::RegridConfig &config, MPI_Comm raw, std::span<const long long> node_ids,
         DistributedOptions options)
        : comm(isolated(raw)), owned(src.n_cells()), destinations(dst.n_cells()) {
        halo::collective_require(
            comm, Kokkos::DefaultHostExecutionSpace().concurrency() == 1 || halo::Environment::thread_support_level() >= MPI_THREAD_FUNNELED,
            "AXIS distributed OpenMP requires MPI_Init_thread with MPI_THREAD_FUNNELED or higher");
        std::vector<int> settings{static_cast<int>(config.method),   static_cast<int>(config.norm_type),     static_cast<int>(config.line_type),
                                  static_cast<int>(config.unmapped), static_cast<int>(config.extrap_method), config.use_limiter};
        auto all = halo::allgather(comm, settings);
        bool valid = true;
        for (int r = 0; r < comm.size(); ++r)
            for (std::size_t k = 0; k < settings.size(); ++k) valid &= all[r * settings.size() + k] == settings[k];
        halo::collective_require(comm, valid, "AXIS distributed configurations differ between ranks");
        auto method_config = config;
        method_config.use_structured_fast_paths = false;
        auto strategies = halo::allgather(comm, std::vector<int>{static_cast<int>(options.geometry)});
        halo::collective_require(comm, std::all_of(strategies.begin(), strategies.end(), [&](int v) { return v == strategies.front(); }),
                                 "AXIS distributed geometry strategies differ");
        halo::collective_require(comm, options.geometry == GeometryStrategy::Selective || options.geometry == GeometryStrategy::ReplicatedReference,
                                 "AXIS distributed: invalid geometry strategy");
        Mesh source_geometry;
        std::vector<long long> source_ids;
        if (options.geometry == GeometryStrategy::Selective) {
            auto imported = detail::selective_geometry(src, dst, node_ids, method_config, comm, options);
            source_geometry = std::move(imported.mesh);
            source_ids = std::move(imported.global_cells);
            statistics = imported.statistics;
        } else {
            auto local_cost = static_cast<long long>(src.n_nodes()) * 256 + static_cast<long long>(src.conn_indices().extent(0)) * 128 +
                              static_cast<long long>(src.n_cells()) * 256;
            auto global_cost = halo::allreduce<long long>(comm, {local_cost}, MPI_SUM)[0];
            halo::collective_require(comm, global_cost >= 0 && static_cast<unsigned long long>(global_cost) <= options.max_import_bytes,
                                     "AXIS replicated reference exceeds geometry budget; use selective setup");
            statistics.estimated_import_bytes = global_cost;
            source_geometry = gather_source(src, node_ids, comm, config.method == solver::InterpolationMethod::Conservative2ndOrder);
            source_ids.resize(source_geometry.n_cells());
            std::iota(source_ids.begin(), source_ids.end(), 0LL);
            statistics.imported_cells = source_geometry.n_cells();
        }
        Matrix full;
        std::string error;
        try {
            if (dst.n_cells())
                full = solver::WeightGenerator::generate(source_geometry, dst, method_config);
            else {
                auto ns = source_geometry.n_cells();
                full = Matrix(view<double>("empty_weights", {}), view<index_t>("empty_rows", {}), view<index_t>("empty_cols", {}),
                              Kokkos::View<double *, Mem>("empty_source_fraction", ns), view<double>("empty_dest_fraction", {}),
                              Kokkos::View<double *, Mem>("empty_source_area", ns), view<double>("empty_dest_area", {}), ns, 0);
                if (config.method == solver::InterpolationMethod::Conservative2ndOrder) {
                    solver::LinearReconstruction<Mem> data;
                    data.enabled = true;
                    data.limited = config.use_limiter;
                    data.neighbor_offsets = Kokkos::View<index_t *, Mem>("empty_neighbor_offsets", ns + 1);
                    data.control_offsets = Kokkos::View<index_t *, Mem>("empty_control_offsets", ns + 1);
                    full.set_linear_reconstruction(std::move(data));
                }
            }
        } catch (const std::exception &e) {
            error = e.what();
        }
        halo::collective_require(comm, error.empty(), "AXIS distributed weight generation failed on a rank; check geometry/configuration");
        matrix = compact(full, ids);
        for (auto &id : ids) id = source_ids.at(static_cast<std::size_t>(id));
        gather = std::make_unique<halo::Indexed_Gather_Plan>(comm, owned, ids);
        std::vector<double> local_fractions(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) local_fractions[i] = matrix.frac_a_view()(i);
        if (config.method == solver::InterpolationMethod::Conservative1stOrder || config.method == solver::InterpolationMethod::Conservative2ndOrder)
            source_fractions = gather->reduce(local_fractions, MPI_SUM, 0);
        gradients = decltype(gradients)("distributed_gradients", ids.size());
        phi = decltype(phi)("distributed_phi", ids.size());
    }
};

DistributedRegridder::DistributedRegridder(const Mesh &s, const Mesh &d, const solver::RegridConfig &c, MPI_Comm comm,
                                           std::span<const long long> nodes, DistributedOptions options)
    : impl_(std::make_unique<Impl>(s, d, c, comm, nodes, options)) {}
DistributedRegridder::~DistributedRegridder() = default;
DistributedRegridder::DistributedRegridder(DistributedRegridder &&) noexcept = default;
DistributedRegridder &DistributedRegridder::operator=(DistributedRegridder &&) noexcept = default;
const GeometryStatistics &DistributedRegridder::geometry_statistics() const noexcept {
    return impl_->statistics;
}
std::size_t DistributedRegridder::local_source_size() const noexcept {
    return impl_->owned;
}
std::size_t DistributedRegridder::local_destination_size() const noexcept {
    return impl_->destinations;
}
const Matrix &DistributedRegridder::local_matrix() const noexcept {
    return impl_->matrix;
}
const std::vector<long long> &DistributedRegridder::requested_source_ids() const noexcept {
    return impl_->ids;
}
const std::vector<double> &DistributedRegridder::owned_source_fractions() const noexcept {
    return impl_->source_fractions;
}

void DistributedRegridder::apply(field_view<const double, 1> src, field_view<double, 1> dst) {
    auto &p = *impl_;
    halo::collective_require(
        p.comm,
        src.extent(0) == p.owned && dst.extent(0) == p.destinations && !overlaps(src.data_handle(), src.extent(0), dst.data_handle(), dst.extent(0)),
        "AXIS distributed apply: extents or buffer alias");
    auto values = p.gather->gather<double>(std::span(src.data_handle(), src.extent(0)));
    halo::collective_require(p.comm, std::all_of(values.begin(), values.end(), [](double x) { return std::isfinite(x); }),
                             "AXIS distributed apply: nonfinite participating source value");
    field_view<const double, 1> field(values.data(), values.size());
    if (!p.matrix.has_linear_reconstruction()) {
        solver::apply(p.matrix, field, dst);
        return;
    }
    Kokkos::View<const double *, Mem, Kokkos::MemoryTraits<Kokkos::Unmanaged>> q(values.data(), values.size());
    Kokkos::View<double *, Mem, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out(dst.data_handle(), dst.extent(0));
    if (p.matrix.linear_reconstruction().limited) {
        solver::compute_linear_gradients<true, Mem>(p.matrix, q, p.gradients, p.phi);
        Kokkos::fence();
        std::vector<double> local_phi(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) local_phi[i] = p.phi(i);
        auto owner_phi = p.gather->reduce(local_phi, MPI_MIN, 1);
        auto global_phi = p.gather->gather<double>(owner_phi);
        for (std::size_t i = 0; i < values.size(); ++i) {
            double ratio = p.phi(i) > 0 ? global_phi[i] / p.phi(i) : 0;
            p.gradients(i, 0) *= ratio;
            p.gradients(i, 1) *= ratio;
        }
    } else
        solver::compute_linear_gradients<false, Mem>(p.matrix, q, p.gradients);
    solver::apply_linear_gradients<Mem>(p.matrix, q, out, p.gradients);
    Kokkos::fence("axis::distributed::complete");
}
void DistributedRegridder::apply(field_view<const double, 2> src, field_view<double, 2> dst) {
    auto &p = *impl_;
    halo::collective_require(p.comm,
                             src.extent(0) == p.owned && dst.extent(0) == p.destinations && src.extent(1) == dst.extent(1) &&
                                 !overlaps(src.data_handle(), src.size(), dst.data_handle(), dst.size()),
                             "AXIS distributed batch: extent mismatch");
    auto fields = halo::allgather(p.comm, std::vector<long long>{static_cast<long long>(src.extent(1))});
    bool valid = std::all_of(fields.begin(), fields.end(), [&](long long n) { return n == static_cast<long long>(src.extent(1)); });
    halo::collective_require(p.comm, valid, "AXIS distributed batch: field counts differ");
    for (std::size_t k = 0; k < src.extent(1); ++k) {
        auto source = src.data_handle();
        auto destination = dst.data_handle();
        if (p.owned) source += k * p.owned;
        if (p.destinations) destination += k * p.destinations;
        apply(field_view<const double, 1>(source, p.owned), field_view<double, 1>(destination, p.destinations));
    }
}
void DistributedRegridder::apply_vector(field_view<const double, 1> u, field_view<const double, 1> v, field_view<const double, 1> sa,
                                        field_view<const double, 1> da, field_view<double, 1> du, field_view<double, 1> dv) {
    auto &p = *impl_;
    bool valid = u.extent(0) == p.owned && v.extent(0) == p.owned && sa.extent(0) == p.owned && da.extent(0) == p.destinations &&
                 du.extent(0) == p.destinations && dv.extent(0) == p.destinations &&
                 !overlaps(du.data_handle(), du.extent(0), dv.data_handle(), dv.extent(0));
    for (auto input : {u, v, sa, da}) {
        valid &= !overlaps(input.data_handle(), input.extent(0), du.data_handle(), du.extent(0));
        valid &= !overlaps(input.data_handle(), input.extent(0), dv.data_handle(), dv.extent(0));
    }
    halo::collective_require(p.comm, valid, "AXIS distributed vector: invalid extents or overlapping buffers");
    for (std::size_t i = 0; i < p.owned; ++i) valid &= std::isfinite(sa(i));
    for (std::size_t j = 0; j < p.destinations; ++j) valid &= std::isfinite(da(j));
    halo::collective_require(p.comm, valid, "AXIS distributed vector: nonfinite rotation");
    std::vector<double> east(p.owned), north(p.owned), e(p.destinations), n(p.destinations);
    for (std::size_t i = 0; i < p.owned; ++i) {
        east[i] = std::cos(sa(i)) * u(i) - std::sin(sa(i)) * v(i);
        north[i] = std::sin(sa(i)) * u(i) + std::cos(sa(i)) * v(i);
    }
    apply(field_view<const double, 1>(east.data(), east.size()), field_view<double, 1>(e.data(), e.size()));
    apply(field_view<const double, 1>(north.data(), north.size()), field_view<double, 1>(n.data(), n.size()));
    for (std::size_t j = 0; j < p.destinations; ++j) {
        du(j) = std::cos(da(j)) * e[j] + std::sin(da(j)) * n[j];
        dv(j) = -std::sin(da(j)) * e[j] + std::cos(da(j)) * n[j];
    }
}
}  // namespace axis::distributed
