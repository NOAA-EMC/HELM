// SPDX-License-Identifier: Apache-2.0
#include "selective_geometry.hpp"

#include <ArborX.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdint>
#include <halo/bounded_peer_exchange.hpp>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <unordered_map>

namespace axis::distributed::detail {
namespace {
using Mem = Kokkos::HostSpace;
using Point = ArborX::Point<3>;
using Box = ArborX::Box<3>;
using Tree = decltype(ArborX::BoundingVolumeHierarchy(Kokkos::DefaultHostExecutionSpace{},
                                                      ArborX::Experimental::attach_indices(Kokkos::View<Point *, Mem>{})));
using BoxTree =
    decltype(ArborX::BoundingVolumeHierarchy(Kokkos::DefaultHostExecutionSpace{}, ArborX::Experimental::attach_indices(Kokkos::View<Box *, Mem>{})));
using Bounds = std::array<float, 6>;
constexpr float inf = std::numeric_limits<float>::infinity();
Bounds empty_bounds() {
    return {inf, inf, inf, -inf, -inf, -inf};
}
void include(Bounds &b, const Bounds &x) {
    for (int d = 0; d < 3; ++d) {
        b[d] = std::min(b[d], x[d]);
        b[d + 3] = std::max(b[d + 3], x[d + 3]);
    }
}
bool intersects(const Bounds &a, const Bounds &b) {
    for (int d = 0; d < 3; ++d)
        if (a[d] > b[d + 3] || b[d] > a[d + 3]) return false;
    return true;
}
Point point(const SearchGeometry &g, std::size_t i) {
    return Point{g.points(i, 0), g.points(i, 1), g.points(i, 2)};
}
Bounds box(const SearchGeometry &g, std::size_t i) {
    Bounds b;
    for (int d = 0; d < 6; ++d) b[d] = g.boxes(i, d);
    return b;
}
float distance(const Point &a, const Point &b) {
    float v = 0;
    for (int d = 0; d < 3; ++d) {
        float x = a[d] - b[d];
        v += x * x;
    }
    return v;
}
float lower_bound(const Bounds &b, const Point &p) {
    float v = 0;
    for (int d = 0; d < 3; ++d) {
        float x = std::max({b[d] - p[d], p[d] - b[d + 3], 0.f});
        v += x * x;
    }
    return v;
}
Box arbor_box(const Bounds &b) {
    return Box{{b[0], b[1], b[2]}, {b[3], b[4], b[5]}};
}
template <class T>
Kokkos::View<T *, Mem> make_view(const char *label, const std::vector<T> &v) {
    Kokkos::View<T *, Mem> out(label, v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out(i) = v[i];
    return out;
}

template <class T, class F>
void stream(const halo::Communicator &comm, const std::vector<std::vector<T>> &out, std::size_t chunk, F consume) {
    for (int step = 0; step < comm.size(); ++step) {
        const int dst = (comm.rank() + step) % comm.size(), src = (comm.rank() - step + comm.size()) % comm.size();
        const auto &v = out[dst];
        long long n = (v.size() + chunk - 1) / chunk;
        auto rounds = halo::allreduce<long long>(comm, {n}, MPI_MAX)[0];
        for (long long b = 0; b < rounds; ++b) {
            const auto begin = std::min(v.size(), static_cast<std::size_t>(b) * chunk), count = std::min(chunk, v.size() - begin);
            std::span<const T> send = count ? std::span<const T>(v.data() + begin, count) : std::span<const T>{};
            auto recv = halo::bounded_peer_exchange<T>(comm, send, dst, src, chunk);
            bool ok = true;
            try {
                consume(src, recv);
            } catch (const std::exception &) {
                ok = false;
            }
            halo::collective_require(comm, ok, "AXIS selective geometry: import/index budget exceeded or malformed response");
        }
    }
}
struct Layout {
    std::vector<long long> cell_offsets, node_offsets;
    int ndim{};
    bool supplied_ids{}, supplied_areas{};
};
Layout validate(const HostMesh &src, const HostMesh &dst, std::span<const long long> ids, const solver::RegridConfig &cfg,
                const halo::Communicator &comm, const DistributedOptions &options) {
    Layout out;
    const auto g = src.geometry_metadata();
    std::vector<int> meta{static_cast<int>(src.coord_system()), static_cast<int>(g.boundary_model), static_cast<int>(g.area_model),
                          static_cast<int>(g.provenance),       g.longitude_periodicity.periodic,   g.longitude_periodicity.increasing};
    auto all = halo::allgather(comm, meta);
    bool valid = src.coord_system() == dst.coord_system();
    for (int r = 0; r < comm.size(); ++r)
        for (std::size_t k = 0; k < meta.size(); ++k) valid &= all[r * meta.size() + k] == meta[k];
    auto periods = halo::allgather(comm, std::vector<double>{g.longitude_periodicity.period});
    for (auto v : periods) valid &= v == g.longitude_periodicity.period;
    valid &= options.query_batch_size > 0 && options.query_batch_size <= 4096 && options.max_message_bytes >= 4096 && options.max_import_bytes > 0;
    auto limits =
        halo::allgather(comm, std::vector<unsigned long long>{options.query_batch_size, options.max_message_bytes, options.max_import_bytes});
    for (int r = 0; r < comm.size(); ++r)
        for (int k = 0; k < 3; ++k) valid &= limits[r * 3 + k] == limits[k];
    auto cc = halo::allgather(comm, std::vector<long long>{static_cast<long long>(src.n_cells())});
    auto nc = halo::allgather(comm, std::vector<long long>{static_cast<long long>(src.n_nodes())});
    out.cell_offsets = {0};
    out.node_offsets = {0};
    for (int r = 0; r < comm.size(); ++r) {
        out.cell_offsets.push_back(out.cell_offsets.back() + cc[r]);
        out.node_offsets.push_back(out.node_offsets.back() + nc[r]);
    }
    auto flags = halo::allgather(
        comm, std::vector<int>{!ids.empty(), src.cell_areas().extent(0) != 0, src.n_nodes() ? static_cast<int>(src.node_coords().extent(1)) : 0});
    for (int r = 0; r < comm.size(); ++r) {
        if (nc[r]) out.supplied_ids |= flags[3 * r];
        if (cc[r]) out.supplied_areas |= flags[3 * r + 1];
        out.ndim = std::max(out.ndim, flags[3 * r + 2]);
    }
    for (int r = 0; r < comm.size(); ++r) {
        if (nc[r]) valid &= flags[3 * r] == out.supplied_ids && flags[3 * r + 2] == out.ndim;
        if (cc[r]) valid &= flags[3 * r + 1] == out.supplied_areas;
    }
    valid &= ids.empty() || ids.size() == src.n_nodes();
    valid &= cfg.method != solver::InterpolationMethod::Conservative2ndOrder || comm.size() == 1 || ids.size() == src.n_nodes();
    auto check_mesh = [&](const HostMesh &m) {
        auto off = m.conn_offsets();
        auto idx = m.conn_indices();
        auto xy = m.node_coords();
        valid &= !m.n_nodes() || (xy.extent(1) == 2 || xy.extent(1) == 3);
        bool empty_csr = m.n_cells() == 0 && idx.extent(0) == 0 && (off.extent(0) == 0 || (off.extent(0) == 1 && off(0) == 0));
        bool csr = empty_csr || (off.extent(0) == m.n_cells() + 1 && off(0) == 0 && off(m.n_cells()) == static_cast<index_t>(idx.extent(0)));
        valid &= csr;
        if (csr)
            for (std::size_t i = 0; i < m.n_cells(); ++i) valid &= off(i) >= 0 && off(i + 1) - off(i) >= 3;
        for (std::size_t i = 0; i < idx.extent(0); ++i) valid &= idx(i) >= 0 && static_cast<std::size_t>(idx(i)) < m.n_nodes();
        for (std::size_t i = 0; i < m.n_nodes(); ++i)
            for (std::size_t d = 0; d < xy.extent(1); ++d) valid &= std::isfinite(xy(i, d));
        valid &= m.cell_mask().extent(0) == 0 || m.cell_mask().extent(0) == m.n_cells();
        valid &= m.cell_areas().extent(0) == 0 || m.cell_areas().extent(0) == m.n_cells();
    };
    check_mesh(src);
    check_mesh(dst);
    for (auto id : ids) valid &= id >= 0;
    halo::collective_require(comm, valid, "AXIS selective setup: invalid mesh, ownership IDs, metadata, or budgets");
    if (!out.ndim) out.ndim = 2;
    return out;
}
struct Node {
    long long id;
    std::array<double, 3> xyz;
};
struct Cell {
    long long id;
    int mask;
    double area;
    std::vector<Node> nodes;
};
struct Importer {
    const HostMesh &src;
    std::span<const long long> node_ids;
    const Layout &layout;
    const halo::Communicator &comm;
    const DistributedOptions &options;
    std::map<long long, Cell> cells;
    GeometryStatistics stats;
    long long node_id(index_t n) const {
        return layout.supplied_ids ? node_ids[n] : layout.node_offsets[comm.rank()] + n;
    }
    int owner(long long id) const {
        return std::upper_bound(layout.cell_offsets.begin(), layout.cell_offsets.end(), id) - layout.cell_offsets.begin() - 1;
    }
    void pull(const std::set<long long> &requested) {
        std::vector<std::vector<long long>> by_owner(comm.size());
        for (auto id : requested)
            if (!cells.contains(id)) by_owner[owner(id)].push_back(id);
        const auto chunk = std::min<std::size_t>(options.query_batch_size, 64);
        for (int step = 0; step < comm.size(); ++step) {
            int dst = (comm.rank() + step) % comm.size(), peer = (comm.rank() - step + comm.size()) % comm.size();
            auto &list = by_owner[dst];
            auto rounds = halo::allreduce<long long>(comm, {static_cast<long long>((list.size() + chunk - 1) / chunk)}, MPI_MAX)[0];
            for (long long b = 0; b < rounds; ++b) {
                auto begin = std::min(list.size(), static_cast<std::size_t>(b) * chunk), n = std::min(chunk, list.size() - begin);
                auto requests = halo::bounded_peer_exchange<long long>(
                    comm, n ? std::span<const long long>(list.data() + begin, n) : std::span<const long long>{}, dst, peer, chunk);
                std::size_t served = 0;
                while (halo::allreduce<int>(comm, {served < requests.size() ? 1 : 0}, MPI_MAX)[0]) {
                    std::vector<long long> ints;
                    std::vector<double> values;
                    bool ok = true;
                    auto off = src.conn_offsets();
                    auto idx = src.conn_indices();
                    auto coords = src.node_coords();
                    auto mask = src.cell_mask();
                    auto areas = src.cell_areas();
                    for (; served < requests.size(); ++served) {
                        auto id = requests[served];
                        long long i = id - layout.cell_offsets[comm.rank()];
                        if (i < 0 || static_cast<std::size_t>(i) >= src.n_cells()) {
                            ok = false;
                            break;
                        }
                        auto nv = static_cast<std::size_t>(off(i + 1) - off(i));
                        if ((ints.size() + 3 + nv) * sizeof(long long) + (values.size() + 1 + 3 * nv) * sizeof(double) > options.max_message_bytes) {
                            if (ints.empty()) ok = false;
                            break;
                        }
                        ints.insert(ints.end(), {id, mask.extent(0) ? mask(i) : 1, static_cast<long long>(nv)});
                        values.push_back(areas.extent(0) ? areas(i) : 0);
                        for (index_t k = off(i); k < off(i + 1); ++k) {
                            auto node = idx(k);
                            ints.push_back(node_id(node));
                            for (int d = 0; d < 3; ++d) values.push_back(d < layout.ndim ? coords(node, d) : 0);
                        }
                    }
                    halo::collective_require(comm, ok, "AXIS selective geometry: polygon transfer exceeds message budget");
                    auto ri = halo::bounded_peer_exchange<long long>(comm, ints, peer, dst, options.max_message_bytes / sizeof(long long));
                    auto rv = halo::bounded_peer_exchange<double>(comm, values, peer, dst, options.max_message_bytes / sizeof(double));
                    stats.peak_received_message_bytes = std::max(stats.peak_received_message_bytes, (ri.size() + rv.size()) * 8);
                    std::size_t a = 0, v = 0;
                    try {
                        while (a < ri.size()) {
                            if (a + 3 > ri.size()) throw std::runtime_error("truncated cell header");
                            auto id = ri[a++];
                            auto mask_value = ri[a++];
                            auto nv = ri[a++];
                            if (nv < 3 || static_cast<std::size_t>(nv) > ri.size() - a || 1 + 3 * static_cast<std::size_t>(nv) > rv.size() - v)
                                throw std::runtime_error("truncated polygon");
                            std::size_t cost = 256 + 128 * static_cast<std::size_t>(nv);
                            if (!cells.contains(id) && cost > options.max_import_bytes - stats.estimated_import_bytes)
                                throw std::runtime_error("import budget exceeded");
                            Cell cell{id, static_cast<int>(mask_value), rv[v++], {}};
                            cell.nodes.reserve(nv);
                            for (long long k = 0; k < nv; ++k) {
                                Node node;
                                node.id = ri[a++];
                                for (int d = 0; d < 3; ++d) node.xyz[d] = rv[v++];
                                cell.nodes.push_back(node);
                            }
                            if (!cells.contains(id)) {
                                stats.estimated_import_bytes += cost;
                                stats.imported_vertices += nv;
                                cells.emplace(id, std::move(cell));
                            }
                        }
                        if (v != rv.size()) throw std::runtime_error("extra coordinates");
                    } catch (const std::exception &) {
                        ok = false;
                    }
                    halo::collective_require(comm, ok, "AXIS selective geometry: retained import budget exceeded or malformed polygon response");
                }
            }
        }
    }
    HostMesh mesh() {
        std::map<long long, index_t> nodes;
        std::vector<Node> unique;
        std::vector<index_t> offsets{0}, indices;
        std::vector<int> masks;
        std::vector<double> areas;
        for (auto &[id, c] : cells) {
            masks.push_back(c.mask);
            if (layout.supplied_areas) areas.push_back(c.area);
            for (auto &node : c.nodes) {
                auto [it, added] = nodes.emplace(node.id, static_cast<index_t>(unique.size()));
                if (added)
                    unique.push_back(node);
                else if (unique[it->second].xyz != node.xyz)
                    throw std::invalid_argument("shared node coordinates differ");
                indices.push_back(it->second);
            }
            offsets.push_back(static_cast<index_t>(indices.size()));
        }
        Kokkos::View<double **, Kokkos::LayoutLeft, Mem> xy("imported_geometry", unique.size(), layout.ndim);
        for (std::size_t i = 0; i < unique.size(); ++i)
            for (int d = 0; d < layout.ndim; ++d) xy(i, d) = unique[i].xyz[d];
        return HostMesh(xy, make_view("imported_offsets", offsets), make_view("imported_connectivity", indices), src.coord_system(),
                        make_view("imported_areas", areas), make_view("imported_masks", masks), src.geometry_metadata());
    }
};
struct SearchIndex {
    Kokkos::View<Point *, Mem> points, wet_points;
    Kokkos::View<Box *, Mem> boxes;
    std::vector<std::size_t> wet_cells;
    Tree tree, wet_tree;
    BoxTree box_tree;
    std::vector<float> summaries;
    SearchIndex(const HostMesh &src, const solver::RegridConfig &cfg, const halo::Communicator &comm) {
        auto g = search_geometry(src, cfg), w = search_geometry(src, cfg, true);
        auto mask = src.cell_mask();
        points = decltype(points)("source_query_points", src.n_cells());
        boxes = decltype(boxes)("source_query_boxes", src.n_cells());
        Bounds pb = empty_bounds(), wb = empty_bounds(), bb = empty_bounds();
        for (std::size_t i = 0; i < src.n_cells(); ++i) {
            points(i) = point(g, i);
            boxes(i) = arbor_box(box(g, i));
            include(bb, box(g, i));
            Bounds p;
            for (int d = 0; d < 3; ++d) p[d] = p[d + 3] = points(i)[d];
            include(pb, p);
            if (!mask.extent(0) || mask(i)) {
                wet_cells.push_back(i);
                for (int d = 0; d < 3; ++d) p[d] = p[d + 3] = w.points(i, d);
                include(wb, p);
            }
        }
        wet_points = decltype(wet_points)("wet_query_points", wet_cells.size());
        for (std::size_t i = 0; i < wet_cells.size(); ++i) wet_points(i) = point(w, wet_cells[i]);
        tree = Tree(Kokkos::DefaultHostExecutionSpace{}, ArborX::Experimental::attach_indices(points));
        wet_tree = Tree(Kokkos::DefaultHostExecutionSpace{}, ArborX::Experimental::attach_indices(wet_points));
        box_tree = BoxTree(Kokkos::DefaultHostExecutionSpace{}, ArborX::Experimental::attach_indices(boxes));
        std::vector<float> bounds;
        for (auto b : {pb, wb, bb}) bounds.insert(bounds.end(), b.begin(), b.end());
        summaries = halo::allgather(comm, bounds);
    }
    Bounds rank_bounds(int rank, int kind) const {
        Bounds b;
        for (int d = 0; d < 6; ++d) b[d] = summaries[rank * 18 + kind * 6 + d];
        return b;
    }
    // Response records are four native int64 elements: query, kind, global ID,
    // and float distance bits. Intersection responses are deduplicated on owner.
    std::vector<long long> query(const std::vector<float> &packet, long long start, std::size_t max_bytes) const {
        std::vector<long long> response;
        std::vector<Box> overlaps;
        Kokkos::DefaultHostExecutionSpace exec;
        for (std::size_t a = 0; a < packet.size(); a += 9) {
            auto q = static_cast<long long>(packet[a]);
            int kind = packet[a + 1], k = packet[a + 2];
            if (kind == 2) {
                Bounds b;
                for (int d = 0; d < 6; ++d) b[d] = packet[a + 3 + d];
                overlaps.push_back(arbor_box(b));
                continue;
            }
            bool wet = kind == 1;
            auto &pts = wet ? wet_points : points;
            auto &t = wet ? wet_tree : tree;
            k = std::min<int>(k, pts.extent(0));
            if (!k) continue;
            Point p{packet[a + 3], packet[a + 4], packet[a + 5]};
            Kokkos::View<decltype(ArborX::nearest(Point{}, 1)) *, Mem> queries("owner_nearest", 1);
            queries(0) = ArborX::nearest(p, k);
            Kokkos::View<typename Tree::value_type *, Mem> values;
            Kokkos::View<int *, Mem> offsets;
            t.query(exec, queries, values, offsets);
            for (std::size_t j = 0; j < values.extent(0); ++j) {
                std::size_t i = values(j).index;
                auto id = start + static_cast<long long>(wet ? wet_cells[i] : i);
                float dist = distance(p, pts(i));
                if (response.size() + 4 > max_bytes / sizeof(long long)) throw std::runtime_error("local discovery response budget exceeded");
                response.insert(response.end(), {q, kind, id, std::bit_cast<std::uint32_t>(dist)});
            }
        }
        if (!overlaps.empty()) {
            Kokkos::View<decltype(ArborX::intersects(Box{})) *, Mem> queries("owner_overlaps", overlaps.size());
            for (std::size_t i = 0; i < overlaps.size(); ++i) queries(i) = ArborX::intersects(overlaps[i]);
            Kokkos::View<int *, Mem> found("owner_overlap_flags", boxes.extent(0));
            box_tree.query(
                exec, ArborX::Experimental::attach_indices(queries),
                KOKKOS_LAMBDA(const auto &, const auto &value) { Kokkos::atomic_exchange(&found(value.index), 1); });
            for (std::size_t i = 0; i < found.extent(0); ++i)
                if (found(i)) {
                    if (response.size() + 4 > max_bytes / sizeof(long long)) throw std::runtime_error("local discovery response budget exceeded");
                    response.insert(response.end(), {0, 2, start + static_cast<long long>(i), 0});
                }
        }
        // Owner responses are at most one entry per owned intersection cell,
        // plus batch*k nearest candidates. No destination/source pair array.
        if (response.size() * 8 > max_bytes) throw std::runtime_error("local discovery response budget exceeded");
        return response;
    }
};
std::set<long long> discover(const HostMesh &src, const HostMesh &dst, const solver::RegridConfig &cfg, const Layout &layout,
                             const halo::Communicator &comm, const DistributedOptions &options) {
    using M = solver::InterpolationMethod;
    const bool conservative = cfg.method == M::Conservative1stOrder || cfg.method == M::Conservative2ndOrder;
    const int k = cfg.method == M::NearestNeighbor ? 1 : cfg.method == M::Bilinear ? 8 : 16;
    const bool extrap = cfg.extrap_method == solver::ExtrapolationAction::NearestWet;
    SearchIndex index(src, cfg, comm);
    auto target = search_geometry(dst, cfg), wet_target = search_geometry(dst, cfg, true);
    std::set<long long> seeds;
    using Candidate = std::pair<float, long long>;
    const bool constant_lat = src.geometry_metadata().boundary_model == topology::BoundaryModel::ConstantLatitude;
    const std::size_t queries_per_cell = (conservative && constant_lat ? 3 : 1) + (extrap ? 1 : 0);
    const auto batch = std::min(options.query_batch_size, options.max_message_bytes / (queries_per_cell * 9 * sizeof(float)));
    const auto batches = halo::allreduce<long long>(comm, {static_cast<long long>((dst.n_cells() + batch - 1) / batch)}, MPI_MAX)[0];
    const std::size_t response_chunk = (options.max_message_bytes / 32) * 4;
    for (long long b = 0; b < batches; ++b) {
        std::size_t begin = std::min(dst.n_cells(), static_cast<std::size_t>(b) * batch), n = std::min(batch, dst.n_cells() - begin);
        std::vector<std::vector<Candidate>> best(n), wet_best(n);
        for (int step = 0; step < comm.size(); ++step) {
            int peer = (comm.rank() + step) % comm.size(), incoming = (comm.rank() - step + comm.size()) % comm.size();
            std::vector<float> packet;
            auto nearest = [&](std::size_t q, int kind, const Point &p, int count, const std::vector<Candidate> &candidates) {
                auto bounds = index.rank_bounds(peer, kind);
                if (bounds[0] > bounds[3]) return;
                float radius = candidates.size() < static_cast<std::size_t>(count) ? inf : candidates.back().first;
                // Inclusive and rounded outward so float ties are not pruned.
                if (lower_bound(bounds, p) > std::nextafter(radius, inf)) return;
                packet.insert(packet.end(), {static_cast<float>(q), static_cast<float>(kind), static_cast<float>(count), p[0], p[1], p[2], 0, 0, 0});
            };
            for (std::size_t q = 0; q < n; ++q) {
                auto j = begin + q;
                if (conservative) {
                    auto bounds = box(target, j);
                    int images = constant_lat ? 3 : 1;
                    for (int image = 0; image < images; ++image) {
                        auto shifted = bounds;
                        if (constant_lat) {
                            float shift = (image - 1) * src.geometry_metadata().longitude_periodicity.period;
                            shifted[0] += shift;
                            shifted[3] += shift;
                        }
                        if (intersects(shifted, index.rank_bounds(peer, 2))) {
                            packet.insert(packet.end(), {static_cast<float>(q), 2, 0});
                            packet.insert(packet.end(), shifted.begin(), shifted.end());
                        }
                    }
                } else
                    nearest(q, 0, point(target, j), k, best[q]);
                if (extrap) nearest(q, 1, point(wet_target, j), 1, wet_best[q]);
            }
            auto received = halo::bounded_peer_exchange<float>(comm, packet, peer, incoming, options.max_message_bytes / sizeof(float));
            std::vector<long long> response;
            bool ok = true;
            try {
                response = index.query(received, layout.cell_offsets[comm.rank()], options.max_import_bytes);
            } catch (const std::exception &) {
                ok = false;
            }
            halo::collective_require(comm, ok, "AXIS selective setup: owner discovery response exceeded budget");
            auto rounds =
                halo::allreduce<long long>(comm, {static_cast<long long>((response.size() + response_chunk - 1) / response_chunk)}, MPI_MAX)[0];
            for (long long r = 0; r < rounds; ++r) {
                auto first = std::min(response.size(), static_cast<std::size_t>(r) * response_chunk),
                     count = std::min(response_chunk, response.size() - first);
                auto reply = halo::bounded_peer_exchange<long long>(
                    comm, count ? std::span<const long long>(response.data() + first, count) : std::span<const long long>{}, incoming, peer,
                    response_chunk);
                for (std::size_t a = 0; a < reply.size(); a += 4) {
                    auto q = reply[a], kind = reply[a + 1], id = reply[a + 2];
                    if (kind == 2) {
                        seeds.insert(id);
                        continue;
                    }
                    if (q < 0 || static_cast<std::size_t>(q) >= n) {
                        ok = false;
                        break;
                    }
                    auto &list = kind == 1 ? wet_best[q] : best[q];
                    list.push_back({std::bit_cast<float>(static_cast<std::uint32_t>(reply[a + 3])), id});
                    std::sort(list.begin(), list.end());
                    list.erase(std::unique(list.begin(), list.end()), list.end());
                    if (list.size() > static_cast<std::size_t>(kind == 1 ? 1 : k)) list.resize(kind == 1 ? 1 : k);
                }
                ok &= seeds.size() <= options.max_import_bytes / 256;
                halo::collective_require(comm, ok, "AXIS selective setup: candidate closure exceeds import budget");
            }
        }
        for (auto &list : best)
            for (auto [dist, id] : list) seeds.insert(id);
        for (auto &list : wet_best)
            for (auto [dist, id] : list) seeds.insert(id);
        halo::collective_require(comm, seeds.size() <= options.max_import_bytes / 256,
                                 "AXIS selective setup: candidate closure exceeds import budget");
    }
    return seeds;
}
void import_neighbors(Importer &importer, const std::set<long long> &seeds) {
    auto &comm = importer.comm;
    const auto &options = importer.options;
    auto off = importer.src.conn_offsets();
    auto idx = importer.src.conn_indices();
    auto mask = importer.src.cell_mask();
    auto shard = [&](long long id) {
        std::uint64_t h = static_cast<std::uint64_t>(id) + 0x9e3779b97f4a7c15ULL;
        h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
        h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<int>((h ^ (h >> 31)) % comm.size());
    };
    halo::collective_require(comm, idx.extent(0) <= options.max_import_bytes / 16, "AXIS selective setup: local node index exceeds budget");
    std::vector<std::vector<long long>> incidence(comm.size());
    for (std::size_t i = 0; i < importer.src.n_cells(); ++i)
        if (!mask.extent(0) || mask(i))
            for (index_t k = off(i); k < off(i + 1); ++k) {
                auto id = importer.node_id(idx(k));
                auto &v = incidence[shard(id)];
                v.push_back(id);
                v.push_back(importer.layout.cell_offsets[comm.rank()] + i);
            }
    std::unordered_map<long long, std::vector<long long>> directory;
    std::size_t entries = 0;
    std::size_t chunk = (options.max_message_bytes / 16) * 2;
    stream<long long>(comm, incidence, chunk, [&](int, const std::vector<long long> &received) {
        if (received.size() / 2 > (options.max_import_bytes / 64) - entries) throw std::runtime_error("node directory budget exceeded");
        for (std::size_t k = 0; k < received.size(); k += 2) directory[received[k]].push_back(received[k + 1]);
        entries += received.size() / 2;
    });
    importer.stats.directory_incidents = entries;
    // Complete first rings of contributing cells, then complete first rings of
    // those neighbors: enough for the native rank-deficient second-ring fit.
    std::set<long long> frontier = seeds;
    for (int ring = 0; ring < 2; ++ring) {
        std::set<long long> nodes;
        for (auto id : frontier) {
            auto &cell = importer.cells.at(id);
            if (cell.mask)
                for (auto &node : cell.nodes) nodes.insert(node.id);
        }
        std::vector<std::vector<long long>> requests(comm.size()), answers(comm.size());
        std::size_t answer_elements = 0;
        for (auto id : nodes) requests[shard(id)].push_back(id);
        stream<long long>(comm, requests, chunk, [&](int peer, const std::vector<long long> &received) {
            auto &out = answers[peer];
            for (auto id : received) {
                auto it = directory.find(id);
                if (it != directory.end()) {
                    if (it->second.size() > (options.max_import_bytes / 8) - answer_elements)
                        throw std::runtime_error("adjacency response budget exceeded");
                    out.insert(out.end(), it->second.begin(), it->second.end());
                    answer_elements += it->second.size();
                }
            }
        });
        std::set<long long> neighbor_cells;
        stream<long long>(comm, answers, chunk, [&](int, const std::vector<long long> &received) {
            for (auto id : received) neighbor_cells.insert(id);
            if (neighbor_cells.size() > options.max_import_bytes / 256) throw std::runtime_error("neighbor closure budget exceeded");
        });
        importer.pull(neighbor_cells);
        frontier = std::move(neighbor_cells);
    }
}
}  // namespace
ImportedGeometry selective_geometry(const HostMesh &src, const HostMesh &dst, std::span<const long long> node_ids, const solver::RegridConfig &cfg,
                                    const halo::Communicator &comm, const DistributedOptions &options) {
    auto layout = validate(src, dst, node_ids, cfg, comm, options);
    auto seeds = discover(src, dst, cfg, layout, comm, options);
    Importer importer{src, node_ids, layout, comm, options, {}, {}};
    importer.pull(seeds);
    if (cfg.method == solver::InterpolationMethod::Conservative2ndOrder) import_neighbors(importer, seeds);
    ImportedGeometry out;
    bool ok = true;
    try {
        out.mesh = importer.mesh();
    } catch (const std::exception &) {
        ok = false;
    }
    halo::collective_require(comm, ok, "AXIS selective setup: imported geometry construction failed");
    for (auto &[id, cell] : importer.cells) out.global_cells.push_back(id);
    importer.stats.imported_cells = importer.cells.size();
    out.statistics = importer.stats;
    return out;
}
}  // namespace axis::distributed::detail
