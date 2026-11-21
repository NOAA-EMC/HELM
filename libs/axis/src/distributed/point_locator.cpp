// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <axis/distributed/point_locator.hpp>
#include <climits>
#include <cmath>
#include <halo/bounded_peer_exchange.hpp>
#include <unordered_set>
namespace axis::distributed {
namespace {
using namespace solver;
using V = std::array<double, 3>;
// Stream IDs to a partitioned directory. No global ID catalog is replicated.
void unique_ids(const halo::Communicator &comm, std::span<const index_t> ids, const PointLocationOptions &opts) {
    std::unordered_set<long long> directory;
    const auto chunk = std::min(opts.max_message_bytes / sizeof(long long), static_cast<size_t>(INT_MAX));
    long long local_rounds = ids.size() / chunk + (ids.size() % chunk != 0);
    auto rounds = halo::allreduce<long long>(comm, {local_rounds}, MPI_MAX)[0];
    for (long long b = 0; b < rounds; ++b) {
        auto begin = std::min(ids.size(), static_cast<size_t>(b) * chunk), end = begin + std::min(chunk, ids.size() - begin);
        for (int step = 0; step < comm.size(); ++step) {
            int dst = (comm.rank() + step) % comm.size(), src = (comm.rank() - step + comm.size()) % comm.size();
            std::vector<long long> send;
            for (size_t k = begin; k < end; ++k)
                if (static_cast<unsigned long long>(ids[k]) % comm.size() == static_cast<unsigned>(dst)) send.push_back(ids[k]);
            auto recv = halo::bounded_peer_exchange<long long>(comm, send, dst, src, chunk);
            bool valid = true;
            for (auto id : recv) {
                if (directory.size() >= opts.max_import_bytes / 64) {
                    valid = false;
                    break;
                }
                valid = directory.insert(id).second && valid;
            }
            // Conservative estimated hash-node/bucket overhead, not an RSS bound.
            valid = valid && directory.size() <= opts.max_import_bytes / 64;
            halo::collective_require(comm, valid, "point locator: duplicate global ID or directory budget exceeded");
        }
    }
}
bool inside(const V &p, const double *b) {
    for (int d = 0; d < 3; ++d)
        if (p[d] < b[d] || p[d] > b[d + 3]) return false;
    return true;
}
}  // namespace
struct DistributedPointLocator::Impl {
    halo::Communicator comm;
    PointLocationOptions opts;
    std::unique_ptr<solver::PointLocator> local;
    std::vector<double> rank_bounds;
    Impl(MPI_Comm c, PointLocationOptions o) : comm(halo::Communicator::duplicate_of(c)), opts(o) {}
};
DistributedPointLocator::DistributedPointLocator(const topology::UnstructuredMesh<Kokkos::HostSpace> &mesh, std::span<const index_t> ids,
                                                 MPI_Comm comm, PointLocationOptions opts)
    : impl_(std::make_unique<Impl>(comm, opts)) {
    auto &p = *impl_;
    bool valid = true;
    valid = std::isfinite(opts.tolerance) && opts.tolerance >= 0 && opts.tolerance < .01 && opts.max_message_bytes >= 24 &&
            opts.query_batch_size > 0 && opts.max_import_bytes >= 64;
    halo::collective_require(p.comm, valid, "point locator: invalid options");
    // All limits and geometric contracts must agree before collective routing.
    std::vector<unsigned long long> settings{opts.max_message_bytes, opts.query_batch_size, opts.max_import_bytes,
                                             mesh.coord_system() == topology::CoordinateSystem::Cartesian3D ? 0ULL : 1ULL};
    for (auto v : settings) {
        auto lo = halo::allreduce<unsigned long long>(p.comm, {v}, MPI_MIN)[0], hi = halo::allreduce<unsigned long long>(p.comm, {v}, MPI_MAX)[0];
        halo::collective_require(p.comm, lo == hi, "point locator: inconsistent collective options/coordinates");
    }
    auto lo = halo::allreduce<double>(p.comm, {opts.tolerance}, MPI_MIN)[0], hi = halo::allreduce<double>(p.comm, {opts.tolerance}, MPI_MAX)[0];
    halo::collective_require(p.comm, lo == hi, "point locator: inconsistent tolerance");
    // Empty mesh ranks may have unspecified boundary metadata.
    int model = mesh.n_cells() ? static_cast<int>(mesh.geometry_metadata().boundary_model) : 0;
    int maxmodel = halo::allreduce<int>(p.comm, {model}, MPI_MAX)[0];
    halo::collective_require(p.comm, !mesh.n_cells() || model == maxmodel, "point locator: inconsistent mesh boundary models");
    try {
        p.local = std::make_unique<solver::PointLocator>(mesh, ids, opts);
    } catch (const std::exception &) {
        valid = false;
    }
    halo::collective_require(p.comm, valid, "point locator: invalid local mesh or IDs");
    unique_ids(p.comm, ids, opts);
    auto bounds = p.local->bounds();
    p.rank_bounds = halo::allgather(p.comm, bounds.data(), 6);
}
DistributedPointLocator::~DistributedPointLocator() = default;
DistributedPointLocator::DistributedPointLocator(DistributedPointLocator &&) noexcept = default;
DistributedPointLocator &DistributedPointLocator::operator=(DistributedPointLocator &&) noexcept = default;
std::vector<solver::PointLocation> DistributedPointLocator::locate(solver::PointSetView points) {
    auto &p = *impl_;
    std::vector<V> xyz;
    bool valid = true;
    try {
        xyz = p.local->canonical_points(points);
    } catch (const std::exception &) {
        valid = false;
    }
    halo::collective_require(p.comm, valid, "point locator: invalid local points/IDs");
    unique_ids(p.comm, points.ids, p.opts);
    std::vector<solver::PointLocation> results(points.ids.size());
    for (size_t k = 0; k < results.size(); ++k) results[k].point_id = points.ids[k];
    size_t batch = std::min({p.opts.query_batch_size, p.opts.max_message_bytes / 24, static_cast<size_t>(INT_MAX / 3)});
    long long n = xyz.size() / batch + (xyz.size() % batch != 0);
    auto rounds = halo::allreduce<long long>(p.comm, {n}, MPI_MAX)[0];
    for (long long b = 0; b < rounds; ++b) {
        auto begin = std::min(xyz.size(), static_cast<size_t>(b) * batch), end = begin + std::min(batch, xyz.size() - begin);
        for (int step = 0; step < p.comm.size(); ++step) {
            int dst = (p.comm.rank() + step) % p.comm.size(), src = (p.comm.rank() - step + p.comm.size()) % p.comm.size();
            std::vector<double> send;
            std::vector<size_t> selected;
            for (size_t k = begin; k < end; ++k)
                if (inside(xyz[k], p.rank_bounds.data() + 6 * dst)) {
                    selected.push_back(k);
                    send.insert(send.end(), xyz[k].begin(), xyz[k].end());
                }
            auto recv = halo::bounded_peer_exchange<double>(p.comm, send, dst, src, batch * 3);
            std::vector<long long> response;
            response.reserve(recv.size());
            std::vector<V> incoming;
            incoming.reserve(recv.size() / 3);
            for (size_t k = 0; k < recv.size(); k += 3) incoming.push_back({recv[k], recv[k + 1], recv[k + 2]});
            for (const auto &match : p.local->locate_canonical(incoming))
                response.insert(response.end(), {match.cell_id, match.owner_local_index, static_cast<long long>(match.status)});
            auto matches = halo::bounded_peer_exchange<long long>(p.comm, response, src, dst, batch * 3);
            for (size_t j = 0; j < selected.size(); ++j) {
                auto &best = results[selected[j]];
                auto id = matches[3 * j];
                if (id >= 0 && (best.cell_id < 0 || id < best.cell_id)) {
                    best.cell_id = id;
                    best.owner_local_index = matches[3 * j + 1];
                    best.owner_rank = dst;
                    best.status = static_cast<solver::PointLocationStatus>(matches[3 * j + 2]);
                }
            }
        }
    }
    return results;
}
}  // namespace axis::distributed
