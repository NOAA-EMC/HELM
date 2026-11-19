// SPDX-License-Identifier: Apache-2.0
#ifndef HALO_INDEXED_GATHER_PLAN_HPP
#define HALO_INDEXED_GATHER_PLAN_HPP

/// @file indexed_gather_plan.hpp
/// @brief Reusable owner/request gather and reduce plan for rank-contiguous
///        global IDs, built on packed HALO neighbor exchanges.

#include <algorithm>
#include <climits>
#include <halo/collectives.hpp>
#include <halo/exchange.hpp>
#include <limits>
#include <span>

namespace halo {

/// Reusable owner/request exchange for rank-contiguous global IDs. Construction
/// and each operation are collective, including on ranks with empty requests.
/// Payloads use packed HALO neighbor exchanges, not replicated field gathers.
/// Owns an isolated communicator; do not destroy while operations are active.
class Indexed_Gather_Plan {
    Communicator comm_;
    std::size_t owned_{};
    std::vector<long long> offsets_;
    std::vector<std::size_t> request_order_, export_indices_;
    Halo_Plan forward_, reverse_;

    static Halo_Plan empty_plan(const Communicator &comm) {
        return Halo_Plan(comm, {}, {});
    }

   public:
    /// @brief Exchange ownership and build the forward/reverse transfer plans.
    ///
    /// Collective, including on ranks with no requests. Global IDs are
    /// rank-contiguous: rank r owns [offsets[r], offsets[r+1]) after an
    /// allgather of local_owned counts.
    /// @param comm         Communicator whose ownership partitions the ID space.
    /// @param local_owned  Number of globally unique cells owned by this rank.
    /// @param requested    Global IDs this rank needs from other ranks.
    /// @throws std::runtime_error collectively when counts overflow int or a
    ///         requested ID lies outside the owned global range.
    Indexed_Gather_Plan(const Communicator &comm, std::size_t local_owned, std::span<const long long> requested)
        : comm_(comm.duplicate()), owned_(local_owned), forward_(empty_plan(comm_)), reverse_(empty_plan(comm_)) {
        collective_require(comm_, local_owned <= static_cast<std::size_t>(INT_MAX) && requested.size() <= static_cast<std::size_t>(INT_MAX),
                           "Indexed_Gather_Plan: MPI count overflow");
        long long n = static_cast<long long>(local_owned);
        auto sizes = allgather(comm_, &n, 1);
        offsets_.push_back(0);
        for (auto count : sizes) offsets_.push_back(offsets_.back() + count);
        bool valid = true;
        for (auto id : requested) valid = valid && id >= 0 && id < offsets_.back();
        collective_require(comm_, valid, "Indexed_Gather_Plan: requested ID has no owner");
        std::vector<std::vector<std::pair<long long, std::size_t>>> by_owner(comm_.size());
        for (std::size_t k = 0; k < requested.size(); ++k) {
            int owner = static_cast<int>(std::upper_bound(offsets_.begin(), offsets_.end(), requested[k]) - offsets_.begin()) - 1;
            by_owner[owner].push_back({requested[k] - offsets_[owner], k});
        }
        std::vector<int> counts(comm_.size());
        std::vector<long long> ids;
        for (int r = 0; r < comm_.size(); ++r) {
            counts[r] = static_cast<int>(by_owner[r].size());
            for (auto [id, k] : by_owner[r]) {
                ids.push_back(id);
                request_order_.push_back(k);
            }
        }
        auto incoming = alltoallv(comm_, ids, counts);
        for (auto id : incoming.buffer) export_indices_.push_back(static_cast<std::size_t>(id));
        std::vector<Neighbor_Info> sends, receives;
        for (int r = 0; r < comm_.size(); ++r) {
            if (incoming.counts[r]) sends.push_back({r, static_cast<std::size_t>(incoming.counts[r])});
            if (counts[r]) receives.push_back({r, static_cast<std::size_t>(counts[r])});
        }
        forward_ = Halo_Plan(comm_, sends, receives);
        reverse_ = Halo_Plan(comm_, receives, sends);
    }
    // Halo_Plan holds references to comm_: moving would invalidate them.
    Indexed_Gather_Plan(const Indexed_Gather_Plan &) = delete;
    Indexed_Gather_Plan &operator=(const Indexed_Gather_Plan &) = delete;
    Indexed_Gather_Plan(Indexed_Gather_Plan &&) = delete;
    Indexed_Gather_Plan &operator=(Indexed_Gather_Plan &&) = delete;
    /// @brief Number of requested IDs on this rank.
    [[nodiscard]] std::size_t size() const noexcept {
        return request_order_.size();
    }
    /// @brief Number of globally unique cells owned by this rank.
    [[nodiscard]] std::size_t local_owned() const noexcept {
        return owned_;
    }
    /// @brief Exclusive-start global ID offset of each rank (size+1 entries).
    [[nodiscard]] const auto &ownership_offsets() const noexcept {
        return offsets_;
    }
    /// @brief Number of owned cells this rank exports to other requestors.
    [[nodiscard]] std::size_t send_elements() const noexcept {
        return export_indices_.size();
    }

    /// @brief Fetch one value per requested global ID (collective).
    ///
    /// Packs the exported subset, runs the forward HALO exchange, and
    /// reassembles results in the caller's request order.
    /// @tparam T           Type with a native MPI datatype.
    /// @param local_values Values on this rank's owned cells, in owned order.
    /// @return Gathered values aligned with the constructor's requested IDs.
    /// @throws std::runtime_error collectively when local_values extent differs
    ///         from the owned count supplied at construction.
    template <typename T>
    [[nodiscard]] std::vector<T> gather(std::span<const T> local_values) const {
        static_assert(detail::has_native_mpi_type_v<T>, "Indexed gather requires a supported MPI type");
        collective_require(comm_, local_values.size() == owned_, "Indexed_Gather_Plan::gather: owned extent mismatch");
        Kokkos::View<T *, Kokkos::HostSpace> packed("halo_indexed_gather", export_indices_.size() + size());
        for (std::size_t k = 0; k < export_indices_.size(); ++k) packed(k) = local_values[export_indices_[k]];
        exchange_blocking(forward_, packed);
        std::vector<T> result(size());
        for (std::size_t k = 0; k < size(); ++k) result[request_order_[k]] = packed(export_indices_.size() + k);
        return result;
    }

    /// @brief Reverse the dependency exchange and reduce requestor values on owners.
    ///
    /// Duplicate requests are reduced too. Unrequested IDs use the supplied
    /// identity (one for limiter minima, zero for inventory sums).
    /// @param requested_values One value per requested ID, in request order.
    /// @param op               MPI_MIN or MPI_SUM.
    /// @param identity         Neutral element for op, filling unrequested owned cells.
    /// @return Reduced values over this rank's owned cells, in owned order.
    /// @throws std::runtime_error collectively on extent mismatch or an unsupported op.
    [[nodiscard]] std::vector<double> reduce(std::span<const double> requested_values, MPI_Op op, double identity) const {
        collective_require(comm_, requested_values.size() == size() && (op == MPI_MIN || op == MPI_SUM),
                           "Indexed_Gather_Plan::reduce: invalid extent or operation");
        Kokkos::View<double *, Kokkos::HostSpace> packed("halo_indexed_reduce", size() + export_indices_.size());
        for (std::size_t k = 0; k < size(); ++k) packed(k) = requested_values[request_order_[k]];
        exchange_blocking(reverse_, packed);
        std::vector<double> result(owned_, identity);
        for (std::size_t k = 0; k < export_indices_.size(); ++k) {
            double x = packed(size() + k);
            auto i = export_indices_[k];
            result[i] = op == MPI_MIN ? std::min(result[i], x) : result[i] + x;
        }
        return result;
    }
};
}  // namespace halo
#endif
