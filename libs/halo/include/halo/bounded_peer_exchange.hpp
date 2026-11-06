// SPDX-License-Identifier: Apache-2.0
#ifndef HALO_BOUNDED_PEER_EXCHANGE_HPP
#define HALO_BOUNDED_PEER_EXCHANGE_HPP

/// @file bounded_peer_exchange.hpp
/// @brief Collective bounded variable-length exchange along a peer permutation.

#include <climits>
#include <halo/collectives.hpp>
#include <halo/exchange.hpp>
#include <span>
namespace halo {
/// @brief Collective, bounded variable-length exchange along a peer permutation.
///
/// Uses existing packed HALO plans; no replicated payload. Header
/// counts are checked on all ranks before allocating or transferring payloads.
/// Caller must isolate the communicator from concurrent exchanges.
/// @tparam T           Type with a native MPI datatype.
/// @param comm         Communicator whose ranks form the peer permutation.
/// @param send         Payload this rank delivers to `destination`.
/// @param destination  Peer receiving this rank's payload.
/// @param source       Peer whose payload this rank receives.
/// @param max_elements Upper bound on any single transfer in this call.
/// @return Received payload from `source`, sized by the exchanged header count.
/// @throws std::runtime_error collectively on invalid peers, budget overflow,
///         or mismatched send/receive bounds.
template <class T>
std::vector<T> bounded_peer_exchange(const Communicator &comm, std::span<const T> send, int destination, int source, std::size_t max_elements) {
    static_assert(detail::has_native_mpi_type_v<T>);
    collective_require(
        comm,
        destination >= 0 && destination < comm.size() && source >= 0 && source < comm.size() && send.size() <= max_elements && send.size() <= INT_MAX,
        "HALO bounded peer exchange: invalid peer or send budget exceeded");
    Halo_Plan header(comm, {{destination, 1}}, {{source, 1}});
    Kokkos::View<long long *, Kokkos::HostSpace> counts("halo_peer_counts", 2);
    counts(0) = static_cast<long long>(send.size());
    exchange_blocking(header, counts);
    collective_require(comm, counts(1) >= 0 && static_cast<unsigned long long>(counts(1)) <= max_elements && counts(1) <= INT_MAX,
                       "HALO bounded peer exchange: receive budget exceeded");
    const auto received = static_cast<std::size_t>(counts(1));
    Halo_Plan payload(comm, send.empty() ? std::vector<Neighbor_Info>{} : std::vector<Neighbor_Info>{{destination, send.size()}},
                      received ? std::vector<Neighbor_Info>{{source, received}} : std::vector<Neighbor_Info>{});
    Kokkos::View<T *, Kokkos::HostSpace> packed(Kokkos::view_alloc(Kokkos::WithoutInitializing, "halo_peer_payload"), send.size() + received);
    for (std::size_t k = 0; k < send.size(); ++k) packed(k) = send[k];
    exchange_blocking(payload, packed);
    std::vector<T> result(received);
    for (std::size_t k = 0; k < received; ++k) result[k] = packed(send.size() + k);
    return result;
}
}  // namespace halo
#endif
