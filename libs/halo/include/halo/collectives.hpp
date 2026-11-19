#ifndef HALO_COLLECTIVES_HPP
#define HALO_COLLECTIVES_HPP

/// @file collectives.hpp
/// @brief Tier 1 low-level collective wrappers and shared non-template helpers.
///
/// This header declares the thin Tier 1 collective primitives (allgather,
/// allgatherv, allreduce) and the shared non-template detail helpers used to
/// build them. The templated wrappers are defined header-only in later tasks;
/// the non-template helpers declared in namespace halo::detail are implemented
/// in src/collectives.cpp.

#include <mpi.h>

#include <climits>
#include <cstddef>
#include <halo/communicator.hpp>
#include <halo/detail/mpi_datatype.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace halo {
namespace detail {

/// @brief Counts-to-displacements prefix sum (Req 2.1).
///
/// Produces a displacement array of the same length as @p counts where
/// displ[0] = 0 and displ[k] = counts[0] + counts[1] + ... + counts[k-1].
/// This is the single shared prefix-sum used by allgatherv and the
/// Replicated_Gather_Plan. Non-template; defined in src/collectives.cpp.
///
/// @param counts Per-rank element counts.
/// @return Displacement array (same length as @p counts).
std::vector<int> prefix_sum(const std::vector<int> &counts);

/// @brief Validate an Allgatherv counts array against the communicator size.
///
/// Performs argument validation BEFORE any MPI call is issued (Req 2.4, 2.5).
/// Non-template; defined in src/collectives.cpp.
///
/// @param counts Per-rank element counts.
/// @param comm_size Expected length of @p counts (the communicator size).
///
/// @throws std::invalid_argument if counts.size() != comm_size (the error
///         message names the expected length, comm_size).
/// @throws std::invalid_argument if any counts[k] is negative (the error
///         message names the offending rank index k).
void validate_counts(const std::vector<int> &counts, int comm_size);

}  // namespace detail

/// @brief Wrapper around MPI_Allgather for a fixed per-rank element count.
///
/// Every rank contributes @p count_per_rank elements from @p send_data; the
/// result is the ascending-rank concatenation of all ranks' send buffers, with
/// length `count_per_rank * comm.size()` (Req 1.1, 1.2). The call is serialized
/// through a detail::Serialized_MPI_Guard when the MPI thread level is below
/// MPI_THREAD_MULTIPLE (Req 1.4).
///
/// Zero-count fast path (Req 1.5): if @p count_per_rank is 0, returns an empty
/// buffer WITHOUT calling MPI_Allgather.
///
/// @tparam T Element type (resolved via detail::mpi_datatype_for<T>()).
/// @param comm           The communicator to gather over (non-owning).
/// @param send_data      Pointer to this rank's @p count_per_rank elements.
/// @param count_per_rank Number of elements each rank contributes.
/// @return Receive buffer of length `count_per_rank * comm.size()` holding all
///         ranks' contributions in ascending rank order.
///
/// @throws std::runtime_error via detail::handle_mpi_error if MPI_Allgather
///         fails (under ErrorPolicy::throw_on_error) (Req 1.3).
template <typename T>
[[nodiscard]] std::vector<T> allgather(const Communicator &comm, const T *send_data, int count_per_rank) {
    // Zero-count fast path (Req 1.5): no MPI call, no guard.
    if (count_per_rank == 0) {
        return {};
    }

    const int comm_size = comm.size();
    std::vector<T> recv(static_cast<std::size_t>(count_per_rank) * static_cast<std::size_t>(comm_size));

    detail::Serialized_MPI_Guard guard;  // Req 1.4
    const MPI_Datatype dt = detail::mpi_datatype_for<T>();
    const int rc = MPI_Allgather(send_data, count_per_rank, dt, recv.data(), count_per_rank, dt, comm.handle());  // Req 1.1, 1.2
    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, comm.rank(), "MPI_Allgather", comm.handle());  // Req 1.3
    }
    return recv;
}

/// @brief Convenience overload accepting a contiguous std::vector send buffer.
///
/// Forwards to the pointer overload using `send_data.data()` and
/// `send_data.size()` as the per-rank element count.
///
/// @tparam T Element type.
/// @param comm      The communicator to gather over (non-owning).
/// @param send_data This rank's contiguous send buffer.
/// @return Receive buffer of length `send_data.size() * comm.size()`.
///
/// @throws std::runtime_error via detail::handle_mpi_error if MPI_Allgather fails.
template <typename T>
[[nodiscard]] std::vector<T> allgather(const Communicator &comm, const std::vector<T> &send_data) {
    return allgather<T>(comm, send_data.data(), static_cast<int>(send_data.size()));
}

/// @brief Result of an allgatherv: the gathered buffer plus per-rank displacements.
///
/// Exposes both the gathered receive buffer and the per-rank displacement array
/// computed once from the per-rank counts via detail::prefix_sum (Req 2.3), so
/// callers can index each rank's contribution without recomputing offsets.
///
/// @tparam T Element type of the gathered buffer.
template <typename T>
struct Allgather_Result {
    /// Gathered contributions from all ranks, each placed at its computed displacement.
    std::vector<T> buffer;
    /// Per-rank displacements (displacements[k] = sum of counts[0..k-1]).
    std::vector<int> displacements;
};

/// @brief Wrapper around MPI_Allgatherv with prefix-sum displacements.
///
/// Gathers a variable, per-rank number of elements. The per-rank displacement
/// array is computed once inside the wrapper from @p counts via
/// detail::prefix_sum (Req 2.1), replacing a hand-rolled offset loop. Argument
/// validation runs BEFORE any MPI call (Req 2.4, 2.5): @p counts must have
/// length comm.size() and contain no negative entry. Exactly one MPI_Allgatherv
/// is issued on comm.handle() using @p counts and the computed displacements
/// (Req 2.2), serialized through a detail::Serialized_MPI_Guard when the MPI
/// thread level is below MPI_THREAD_MULTIPLE (Req 2.7). The returned
/// Allgather_Result carries both the gathered buffer and the displacements
/// (Req 2.3).
///
/// @tparam T Element type (resolved via detail::mpi_datatype_for<T>()).
/// @param comm      The communicator to gather over (non-owning).
/// @param send_data Pointer to this rank's `counts[comm.rank()]` elements.
/// @param counts    Per-rank element counts; length MUST equal comm.size().
/// @return Allgather_Result whose buffer has length `displs.back() + counts.back()`
///         (the total across all ranks) and whose displacements are the computed
///         per-rank offsets.
///
/// @throws std::invalid_argument via detail::validate_counts if @p counts has
///         the wrong length or contains a negative entry (before any MPI call)
///         (Req 2.4, 2.5).
/// @throws std::runtime_error via detail::handle_mpi_error if MPI_Allgatherv
///         fails (under ErrorPolicy::throw_on_error) (Req 2.6).
template <typename T>
[[nodiscard]] Allgather_Result<T> allgatherv(const Communicator &comm, const T *send_data, const std::vector<int> &counts) {
    const int comm_size = comm.size();
    detail::validate_counts(counts, comm_size);            // Req 2.4, 2.5 — BEFORE any MPI
    std::vector<int> displs = detail::prefix_sum(counts);  // Req 2.1 — computed once

    const int my_count = counts[static_cast<std::size_t>(comm.rank())];
    // counts is non-empty (comm_size >= 1 guaranteed), so back() is valid.
    const std::size_t total = static_cast<std::size_t>(displs.back()) + static_cast<std::size_t>(counts.back());

    Allgather_Result<T> result;
    result.buffer.resize(total);
    result.displacements = displs;  // Req 2.3

    detail::Serialized_MPI_Guard guard;  // Req 2.7
    const MPI_Datatype dt = detail::mpi_datatype_for<T>();
    const int rc = MPI_Allgatherv(send_data, my_count, dt, result.buffer.data(), counts.data(), displs.data(), dt,
                                  comm.handle());  // Req 2.2
    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, comm.rank(), "MPI_Allgatherv", comm.handle());  // Req 2.6
    }
    return result;
}

/// @brief Convenience overload accepting a contiguous std::vector send buffer.
///
/// Forwards to the pointer overload using `send_data.data()`. The number of
/// elements this rank contributes is taken from @p counts (specifically
/// `counts[comm.rank()]`), not from @p send_data.size().
///
/// @tparam T Element type.
/// @param comm      The communicator to gather over (non-owning).
/// @param send_data This rank's contiguous send buffer (must hold at least
///                  `counts[comm.rank()]` elements).
/// @param counts    Per-rank element counts; length MUST equal comm.size().
/// @return Allgather_Result with the gathered buffer and computed displacements.
///
/// @throws std::invalid_argument via detail::validate_counts on bad @p counts.
/// @throws std::runtime_error via detail::handle_mpi_error if MPI_Allgatherv fails.
template <typename T>
[[nodiscard]] Allgather_Result<T> allgatherv(const Communicator &comm, const std::vector<T> &send_data, const std::vector<int> &counts) {
    return allgatherv<T>(comm, send_data.data(), counts);
}

/// @brief Wrapper around MPI_Allreduce that reduces a whole vector in one call.
///
/// Issues exactly one MPI_Allreduce over the entire @p input buffer with the
/// caller-supplied @p op (Req 3.1), returning an output buffer of the same
/// length whose k-th element is the elementwise reduction of every rank's
/// input[k] across the communicator (Req 3.2). Supports at minimum MPI_MIN,
/// MPI_MAX, and MPI_SUM for integer element types (Req 3.3). The call is
/// serialized through a detail::Serialized_MPI_Guard when the MPI thread level
/// is below MPI_THREAD_MULTIPLE (Req 3.5).
///
/// Empty-input fast path: if @p input is empty, returns an empty buffer WITHOUT
/// calling MPI_Allreduce, mirroring the zero-count allgather fast path.
///
/// @tparam T Element type (resolved via detail::mpi_datatype_for<T>()).
/// @param comm  The communicator to reduce over (non-owning).
/// @param input This rank's contribution; every rank MUST supply the same length.
/// @param op    The MPI reduction operation (e.g. MPI_MIN, MPI_MAX, MPI_SUM).
/// @return Output buffer of length `input.size()` holding the elementwise
///         reduction across all ranks (identical on every rank).
///
/// @throws std::runtime_error via detail::handle_mpi_error if MPI_Allreduce
///         fails (under ErrorPolicy::throw_on_error) (Req 3.4).
template <typename T>
[[nodiscard]] std::vector<T> allreduce(const Communicator &comm, const std::vector<T> &input, MPI_Op op) {
    // Empty-input fast path: no MPI call, no guard.
    if (input.empty()) {
        return {};
    }

    std::vector<T> output(input.size());

    detail::Serialized_MPI_Guard guard;  // Req 3.5
    const MPI_Datatype dt = detail::mpi_datatype_for<T>();
    const int rc = MPI_Allreduce(input.data(), output.data(), static_cast<int>(input.size()), dt, op,
                                 comm.handle());  // Req 3.1, 3.2, 3.3
    if (rc != MPI_SUCCESS) {
        detail::handle_mpi_error(rc, comm.rank(), "MPI_Allreduce", comm.handle());  // Req 3.4
    }
    return output;
}

/// Collective precondition: all ranks throw together, before the next exchange.
inline void collective_require(const Communicator &comm, bool valid, const char *message) {
    if (allreduce<int>(comm, {valid ? 1 : 0}, MPI_MIN)[0] == 0) throw std::invalid_argument(message);
}

/// Variable-size peer exchange. Counts and displacements are in T elements.
/// The send buffer is grouped by destination rank; output is grouped by source.
/// Unlike allgatherv, this does not replicate each payload on every rank.
template <typename T>
struct Alltoall_Result {
    std::vector<T> buffer;
    std::vector<int> counts, displacements;
};

template <typename T>
[[nodiscard]] Alltoall_Result<T> alltoallv(const Communicator &comm, const std::vector<T> &send, const std::vector<int> &counts) {
    static_assert(detail::has_native_mpi_type_v<T>, "HALO alltoallv requires a supported MPI type");
    bool valid = counts.size() == static_cast<std::size_t>(comm.size());
    std::size_t total = 0;
    for (int n : counts) {
        valid = valid && n >= 0;
        if (n >= 0) total += static_cast<std::size_t>(n);
    }
    valid = valid && total == send.size() && total <= static_cast<std::size_t>(INT_MAX);
    collective_require(comm, valid, "halo::alltoallv: invalid counts, buffer length, or MPI count overflow");
    auto displacements = detail::prefix_sum(counts);
    Alltoall_Result<T> out;
    out.counts.resize(comm.size());
    {
        detail::Serialized_MPI_Guard guard;
        int rc = MPI_Alltoall(counts.data(), 1, MPI_INT, out.counts.data(), 1, MPI_INT, comm.handle());
        if (rc != MPI_SUCCESS) detail::handle_mpi_error(rc, comm.rank(), "MPI_Alltoall", comm.handle());
    }
    std::size_t received = 0;
    for (int n : out.counts) received += static_cast<std::size_t>(n);
    collective_require(comm, received <= static_cast<std::size_t>(INT_MAX), "halo::alltoallv: receive count overflow");
    out.displacements = detail::prefix_sum(out.counts);
    out.buffer.resize(received);
    {
        detail::Serialized_MPI_Guard guard;
        const auto dt = detail::mpi_datatype_for<T>();
        int rc = MPI_Alltoallv(send.data(), counts.data(), displacements.data(), dt, out.buffer.data(), out.counts.data(), out.displacements.data(),
                               dt, comm.handle());
        if (rc != MPI_SUCCESS) detail::handle_mpi_error(rc, comm.rank(), "MPI_Alltoallv", comm.handle());
    }
    return out;
}

}  // namespace halo

#endif  // HALO_COLLECTIVES_HPP
