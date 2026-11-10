#ifndef HALO_EXCHANGE_HPP
#define HALO_EXCHANGE_HPP

/// @file exchange.hpp
/// @brief Blocking and non-blocking halo exchange function templates.
///
/// Provides exchange_blocking<ViewType> which executes a synchronous halo
/// exchange using a precomputed Halo_Plan. Posts all MPI_Irecv before any
/// MPI_Isend, uses deterministic tag computation, and dispatches between
/// GPU-direct and host-staged paths at compile time via requires_staging_v.
///
/// Thread safety is ensured via detail::Serialized_MPI_Guard when the MPI
/// thread level is below MPI_THREAD_MULTIPLE.

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <chrono>
#include <cstddef>
#include <halo/communicator.hpp>
#include <halo/detail/compute_tag.hpp>
#include <halo/detail/gpu_aware_probe.hpp>
#include <halo/detail/memory_traits.hpp>
#include <halo/detail/staging.hpp>
#include <halo/diagnostics.hpp>
#include <halo/environment.hpp>
#include <halo/error_policy.hpp>
#include <halo/halo_handle.hpp>
#include <halo/halo_plan.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace halo {

namespace detail {

/// @brief Throw a std::runtime_error with MPI error string and failing rank.
///
/// @deprecated Use detail::handle_mpi_error() which respects the active ErrorPolicy.
/// This function is retained for backward compatibility and delegates to
/// handle_mpi_error() directly.
///
/// @param mpi_error_code The MPI error code returned by the failing call.
/// @param neighbor_rank  The rank of the neighbor involved in the failure.
/// @param operation      Description of the failing operation (e.g., "MPI_Irecv").
[[noreturn]] inline void throw_mpi_error(int mpi_error_code, int neighbor_rank, const char *operation) {
    handle_mpi_error(mpi_error_code, neighbor_rank, operation);
}

}  // namespace detail

/// @brief Blocking halo exchange.
///
/// Posts all MPI_Irecv before any MPI_Isend, then calls MPI_Waitall.
/// Dispatches GPU-direct or host-staged path at compile time based on
/// requires_staging_v<ViewType>.
///
/// @tparam ViewType A Kokkos::View type (1D).
/// @param plan The precomputed halo exchange plan.
/// @param view The Kokkos view containing data to exchange.
///
/// @throws std::runtime_error if any MPI operation fails.
///
/// @note Uses detail::Serialized_MPI_Guard for thread safety when MPI
///       thread level < MPI_THREAD_MULTIPLE.
template <typename ViewType>
void exchange_blocking(const Halo_Plan &plan, ViewType &view) {
    // Early return for empty plans (no neighbors in either direction)
    if (plan.num_send_neighbors() == 0 && plan.num_recv_neighbors() == 0) {
        return;
    }

    // ─── Diagnostics: begin event ───────────────────────────────────────────
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        const int num_neighbors = static_cast<int>(plan.num_send_neighbors() + plan.num_recv_neighbors());
        const std::size_t total_bytes = (plan.total_send_elements() + plan.total_recv_elements()) * sizeof(typename ViewType::value_type);
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::begin, plan.communicator().rank(), num_neighbors, total_bytes,
                                         std::chrono::nanoseconds{0},
                                         /*is_async=*/false});
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const auto &comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();

    // Determine MPI datatype from the view's value type
    using value_type = typename ViewType::value_type;
    MPI_Datatype mpi_dtype;
    if constexpr (std::is_same_v<value_type, double>) {
        mpi_dtype = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<value_type, float>) {
        mpi_dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<value_type, int>) {
        mpi_dtype = MPI_INT;
    } else if constexpr (std::is_same_v<value_type, long>) {
        mpi_dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<value_type, long long>) {
        mpi_dtype = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<value_type, unsigned int>) {
        mpi_dtype = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<value_type, unsigned long>) {
        mpi_dtype = MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<value_type, unsigned long long>) {
        mpi_dtype = MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<value_type, char>) {
        mpi_dtype = MPI_CHAR;
    } else {
        // Fallback: treat as raw bytes
        mpi_dtype = MPI_BYTE;
    }

    const auto recv_info = plan.recv_info();
    const auto send_info = plan.send_info();

    const std::size_t num_recv = recv_info.size();
    const std::size_t num_send = send_info.size();
    const std::size_t total_requests = num_recv + num_send;

    // Allocate request array for MPI_Waitall
    std::vector<MPI_Request> requests(total_requests, MPI_REQUEST_NULL);

    if constexpr (detail::requires_staging_v<ViewType>) {
        // The compile-time trait says staging is needed (device view, no
        // HALO_GPU_AWARE_MPI flag). However, runtime detection may reveal that
        // GPU-aware MPI IS available — in that case, skip staging and use
        // the direct path.
        if (Environment::is_gpu_aware_mpi()) {
            // ─── Runtime GPU-aware path: pass device pointers directly ───────
            std::size_t recv_offset = plan.total_send_elements();
            for (std::size_t i = 0; i < num_recv; ++i) {
                const auto &neighbor = recv_info[i];
                const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

                int rc =
                    MPI_Irecv(view.data() + recv_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests[i]);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
                }

                recv_offset += neighbor.count;
            }

            std::size_t send_offset = 0;
            for (std::size_t i = 0; i < num_send; ++i) {
                const auto &neighbor = send_info[i];
                const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

                int rc = MPI_Isend(view.data() + send_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm,
                                   &requests[num_recv + i]);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
                }

                send_offset += neighbor.count;
            }

            int rc = MPI_Waitall(static_cast<int>(total_requests), requests.data(), MPI_STATUSES_IGNORE);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, my_rank, "MPI_Waitall");
            }
        } else {
            // ─── Staged path: device view requires host buffers ─────────────────

            // Allocate host receive buffers
            std::vector<detail::host_mirror_t<ViewType>> recv_buffers;
            recv_buffers.reserve(num_recv);

            // Post all MPI_Irecv first (into host buffers)
            std::size_t recv_offset = plan.total_send_elements();
            for (std::size_t i = 0; i < num_recv; ++i) {
                const auto &neighbor = recv_info[i];
                const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

                // Allocate host buffer for this receive
                auto host_buf =
                    Kokkos::View<value_type *, Kokkos::HostSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "recv_buf"), neighbor.count);
                recv_buffers.push_back(host_buf);

                int rc = MPI_Irecv(host_buf.data(), static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests[i]);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
                }
            }

            // Stage send data from device to host, then post MPI_Isend
            std::vector<detail::host_mirror_t<ViewType>> send_buffers;
            send_buffers.reserve(num_send);

            std::size_t send_offset = 0;
            for (std::size_t i = 0; i < num_send; ++i) {
                const auto &neighbor = send_info[i];
                const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

                // Deep-copy send region from device to host
                auto host_buf = detail::stage_send(view, send_offset, neighbor.count);
                send_buffers.push_back(host_buf);

                int rc =
                    MPI_Isend(host_buf.data(), static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests[num_recv + i]);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
                }

                send_offset += neighbor.count;
            }

            // Wait for all operations to complete
            int rc = MPI_Waitall(static_cast<int>(total_requests), requests.data(), MPI_STATUSES_IGNORE);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, my_rank, "MPI_Waitall");
            }

            // Deep-copy received data from host buffers back to device view
            recv_offset = plan.total_send_elements();
            for (std::size_t i = 0; i < num_recv; ++i) {
                const auto &neighbor = recv_info[i];
                detail::stage_recv<ViewType>(recv_buffers[i], view, recv_offset, neighbor.count);
                recv_offset += neighbor.count;
            }

        }  // end staged path
    } else {
        // ─── Direct path: host view or GPU-aware MPI ────────────────────────

        // Post all MPI_Irecv first (directly into view)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto &neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            int rc = MPI_Irecv(view.data() + recv_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &requests[i]);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
            }

            recv_offset += neighbor.count;
        }

        // Post all MPI_Isend (directly from view)
        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto &neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            int rc = MPI_Isend(view.data() + send_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm,
                               &requests[num_recv + i]);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
            }

            send_offset += neighbor.count;
        }

        // Wait for all operations to complete
        int rc = MPI_Waitall(static_cast<int>(total_requests), requests.data(), MPI_STATUSES_IGNORE);

        if (rc != MPI_SUCCESS) {
            detail::throw_mpi_error(rc, my_rank, "MPI_Waitall");
        }
    }

    // ─── Diagnostics: end event ─────────────────────────────────────────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        const int num_neighbors = static_cast<int>(plan.num_send_neighbors() + plan.num_recv_neighbors());
        const std::size_t total_bytes = (plan.total_send_elements() + plan.total_recv_elements()) * sizeof(typename ViewType::value_type);
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::end, plan.communicator().rank(), num_neighbors, total_bytes,
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
                                         /*is_async=*/false});
    }
}

/// @brief Non-blocking asynchronous halo exchange.
///
/// Posts all MPI_Irecv before any MPI_Isend, wraps each request in a
/// Request_Guard, and returns a Halo_Handle that owns all pending operations.
/// Dispatches GPU-direct or host-staged path at compile time based on
/// requires_staging_v<ViewType>.
///
/// @tparam ViewType A Kokkos::View type (1D).
/// @param plan The precomputed halo exchange plan.
/// @param view The Kokkos view containing data to exchange.
/// @return A Halo_Handle owning all pending Request_Guard objects.
///
/// @throws std::runtime_error if any MPI operation fails.
///
/// @note Uses detail::Serialized_MPI_Guard for thread safety when MPI
///       thread level < MPI_THREAD_MULTIPLE.
template <typename ViewType>
[[nodiscard]] Halo_Handle exchange_async(const Halo_Plan &plan, ViewType &view) {
    // Early return for empty plans (no neighbors in either direction)
    if (plan.num_send_neighbors() == 0 && plan.num_recv_neighbors() == 0) {
        return Halo_Handle{};
    }

    // ─── Diagnostics: begin event ───────────────────────────────────────────
    const bool diag_active = Diagnostics::is_active();
    std::chrono::steady_clock::time_point diag_t0;
    if (diag_active) {
        diag_t0 = std::chrono::steady_clock::now();
        const int num_neighbors = static_cast<int>(plan.num_send_neighbors() + plan.num_recv_neighbors());
        const std::size_t total_bytes = (plan.total_send_elements() + plan.total_recv_elements()) * sizeof(typename ViewType::value_type);
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::begin, plan.communicator().rank(), num_neighbors, total_bytes,
                                         std::chrono::nanoseconds{0},
                                         /*is_async=*/true});
    }

    // Acquire serialization guard for thread safety
    detail::Serialized_MPI_Guard guard;

    const auto &comm = plan.communicator();
    const int my_rank = comm.rank();
    const int comm_size = comm.size();
    const MPI_Comm mpi_comm = comm.handle();

    // Determine MPI datatype from the view's value type
    using value_type = typename ViewType::value_type;
    MPI_Datatype mpi_dtype;
    if constexpr (std::is_same_v<value_type, double>) {
        mpi_dtype = MPI_DOUBLE;
    } else if constexpr (std::is_same_v<value_type, float>) {
        mpi_dtype = MPI_FLOAT;
    } else if constexpr (std::is_same_v<value_type, int>) {
        mpi_dtype = MPI_INT;
    } else if constexpr (std::is_same_v<value_type, long>) {
        mpi_dtype = MPI_LONG;
    } else if constexpr (std::is_same_v<value_type, long long>) {
        mpi_dtype = MPI_LONG_LONG;
    } else if constexpr (std::is_same_v<value_type, unsigned int>) {
        mpi_dtype = MPI_UNSIGNED;
    } else if constexpr (std::is_same_v<value_type, unsigned long>) {
        mpi_dtype = MPI_UNSIGNED_LONG;
    } else if constexpr (std::is_same_v<value_type, unsigned long long>) {
        mpi_dtype = MPI_UNSIGNED_LONG_LONG;
    } else if constexpr (std::is_same_v<value_type, char>) {
        mpi_dtype = MPI_CHAR;
    } else {
        // Fallback: treat as raw bytes
        mpi_dtype = MPI_BYTE;
    }

    const auto recv_info = plan.recv_info();
    const auto send_info = plan.send_info();

    const std::size_t num_recv = recv_info.size();
    const std::size_t num_send = send_info.size();

    // Build the Halo_Handle that will own all Request_Guards
    Halo_Handle handle;
    handle.requests_.reserve(num_recv + num_send);

    if constexpr (detail::requires_staging_v<ViewType>) {
        // The compile-time trait says staging is needed (device view, no
        // HALO_GPU_AWARE_MPI flag). However, runtime detection may reveal that
        // GPU-aware MPI IS available — in that case, skip staging and use
        // the direct path.
        if (Environment::is_gpu_aware_mpi()) {
            // ─── Runtime GPU-aware path: pass device pointers directly ───────
            std::size_t recv_offset = plan.total_send_elements();
            for (std::size_t i = 0; i < num_recv; ++i) {
                const auto &neighbor = recv_info[i];
                const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

                MPI_Request req = MPI_REQUEST_NULL;
                int rc = MPI_Irecv(view.data() + recv_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &req);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
                }

                handle.requests_.emplace_back(req);
                recv_offset += neighbor.count;
            }

            std::size_t send_offset = 0;
            for (std::size_t i = 0; i < num_send; ++i) {
                const auto &neighbor = send_info[i];
                const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

                MPI_Request req = MPI_REQUEST_NULL;
                int rc = MPI_Isend(view.data() + send_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &req);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
                }

                handle.requests_.emplace_back(req);
                send_offset += neighbor.count;
            }
        } else {
            // ─── Staged path: device view requires host buffers ─────────────────

            // Allocate host receive buffers
            std::vector<detail::host_mirror_t<ViewType>> recv_buffers;
            recv_buffers.reserve(num_recv);

            // Post all MPI_Irecv first (into host buffers)
            std::size_t recv_offset = plan.total_send_elements();
            for (std::size_t i = 0; i < num_recv; ++i) {
                const auto &neighbor = recv_info[i];
                const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

                // Allocate host buffer for this receive
                auto host_buf =
                    Kokkos::View<value_type *, Kokkos::HostSpace>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "recv_buf"), neighbor.count);
                recv_buffers.push_back(host_buf);

                MPI_Request req = MPI_REQUEST_NULL;
                int rc = MPI_Irecv(host_buf.data(), static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &req);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
                }

                handle.requests_.emplace_back(req);
            }

            // Stage send data from device to host, then post MPI_Isend
            std::vector<detail::host_mirror_t<ViewType>> send_buffers;
            send_buffers.reserve(num_send);

            std::size_t send_offset = 0;
            for (std::size_t i = 0; i < num_send; ++i) {
                const auto &neighbor = send_info[i];
                const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

                // Deep-copy send region from device to host
                auto host_buf = detail::stage_send(view, send_offset, neighbor.count);
                send_buffers.push_back(host_buf);

                MPI_Request req = MPI_REQUEST_NULL;
                int rc = MPI_Isend(host_buf.data(), static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &req);

                if (rc != MPI_SUCCESS) {
                    detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
                }

                handle.requests_.emplace_back(req);
                send_offset += neighbor.count;
            }

            // Attach staged receive state: captures recv buffers and device view
            // reference for post-receive deep-copy upon completion.
            auto staged = std::make_unique<Halo_Handle::Staged_Recv>();
            staged->post_recv_copy = [recv_buffers = std::move(recv_buffers), send_buffers = std::move(send_buffers), &view, recv_info,
                                      total_send = plan.total_send_elements()]() mutable {
                std::size_t offset = total_send;
                for (std::size_t i = 0; i < recv_info.size(); ++i) {
                    const auto &neighbor = recv_info[i];
                    detail::stage_recv<ViewType>(recv_buffers[i], view, offset, neighbor.count);
                    offset += neighbor.count;
                }
            };
            handle.staged_recv_ = std::move(staged);

        }  // end staged path (runtime check: not gpu-aware)
    } else {
        // ─── Direct path: host view or GPU-aware MPI ────────────────────────

        // Post all MPI_Irecv first (directly into view)
        std::size_t recv_offset = plan.total_send_elements();
        for (std::size_t i = 0; i < num_recv; ++i) {
            const auto &neighbor = recv_info[i];
            const int tag = detail::compute_tag(neighbor.rank, my_rank, comm_size);

            MPI_Request req = MPI_REQUEST_NULL;
            int rc = MPI_Irecv(view.data() + recv_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &req);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Irecv");
            }

            handle.requests_.emplace_back(req);
            recv_offset += neighbor.count;
        }

        // Post all MPI_Isend (directly from view)
        std::size_t send_offset = 0;
        for (std::size_t i = 0; i < num_send; ++i) {
            const auto &neighbor = send_info[i];
            const int tag = detail::compute_tag(my_rank, neighbor.rank, comm_size);

            MPI_Request req = MPI_REQUEST_NULL;
            int rc = MPI_Isend(view.data() + send_offset, static_cast<int>(neighbor.count), mpi_dtype, neighbor.rank, tag, mpi_comm, &req);

            if (rc != MPI_SUCCESS) {
                detail::throw_mpi_error(rc, neighbor.rank, "MPI_Isend");
            }

            handle.requests_.emplace_back(req);
            send_offset += neighbor.count;
        }
    }

    // ─── Diagnostics: end event ─────────────────────────────────────────────
    if (diag_active) {
        const auto elapsed = std::chrono::steady_clock::now() - diag_t0;
        const int num_neighbors = static_cast<int>(plan.num_send_neighbors() + plan.num_recv_neighbors());
        const std::size_t total_bytes = (plan.total_send_elements() + plan.total_recv_elements()) * sizeof(typename ViewType::value_type);
        Diagnostics::emit(Exchange_Event{Exchange_Event::Phase::end, plan.communicator().rank(), num_neighbors, total_bytes,
                                         std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed),
                                         /*is_async=*/true});
    }

    return handle;
}

}  // namespace halo

#endif  // HALO_EXCHANGE_HPP
