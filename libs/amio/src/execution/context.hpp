#ifndef AMIO_SRC_EXECUTION_CONTEXT_HPP
#define AMIO_SRC_EXECUTION_CONTEXT_HPP

#include "amio/amio_errors.h"

#include <cstdint>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#include <halo/communicator.hpp>
#endif

namespace amio::detail {

class RoleMap;

enum class ExecutionMode : std::uint8_t { Direct, Aggregated, Stream };
enum class ProgressPolicy : std::uint8_t { Caller, Background };

struct RoleConfiguration {
    ExecutionMode mode = ExecutionMode::Direct;
    std::uint32_t rank_count = 1;
    std::vector<int> producers;
    std::vector<int> writers;
    std::vector<int> consumers;
};

/** Validate a disjoint static role partition in linear time, capped at one million ranks. */
[[nodiscard]] amio_status_t validate_roles(const RoleConfiguration &roles);
[[nodiscard]] amio_status_t validate_progress_support(ProgressPolicy policy, int provided_thread_level) noexcept;

class ExecutionContext {
   public:
    ExecutionContext(const ExecutionContext &) = delete;
    ExecutionContext &operator=(const ExecutionContext &) = delete;

    [[nodiscard]] const RoleConfiguration &roles() const noexcept { return roles_; }
    [[nodiscard]] ProgressPolicy progress_policy() const noexcept { return progress_; }
    /** Return the immutable, validated rank-role and producer-to-writer map. */
    [[nodiscard]] const RoleMap &role_map() const noexcept;
    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
    [[nodiscard]] std::uint32_t size() const noexcept { return roles_.rank_count; }
#ifdef AMIO_HAS_MPI
    /** Return the parent ordering communicator used by existing collective paths. */
    [[nodiscard]] MPI_Comm communicator() const noexcept { return parent_.handle(); }
    /** Return the context-owned communicator reserved for control messages. */
    [[nodiscard]] MPI_Comm control_communicator() const noexcept { return control_.handle(); }
    /** Return the context-owned communicator reserved for payload messages. */
    [[nodiscard]] MPI_Comm data_communicator() const noexcept { return data_.handle(); }
    /** Return the HALO-owned producer group, or MPI_COMM_NULL for service-only ranks. */
    [[nodiscard]] MPI_Comm producer_communicator() const noexcept {
        return producer_group_.has_value() ? producer_group_->handle() : MPI_COMM_NULL;
    }
    /** Return the HALO-owned storage-writer group, or MPI_COMM_NULL for non-writers. */
    [[nodiscard]] MPI_Comm writer_communicator() const noexcept {
        return writer_group_.has_value() ? writer_group_->handle() : MPI_COMM_NULL;
    }
    /** Return the HALO-owned consumer group, or MPI_COMM_NULL for non-consumers. */
    [[nodiscard]] MPI_Comm consumer_communicator() const noexcept {
        return consumer_group_.has_value() ? consumer_group_->handle() : MPI_COMM_NULL;
    }
    /** Return the HALO-owned non-producing service group (writers or consumers). */
    [[nodiscard]] MPI_Comm service_communicator() const noexcept {
        return service_group_.has_value() ? service_group_->handle() : MPI_COMM_NULL;
    }
#endif
    /** Return whether this rank may open a persistent storage destination. */
    [[nodiscard]] bool can_open_storage() const noexcept;
    /** Return the first configured writer, or rank zero for a direct context. */
    [[nodiscard]] std::uint32_t storage_root_rank() const noexcept;

    /** Collectively validate config agreement and duplicate the caller communicator. */
    static amio_status_t create(const RoleConfiguration &roles, ProgressPolicy progress, const std::string &canonical_configuration,
#ifdef AMIO_HAS_MPI
                                MPI_Comm caller_communicator,
#else
                                int caller_communicator,
#endif
                                std::unique_ptr<ExecutionContext> &out_context);

   private:
    ExecutionContext(RoleConfiguration roles, ProgressPolicy progress, std::uint32_t rank
#ifdef AMIO_HAS_MPI
                     , halo::Communicator parent, halo::Communicator control, halo::Communicator data,
                     std::optional<halo::Communicator> producer_group, std::optional<halo::Communicator> writer_group,
                     std::optional<halo::Communicator> consumer_group, std::optional<halo::Communicator> service_group
#endif
                     , std::shared_ptr<const RoleMap> role_map
                     );

    RoleConfiguration roles_;
    ProgressPolicy progress_;
    std::uint32_t rank_;
    std::shared_ptr<const RoleMap> role_map_;
#ifdef AMIO_HAS_MPI
    halo::Communicator parent_;
    halo::Communicator control_;
    halo::Communicator data_;
    std::optional<halo::Communicator> producer_group_;
    std::optional<halo::Communicator> writer_group_;
    std::optional<halo::Communicator> consumer_group_;
    std::optional<halo::Communicator> service_group_;
#endif
};

/** Compare bounded metadata against the storage root without allocating a second copy.
 * All context members must call in the same order with the same byte limit.
 */
[[nodiscard]] amio_status_t compare_context_metadata(const ExecutionContext &context, std::string_view value,
                                                      std::size_t max_bytes) noexcept;

} // namespace amio::detail

#endif // AMIO_SRC_EXECUTION_CONTEXT_HPP
