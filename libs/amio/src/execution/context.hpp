#ifndef AMIO_SRC_EXECUTION_CONTEXT_HPP
#define AMIO_SRC_EXECUTION_CONTEXT_HPP

#include "amio/amio_errors.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#include <halo/communicator.hpp>
#endif

namespace amio::detail {

enum class ExecutionMode : std::uint8_t { Direct, Aggregated, Stream };
enum class ProgressPolicy : std::uint8_t { Caller, Background };

struct RoleConfiguration {
    ExecutionMode mode = ExecutionMode::Direct;
    std::uint32_t rank_count = 1;
    std::vector<int> producers;
    std::vector<int> writers;
    std::vector<int> consumers;
};

[[nodiscard]] amio_status_t validate_roles(const RoleConfiguration &roles);
[[nodiscard]] amio_status_t validate_progress_support(ProgressPolicy policy, int provided_thread_level) noexcept;

class ExecutionContext {
   public:
    ExecutionContext(const ExecutionContext &) = delete;
    ExecutionContext &operator=(const ExecutionContext &) = delete;

    [[nodiscard]] const RoleConfiguration &roles() const noexcept { return roles_; }
    [[nodiscard]] ProgressPolicy progress_policy() const noexcept { return progress_; }
    [[nodiscard]] std::uint32_t rank() const noexcept { return rank_; }
    [[nodiscard]] std::uint32_t size() const noexcept { return roles_.rank_count; }
#ifdef AMIO_HAS_MPI
    [[nodiscard]] MPI_Comm communicator() const noexcept { return parent_.handle(); }
#endif

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
                     , halo::Communicator parent
#endif
                     );

    RoleConfiguration roles_;
    ProgressPolicy progress_;
    std::uint32_t rank_;
#ifdef AMIO_HAS_MPI
    halo::Communicator parent_;
#endif
};

} // namespace amio::detail

#endif // AMIO_SRC_EXECUTION_CONTEXT_HPP
