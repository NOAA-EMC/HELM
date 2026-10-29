#include "execution/context.hpp"
#include "workers/comm_split.hpp"

#include <algorithm>
#include <array>
#include <sstream>
#include <set>
#include <utility>

namespace amio::detail {
namespace {

bool add_role(const std::vector<int> &ranks, std::uint32_t rank_count, std::set<int> &assigned) {
    for (const int rank : ranks) {
        if (rank < 0 || static_cast<std::uint32_t>(rank) >= rank_count || !assigned.insert(rank).second) {
            return false;
        }
    }
    return true;
}

std::string full_configuration(const RoleConfiguration &roles, ProgressPolicy progress, const std::string &canonical) {
    std::ostringstream out;
    out << canonical << "\nmode=" << static_cast<unsigned int>(roles.mode) << ";progress=" << static_cast<unsigned int>(progress)
        << ";ranks=" << roles.rank_count << ";producers=";
    for (int rank : roles.producers) out << rank << ',';
    out << ";writers=";
    for (int rank : roles.writers) out << rank << ',';
    out << ";consumers=";
    for (int rank : roles.consumers) out << rank << ',';
    return out.str();
}

#ifdef AMIO_HAS_MPI
amio_status_t agree_on_configuration(MPI_Comm comm, const std::string &canonical) {
    const unsigned long long local_size = static_cast<unsigned long long>(canonical.size());
    unsigned long long min_size = 0;
    unsigned long long max_size = 0;
    if (MPI_Allreduce(&local_size, &min_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm) != MPI_SUCCESS ||
        MPI_Allreduce(&local_size, &max_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm) != MPI_SUCCESS) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    if (min_size != max_size) {
        return AMIO_ERR_INVALID_POLICY;
    }

    constexpr std::size_t kChunkSize = 4096;
    std::array<unsigned char, kChunkSize> local{};
    std::array<unsigned char, kChunkSize> min_values{};
    std::array<unsigned char, kChunkSize> max_values{};
    for (std::size_t offset = 0; offset < canonical.size(); offset += kChunkSize) {
        local.fill(0);
        min_values.fill(0);
        max_values.fill(0);
        const auto count = std::min(kChunkSize, canonical.size() - offset);
        for (std::size_t i = 0; i < count; ++i) {
            local[i] = static_cast<unsigned char>(canonical[offset + i]);
        }
        if (MPI_Allreduce(local.data(), min_values.data(), static_cast<int>(kChunkSize), MPI_UNSIGNED_CHAR, MPI_MIN, comm) != MPI_SUCCESS ||
            MPI_Allreduce(local.data(), max_values.data(), static_cast<int>(kChunkSize), MPI_UNSIGNED_CHAR, MPI_MAX, comm) != MPI_SUCCESS) {
            return AMIO_ERR_COMM_SPLIT_FAILED;
        }
        if (!std::equal(local.begin(), local.begin() + static_cast<std::ptrdiff_t>(count), min_values.begin()) ||
            !std::equal(local.begin(), local.begin() + static_cast<std::ptrdiff_t>(count), max_values.begin())) {
            return AMIO_ERR_INVALID_POLICY;
        }
    }
    return AMIO_OK;
}
#endif

} // namespace

amio_status_t validate_roles(const RoleConfiguration &roles) {
    if (roles.rank_count == 0) {
        return AMIO_ERR_INVALID_POLICY;
    }
    std::set<int> assigned;
    if (!add_role(roles.producers, roles.rank_count, assigned) || !add_role(roles.writers, roles.rank_count, assigned) ||
        !add_role(roles.consumers, roles.rank_count, assigned)) {
        return AMIO_ERR_INVALID_POLICY;
    }

    switch (roles.mode) {
        case ExecutionMode::Direct:
            if (!roles.writers.empty() || !roles.consumers.empty() || (!roles.producers.empty() && assigned.size() != roles.rank_count)) {
                return AMIO_ERR_INVALID_POLICY;
            }
            return AMIO_OK;
        case ExecutionMode::Aggregated:
            if (roles.rank_count < 2 || roles.producers.empty() || roles.writers.empty() || !roles.consumers.empty() ||
                assigned.size() != roles.rank_count) {
                return AMIO_ERR_INVALID_POLICY;
            }
            return AMIO_OK;
        case ExecutionMode::Stream:
            if (roles.producers.empty() || roles.consumers.empty() || !roles.writers.empty() || assigned.size() != roles.rank_count) {
                return AMIO_ERR_INVALID_POLICY;
            }
            return AMIO_OK;
    }
    return AMIO_ERR_INVALID_POLICY;
}

amio_status_t validate_progress_support(ProgressPolicy policy, int provided_thread_level) noexcept {
    if (provided_thread_level < 0 || provided_thread_level > 3) {
        return AMIO_ERR_THREADING_UNSUPPORTED;
    }
    switch (policy) {
        case ProgressPolicy::Background:
            return provided_thread_level < 3 ? AMIO_ERR_THREADING_UNSUPPORTED : AMIO_OK;
        case ProgressPolicy::Caller:
            return provided_thread_level < 1 ? AMIO_ERR_THREADING_UNSUPPORTED : AMIO_OK;
    }
    return AMIO_ERR_INVALID_POLICY;
}

ExecutionContext::ExecutionContext(RoleConfiguration roles, ProgressPolicy progress, std::uint32_t rank
#ifdef AMIO_HAS_MPI
                                   , halo::Communicator parent
#endif
                                   )
    : roles_(std::move(roles)), progress_(progress), rank_(rank)
#ifdef AMIO_HAS_MPI
      , parent_(std::move(parent))
#endif
{}

amio_status_t ExecutionContext::create(const RoleConfiguration &roles, ProgressPolicy progress, const std::string &canonical_configuration,
#ifdef AMIO_HAS_MPI
                                        MPI_Comm caller_communicator,
#else
                                        int caller_communicator,
#endif
                                        std::unique_ptr<ExecutionContext> &out_context) {
    out_context.reset();
#ifdef AMIO_HAS_MPI
    int initialized = 0;
    int finalized = 0;
    if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized || MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
        caller_communicator == MPI_COMM_NULL) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    int rank = 0;
    int size = 0;
    int thread_level = MPI_THREAD_SINGLE;
    if (MPI_Comm_rank(caller_communicator, &rank) != MPI_SUCCESS || MPI_Comm_size(caller_communicator, &size) != MPI_SUCCESS ||
        MPI_Query_thread(&thread_level) != MPI_SUCCESS) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    amio_status_t local_status = validate_roles(roles);
    if (local_status == AMIO_OK && (roles.rank_count != static_cast<std::uint32_t>(size) || canonical_configuration.empty() ||
                                    canonical_configuration.size() > 16 * 1024 * 1024)) {
        local_status = AMIO_ERR_INVALID_POLICY;
    }
    if (local_status == AMIO_OK) {
        local_status = validate_progress_support(progress, thread_level);
    }
    int local_code = local_status;
    int global_code = AMIO_OK;
    if (MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX, caller_communicator) != MPI_SUCCESS) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    if (global_code != AMIO_OK) {
        return global_code;
    }
    const std::string complete_configuration = full_configuration(roles, progress, canonical_configuration);
    const amio_status_t agreement = agree_on_configuration(caller_communicator, complete_configuration);
    if (agreement != AMIO_OK) {
        return agreement;
    }

    auto owned_duplicate = duplicate_context_communicator(caller_communicator);
    if (!owned_duplicate.has_value()) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    try {
        out_context.reset(new ExecutionContext(roles, progress, static_cast<std::uint32_t>(rank), std::move(*owned_duplicate)));
    } catch (...) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    return AMIO_OK;
#else
    (void)caller_communicator;
    if (validate_roles(roles) != AMIO_OK || roles.rank_count != 1 || !roles.writers.empty() || !roles.consumers.empty()) {
        return AMIO_ERR_UNSUPPORTED;
    }
    const auto thread_status = validate_progress_support(progress, 1);
    if (thread_status != AMIO_OK || canonical_configuration.empty() || canonical_configuration.size() > 16 * 1024 * 1024) {
        return thread_status == AMIO_OK ? AMIO_ERR_INVALID_POLICY : thread_status;
    }
    out_context.reset(new ExecutionContext(roles, progress, 0));
    return AMIO_OK;
#endif
}

} // namespace amio::detail
