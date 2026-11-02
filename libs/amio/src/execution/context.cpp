#include "execution/context.hpp"
#include "execution/role_map.hpp"
#include "workers/comm_split.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <sstream>
#include <utility>

namespace amio::detail {

amio_status_t compare_context_metadata(const ExecutionContext &context, std::string_view value,
                                       std::size_t max_bytes) noexcept {
#ifdef AMIO_HAS_MPI
    const int root = static_cast<int>(context.storage_root_rank());
    const MPI_Comm communicator = context.communicator();
    unsigned long long total = context.rank() == static_cast<std::uint32_t>(root) ? value.size() : 0;
    if (MPI_Bcast(&total, 1, MPI_UNSIGNED_LONG_LONG, root, communicator) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (total > max_bytes) return AMIO_ERR_INVALID_INPUT;
    std::array<char, 4096> chunk{};
    int mismatch = value.size() != total ? 1 : 0;
    for (std::size_t offset = 0; offset < static_cast<std::size_t>(total);) {
        const auto count = std::min(chunk.size(), static_cast<std::size_t>(total) - offset);
        if (context.rank() == static_cast<std::uint32_t>(root)) std::memcpy(chunk.data(), value.data() + offset, count);
        if (MPI_Bcast(chunk.data(), static_cast<int>(count), MPI_CHAR, root, communicator) != MPI_SUCCESS)
            return AMIO_ERR_COMM_SPLIT_FAILED;
        if (offset > value.size() || count > value.size() - offset ||
            std::memcmp(chunk.data(), value.data() + offset, count) != 0) mismatch = 1;
        offset += count;
    }
    int any_mismatch = 0;
    if (MPI_Allreduce(&mismatch, &any_mismatch, 1, MPI_INT, MPI_MAX, communicator) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    return any_mismatch ? AMIO_ERR_SCHEMA_MISMATCH : AMIO_OK;
#else
    (void)context;
    return value.size() > max_bytes ? AMIO_ERR_INVALID_INPUT : AMIO_OK;
#endif
}

namespace {

bool add_role(const std::vector<int> &ranks, std::uint32_t rank_count, std::vector<std::uint8_t> &assigned,
              std::size_t &assigned_count) {
    for (const int rank : ranks) {
        if (rank < 0 || static_cast<std::uint32_t>(rank) >= rank_count || assigned[static_cast<std::size_t>(rank)] != 0)
            return false;
        assigned[static_cast<std::size_t>(rank)] = 1;
        ++assigned_count;
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
    if (roles.rank_count == 0 || roles.rank_count > 1'000'000) return AMIO_ERR_INVALID_POLICY;
    try {
        std::vector<std::uint8_t> assigned(roles.rank_count, 0);
        std::size_t assigned_count = 0;
        if (!add_role(roles.producers, roles.rank_count, assigned, assigned_count) ||
            !add_role(roles.writers, roles.rank_count, assigned, assigned_count) ||
            !add_role(roles.consumers, roles.rank_count, assigned, assigned_count)) return AMIO_ERR_INVALID_POLICY;

        switch (roles.mode) {
            case ExecutionMode::Direct:
                if (!roles.writers.empty() || !roles.consumers.empty() ||
                    (!roles.producers.empty() && assigned_count != roles.rank_count)) return AMIO_ERR_INVALID_POLICY;
                return AMIO_OK;
            case ExecutionMode::Aggregated:
                if (roles.rank_count < 2 || roles.producers.empty() || roles.writers.empty() || !roles.consumers.empty() ||
                    assigned_count != roles.rank_count) return AMIO_ERR_INVALID_POLICY;
                return AMIO_OK;
            case ExecutionMode::Stream:
                if (roles.producers.empty() || roles.consumers.empty() || !roles.writers.empty() ||
                    assigned_count != roles.rank_count) return AMIO_ERR_INVALID_POLICY;
                return AMIO_OK;
        }
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
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
                                   , halo::Communicator parent, halo::Communicator control, halo::Communicator data
                                   , std::optional<halo::Communicator> producer_group,
                                   std::optional<halo::Communicator> writer_group,
                                   std::optional<halo::Communicator> consumer_group,
                                   std::optional<halo::Communicator> service_group
#endif
                                   , std::shared_ptr<const RoleMap> role_map
                                   )
    : roles_(std::move(roles)), progress_(progress), rank_(rank), role_map_(std::move(role_map))
#ifdef AMIO_HAS_MPI
      , parent_(std::move(parent)), control_(std::move(control)), data_(std::move(data)),
      producer_group_(std::move(producer_group)), writer_group_(std::move(writer_group))
      , consumer_group_(std::move(consumer_group)), service_group_(std::move(service_group))
#endif
{}

const RoleMap &ExecutionContext::role_map() const noexcept { return *role_map_; }

bool ExecutionContext::can_open_storage() const noexcept { return role_map_->can_open_storage(rank_); }

std::uint32_t ExecutionContext::storage_root_rank() const noexcept {
    if (role_map_->writers().empty()) return 0;
    const int root = role_map_->writers().front();
    return root < 0 ? 0u : static_cast<std::uint32_t>(root);
}

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
    std::shared_ptr<const RoleMap> role_map;
    if (local_status == AMIO_OK) {
        try {
            RoleMap built;
            local_status = RoleMap::create(roles, built);
            if (local_status == AMIO_OK) role_map = std::make_shared<RoleMap>(std::move(built));
        } catch (...) {
            local_status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
    int local_code = local_status;
    int global_code = AMIO_OK;
    if (MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX, caller_communicator) != MPI_SUCCESS) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    if (global_code != AMIO_OK) {
        return global_code;
    }
    std::string complete_configuration;
    local_code = AMIO_OK;
    try { complete_configuration = full_configuration(roles, progress, canonical_configuration); }
    catch (...) { local_code = AMIO_ERR_BACKEND_FAILURE; }
    global_code = AMIO_OK;
    if (MPI_Allreduce(&local_code, &global_code, 1, MPI_INT, MPI_MAX, caller_communicator) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (global_code != AMIO_OK) return global_code;
    const amio_status_t agreement = agree_on_configuration(caller_communicator, complete_configuration);
    if (agreement != AMIO_OK) {
        return agreement;
    }

    auto owned_duplicate = duplicate_context_communicator(caller_communicator);
    if (!owned_duplicate.has_value()) {
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    try {
        auto control = owned_duplicate->duplicate();
        auto data = owned_duplicate->duplicate();
        const auto local_role = role_map->role(static_cast<std::uint32_t>(rank));
        const int producer_color = local_role == RankRole::Producer ? 1 : MPI_UNDEFINED;
        const int writer_color = role_map->can_open_storage(static_cast<std::uint32_t>(rank)) ? 1 : MPI_UNDEFINED;
        const int consumer_color = local_role == RankRole::Consumer ? 1 : MPI_UNDEFINED;
        const int service_color = role_map->is_service_only(static_cast<std::uint32_t>(rank)) ? 1 : MPI_UNDEFINED;
        MPI_Comm producer_raw = MPI_COMM_NULL;
        MPI_Comm writer_raw = MPI_COMM_NULL;
        MPI_Comm consumer_raw = MPI_COMM_NULL;
        MPI_Comm service_raw = MPI_COMM_NULL;
        if (MPI_Comm_split(owned_duplicate->handle(), producer_color, rank, &producer_raw) != MPI_SUCCESS ||
            MPI_Comm_split(owned_duplicate->handle(), writer_color, rank, &writer_raw) != MPI_SUCCESS ||
            MPI_Comm_split(owned_duplicate->handle(), consumer_color, rank, &consumer_raw) != MPI_SUCCESS ||
            MPI_Comm_split(owned_duplicate->handle(), service_color, rank, &service_raw) != MPI_SUCCESS) {
            if (producer_raw != MPI_COMM_NULL) MPI_Comm_free(&producer_raw);
            if (writer_raw != MPI_COMM_NULL) MPI_Comm_free(&writer_raw);
            if (consumer_raw != MPI_COMM_NULL) MPI_Comm_free(&consumer_raw);
            if (service_raw != MPI_COMM_NULL) MPI_Comm_free(&service_raw);
            return AMIO_ERR_COMM_SPLIT_FAILED;
        }
        std::optional<halo::Communicator> producer_group;
        std::optional<halo::Communicator> writer_group;
        std::optional<halo::Communicator> consumer_group;
        std::optional<halo::Communicator> service_group;
        if (producer_raw != MPI_COMM_NULL) {
            producer_group.emplace(producer_raw);
            producer_raw = MPI_COMM_NULL;
        }
        if (writer_raw != MPI_COMM_NULL) {
            writer_group.emplace(writer_raw);
            writer_raw = MPI_COMM_NULL;
        }
        if (consumer_raw != MPI_COMM_NULL) {
            consumer_group.emplace(consumer_raw);
            consumer_raw = MPI_COMM_NULL;
        }
        if (service_raw != MPI_COMM_NULL) {
            service_group.emplace(service_raw);
            service_raw = MPI_COMM_NULL;
        }
        out_context.reset(new ExecutionContext(roles, progress, static_cast<std::uint32_t>(rank), std::move(*owned_duplicate),
                                               std::move(control), std::move(data), std::move(producer_group),
                                               std::move(writer_group), std::move(consumer_group), std::move(service_group),
                                               std::move(role_map)));
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
    std::shared_ptr<const RoleMap> role_map;
    try {
        RoleMap built;
        const auto map_status = RoleMap::create(roles, built);
        if (map_status != AMIO_OK) return map_status;
        role_map = std::make_shared<RoleMap>(std::move(built));
        out_context.reset(new ExecutionContext(roles, progress, 0, std::move(role_map)));
    } catch (...) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    return AMIO_OK;
#endif
}

} // namespace amio::detail
