#include "execution/role_map.hpp"

#include <new>
#include <optional>
#include <utility>

namespace amio::detail {

amio_status_t RoleMap::create(const RoleConfiguration &configuration, RoleMap &out) {
    const auto validation = validate_roles(configuration);
    if (validation != AMIO_OK) return validation;
    try {
        RoleMap candidate;
        candidate.rank_count_ = configuration.rank_count;
        candidate.mode_ = configuration.mode;
        candidate.producers_ = configuration.producers;
        candidate.writers_ = configuration.writers;
        if (candidate.mode_ == ExecutionMode::Direct && candidate.producers_.empty()) {
            candidate.producers_.reserve(candidate.rank_count_);
            for (std::uint32_t rank = 0; rank < candidate.rank_count_; ++rank)
                candidate.producers_.push_back(static_cast<int>(rank));
            candidate.writers_ = candidate.producers_;
        }
        if (candidate.mode_ == ExecutionMode::Direct) candidate.writers_ = candidate.producers_;
        candidate.rank_roles_.assign(candidate.rank_count_, RankRole::Inactive);
        candidate.writer_by_rank_.assign(candidate.rank_count_, -1);
        for (std::size_t i = 0; i < candidate.producers_.size(); ++i) {
            const auto rank = static_cast<std::uint32_t>(candidate.producers_[i]);
            candidate.rank_roles_[rank] = RankRole::Producer;
            if (candidate.mode_ == ExecutionMode::Direct) {
                candidate.writer_by_rank_[rank] = static_cast<int>(rank);
            } else if (candidate.mode_ == ExecutionMode::Aggregated) {
                const auto writer = static_cast<std::uint32_t>(candidate.writers_[i % candidate.writers_.size()]);
                candidate.writer_by_rank_[rank] = static_cast<int>(writer);
            } else if (candidate.mode_ == ExecutionMode::Stream) {
                const auto consumer = static_cast<std::uint32_t>(configuration.consumers[i % configuration.consumers.size()]);
                candidate.writer_by_rank_[rank] = static_cast<int>(consumer);
            }
        }
        if (candidate.mode_ == ExecutionMode::Aggregated) {
            for (int rank : candidate.writers_) candidate.rank_roles_[static_cast<std::size_t>(rank)] = RankRole::Writer;
        } else if (candidate.mode_ == ExecutionMode::Stream) {
            for (int rank : configuration.consumers) candidate.rank_roles_[static_cast<std::size_t>(rank)] = RankRole::Consumer;
        }
        out = std::move(candidate);
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    return AMIO_OK;
}

RankRole RoleMap::role(std::uint32_t world_rank) const noexcept {
    return world_rank < rank_roles_.size() ? rank_roles_[world_rank] : RankRole::Inactive;
}

bool RoleMap::can_open_storage(std::uint32_t world_rank) const noexcept {
    const auto current = role(world_rank);
    return current == RankRole::Writer || (mode_ == ExecutionMode::Direct && current == RankRole::Producer);
}

bool RoleMap::is_service_only(std::uint32_t world_rank) const noexcept {
    const auto current = role(world_rank);
    return current == RankRole::Writer || current == RankRole::Consumer;
}

std::optional<std::uint32_t> RoleMap::writer_for(std::uint32_t producer_rank) const noexcept {
    if (producer_rank >= writer_by_rank_.size() || writer_by_rank_[producer_rank] < 0) return std::nullopt;
    return static_cast<std::uint32_t>(writer_by_rank_[producer_rank]);
}

} // namespace amio::detail
