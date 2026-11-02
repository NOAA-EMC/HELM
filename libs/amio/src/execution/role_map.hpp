#ifndef AMIO_SRC_EXECUTION_ROLE_MAP_HPP
#define AMIO_SRC_EXECUTION_ROLE_MAP_HPP

#include "execution/context.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace amio::detail {

/** Static responsibility assigned to one rank in an execution context. */
enum class RankRole : std::uint8_t { Inactive, Producer, Writer, Consumer };

/** Immutable, deterministic mapping for a statically configured MPI context. */
class RoleMap {
public:
    /** Validate and build a deterministic mapping, leaving p out unchanged on failure. */
    [[nodiscard]] static amio_status_t create(const RoleConfiguration &configuration, RoleMap &out);
    /** Return the configured role for this rank, or Inactive when it is out of range. */
    [[nodiscard]] RankRole role(std::uint32_t world_rank) const noexcept;
    /** Return whether this rank is permitted to create or open a destination. */
    [[nodiscard]] bool can_open_storage(std::uint32_t world_rank) const noexcept;
    /** Return whether this rank serves output/consumption without producing model fields. */
    [[nodiscard]] bool is_service_only(std::uint32_t world_rank) const noexcept;
    /** Return the statically assigned writer rank for a producer, if configured. */
    [[nodiscard]] std::optional<std::uint32_t> writer_for(std::uint32_t producer_rank) const noexcept;
    /** Return producer ranks in the ordering used for deterministic assignment. */
    [[nodiscard]] const std::vector<int> &producers() const noexcept { return producers_; }
    /** Return writer ranks in the ordering used for deterministic assignment. */
    [[nodiscard]] const std::vector<int> &writers() const noexcept { return writers_; }
private:
    std::uint32_t rank_count_ = 0;
    ExecutionMode mode_ = ExecutionMode::Direct;
    std::vector<int> producers_;
    std::vector<int> writers_;
    std::vector<RankRole> rank_roles_;
    /** Direct world-rank lookup avoids scanning all producers on every forwarded field. */
    std::vector<int> writer_by_rank_;
};

} // namespace amio::detail

#endif
