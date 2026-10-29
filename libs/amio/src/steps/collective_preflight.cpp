#include "steps/collective_preflight.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <utility>

namespace amio::detail {
namespace {

struct RankMetadata {
    std::string digest;
    std::vector<std::string> variables;
    std::vector<std::string> required;
    std::vector<ContributionRegion> contributions;
};

bool validate_local(const PreflightRequest &request) {
    if (request.local_admission_status != AMIO_OK || request.schema_digest.empty() || request.schema_digest.size() > 1024 * 1024 ||
        request.descriptor_limit == 0 || request.metadata_byte_limit == 0 || request.contributions.size() > request.descriptor_limit ||
        request.schema_variables.size() > std::numeric_limits<std::uint32_t>::max() ||
        request.required_variables.size() > std::numeric_limits<std::uint32_t>::max() ||
        request.contributions.size() > std::numeric_limits<std::uint32_t>::max()) return false;
    std::set<std::string> variables;
    for (const auto &name : request.schema_variables) {
        if (name.empty() || !variables.insert(name).second) return false;
    }
    std::set<std::string> required;
    for (const auto &name : request.required_variables) {
        if (variables.find(name) == variables.end() || !required.insert(name).second) return false;
    }
    for (const auto &contribution : request.contributions) {
        if (variables.find(contribution.variable) == variables.end() || contribution.global_extents.empty() ||
            contribution.global_extents.size() > 7 || contribution.offsets.size() != contribution.global_extents.size() ||
            contribution.extents.size() != contribution.global_extents.size()) return false;
        bool has_zero = false;
        for (std::size_t dimension = 0; dimension < contribution.global_extents.size(); ++dimension) {
            const auto global = contribution.global_extents[dimension];
            const auto offset = contribution.offsets[dimension];
            const auto extent = contribution.extents[dimension];
            if (global == 0 || offset > global || extent > global - offset) return false;
            has_zero = has_zero || extent == 0;
        }
        if (contribution.empty != has_zero) return false;
    }
    return true;
}

void append_u32(std::vector<std::uint8_t> &out, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8) out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
}
void append_u64(std::vector<std::uint8_t> &out, std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
}
bool append_string(std::vector<std::uint8_t> &out, const std::string &value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) return false;
    append_u32(out, static_cast<std::uint32_t>(value.size()));
    out.insert(out.end(), value.begin(), value.end());
    return true;
}

std::vector<std::uint8_t> serialize(const PreflightRequest &request) {
    std::vector<std::uint8_t> out;
    append_string(out, request.schema_digest);
    append_u32(out, static_cast<std::uint32_t>(request.schema_variables.size()));
    for (const auto &name : request.schema_variables) append_string(out, name);
    append_u32(out, static_cast<std::uint32_t>(request.required_variables.size()));
    for (const auto &name : request.required_variables) append_string(out, name);
    append_u32(out, static_cast<std::uint32_t>(request.contributions.size()));
    for (const auto &contribution : request.contributions) {
        append_string(out, contribution.variable);
        append_u32(out, contribution.empty ? 1u : 0u);
        append_u32(out, static_cast<std::uint32_t>(contribution.global_extents.size()));
        for (const auto extent : contribution.global_extents) append_u64(out, extent);
        for (const auto offset : contribution.offsets) append_u64(out, offset);
        for (const auto extent : contribution.extents) append_u64(out, extent);
    }
    return out;
}

class Reader {
   public:
    explicit Reader(const std::vector<std::uint8_t> &bytes) : bytes_(bytes) {}
    bool u32(std::uint32_t &value) {
        if (bytes_.size() - cursor_ < 4) return false;
        value = 0;
        for (unsigned i = 0; i < 4; ++i) value |= static_cast<std::uint32_t>(bytes_[cursor_++]) << (8 * i);
        return true;
    }
    bool u64(std::uint64_t &value) {
        if (bytes_.size() - cursor_ < 8) return false;
        value = 0;
        for (unsigned i = 0; i < 8; ++i) value |= static_cast<std::uint64_t>(bytes_[cursor_++]) << (8 * i);
        return true;
    }
    bool string(std::string &value) {
        std::uint32_t size = 0;
        if (!u32(size) || size > bytes_.size() - cursor_) return false;
        value.assign(reinterpret_cast<const char *>(bytes_.data() + cursor_), size);
        cursor_ += size;
        return true;
    }
    bool done() const noexcept { return cursor_ == bytes_.size(); }

   private:
    const std::vector<std::uint8_t> &bytes_;
    std::size_t cursor_ = 0;
};

bool deserialize(const std::vector<std::uint8_t> &bytes, RankMetadata &out) {
    Reader reader(bytes);
    std::uint32_t count = 0;
    if (!reader.string(out.digest) || !reader.u32(count) || count > 100000) return false;
    out.variables.resize(count);
    for (auto &name : out.variables) if (!reader.string(name)) return false;
    if (!reader.u32(count) || count > out.variables.size()) return false;
    out.required.resize(count);
    for (auto &name : out.required) if (!reader.string(name)) return false;
    if (!reader.u32(count) || count > 100000) return false;
    out.contributions.resize(count);
    for (auto &contribution : out.contributions) {
        std::uint32_t empty = 0;
        std::uint32_t rank = 0;
        if (!reader.string(contribution.variable) || !reader.u32(empty) || empty > 1 || !reader.u32(rank) || rank == 0 || rank > 7) return false;
        contribution.empty = empty != 0;
        contribution.global_extents.resize(rank);
        contribution.offsets.resize(rank);
        contribution.extents.resize(rank);
        for (auto &value : contribution.global_extents) if (!reader.u64(value)) return false;
        for (auto &value : contribution.offsets) if (!reader.u64(value)) return false;
        for (auto &value : contribution.extents) if (!reader.u64(value)) return false;
    }
    return reader.done();
}

PreflightResult build_rounds(const std::vector<RankMetadata> &ranks, int local_rank, const PreflightRequest &request) {
    PreflightResult result;
    const auto &canonical = ranks.front();
    for (const auto &rank : ranks) {
        if (rank.digest != canonical.digest || rank.variables != canonical.variables || rank.required != canonical.required) {
            result.status = AMIO_ERR_SCHEMA_MISMATCH;
            return result;
        }
    }
    std::set<std::string> required(canonical.required.begin(), canonical.required.end());
    std::map<std::string, std::vector<std::vector<ContributionRegion>>, std::less<>> fields;
    for (const auto &name : canonical.variables) fields.emplace(name, std::vector<std::vector<ContributionRegion>>(ranks.size()));
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        for (const auto &region : ranks[rank].contributions) {
            if (region.empty) continue;
            auto found = fields.find(region.variable);
            if (found == fields.end()) {
                result.status = AMIO_ERR_SCHEMA_MISMATCH;
                return result;
            }
            auto &local = found->second[rank];
            if (!local.empty() && local.front().global_extents != region.global_extents) {
                result.status = AMIO_ERR_PARTITION_INVALID;
                return result;
            }
            local.push_back(region);
        }
    }
    std::size_t descriptors = 0;
    for (auto &[name, per_rank] : fields) {
        std::size_t max_fragments = 0;
        bool present = false;
        std::optional<std::vector<std::uint64_t>> global_shape;
        for (auto &regions : per_rank) {
            std::sort(regions.begin(), regions.end(), [](const auto &left, const auto &right) {
                return left.offsets == right.offsets ? left.extents < right.extents : left.offsets < right.offsets;
            });
            max_fragments = std::max(max_fragments, regions.size());
            present = present || !regions.empty();
            for (const auto &region : regions) {
                if (global_shape && *global_shape != region.global_extents) {
                    result.status = AMIO_ERR_PARTITION_INVALID;
                    return result;
                }
                global_shape = region.global_extents;
                if (++descriptors > request.descriptor_limit) {
                    result.status = AMIO_ERR_BUDGET_EXCEEDED;
                    return result;
                }
            }
        }
        if (required.find(name) != required.end() && !present) {
            result.status = AMIO_ERR_FIELD_NOT_PRESENT;
            return result;
        }
        for (std::size_t fragment = 0; fragment < max_fragments; ++fragment) {
            PreflightRound round;
            round.variable = name;
            round.fragment_index = fragment;
            round.participants.reserve(ranks.size());
            for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
                round.participants.push_back(static_cast<int>(rank));
                if (static_cast<int>(rank) == local_rank && fragment < per_rank[rank].size()) round.local_region = per_rank[rank][fragment];
            }
            result.rounds.push_back(std::move(round));
        }
    }
    return result;
}

} // namespace

#ifdef AMIO_HAS_MPI
PreflightResult CollectivePreflight::run(MPI_Comm communicator, const PreflightRequest &request) {
    PreflightResult result;
    int local_rank = 0;
    int size = 0;
    if (MPI_Comm_rank(communicator, &local_rank) != MPI_SUCCESS || MPI_Comm_size(communicator, &size) != MPI_SUCCESS || size <= 0) {
        result.status = AMIO_ERR_COMM_SPLIT_FAILED;
        return result;
    }
    int local_status = request.local_admission_status;
    if (local_status == AMIO_OK && !validate_local(request)) local_status = AMIO_ERR_INVALID_POLICY;
    int common_status = AMIO_OK;
    if (MPI_Allreduce(&local_status, &common_status, 1, MPI_INT, MPI_MAX, communicator) != MPI_SUCCESS) {
        result.status = AMIO_ERR_COMM_SPLIT_FAILED;
        return result;
    }
    if (common_status != AMIO_OK) {
        result.status = static_cast<amio_status_t>(common_status);
        return result;
    }
    const auto packet = serialize(request);
    int local_bytes = packet.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ? -1 : static_cast<int>(packet.size());
    int sizes_status = local_bytes < 0 ? AMIO_ERR_BUDGET_EXCEEDED : AMIO_OK;
    int common_sizes_status = AMIO_OK;
    if (MPI_Allreduce(&sizes_status, &common_sizes_status, 1, MPI_INT, MPI_MAX, communicator) != MPI_SUCCESS) {
        result.status = AMIO_ERR_COMM_SPLIT_FAILED;
        return result;
    }
    if (common_sizes_status != AMIO_OK) {
        result.status = static_cast<amio_status_t>(common_sizes_status);
        return result;
    }
    std::vector<int> counts(static_cast<std::size_t>(size));
    if (MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator) != MPI_SUCCESS) {
        result.status = AMIO_ERR_COMM_SPLIT_FAILED;
        return result;
    }
    unsigned long long local_limit = static_cast<unsigned long long>(request.metadata_byte_limit);
    unsigned long long min_limit = 0;
    if (MPI_Allreduce(&local_limit, &min_limit, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator) != MPI_SUCCESS) {
        result.status = AMIO_ERR_COMM_SPLIT_FAILED;
        return result;
    }
    std::vector<int> displacements(static_cast<std::size_t>(size));
    std::size_t total = 0;
    for (int rank = 0; rank < size; ++rank) {
        if (counts[rank] < 0 || static_cast<std::size_t>(counts[rank]) > min_limit - std::min<unsigned long long>(min_limit, total) ||
            total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            result.status = AMIO_ERR_BUDGET_EXCEEDED;
            return result;
        }
        displacements[rank] = static_cast<int>(total);
        total += static_cast<std::size_t>(counts[rank]);
    }
    if (total > min_limit || total > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        result.status = AMIO_ERR_BUDGET_EXCEEDED;
        return result;
    }
    std::vector<std::uint8_t> gathered(total);
    if (MPI_Allgatherv(packet.data(), local_bytes, MPI_UNSIGNED_CHAR, gathered.data(), counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                       communicator) != MPI_SUCCESS) {
        result.status = AMIO_ERR_COMM_SPLIT_FAILED;
        return result;
    }
    std::vector<RankMetadata> ranks(static_cast<std::size_t>(size));
    for (int rank = 0; rank < size; ++rank) {
        std::vector<std::uint8_t> bytes(gathered.begin() + displacements[rank], gathered.begin() + displacements[rank] + counts[rank]);
        if (!deserialize(bytes, ranks[rank])) {
            result.status = AMIO_ERR_INVALID_INPUT;
            return result;
        }
    }
    return build_rounds(ranks, local_rank, request);
}
#else
PreflightResult CollectivePreflight::run(int communicator, const PreflightRequest &request) {
    PreflightResult result;
    if (communicator != 0 || !validate_local(request)) {
        result.status = request.local_admission_status == AMIO_OK ? AMIO_ERR_INVALID_POLICY : request.local_admission_status;
        return result;
    }
    RankMetadata local{request.schema_digest, request.schema_variables, request.required_variables, request.contributions};
    return build_rounds({local}, 0, request);
}
#endif

} // namespace amio::detail
