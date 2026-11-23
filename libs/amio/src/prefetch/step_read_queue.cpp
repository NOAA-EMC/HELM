#include "prefetch/step_read_queue.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace amio::detail {

bool StepReadQueue::valid(const StepReadKey &key, const StepReadMetadata &metadata) noexcept {
    if (key.dataset_uuid.empty() || key.schema_digest.empty() || key.field.empty() || key.dataset_uuid.size() > 4096 ||
        key.schema_digest.size() > 4096 || key.field.size() > 4096 || key.dataset_uuid != metadata.dataset_uuid || key.step_id != metadata.step_id ||
        key.schema_digest != metadata.schema_digest || key.field != metadata.field)
        return false;
    if (key.selection) {
        const auto &b = *key.selection;
        if (b.rank < 1 || b.rank > AMIO_MAX_RANK || metadata.global_extents.size() != static_cast<std::size_t>(b.rank)) return false;
        for (int d = 0; d < b.rank; ++d) {
            if (b.offsets[d] < 0 || b.extents[d] <= 0 || b.strides[d] <= 0) return false;
            const auto offset = static_cast<std::uint64_t>(b.offsets[d]);
            const auto extent = static_cast<std::uint64_t>(b.extents[d]);
            const auto stride = static_cast<std::uint64_t>(b.strides[d]);
            const auto global = metadata.global_extents[static_cast<std::size_t>(d)];
            if (offset >= global || extent - 1 > (std::numeric_limits<std::uint64_t>::max() - offset) / stride) return false;
            const auto last = offset + (extent - 1) * stride;
            if (last >= global) return false;
        }
        for (int d = b.rank; d < AMIO_MAX_RANK; ++d)
            if (b.offsets[d] != 0 || b.extents[d] != 0 || b.strides[d] != 0) return false;
    }
    return true;
}

std::string StepReadQueue::encode(const StepReadKey &key) {
    // Length prefixes make arbitrary field/UUID strings unambiguous.
    std::string out;
    auto part = [&out](const std::string &s) {
        out.append(std::to_string(s.size())).push_back(':');
        out.append(s).push_back('|');
    };
    part(key.dataset_uuid);
    out.append(std::to_string(key.step_id)).push_back('|');
    part(key.schema_digest);
    part(key.field);
    if (!key.selection) {
        out.push_back('F');
        return out;
    }
    const auto &b = *key.selection;
    out.push_back('S');
    out.append(std::to_string(b.rank)).push_back('|');
    if (b.rank < 1 || b.rank > AMIO_MAX_RANK) return out + "invalid";
    for (int d = 0; d < b.rank; ++d) {
        out.append(std::to_string(b.offsets[d])).push_back(',');
        out.append(std::to_string(b.extents[d])).push_back(',');
        out.append(std::to_string(b.strides[d])).push_back(';');
    }
    return out;
}

void StepReadQueue::evict_for(std::size_t bytes) {
    while (bytes_used_ > byte_limit_ - bytes && !lru_.empty()) {
        auto last = std::prev(lru_.end());
        auto it = entries_.find(*last);
        if (it != entries_.end()) {
            bytes_used_ -= it->second.payload.size();
            entries_.erase(it);
        }
        lru_.erase(last);
    }
}

amio_status_t StepReadQueue::store(const StepReadKey &key, const StepReadMetadata &metadata, const void *payload, std::size_t bytes) {
    if (!valid(key, metadata)) return AMIO_ERR_SCHEMA_MISMATCH;
    if (!metadata.field_present) return AMIO_ERR_FIELD_NOT_PRESENT;
    if (bytes == 0 || bytes != metadata.payload_bytes || payload == nullptr) return AMIO_ERR_INVALID_INPUT;
    if (bytes > byte_limit_) return AMIO_ERR_BUDGET_EXCEEDED;

    const std::string id = encode(key);
    std::lock_guard<std::mutex> lock(mutex_);
    auto old = entries_.find(id);
    if (old != entries_.end()) {
        bytes_used_ -= old->second.payload.size();
        lru_.erase(old->second.lru);
        entries_.erase(old);
    }
    evict_for(bytes);
    std::vector<std::byte> copy(bytes);  // allocation occurs only after metadata and budget checks
    if (bytes) std::memcpy(copy.data(), payload, bytes);
    lru_.push_front(id);
    bytes_used_ += bytes;
    entries_.emplace(id, Entry{std::move(copy), lru_.begin()});
    return AMIO_OK;
}

amio_status_t StepReadQueue::load(const StepReadKey &key, const StepReadMetadata &metadata, std::vector<std::byte> &payload) {
    if (!valid(key, metadata)) return AMIO_ERR_SCHEMA_MISMATCH;
    if (!metadata.field_present) return AMIO_ERR_FIELD_NOT_PRESENT;
    if (metadata.payload_bytes > byte_limit_) return AMIO_ERR_BUDGET_EXCEEDED;
    const std::string id = encode(key);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(id);
    if (it == entries_.end()) return AMIO_ERR_BACKEND_FAILURE;
    if (it->second.payload.size() != metadata.payload_bytes) return AMIO_ERR_SCHEMA_MISMATCH;
    lru_.splice(lru_.begin(), lru_, it->second.lru);
    payload = it->second.payload;
    return AMIO_OK;
}

void StepReadQueue::clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    lru_.clear();
    bytes_used_ = 0;
}

std::size_t StepReadQueue::bytes_used() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_used_;
}
std::size_t StepReadQueue::entry_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

}  // namespace amio::detail
