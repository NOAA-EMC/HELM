#include "execution/stream_retention.hpp"

#include <algorithm>
#include <limits>

namespace amio::detail {

StreamRetention::StreamRetention(StreamRetentionPolicy policy, std::uint32_t consumers, std::size_t max_steps, std::size_t max_bytes,
                                 std::size_t max_descriptors)
    : policy_(policy),
      consumer_count_(consumers),
      max_steps_(max_steps),
      max_bytes_(max_bytes),
      max_descriptors_(max_descriptors),
      cursors_(consumers) {}

void StreamRetention::reclaim_released_front() noexcept {
    while (!steps_.empty()) {
        const auto it = steps_.begin();
        if (it->second.active_leases != 0 || !std::all_of(it->second.released.begin(), it->second.released.end(), [](bool v) { return v; })) break;
        bytes_ -= it->second.step->payload_bytes;
        descriptors_ -= 1 + it->second.step->fields.size() + it->second.step->absent_optional_fields.size();
        steps_.erase(it);
    }
}

amio_status_t StreamRetention::publish(RetainedStreamStep step) {
    std::lock_guard lock(mutex_);
    if (closed_ || finished_ || consumer_count_ == 0 || max_steps_ == 0 || max_bytes_ == 0 || max_descriptors_ == 0 || step.schema_digest.empty() ||
        step.payload_bytes > max_bytes_ || steps_.contains(step.step_id))
        return step.payload_bytes > max_bytes_ ? AMIO_ERR_BUDGET_EXCEEDED : AMIO_ERR_INVALID_INPUT;
    if (final_step_id_ && step.step_id <= *final_step_id_) return AMIO_ERR_INVALID_INPUT;
    std::size_t actual_bytes = 0;
    for (const auto &[name, data] : step.fields) {
        (void)name;
        if (data.size() > std::numeric_limits<std::size_t>::max() - actual_bytes) return AMIO_ERR_BUDGET_EXCEEDED;
        actual_bytes += data.size();
    }
    if (step.payload_bytes != actual_bytes) return AMIO_ERR_INVALID_INPUT;
    if (step.payload_bytes > max_bytes_) return AMIO_ERR_BUDGET_EXCEEDED;
    const auto step_descriptors = 1 + step.fields.size() + step.absent_optional_fields.size();
    if (step_descriptors > max_descriptors_) return AMIO_ERR_BUDGET_EXCEEDED;
    while (steps_.size() >= max_steps_ || bytes_ > max_bytes_ - step.payload_bytes || descriptors_ > max_descriptors_ - step_descriptors) {
        reclaim_released_front();
        if (steps_.size() < max_steps_ && bytes_ <= max_bytes_ - step.payload_bytes && descriptors_ <= max_descriptors_ - step_descriptors) break;
        if (policy_ != StreamRetentionPolicy::Latest || steps_.empty()) return AMIO_ERR_QUEUE_FULL;
        const auto evictable = std::find_if(steps_.begin(), steps_.end(), [](const auto &entry) { return entry.second.active_leases == 0; });
        if (evictable == steps_.end()) return AMIO_ERR_QUEUE_FULL;
        bytes_ -= evictable->second.step->payload_bytes;
        descriptors_ -= 1 + evictable->second.step->fields.size() + evictable->second.step->absent_optional_fields.size();
        steps_.erase(evictable);
    }
    if (!first_step_id_) first_step_id_ = step.step_id;
    final_step_id_ = step.step_id;
    Entry entry;
    entry.step = std::make_shared<const RetainedStreamStep>(std::move(step));
    entry.released.resize(consumer_count_, false);
    bytes_ += entry.step->payload_bytes;
    descriptors_ += 1 + entry.step->fields.size() + entry.step->absent_optional_fields.size();
    steps_.emplace(entry.step->step_id, std::move(entry));
    changed_.notify_all();
    return AMIO_OK;
}

amio_status_t StreamRetention::append_field(std::uint64_t step_id, std::string field, std::string schema_digest, std::vector<std::byte> bytes,
                                            std::vector<std::uint64_t> shape, std::vector<std::uint64_t> origin) {
    std::lock_guard lock(mutex_);
    if (closed_ || finished_) return AMIO_ERR_INVALID_POLICY;
    const auto found = steps_.find(step_id);
    if (found == steps_.end() || found->second.active_leases != 0) return AMIO_ERR_INVALID_HANDLE;
    if (found->second.step->schema_digest != schema_digest) return AMIO_ERR_SCHEMA_MISMATCH;
    if (found->second.step->fields.contains(field) || shape.empty() || shape.size() != origin.size()) return AMIO_ERR_INVALID_INPUT;
    if (bytes.size() > max_bytes_ - std::min(bytes_, max_bytes_) || descriptors_ >= max_descriptors_) return AMIO_ERR_QUEUE_FULL;
    auto updated = std::make_shared<RetainedStreamStep>(*found->second.step);
    updated->payload_bytes += bytes.size();
    updated->fields.emplace(field, std::move(bytes));
    updated->field_shapes.emplace(field, std::move(shape));
    updated->field_offsets.emplace(field, std::move(origin));
    bytes_ += updated->payload_bytes - found->second.step->payload_bytes;
    ++descriptors_;
    found->second.step = std::move(updated);
    changed_.notify_all();
    return AMIO_OK;
}

StreamAcquireResult StreamRetention::acquire_next(std::uint32_t consumer, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    if (consumer >= consumer_count_) return {StreamAcquireState::Failed, {}, {}, AMIO_ERR_INVALID_INPUT};
    const auto deadline = std::chrono::steady_clock::now() + std::max(timeout, std::chrono::milliseconds::zero());
    for (;;) {
        auto begin = cursors_[consumer] ? steps_.upper_bound(*cursors_[consumer]) : steps_.begin();
        if (begin != steps_.end()) {
            const auto next_id = begin->first;
            std::optional<StreamSkipRange> skipped;
            if (cursors_[consumer] && *cursors_[consumer] != std::numeric_limits<std::uint64_t>::max() && next_id > *cursors_[consumer] + 1) {
                skipped = StreamSkipRange{*cursors_[consumer] + 1, next_id - 1};
            } else if (!cursors_[consumer] && first_step_id_ && next_id > *first_step_id_) {
                skipped = StreamSkipRange{*first_step_id_, next_id - 1};
            }
            if (skipped) {
                skipped_steps_ += skipped->last - skipped->first + 1;
                cursors_[consumer] = skipped->last;
                return {StreamAcquireState::Skipped, {}, skipped, AMIO_OK};
            }
            Entry &entry = begin->second;
            if (entry.released[consumer]) {
                cursors_[consumer] = next_id;
                continue;
            }
            const auto lease_id = next_lease_id_++;
            if (lease_id == 0) return {StreamAcquireState::Failed, {}, {}, AMIO_ERR_BUDGET_EXCEEDED};
            if (descriptors_ >= max_descriptors_) return {StreamAcquireState::Failed, {}, {}, AMIO_ERR_BUDGET_EXCEEDED};
            ++entry.active_leases;
            ++descriptors_;
            leases_.emplace(lease_id, LeaseRecord{next_id, consumer});
            cursors_[consumer] = next_id;
            return {StreamAcquireState::Available, StreamLease{lease_id, consumer, entry.step}, {}, AMIO_OK};
        }
        if (finished_) {
            if (terminal_status_ != AMIO_OK) return {StreamAcquireState::Failed, {}, {}, terminal_status_};
            return {StreamAcquireState::EndOfStream, {}, {}, AMIO_ERR_STREAM_END};
        }
        if (changed_.wait_until(lock, deadline) == std::cv_status::timeout) return {StreamAcquireState::Timeout, {}, {}, AMIO_ERR_TIMEOUT};
    }
}

amio_status_t StreamRetention::release(std::uint64_t lease_id) {
    std::lock_guard lock(mutex_);
    const auto lease = leases_.find(lease_id);
    if (lease == leases_.end()) return AMIO_ERR_INVALID_HANDLE;
    const auto step = steps_.find(lease->second.step_id);
    if (step == steps_.end() || step->second.active_leases == 0) return AMIO_ERR_BACKEND_FAILURE;
    --step->second.active_leases;
    step->second.released[lease->second.consumer] = true;
    leases_.erase(lease);
    --descriptors_;
    reclaim_released_front();
    changed_.notify_all();
    return AMIO_OK;
}

amio_status_t StreamRetention::acknowledge(std::uint64_t step_id, std::uint32_t consumer) {
    std::lock_guard lock(mutex_);
    if (consumer >= consumer_count_) return AMIO_ERR_INVALID_INPUT;
    const auto step = steps_.find(step_id);
    if (step == steps_.end()) return AMIO_ERR_INVALID_HANDLE;
    if (step->second.active_leases != 0 || step->second.released[consumer]) return AMIO_ERR_INVALID_POLICY;
    step->second.released[consumer] = true;
    reclaim_released_front();
    changed_.notify_all();
    return AMIO_OK;
}

std::optional<std::uint64_t> StreamRetention::lease_step_id(std::uint64_t lease_id, std::uint32_t consumer) const noexcept {
    std::lock_guard lock(mutex_);
    const auto lease = leases_.find(lease_id);
    if (lease == leases_.end() || lease->second.consumer != consumer) return std::nullopt;
    return lease->second.step_id;
}

void StreamRetention::finish(amio_status_t status) noexcept {
    std::lock_guard lock(mutex_);
    finished_ = true;
    terminal_status_ = status;
    changed_.notify_all();
}

amio_status_t StreamRetention::close(bool drain) noexcept {
    std::lock_guard lock(mutex_);
    if (!leases_.empty()) return AMIO_ERR_VIEWS_OUTSTANDING;
    if (!drain && !steps_.empty()) return AMIO_ERR_OPERATION_PENDING;
    bytes_ = 0;
    descriptors_ = 0;
    steps_.clear();
    closed_ = true;
    changed_.notify_all();
    return AMIO_OK;
}

StreamRetention::Snapshot StreamRetention::snapshot() const noexcept {
    std::lock_guard lock(mutex_);
    return {bytes_, steps_.size(), leases_.size(), descriptors_, skipped_steps_};
}

}  // namespace amio::detail
