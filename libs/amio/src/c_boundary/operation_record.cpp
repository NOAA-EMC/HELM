#include "c_boundary/operation_record.hpp"

#include <limits>

namespace amio::detail {

bool OperationRecord::retain_observer() noexcept {
    auto current = observer_references_.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint32_t>::max()) {
        if (observer_references_.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

bool OperationRecord::release_observer() noexcept {
    auto current = observer_references_.load(std::memory_order_acquire);
    while (current != 0) {
        if (observer_references_.compare_exchange_weak(current, current - 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            changed_.notify_all();
            return true;
        }
    }
    return false;
}

bool OperationRecord::mark(OperationMilestone milestone) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.outcome != OperationOutcome::Pending) {
        return false;
    }
    switch (milestone) {
        case OperationMilestone::SourceReleased:
            state_.source_released = true;
            break;
        case OperationMilestone::PayloadComplete:
            state_.payload_complete = true;
            break;
        case OperationMilestone::ReaderVisible:
            state_.reader_visible = true;
            break;
        case OperationMilestone::StorageSynced:
            state_.storage_synced = true;
            break;
    }
    changed_.notify_all();
    return true;
}

bool OperationRecord::finish_success() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.outcome != OperationOutcome::Pending) {
        return false;
    }
    state_.outcome = OperationOutcome::Succeeded;
    state_.error = AMIO_OK;
    changed_.notify_all();
    return true;
}

bool OperationRecord::finish_failure(amio_status_t error) noexcept {
    if (error == AMIO_OK) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_.outcome != OperationOutcome::Pending) {
        return false;
    }
    state_.outcome = OperationOutcome::Failed;
    state_.error = error;
    changed_.notify_all();
    return true;
}

OperationSnapshot OperationRecord::snapshot() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    OperationSnapshot result = state_;
    result.observer_references = observer_references_.load(std::memory_order_acquire);
    return result;
}

bool OperationRecord::reached_locked(OperationMilestone milestone) const noexcept {
    switch (milestone) {
        case OperationMilestone::SourceReleased:
            return state_.source_released;
        case OperationMilestone::PayloadComplete:
            return state_.payload_complete;
        case OperationMilestone::ReaderVisible:
            return state_.reader_visible;
        case OperationMilestone::StorageSynced:
            return state_.storage_synced;
    }
    return false;
}

WaitOutcome OperationRecord::wait_for(OperationMilestone milestone, std::chrono::milliseconds timeout) const {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto changed = [this, milestone] { return reached_locked(milestone) || state_.outcome != OperationOutcome::Pending; };
    if (!changed_.wait_for(lock, timeout, changed)) {
        return WaitOutcome::Timeout;
    }
    return reached_locked(milestone) ? WaitOutcome::Reached : WaitOutcome::Failed;
}

}  // namespace amio::detail
