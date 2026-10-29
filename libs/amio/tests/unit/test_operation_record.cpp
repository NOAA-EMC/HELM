#include "c_boundary/operation_record.hpp"

#include <cassert>
#include <chrono>

namespace {

void test_milestones_are_independent_and_monotonic() {
    amio::detail::OperationRecord operation;
    assert(operation.retain_observer());
    assert(operation.snapshot().observer_references == 1);
    assert(operation.mark(amio::detail::OperationMilestone::SourceReleased));
    auto snapshot = operation.snapshot();
    assert(snapshot.source_released);
    assert(!snapshot.payload_complete && !snapshot.reader_visible && !snapshot.storage_synced);
    assert(operation.mark(amio::detail::OperationMilestone::PayloadComplete));
    assert(operation.wait_for(amio::detail::OperationMilestone::PayloadComplete, std::chrono::milliseconds(1)) ==
           amio::detail::WaitOutcome::Reached);
    assert(operation.finish_success());
    assert(!operation.finish_failure(AMIO_ERR_BACKEND_FAILURE));
    assert(operation.release_observer());
    assert(operation.snapshot().observer_references == 0);
    assert(!operation.release_observer());
}

void test_timeout_does_not_change_operation_state() {
    amio::detail::OperationRecord operation;
    assert(operation.wait_for(amio::detail::OperationMilestone::ReaderVisible, std::chrono::milliseconds(1)) ==
           amio::detail::WaitOutcome::Timeout);
    const auto snapshot = operation.snapshot();
    assert(snapshot.outcome == amio::detail::OperationOutcome::Pending);
    assert(!snapshot.reader_visible);
}

void test_terminal_failure_is_immutable_and_wakes_waiters() {
    amio::detail::OperationRecord operation;
    assert(operation.finish_failure(AMIO_ERR_BACKEND_FAILURE));
    assert(!operation.finish_success());
    assert(operation.wait_for(amio::detail::OperationMilestone::StorageSynced, std::chrono::milliseconds(1)) ==
           amio::detail::WaitOutcome::Failed);
    const auto snapshot = operation.snapshot();
    assert(snapshot.outcome == amio::detail::OperationOutcome::Failed);
    assert(snapshot.error == AMIO_ERR_BACKEND_FAILURE);
}

} // namespace

int main() {
    test_milestones_are_independent_and_monotonic();
    test_timeout_does_not_change_operation_state();
    test_terminal_failure_is_immutable_and_wakes_waiters();
    return 0;
}
