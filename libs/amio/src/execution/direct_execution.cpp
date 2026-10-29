#include "execution/direct_execution.hpp"

#include <exception>
#include <utility>

namespace amio::detail {

PublicationResult DirectExecution::publish(StepCoordinator &step, CompletionRecord record,
                                           const PayloadWriter &write_payloads,
                                           const WriterSuccessVote &writer_success_vote) {
    if (!catalog_ || !write_payloads || !writer_success_vote || step.state() != StepState::Sealed ||
        !pending_commit_.empty()) {
        return {AMIO_ERR_INVALID_POLICY, PublicationState::Failed, {}};
    }

    record.dataset_uuid = step.dataset_uuid();
    record.step_id = step.active_step_id();
    record.schema_digest = step.active_schema_digest();
    if (!record.previous_commit.empty() && record.previous_commit != previous_commit_) {
        return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, {}};
    }
    record.previous_commit = previous_commit_;

    amio_status_t local_status = AMIO_OK;
    try {
        local_status = write_payloads(step.contributions(), record);
    } catch (...) {
        local_status = AMIO_ERR_BACKEND_FAILURE;
    }

    // This vote is mandatory even after local failure: all producers learn the
    // same outcome before anyone attempts the authoritative catalog update.
    amio_status_t agreed_status = AMIO_ERR_BACKEND_FAILURE;
    try {
        agreed_status = writer_success_vote(local_status);
    } catch (...) {
        agreed_status = AMIO_ERR_BACKEND_FAILURE;
    }
    if (local_status != AMIO_OK || agreed_status != AMIO_OK) {
        return fail(step, agreed_status == AMIO_OK ? local_status : agreed_status);
    }

    auto status = step.mark_payload_complete();
    if (status != AMIO_OK) return fail(step, status);
    status = step.mark_publishing();
    if (status != AMIO_OK) return fail(step, status);

    auto result = catalog_->publish(std::move(record), guarantee_);
    if (result.state == PublicationState::Complete && result.status == AMIO_OK) {
        status = step.mark_publication_visible();
        if (status != AMIO_OK) return fail(step, status);
        previous_commit_ = result.commit_path;
        return result;
    }
    if (result.state == PublicationState::Reconciling || result.status == AMIO_ERR_PUBLICATION_PENDING) {
        status = step.mark_reconciling();
        if (status != AMIO_OK) return fail(step, status);
        pending_commit_ = result.commit_path;
        result.status = AMIO_ERR_PUBLICATION_PENDING;
        result.state = PublicationState::Reconciling;
        return result;
    }
    return fail(step, result.status == AMIO_OK ? AMIO_ERR_BACKEND_FAILURE : result.status);
}

PublicationResult DirectExecution::reconcile(StepCoordinator &step) {
    if (!catalog_ || pending_commit_.empty() || step.state() != StepState::Reconciling) {
        return {AMIO_ERR_INVALID_POLICY, PublicationState::Failed, pending_commit_};
    }
    auto result = catalog_->reconcile(pending_commit_);
    if (result.state == PublicationState::Complete && result.status == AMIO_OK) {
        const auto status = step.mark_publication_visible();
        if (status != AMIO_OK) return fail(step, status);
        previous_commit_ = pending_commit_;
        pending_commit_.clear();
        return result;
    }
    if (result.state == PublicationState::Reconciling || result.status == AMIO_ERR_PUBLICATION_PENDING) {
        result.status = AMIO_ERR_PUBLICATION_PENDING;
        result.state = PublicationState::Reconciling;
        return result;
    }
    const auto failed = fail(step, result.status == AMIO_OK ? AMIO_ERR_BACKEND_FAILURE : result.status);
    pending_commit_.clear();
    return failed;
}

PublicationResult DirectExecution::fail(StepCoordinator &step, amio_status_t status) {
    if (status == AMIO_OK) status = AMIO_ERR_BACKEND_FAILURE;
    if (step.state() == StepState::Publishing || step.state() == StepState::Reconciling) {
        (void)step.fail_publication(status);
    }
    (void)step.poison_dataset(status);
    pending_commit_.clear();
    return {status, PublicationState::Failed, pending_commit_};
}

} // namespace amio::detail
