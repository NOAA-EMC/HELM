#include "steps/step_coordinator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace amio::detail {

namespace {
struct DatasetOwners {
    std::mutex mutex;
    std::unordered_map<std::string, const StepCoordinator *> owners;
};

DatasetOwners &dataset_owners() {
    static DatasetOwners registry;
    return registry;
}

bool active(StepState state) noexcept {
    return state == StepState::Open || state == StepState::Sealed || state == StepState::Transferring ||
           state == StepState::PayloadReady || state == StepState::Publishing || state == StepState::Reconciling;
}
} // namespace

StepCoordinator::StepCoordinator(std::string dataset_uuid, ResourceBudget &budget)
    : dataset_uuid_(std::move(dataset_uuid)), budget_(budget) {
    if (dataset_uuid_.empty()) throw std::invalid_argument("dataset UUID cannot be empty");
    auto &registry = dataset_owners();
    std::lock_guard<std::mutex> lock(registry.mutex);
    if (!registry.owners.emplace(dataset_uuid_, this).second) throw std::invalid_argument("dataset already has an active producer owner");
}

StepCoordinator::~StepCoordinator() {
    if (active(state_)) (void)fail_step(AMIO_ERR_STEP_FAILED);
    auto &registry = dataset_owners();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto found = registry.owners.find(dataset_uuid_);
    if (found != registry.owners.end() && found->second == this) registry.owners.erase(found);
}

amio_status_t StepCoordinator::register_schema(std::shared_ptr<const Schema> schema) {
    if (!schema || !schema->sealed() || schema->digest().empty() || active(state_)) return AMIO_ERR_INVALID_POLICY;
    const auto found = schemas_.find(schema->digest());
    if (found != schemas_.end() && !found->second->equivalent(*schema)) return AMIO_ERR_SCHEMA_MISMATCH;
    schemas_[schema->digest()] = std::move(schema);
    return AMIO_OK;
}

amio_status_t StepCoordinator::begin_step(const StepBeginRequest &request) {
    if (dataset_poisoned_) return AMIO_ERR_STEP_FAILED;
    if (active(state_)) return AMIO_ERR_INVALID_POLICY;
    if (has_attempted_step_ && request.step_id <= last_step_id_) return AMIO_ERR_INVALID_POLICY;
    has_attempted_step_ = true;
    last_step_id_ = request.step_id;
    contributions_.clear();
    plan_.reset();
    completion_operation_.reset();
    active_schema_digest_.clear();
    state_ = StepState::Failed;

    if (!std::isfinite(request.model_time) || request.snapshot_descriptors == 0) return AMIO_ERR_INVALID_POLICY;
    const auto schema = schemas_.find(request.schema_digest);
    if (schema == schemas_.end()) return AMIO_ERR_SCHEMA_MISMATCH;
    const auto plan_status = plan_.configure(*schema->second, request.contributions, request.snapshot_bytes,
                                             request.empty_contributor, request.defer_required_field_validation);
    if (plan_status != AMIO_OK) return plan_status;
    std::shared_ptr<OperationRecord> completion;
    try {
        completion = std::make_shared<OperationRecord>();
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    }
    const auto reservation = budget_.reserve(request.snapshot_bytes, request.snapshot_descriptors, ResourceBudget::Class::Data);
    if (!reservation) return AMIO_ERR_BUDGET_EXCEEDED;

    reservation_ = reservation;
    active_step_id_ = request.step_id;
    active_model_time_ = request.model_time;
    active_schema_digest_ = request.schema_digest;
    completion_operation_ = std::move(completion);
    if (!completion_operation_->retain_observer()) return fail_step(AMIO_ERR_BACKEND_FAILURE);
    state_ = StepState::Open;
    return AMIO_OK;
}

StepPutResult StepCoordinator::put(const std::string &variable_name, const void *source, std::size_t bytes) {
    if (state_ != StepState::Open) return {state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY};
    const auto schema = schemas_.find(active_schema_digest_);
    if (schema == schemas_.end()) return {fail_step(AMIO_ERR_SCHEMA_MISMATCH)};
    const auto variable = schema->second->variables().find(variable_name);
    if (variable == schema->second->variables().end()) return {AMIO_ERR_INVALID_INPUT};
    if (contributions_.find(variable_name) != contributions_.end()) return {AMIO_ERR_INVALID_POLICY};
    const auto expected = plan_.expected_bytes(variable_name);
    if (!expected) return {AMIO_ERR_INVALID_POLICY};
    if (source == nullptr || bytes == 0 || bytes != *expected) return {AMIO_ERR_INVALID_INPUT};

    std::size_t already_admitted = 0;
    for (const auto &[name, contribution] : contributions_) {
        (void)name;
        if (contribution.bytes.size() > std::numeric_limits<std::size_t>::max() - already_admitted) return {fail_step(AMIO_ERR_INVALID_INPUT)};
        already_admitted += contribution.bytes.size();
    }
    if (!reservation_ || already_admitted > reservation_->bytes || bytes > reservation_->bytes - already_admitted) {
        return {fail_step(AMIO_ERR_BUDGET_EXCEEDED)};
    }

    try {
        auto snapshot = snapshot_host_buffer(source, bytes);
        if (snapshot.status != AMIO_OK) return {fail_step(snapshot.status)};
        if (!plan_.mark_present(variable_name)) return {AMIO_ERR_INVALID_POLICY};
        StepContributionSnapshot contribution;
        contribution.bytes = std::move(snapshot.bytes);
        if (const auto *region = plan_.region(variable_name)) contribution.region = *region;
        contribution.operation_id = add_operation();
        if (contribution.operation_id == 0) return {fail_step(AMIO_ERR_BACKEND_FAILURE)};
        auto operation = operations_.at(contribution.operation_id);
        operation->mark(OperationMilestone::SourceReleased);
        const auto id = contribution.operation_id;
        contributions_.emplace(variable_name, std::move(contribution));
        return {AMIO_OK, id, true, false};
    } catch (const std::bad_alloc &) {
        return {fail_step(AMIO_ERR_BACKEND_FAILURE)};
    } catch (...) {
        return {fail_step(AMIO_ERR_BACKEND_FAILURE)};
    }
}

amio_status_t StepCoordinator::end_step() {
    if (state_ != StepState::Open) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    const auto schema = schemas_.find(active_schema_digest_);
    if (schema == schemas_.end()) return fail_step(AMIO_ERR_SCHEMA_MISMATCH);
    if (!plan_.required_fields_present()) return fail_step(AMIO_ERR_FIELD_NOT_PRESENT);
    state_ = StepState::Sealed;
    return AMIO_OK;
}

amio_status_t StepCoordinator::mark_payload_complete() {
    if (state_ != StepState::Sealed && state_ != StepState::Transferring) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    for (auto &[id, operation] : operations_) {
        (void)id;
        if (operation->snapshot().outcome == OperationOutcome::Pending) operation->mark(OperationMilestone::PayloadComplete);
    }
    if (completion_operation_) completion_operation_->mark(OperationMilestone::PayloadComplete);
    state_ = StepState::PayloadReady;
    return AMIO_OK;
}

amio_status_t StepCoordinator::mark_publishing() {
    if (state_ != StepState::PayloadReady) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    state_ = StepState::Publishing;
    return AMIO_OK;
}

amio_status_t StepCoordinator::mark_reconciling() {
    if (state_ != StepState::Publishing) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    state_ = StepState::Reconciling;
    return AMIO_OK;
}

amio_status_t StepCoordinator::mark_publication_visible(bool storage_synced) {
    if (state_ != StepState::Publishing && state_ != StepState::Reconciling) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    for (auto &[id, operation] : operations_) {
        (void)id;
        if (operation->snapshot().outcome == OperationOutcome::Pending) {
            operation->mark(OperationMilestone::ReaderVisible);
            if (storage_synced) operation->mark(OperationMilestone::StorageSynced);
            operation->finish_success();
        }
    }
    if (completion_operation_) {
        completion_operation_->mark(OperationMilestone::ReaderVisible);
        if (storage_synced) completion_operation_->mark(OperationMilestone::StorageSynced);
        completion_operation_->finish_success();
    }
    state_ = StepState::Complete;
    release_reservation();
    contributions_.clear();
    return AMIO_OK;
}

amio_status_t StepCoordinator::fail_publication(amio_status_t error) {
    if (state_ != StepState::Publishing && state_ != StepState::Reconciling) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    return fail_step(error == AMIO_OK ? AMIO_ERR_BACKEND_FAILURE : error);
}

amio_status_t StepCoordinator::poison_dataset(amio_status_t error) {
    if (state_ == StepState::Complete) return AMIO_ERR_INVALID_POLICY;
    dataset_poisoned_ = true;
    if (active(state_)) return fail_step(error == AMIO_OK ? AMIO_ERR_BACKEND_FAILURE : error);
    return AMIO_OK;
}

amio_status_t StepCoordinator::cancel_step() {
    if (!active(state_) || state_ == StepState::Publishing || state_ == StepState::Reconciling) return AMIO_ERR_STEP_FAILED;
    (void)fail_step(AMIO_ERR_STEP_FAILED);
    return AMIO_OK;
}

amio_status_t StepCoordinator::wait_for_publication(std::uint64_t timeout_ms) const {
    if (!completion_operation_) return state_ == StepState::Failed ? AMIO_ERR_STEP_FAILED : AMIO_ERR_INVALID_POLICY;
    if (timeout_ms > static_cast<std::uint64_t>(std::numeric_limits<std::chrono::milliseconds::rep>::max())) return AMIO_ERR_INVALID_INPUT;
    const auto outcome = completion_operation_->wait_for(OperationMilestone::ReaderVisible, std::chrono::milliseconds(timeout_ms));
    if (outcome == WaitOutcome::Reached) return AMIO_OK;
    if (outcome == WaitOutcome::Timeout) return AMIO_ERR_TIMEOUT;
    const auto snapshot = completion_operation_->snapshot();
    return snapshot.error == AMIO_OK ? AMIO_ERR_STEP_FAILED : snapshot.error;
}

OperationSnapshot StepCoordinator::operation_snapshot(std::uint64_t handle) const noexcept {
    const auto found = operations_.find(handle);
    return found == operations_.end() ? OperationSnapshot{} : found->second->snapshot();
}

std::shared_ptr<OperationRecord> StepCoordinator::operation_record(std::uint64_t handle) const noexcept {
    const auto found = operations_.find(handle);
    return found == operations_.end() ? std::shared_ptr<OperationRecord>{} : found->second;
}

amio_status_t StepCoordinator::fail_step(amio_status_t error) noexcept {
    state_ = StepState::Failed;
    for (auto &[id, operation] : operations_) {
        (void)id;
        if (operation->snapshot().outcome == OperationOutcome::Pending) operation->finish_failure(error);
    }
    if (completion_operation_) completion_operation_->finish_failure(error);
    release_reservation();
    contributions_.clear();
    return error;
}

void StepCoordinator::release_reservation() noexcept {
    if (!reservation_) return;
    budget_.release(*reservation_);
    reservation_.reset();
}

std::uint64_t StepCoordinator::add_operation() {
    if (next_operation_id_ == 0 || next_operation_id_ == std::numeric_limits<std::uint64_t>::max()) return 0;
    const auto id = next_operation_id_++;
    auto operation = std::make_shared<OperationRecord>();
    operation->retain_observer();
    operations_.emplace(id, std::move(operation));
    return id;
}

} // namespace amio::detail
