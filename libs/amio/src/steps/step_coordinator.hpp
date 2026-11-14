#ifndef AMIO_SRC_STEPS_STEP_COORDINATOR_HPP
#define AMIO_SRC_STEPS_STEP_COORDINATOR_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "amio/amio_errors.h"
#include "c_boundary/operation_record.hpp"
#include "memory/host_snapshot.hpp"
#include "memory/memory_descriptor.hpp"
#include "schema/schema.hpp"
#include "staging/resource_budget.hpp"
#include "steps/contribution_plan.hpp"

namespace amio::detail {

enum class StepState : std::uint8_t { Declared, Open, Sealed, Transferring, PayloadReady, Publishing, Reconciling, Complete, Failed };

struct StepBeginRequest {
    std::uint64_t step_id = 0;
    double model_time = 0.0;
    std::string schema_digest;
    std::size_t snapshot_bytes = 0;
    std::size_t snapshot_descriptors = 0;
    bool empty_contributor = false;
    bool defer_required_field_validation = false;
    std::vector<ContributionDeclaration> contributions;
};

struct StepPutResult {
    amio_status_t status = AMIO_ERR_INVALID_INPUT;
    std::uint64_t handle = 0;
    bool source_released = false;
    bool payload_complete = false;
    bool used_fallback = false;
};

struct StepContributionSnapshot {
    std::vector<std::byte> bytes;
    RegionSelection region;
    std::uint64_t operation_id = 0;
};

/** Per-dataset lifecycle owner; storage publication is advanced by its execution adapter. */
class StepCoordinator {
   public:
    StepCoordinator(std::string dataset_uuid, ResourceBudget &budget);
    ~StepCoordinator();
    StepCoordinator(const StepCoordinator &) = delete;
    StepCoordinator &operator=(const StepCoordinator &) = delete;

    [[nodiscard]] amio_status_t register_schema(std::shared_ptr<const Schema> schema);
    [[nodiscard]] amio_status_t begin_step(const StepBeginRequest &request);
    [[nodiscard]] StepPutResult put(const std::string &variable, const void *source, std::size_t bytes);
    [[nodiscard]] StepPutResult put_memory(const std::string &variable, const MemoryDescriptor &descriptor, const void *source, std::size_t bytes);
    [[nodiscard]] amio_status_t end_step();
    [[nodiscard]] amio_status_t mark_payload_complete();
    [[nodiscard]] amio_status_t mark_publishing();
    [[nodiscard]] amio_status_t mark_reconciling();
    [[nodiscard]] amio_status_t mark_publication_visible(bool storage_synced = false);
    [[nodiscard]] amio_status_t fail_publication(amio_status_t error);
    [[nodiscard]] amio_status_t poison_dataset(amio_status_t error);
    [[nodiscard]] amio_status_t cancel_step();
    [[nodiscard]] amio_status_t wait_for_publication(std::uint64_t timeout_ms) const;
    [[nodiscard]] StepState state() const noexcept {
        return state_;
    }
    [[nodiscard]] const std::string &dataset_uuid() const noexcept {
        return dataset_uuid_;
    }
    [[nodiscard]] std::uint64_t active_step_id() const noexcept {
        return active_step_id_;
    }
    [[nodiscard]] double active_model_time() const noexcept {
        return active_model_time_;
    }
    [[nodiscard]] const std::string &active_schema_digest() const noexcept {
        return active_schema_digest_;
    }
    [[nodiscard]] bool dataset_poisoned() const noexcept {
        return dataset_poisoned_;
    }
    [[nodiscard]] OperationSnapshot operation_snapshot(std::uint64_t handle) const noexcept;
    [[nodiscard]] std::shared_ptr<OperationRecord> operation_record(std::uint64_t handle) const noexcept;
    [[nodiscard]] const std::map<std::string, StepContributionSnapshot, std::less<>> &contributions() const noexcept {
        return contributions_;
    }

   private:
    [[nodiscard]] amio_status_t fail_step(amio_status_t error) noexcept;
    [[nodiscard]] amio_status_t validate_put(const std::string &variable, std::size_t bytes);
    [[nodiscard]] StepPutResult put_snapshot(const std::string &variable, std::vector<std::byte> &&snapshot);
    void release_reservation() noexcept;
    [[nodiscard]] std::uint64_t add_operation();

    std::string dataset_uuid_;
    ResourceBudget &budget_;
    std::map<std::string, std::shared_ptr<const Schema>, std::less<>> schemas_;
    std::map<std::string, StepContributionSnapshot, std::less<>> contributions_;
    std::map<std::uint64_t, std::shared_ptr<OperationRecord>> operations_;
    ContributionPlan plan_;
    std::shared_ptr<OperationRecord> completion_operation_;
    std::optional<ResourceBudget::Reservation> reservation_;
    std::string active_schema_digest_;
    std::uint64_t last_step_id_ = 0;
    std::uint64_t active_step_id_ = 0;
    double active_model_time_ = 0.0;
    std::uint64_t next_operation_id_ = 1;
    StepState state_ = StepState::Declared;
    bool has_attempted_step_ = false;
    bool dataset_poisoned_ = false;
};

}  // namespace amio::detail

#endif  // AMIO_SRC_STEPS_STEP_COORDINATOR_HPP
