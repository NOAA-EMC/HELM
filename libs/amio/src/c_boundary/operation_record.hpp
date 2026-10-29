#ifndef AMIO_SRC_C_BOUNDARY_OPERATION_RECORD_HPP
#define AMIO_SRC_C_BOUNDARY_OPERATION_RECORD_HPP

#include "amio/amio_errors.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace amio::detail {

enum class OperationMilestone : std::uint8_t { SourceReleased, PayloadComplete, ReaderVisible, StorageSynced };
enum class OperationOutcome : std::uint8_t { Pending, Succeeded, Failed };
enum class WaitOutcome : std::uint8_t { Reached, Failed, Timeout };

struct OperationSnapshot {
    OperationOutcome outcome = OperationOutcome::Pending;
    amio_status_t error = AMIO_OK;
    bool source_released = false;
    bool payload_complete = false;
    bool reader_visible = false;
    bool storage_synced = false;
    std::uint32_t observer_references = 0;
};

/**
 * Shared operation state. Runtime owners retain the shared record until work
 * and step accounting are finished; caller references are observations only.
 */
class OperationRecord {
   public:
    OperationRecord() = default;
    OperationRecord(const OperationRecord &) = delete;
    OperationRecord &operator=(const OperationRecord &) = delete;

    bool retain_observer() noexcept;
    bool release_observer() noexcept;
    bool mark(OperationMilestone milestone) noexcept;
    bool finish_success() noexcept;
    bool finish_failure(amio_status_t error) noexcept;
    [[nodiscard]] OperationSnapshot snapshot() const noexcept;
    [[nodiscard]] WaitOutcome wait_for(OperationMilestone milestone, std::chrono::milliseconds timeout) const;

   private:
    [[nodiscard]] bool reached_locked(OperationMilestone milestone) const noexcept;

    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    OperationSnapshot state_;
    std::atomic<std::uint32_t> observer_references_{0};
};

} // namespace amio::detail

#endif // AMIO_SRC_C_BOUNDARY_OPERATION_RECORD_HPP
