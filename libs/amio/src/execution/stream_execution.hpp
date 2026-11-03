#ifndef AMIO_SRC_EXECUTION_STREAM_EXECUTION_HPP
#define AMIO_SRC_EXECUTION_STREAM_EXECUTION_HPP

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "execution/aggregated_execution.hpp"
#include "execution/stream_retention.hpp"

namespace amio::detail {

/** Same-job static producer-to-consumer stream using bounded v1 MPI frames. */
class StreamExecution {
   public:
    StreamExecution(const ExecutionContext &context, MpiTransport &transport, StreamRetention &retention, std::uint64_t epoch,
                    std::array<std::uint8_t, kDatasetUuidBytes> dataset_uuid, std::size_t max_step_bytes);

    /** Submit one producer contribution; the caller retains the source through RECEIVED. */
    [[nodiscard]] amio_status_t submit(TransportEnvelope frame, std::span<const std::byte> bytes, std::size_t element_bytes = 1);
    /** Receive this consumer's canonical producer round, validate it, and retain the completed step. */
    [[nodiscard]] amio_status_t receive_step(std::uint64_t step_id, const std::array<std::uint8_t, kSchemaDigestBytes> &schema_digest,
                                             const std::vector<std::string> &absent_optional_fields = {}, std::uint32_t fields_per_producer = 1,
                                             std::uint64_t *received_bytes = nullptr);
    /** Acknowledge release only after every local view of the retained step is released. */
    [[nodiscard]] amio_status_t release_lease(std::uint64_t lease_id);
    /** Producer waits for consumer release before retiring its source snapshot. */
    [[nodiscard]] amio_status_t wait_for_release(std::uint64_t step_id, std::chrono::milliseconds timeout = std::chrono::milliseconds::max());
    /** Send normal EOS or a graceful-disconnect error for this producer. */
    [[nodiscard]] amio_status_t finish_producer(bool graceful = true);
    /** Cancel the stream after all already accepted producer steps are released. */
    [[nodiscard]] amio_status_t cancel_producer();
    /** Receive EOS/disconnect from all statically assigned producers. */
    [[nodiscard]] amio_status_t finish_consumer();

   private:
    const ExecutionContext *context_;
    MpiTransport *transport_;
    StreamRetention *retention_;
    const std::uint64_t epoch_;
    const std::array<std::uint8_t, kDatasetUuidBytes> dataset_uuid_;
    const std::size_t max_step_bytes_;
    std::vector<std::uint64_t> sent_sequences_;
    SequenceLedger received_sequences_;
    SequenceLedger control_sequences_;
    std::vector<std::uint64_t> release_sequences_;
    std::uint64_t next_control_sequence_ = 1;
};

}  // namespace amio::detail

#endif
