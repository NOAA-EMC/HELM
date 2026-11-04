#ifndef AMIO_SRC_EXECUTION_STREAM_RETENTION_HPP
#define AMIO_SRC_EXECUTION_STREAM_RETENTION_HPP

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

enum class StreamRetentionPolicy : std::uint8_t { Reliable, Latest };

struct RetainedStreamStep {
    std::uint64_t step_id = 0;
    std::string schema_digest;
    std::map<std::string, std::vector<std::byte>, std::less<>> fields;
    /** Per-contribution logical shape and origin, retained with the schema identity. */
    std::map<std::string, std::vector<std::uint64_t>, std::less<>> field_shapes;
    std::map<std::string, std::vector<std::uint64_t>, std::less<>> field_offsets;
    std::set<std::string, std::less<>> absent_optional_fields;
    std::size_t payload_bytes = 0;
};

struct StreamSkipRange {
    std::uint64_t first = 0;
    std::uint64_t last = 0;
};

struct StreamLease {
    std::uint64_t lease_id = 0;
    std::uint32_t consumer = 0;
    std::shared_ptr<const RetainedStreamStep> step;
};

enum class StreamAcquireState : std::uint8_t { Available, Skipped, Timeout, EndOfStream, Failed };
struct StreamAcquireResult {
    StreamAcquireState state = StreamAcquireState::Timeout;
    std::optional<StreamLease> lease;
    std::optional<StreamSkipRange> skipped;
    amio_status_t failure = AMIO_OK;
};

/** Finite step retention with per-consumer leases and cursor-derived skip ranges. */
class StreamRetention {
   public:
    struct Snapshot {
        std::size_t bytes = 0;
        std::size_t steps = 0;
        std::size_t leases = 0;
        std::size_t descriptors = 0;
        std::uint64_t skipped_steps = 0;
    };
    StreamRetention(StreamRetentionPolicy policy, std::uint32_t consumers, std::size_t max_steps, std::size_t max_bytes,
                    std::size_t max_descriptors = 4096);

    /** Publish an immutable completed step. Queue/budget exhaustion is explicit. */
    [[nodiscard]] amio_status_t publish(RetainedStreamStep step);
    /** Append one non-overlapping field contribution to an unpublished step. */
    [[nodiscard]] amio_status_t append_field(std::uint64_t step_id, std::string field, std::string schema_digest, std::vector<std::byte> bytes,
                                             std::vector<std::uint64_t> shape, std::vector<std::uint64_t> origin);
    /** Return the next available step, a cursor gap, timeout, EOS, or failure. */
    [[nodiscard]] StreamAcquireResult acquire_next(std::uint32_t consumer, std::chrono::milliseconds timeout);
    /** Release this consumer's lease; bytes remain pinned until all consumers release. */
    [[nodiscard]] amio_status_t release(std::uint64_t lease_id);
    [[nodiscard]] std::optional<std::uint64_t> lease_step_id(std::uint64_t lease_id, std::uint32_t consumer) const noexcept;
    /** Acknowledge remote release of a producer-retained step. */
    [[nodiscard]] amio_status_t acknowledge(std::uint64_t step_id, std::uint32_t consumer);
    /** Mark normal EOS or a terminal failure after accepted steps have drained. */
    void finish(amio_status_t status = AMIO_OK) noexcept;
    /** Close refuses to discard live views. */
    [[nodiscard]] amio_status_t close(bool drain = false) noexcept;
    [[nodiscard]] Snapshot snapshot() const noexcept;
    [[nodiscard]] StreamRetentionPolicy policy() const noexcept {
        return policy_;
    }

   private:
    struct Entry {
        std::shared_ptr<const RetainedStreamStep> step;
        std::vector<bool> released;
        std::size_t active_leases = 0;
    };
    struct LeaseRecord {
        std::uint64_t step_id;
        std::uint32_t consumer;
    };
    void reclaim_released_front() noexcept;

    const StreamRetentionPolicy policy_;
    const std::uint32_t consumer_count_;
    const std::size_t max_steps_;
    const std::size_t max_bytes_;
    const std::size_t max_descriptors_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::map<std::uint64_t, Entry> steps_;
    std::map<std::uint64_t, LeaseRecord> leases_;
    std::vector<std::optional<std::uint64_t>> cursors_;
    std::size_t bytes_ = 0;
    std::size_t descriptors_ = 0;
    std::uint64_t next_lease_id_ = 1;
    std::optional<std::uint64_t> first_step_id_;
    std::optional<std::uint64_t> final_step_id_;
    std::uint64_t skipped_steps_ = 0;
    amio_status_t terminal_status_ = AMIO_OK;
    bool finished_ = false;
    bool closed_ = false;
};

}  // namespace amio::detail

#endif
