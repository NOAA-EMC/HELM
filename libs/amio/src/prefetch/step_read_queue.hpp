#ifndef AMIO_SRC_PREFETCH_STEP_READ_QUEUE_HPP
#define AMIO_SRC_PREFETCH_STEP_READ_QUEUE_HPP

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "amio/amio_errors.h"
#include "amio/amio_types.h"

namespace amio::detail {

/** Identity for a step payload; selection participates in cache identity. */
struct StepReadKey {
    std::string dataset_uuid;
    std::uint64_t step_id = 0;
    std::string schema_digest;
    std::string field;
    std::optional<amio_bbox_t> selection;
};

/** Metadata discovered from a committed step before payload allocation. */
struct StepReadMetadata {
    std::string dataset_uuid;
    std::uint64_t step_id = 0;
    std::string schema_digest;
    std::string field;
    bool field_present = false;
    std::size_t payload_bytes = 0;
    // Global field shape from the committed schema. Required for selected reads
    // so bounds can be checked before a cache allocation or backend read.
    std::vector<std::uint64_t> global_extents;
};

/** Thread-safe byte-bounded LRU of immutable step read payloads. */
class StepReadQueue {
   public:
    explicit StepReadQueue(std::size_t byte_limit) : byte_limit_(byte_limit) {}
    StepReadQueue(const StepReadQueue &) = delete;
    StepReadQueue &operator=(const StepReadQueue &) = delete;

    // Checks identity, presence, and byte bounds before allocating the payload.
    [[nodiscard]] amio_status_t store(const StepReadKey &key, const StepReadMetadata &metadata, const void *payload, std::size_t bytes);
    // Returns FIELD_NOT_PRESENT when metadata says the optional field is absent.
    [[nodiscard]] amio_status_t load(const StepReadKey &key, const StepReadMetadata &metadata, std::vector<std::byte> &payload);
    void clear() noexcept;
    [[nodiscard]] std::size_t bytes_used() const noexcept;
    [[nodiscard]] std::size_t entry_count() const noexcept;

   private:
    struct Entry {
        std::vector<std::byte> payload;
        std::list<std::string>::iterator lru;
    };
    static bool valid(const StepReadKey &key, const StepReadMetadata &metadata) noexcept;
    static std::string encode(const StepReadKey &key);
    void evict_for(std::size_t bytes);

    const std::size_t byte_limit_;
    mutable std::mutex mutex_;
    std::size_t bytes_used_ = 0;
    std::list<std::string> lru_;
    std::unordered_map<std::string, Entry> entries_;
};

}  // namespace amio::detail

#endif
