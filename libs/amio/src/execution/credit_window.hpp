#ifndef AMIO_SRC_EXECUTION_CREDIT_WINDOW_HPP
#define AMIO_SRC_EXECUTION_CREDIT_WINDOW_HPP

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace amio::detail {

/** Resource owner charged against either payload or reserved control capacity. */
enum class CreditOwner : std::uint8_t { Sender, Receiver, Control };

/** Per-peer payload credits plus a physically separate reserved control pool. */
class CreditWindow {
public:
    /** Finite byte/descriptor limits for each peer and the context as a whole. */
    struct Limits {
        std::size_t peer_count;
        std::size_t per_peer_bytes;
        std::size_t per_peer_descriptors;
        std::size_t total_data_bytes;
        std::size_t total_data_descriptors;
        std::size_t control_bytes;
        std::size_t control_descriptors;
    };
    struct Ticket { std::uint64_t id; std::size_t peer; CreditOwner owner; std::size_t bytes; std::size_t descriptors; };
    struct Snapshot {
        std::size_t sender_bytes = 0, receiver_bytes = 0, data_bytes = 0, control_bytes = 0;
        std::size_t sender_descriptors = 0, receiver_descriptors = 0, data_descriptors = 0, control_descriptors = 0;
    };
    /** Construct an empty bounded credit pool; invalid zero or inconsistent limits throw. */
    explicit CreditWindow(Limits limits);
    /** Reserve payload capacity for one peer and owner, or return empty if exhausted. */
    [[nodiscard]] std::optional<Ticket> acquire(std::size_t peer, CreditOwner owner, std::size_t bytes, std::size_t descriptors);
    /** Reserve from capacity kept separate for progress and protocol control traffic. */
    [[nodiscard]] std::optional<Ticket> acquire_control(std::size_t bytes, std::size_t descriptors);
    /** Release a live ticket exactly once; stale or duplicate tickets return false. */
    bool release(const Ticket &ticket) noexcept;
    /** Return current reserved usage without changing any credit. */
    [[nodiscard]] Snapshot snapshot() const noexcept;
    /** Return per-counter peak reservations since construction. */
    [[nodiscard]] Snapshot high_water() const noexcept;
private:
    struct Record { std::size_t peer, bytes, descriptors; CreditOwner owner; };
    struct PeerUse { std::size_t sender_bytes=0, receiver_bytes=0, sender_descriptors=0, receiver_descriptors=0; };
    Limits limits_;
    mutable std::mutex mutex_;
    std::vector<PeerUse> peers_;
    std::unordered_map<std::uint64_t, Record> live_;
    std::uint64_t next_id_ = 1;
    Snapshot usage_;
    Snapshot high_water_;
};

} // namespace amio::detail

#endif
