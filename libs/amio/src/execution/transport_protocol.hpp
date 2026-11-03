#ifndef AMIO_SRC_EXECUTION_TRANSPORT_PROTOCOL_HPP
#define AMIO_SRC_EXECUTION_TRANSPORT_PROTOCOL_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "amio/amio_types.h"

namespace amio::detail {

inline constexpr std::uint16_t kTransportProtocolVersion = 1;
inline constexpr std::size_t kDatasetUuidBytes = 16;
inline constexpr std::size_t kSchemaDigestBytes = 32;
/** Serialized size of the canonical big-endian protocol envelope. */
inline constexpr std::size_t kTransportEnvelopeWireBytes = 2 + 8 + kDatasetUuidBytes + 8 + kSchemaDigestBytes + 4 + 1 + 4 + 4 + 8 + 4 + 4 +
                                                            4 + 2 * AMIO_MAX_RANK * 8 + 8 + 8 + 8;

/** Version 1 control and payload messages exchanged by an AMIO context. */
enum class MessageKind : std::uint8_t {
    Config = 1, Begin, Contribute, Credit, Chunk, Received, Seal, PayloadResult, Published, Cancel, Close, Error, EndOfStream, Release
};

enum class ProtocolError : std::uint8_t {
    None = 0, UnsupportedVersion, StaleEpoch, WrongDataset, InvalidKind, InvalidProducer, InvalidWriterMapping, InvalidWireFrame,
    InvalidSequence, InvalidRegion, FragmentTooLarge, PayloadTooLarge, WrongSchema, WrongStep, WrongField
};

/** Fixed-size frame metadata; validate it before allocating or receiving payload bytes. */
struct TransportEnvelope {
    std::uint16_t version = kTransportProtocolVersion;
    std::uint64_t context_epoch = 0;
    std::array<std::uint8_t, kDatasetUuidBytes> dataset_uuid{};
    std::uint64_t step_id = 0;
    std::array<std::uint8_t, kSchemaDigestBytes> schema_digest{};
    std::uint32_t field_id = 0;
    MessageKind kind = MessageKind::Error;
    std::uint32_t producer_rank = 0;
    std::uint32_t writer_rank = 0;
    std::uint64_t sequence = 0;
    std::uint32_t fragment_index = 0;
    std::uint32_t fragment_count = 1;
    std::uint32_t dimension_count = 0;
    std::array<std::uint64_t, AMIO_MAX_RANK> offsets{};
    std::array<std::uint64_t, AMIO_MAX_RANK> extents{};
    /** Linear byte offset of this fragment in the original contribution. */
    std::uint64_t fragment_offset = 0;
    /** Total byte length of the original contribution represented by fragments. */
    std::uint64_t contribution_bytes = 0;
    /** Byte length carried by this individual fragment. */
    std::uint64_t payload_bytes = 0;
};

/** Agreed context state used to reject stale or misrouted frames before allocation. */
struct TransportValidationContext {
    std::uint64_t expected_epoch = 0;
    std::array<std::uint8_t, kDatasetUuidBytes> expected_dataset_uuid{};
    std::optional<std::array<std::uint8_t, kSchemaDigestBytes>> expected_schema_digest;
    std::optional<std::uint64_t> expected_step_id;
    std::optional<std::uint32_t> expected_field_id;
    /** World ranks in the agreed producer order; envelope producer_rank is a world rank. */
    std::vector<std::uint32_t> producer_ranks;
    /** World ranks in the agreed writer order. */
    std::vector<std::uint32_t> writer_ranks;
    std::uint64_t max_payload_bytes = 0;
    std::uint64_t max_fragment_bytes = 0;
};

/** Validation outcome and the payload allocation size permitted by the envelope. */
struct ProtocolValidationResult {
    ProtocolError error = ProtocolError::None;
    std::size_t payload_allocation_bytes = 0;
    explicit operator bool() const noexcept { return error == ProtocolError::None; }
};

/** Validate identity, routing, region, sequence, and byte limits before payload allocation. */
[[nodiscard]] ProtocolValidationResult validate_envelope(const TransportEnvelope &envelope,
                                                          const TransportValidationContext &context) noexcept;
/** Encode fixed frame metadata as canonical big-endian bytes for same-job MPI transport. */
[[nodiscard]] std::array<std::uint8_t, kTransportEnvelopeWireBytes> encode_envelope(const TransportEnvelope &envelope) noexcept;
/** Decode one exact-size wire envelope; reject malformed lengths before reading fields. */
[[nodiscard]] ProtocolError decode_envelope(std::span<const std::uint8_t> bytes, TransportEnvelope &out) noexcept;
/** Match an acknowledgment to every request field except the message kind. */
[[nodiscard]] bool acknowledgment_matches(const TransportEnvelope &request, const TransportEnvelope &ack) noexcept;

enum class SequenceResult : std::uint8_t { Accepted, Duplicate, Stale, InvalidPeer };

/** Tracks the last accepted transfer sequence for each peer without growing after construction. */
class SequenceLedger {
public:
    /** Create a ledger with a fixed number of peers; no storage grows during use. */
    explicit SequenceLedger(std::size_t peer_count);
    /** Accept a strictly increasing nonzero sequence for the selected peer. */
    [[nodiscard]] SequenceResult accept(std::size_t peer, std::uint64_t sequence) noexcept;
private:
    std::vector<std::uint64_t> last_sequence_;
    std::vector<bool> initialized_;
};

/** One bounded payload slice, with an offset relative to its original contribution. */
struct PayloadFragment {
    std::uint64_t offset = 0;
    std::uint32_t length = 0;
};

/** Split a payload into MPI-count-safe ranges, rejecting descriptor-count overflow. */
[[nodiscard]] std::vector<PayloadFragment> make_fragments(std::uint64_t total_bytes, std::uint64_t max_fragment_bytes);

enum class AckMilestone : std::uint8_t { None = 0, ReceiverOwnsPayload, BackendCompleted, ReaderCanDiscoverStep };
/** Map a protocol message to its strongest acknowledged ownership/completion milestone. */
[[nodiscard]] AckMilestone acknowledgment_milestone(MessageKind kind) noexcept;

} // namespace amio::detail

#endif
