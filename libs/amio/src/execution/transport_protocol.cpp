#include "execution/transport_protocol.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace amio::detail {
namespace {

template <std::size_t N>
void put_unsigned(std::array<std::uint8_t, N> &bytes, std::size_t &offset, std::uint64_t value, std::size_t width) noexcept {
    for (std::size_t i = 0; i < width; ++i) {
        const auto shift = static_cast<unsigned int>((width - i - 1) * 8);
        bytes[offset++] = static_cast<std::uint8_t>((value >> shift) & 0xffu);
    }
}

std::uint64_t get_unsigned(std::span<const std::uint8_t> bytes, std::size_t &offset, std::size_t width) noexcept {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i) value = (value << 8) | bytes[offset++];
    return value;
}

template <std::size_t N>
void put_array(std::array<std::uint8_t, N> &wire, std::size_t &offset, const std::uint8_t *data, std::size_t length) noexcept {
    std::copy_n(data, length, wire.begin() + static_cast<std::ptrdiff_t>(offset));
    offset += length;
}

void get_array(std::span<const std::uint8_t> wire, std::size_t &offset, std::uint8_t *data, std::size_t length) noexcept {
    std::copy_n(wire.begin() + static_cast<std::ptrdiff_t>(offset), length, data);
    offset += length;
}

bool known_kind(MessageKind kind) noexcept {
    const auto value = static_cast<std::uint8_t>(kind);
    return value >= static_cast<std::uint8_t>(MessageKind::Config) && value <= static_cast<std::uint8_t>(MessageKind::EndOfStream);
}

} // namespace

std::array<std::uint8_t, kTransportEnvelopeWireBytes> encode_envelope(const TransportEnvelope &envelope) noexcept {
    std::array<std::uint8_t, kTransportEnvelopeWireBytes> bytes{};
    std::size_t offset = 0;
    put_unsigned(bytes, offset, envelope.version, 2);
    put_unsigned(bytes, offset, envelope.context_epoch, 8);
    put_array(bytes, offset, envelope.dataset_uuid.data(), envelope.dataset_uuid.size());
    put_unsigned(bytes, offset, envelope.step_id, 8);
    put_array(bytes, offset, envelope.schema_digest.data(), envelope.schema_digest.size());
    put_unsigned(bytes, offset, envelope.field_id, 4);
    put_unsigned(bytes, offset, static_cast<std::uint8_t>(envelope.kind), 1);
    put_unsigned(bytes, offset, envelope.producer_rank, 4);
    put_unsigned(bytes, offset, envelope.writer_rank, 4);
    put_unsigned(bytes, offset, envelope.sequence, 8);
    put_unsigned(bytes, offset, envelope.fragment_index, 4);
    put_unsigned(bytes, offset, envelope.fragment_count, 4);
    put_unsigned(bytes, offset, envelope.dimension_count, 4);
    for (const auto value : envelope.offsets) put_unsigned(bytes, offset, value, 8);
    for (const auto value : envelope.extents) put_unsigned(bytes, offset, value, 8);
    put_unsigned(bytes, offset, envelope.fragment_offset, 8);
    put_unsigned(bytes, offset, envelope.contribution_bytes, 8);
    put_unsigned(bytes, offset, envelope.payload_bytes, 8);
    return bytes;
}

ProtocolError decode_envelope(std::span<const std::uint8_t> bytes, TransportEnvelope &out) noexcept {
    if (bytes.size() != kTransportEnvelopeWireBytes) return ProtocolError::InvalidWireFrame;
    std::size_t offset = 0;
    TransportEnvelope decoded;
    decoded.version = static_cast<std::uint16_t>(get_unsigned(bytes, offset, 2));
    decoded.context_epoch = get_unsigned(bytes, offset, 8);
    get_array(bytes, offset, decoded.dataset_uuid.data(), decoded.dataset_uuid.size());
    decoded.step_id = get_unsigned(bytes, offset, 8);
    get_array(bytes, offset, decoded.schema_digest.data(), decoded.schema_digest.size());
    decoded.field_id = static_cast<std::uint32_t>(get_unsigned(bytes, offset, 4));
    decoded.kind = static_cast<MessageKind>(get_unsigned(bytes, offset, 1));
    decoded.producer_rank = static_cast<std::uint32_t>(get_unsigned(bytes, offset, 4));
    decoded.writer_rank = static_cast<std::uint32_t>(get_unsigned(bytes, offset, 4));
    decoded.sequence = get_unsigned(bytes, offset, 8);
    decoded.fragment_index = static_cast<std::uint32_t>(get_unsigned(bytes, offset, 4));
    decoded.fragment_count = static_cast<std::uint32_t>(get_unsigned(bytes, offset, 4));
    decoded.dimension_count = static_cast<std::uint32_t>(get_unsigned(bytes, offset, 4));
    for (auto &value : decoded.offsets) value = get_unsigned(bytes, offset, 8);
    for (auto &value : decoded.extents) value = get_unsigned(bytes, offset, 8);
    decoded.fragment_offset = get_unsigned(bytes, offset, 8);
    decoded.contribution_bytes = get_unsigned(bytes, offset, 8);
    decoded.payload_bytes = get_unsigned(bytes, offset, 8);
    out = std::move(decoded);
    return ProtocolError::None;
}

bool acknowledgment_matches(const TransportEnvelope &request, const TransportEnvelope &ack) noexcept {
    return ack.version == request.version && ack.context_epoch == request.context_epoch &&
           ack.dataset_uuid == request.dataset_uuid && ack.step_id == request.step_id &&
           ack.schema_digest == request.schema_digest && ack.field_id == request.field_id &&
           ack.producer_rank == request.producer_rank && ack.writer_rank == request.writer_rank &&
           ack.sequence == request.sequence && ack.fragment_index == request.fragment_index &&
           ack.fragment_count == request.fragment_count && ack.dimension_count == request.dimension_count &&
           ack.offsets == request.offsets && ack.extents == request.extents &&
           ack.fragment_offset == request.fragment_offset && ack.contribution_bytes == request.contribution_bytes &&
           ack.payload_bytes == request.payload_bytes;
}

ProtocolValidationResult validate_envelope(const TransportEnvelope &envelope, const TransportValidationContext &context) noexcept {
    if (envelope.version != kTransportProtocolVersion) return {ProtocolError::UnsupportedVersion, 0};
    if (envelope.context_epoch != context.expected_epoch) return {ProtocolError::StaleEpoch, 0};
    if (envelope.dataset_uuid != context.expected_dataset_uuid) return {ProtocolError::WrongDataset, 0};
    if (context.expected_schema_digest && envelope.schema_digest != *context.expected_schema_digest) return {ProtocolError::WrongSchema, 0};
    if (context.expected_step_id && envelope.step_id != *context.expected_step_id) return {ProtocolError::WrongStep, 0};
    if (context.expected_field_id && envelope.field_id != *context.expected_field_id) return {ProtocolError::WrongField, 0};
    if (!known_kind(envelope.kind)) return {ProtocolError::InvalidKind, 0};
    const auto producer = std::find(context.producer_ranks.begin(), context.producer_ranks.end(), envelope.producer_rank);
    if (producer == context.producer_ranks.end() || context.writer_ranks.empty()) return {ProtocolError::InvalidProducer, 0};
    const auto producer_index = static_cast<std::size_t>(producer - context.producer_ranks.begin());
    const auto expected_writer = context.writer_ranks[producer_index % context.writer_ranks.size()];
    const bool writer_is_member = std::find(context.writer_ranks.begin(), context.writer_ranks.end(), envelope.writer_rank) !=
                                  context.writer_ranks.end();
    if (!writer_is_member || (envelope.kind == MessageKind::Chunk && envelope.writer_rank != expected_writer)) {
        return {ProtocolError::InvalidWriterMapping, 0};
    }
    if (envelope.sequence == 0 || envelope.fragment_count == 0 || envelope.fragment_index >= envelope.fragment_count) {
        return {ProtocolError::InvalidSequence, 0};
    }
    if (envelope.dimension_count > AMIO_MAX_RANK) return {ProtocolError::InvalidRegion, 0};
    for (std::uint32_t dimension = 0; dimension < envelope.dimension_count; ++dimension) {
        const auto offset = envelope.offsets[dimension];
        const auto extent = envelope.extents[dimension];
        if ((extent == 0 && (envelope.contribution_bytes != 0 || envelope.payload_bytes != 0)) ||
            offset > std::numeric_limits<std::uint64_t>::max() - extent) {
            return {ProtocolError::InvalidRegion, 0};
        }
    }
    if (envelope.fragment_offset > envelope.contribution_bytes ||
        envelope.payload_bytes > envelope.contribution_bytes - envelope.fragment_offset) return {ProtocolError::InvalidSequence, 0};
    if (envelope.kind == MessageKind::Chunk) {
        // Every nonempty fragment carries at least one byte.  Otherwise a peer
        // could advertise an arbitrarily long drain for a tiny contribution.
        if (envelope.contribution_bytes == 0) {
            if (envelope.fragment_count != 1 || envelope.fragment_index != 0 || envelope.payload_bytes != 0)
                return {ProtocolError::InvalidSequence, 0};
        } else if (envelope.payload_bytes == 0 || envelope.fragment_count > envelope.contribution_bytes ||
                   (envelope.fragment_index == 0 && envelope.fragment_offset != 0) ||
                   (envelope.fragment_index + 1 == envelope.fragment_count &&
                    envelope.payload_bytes != envelope.contribution_bytes - envelope.fragment_offset)) {
            return {ProtocolError::InvalidSequence, 0};
        }
    }
    if (envelope.contribution_bytes > context.max_payload_bytes || envelope.payload_bytes > context.max_payload_bytes)
        return {ProtocolError::PayloadTooLarge, 0};
    if (envelope.payload_bytes > context.max_fragment_bytes || envelope.payload_bytes > std::numeric_limits<std::uint32_t>::max()) {
        return {ProtocolError::FragmentTooLarge, 0};
    }
    // The caller may allocate payload storage only after this result is clear.
    if (envelope.payload_bytes > std::numeric_limits<std::size_t>::max()) return {ProtocolError::PayloadTooLarge, 0};
    return {ProtocolError::None, static_cast<std::size_t>(envelope.payload_bytes)};
}

SequenceLedger::SequenceLedger(std::size_t peer_count) : last_sequence_(peer_count), initialized_(peer_count, false) {
    if (peer_count == 0) throw std::invalid_argument("sequence ledger requires at least one peer");
}

SequenceResult SequenceLedger::accept(std::size_t peer, std::uint64_t sequence) noexcept {
    if (peer >= last_sequence_.size() || sequence == 0) return SequenceResult::InvalidPeer;
    if (initialized_[peer]) {
        if (sequence == last_sequence_[peer]) return SequenceResult::Duplicate;
        if (sequence < last_sequence_[peer]) return SequenceResult::Stale;
    }
    last_sequence_[peer] = sequence;
    initialized_[peer] = true;
    return SequenceResult::Accepted;
}

std::vector<PayloadFragment> make_fragments(std::uint64_t total_bytes, std::uint64_t max_fragment_bytes) {
    if (max_fragment_bytes == 0 || max_fragment_bytes > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("fragment size must be within the positive MPI count range");
    }
    if (total_bytes == 0) return {};
    const std::uint64_t count = total_bytes / max_fragment_bytes + (total_bytes % max_fragment_bytes != 0 ? 1 : 0);
    constexpr std::uint64_t kMaximumFragments = 1u << 20;
    if (count > kMaximumFragments || count > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("fragment descriptor limit exceeded");
    }
    std::vector<PayloadFragment> result;
    result.reserve(static_cast<std::size_t>(count));
    std::uint64_t offset = 0;
    while (offset < total_bytes) {
        const auto length = std::min(max_fragment_bytes, total_bytes - offset);
        result.push_back({offset, static_cast<std::uint32_t>(length)});
        offset += length;
    }
    return result;
}

AckMilestone acknowledgment_milestone(MessageKind kind) noexcept {
    switch (kind) {
        case MessageKind::Received: return AckMilestone::ReceiverOwnsPayload;
        case MessageKind::PayloadResult: return AckMilestone::BackendCompleted;
        case MessageKind::Published: return AckMilestone::ReaderCanDiscoverStep;
        default: return AckMilestone::None;
    }
}

} // namespace amio::detail
