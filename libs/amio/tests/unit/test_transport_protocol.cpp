#include "execution/transport_protocol.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

using namespace amio::detail;

namespace {

TransportEnvelope valid_envelope() {
    TransportEnvelope frame;
    frame.version = kTransportProtocolVersion;
    frame.context_epoch = 7;
    frame.dataset_uuid[0] = 0x42;
    frame.step_id = 11;
    frame.schema_digest[0] = 0xA5;
    frame.field_id = 3;
    frame.kind = MessageKind::Chunk;
    frame.producer_rank = 1;
    frame.writer_rank = 3;
    frame.sequence = 19;
    frame.fragment_index = 0;
    frame.fragment_count = 1;
    frame.dimension_count = 2;
    frame.offsets[0] = 4;
    frame.extents[0] = 8;
    frame.offsets[1] = 0;
    frame.extents[1] = 16;
    frame.contribution_bytes = 512;
    frame.payload_bytes = 512;
    return frame;
}

TransportValidationContext validation_context() {
    TransportValidationContext context;
    context.expected_epoch = 7;
    context.expected_dataset_uuid[0] = 0x42;
    context.expected_schema_digest.emplace();
    (*context.expected_schema_digest)[0] = 0xA5;
    context.expected_step_id = 11;
    context.expected_field_id = 3;
    context.producer_ranks = {0, 1};
    context.writer_ranks = {3};
    context.max_payload_bytes = 1024;
    context.max_fragment_bytes = 512;
    return context;
}

void test_rejects_stale_epoch_before_payload_handling() {
    auto frame = valid_envelope();
    frame.context_epoch = 6;
    auto result = validate_envelope(frame, validation_context());
    assert(result.error == ProtocolError::StaleEpoch);
    assert(result.payload_allocation_bytes == 0);
}

void test_rejects_schema_step_and_field_mismatch_before_allocation() {
    auto frame = valid_envelope();
    auto context = validation_context();
    frame.schema_digest[0]++;
    assert(validate_envelope(frame, context).error == ProtocolError::WrongSchema);
    frame = valid_envelope();
    frame.step_id++;
    assert(validate_envelope(frame, context).error == ProtocolError::WrongStep);
    frame = valid_envelope();
    frame.field_id++;
    assert(validate_envelope(frame, context).error == ProtocolError::WrongField);
}

void test_rejects_oversize_and_invalid_mapping_before_payload_handling() {
    auto frame = valid_envelope();
    frame.payload_bytes = 513;
    frame.contribution_bytes = 513;
    auto result = validate_envelope(frame, validation_context());
    assert(result.error == ProtocolError::FragmentTooLarge);
    assert(result.payload_allocation_bytes == 0);

    frame = valid_envelope();
    frame.producer_rank = 9;
    result = validate_envelope(frame, validation_context());
    assert(result.error == ProtocolError::InvalidProducer);
    assert(result.payload_allocation_bytes == 0);

    frame = valid_envelope();
    frame.writer_rank = 4;
    result = validate_envelope(frame, validation_context());
    assert(result.error == ProtocolError::InvalidWriterMapping);
    assert(result.payload_allocation_bytes == 0);
}

void test_rejects_overflowing_or_invalid_regions() {
    auto frame = valid_envelope();
    frame.offsets[0] = std::numeric_limits<std::uint64_t>::max() - 3;
    frame.extents[0] = 8;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidRegion);

    frame = valid_envelope();
    frame.dimension_count = AMIO_MAX_RANK + 1;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidRegion);

    frame = valid_envelope();
    frame.extents[0] = 0;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidRegion);
    frame.contribution_bytes = 0;
    frame.payload_bytes = 0;
    frame.fragment_offset = 0;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::None);
}

void test_rejects_duplicate_and_stale_sequences() {
    SequenceLedger ledger(2);
    assert(ledger.accept(0, 10) == SequenceResult::Accepted);
    assert(ledger.accept(0, 10) == SequenceResult::Duplicate);
    assert(ledger.accept(0, 9) == SequenceResult::Stale);
    assert(ledger.accept(1, 10) == SequenceResult::Accepted);
}

void test_rejects_impossible_fragment_drains() {
    auto frame = valid_envelope();
    frame.contribution_bytes = 2;
    frame.payload_bytes = 1;
    frame.fragment_count = 3;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidSequence);
    frame.fragment_count = 2;
    assert(validate_envelope(frame, validation_context()));
    frame.payload_bytes = 0;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidSequence);
    frame.payload_bytes = 1;
    frame.fragment_offset = 1;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidSequence);
    frame.fragment_index = 1;
    assert(validate_envelope(frame, validation_context()));
    frame.fragment_offset = 0;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidSequence);
    frame.contribution_bytes = 0;
    frame.payload_bytes = 0;
    frame.fragment_offset = 0;
    assert(validate_envelope(frame, validation_context()).error == ProtocolError::InvalidSequence);
}

void test_arbitrary_rank_sets_map_deterministically() {
    auto context = validation_context();
    context.producer_ranks = {2, 7, 11};
    context.writer_ranks = {4, 9};
    auto frame = valid_envelope();
    frame.producer_rank = 7;
    frame.writer_rank = 9;
    assert(validate_envelope(frame, context));
    frame.writer_rank = 4;
    assert(validate_envelope(frame, context).error == ProtocolError::InvalidWriterMapping);
    frame.kind = MessageKind::Begin;
    frame.dimension_count = 0;
    frame.contribution_bytes = 0;
    frame.payload_bytes = 0;
    assert(validate_envelope(frame, context));
    frame.writer_rank = 5;
    assert(validate_envelope(frame, context).error == ProtocolError::InvalidWriterMapping);
}

void test_wire_envelope_round_trip_and_exact_length() {
    auto frame = valid_envelope();
    const auto wire = encode_envelope(frame);
    TransportEnvelope decoded;
    assert(decode_envelope(wire, decoded) == ProtocolError::None);
    assert(decoded.version == frame.version);
    assert(decoded.context_epoch == frame.context_epoch);
    assert(decoded.dataset_uuid == frame.dataset_uuid);
    assert(decoded.step_id == frame.step_id);
    assert(decoded.schema_digest == frame.schema_digest);
    assert(decoded.field_id == frame.field_id && decoded.kind == frame.kind);
    assert(decoded.producer_rank == frame.producer_rank && decoded.writer_rank == frame.writer_rank);
    assert(decoded.sequence == frame.sequence && decoded.payload_bytes == frame.payload_bytes);
    assert(decoded.fragment_offset == frame.fragment_offset && decoded.contribution_bytes == frame.contribution_bytes);
    assert(decoded.offsets == frame.offsets && decoded.extents == frame.extents);
    assert(wire[0] == 0 && wire[1] == kTransportProtocolVersion);
    assert(decode_envelope(std::span<const std::uint8_t>(wire.data(), wire.size() - 1), decoded) == ProtocolError::InvalidWireFrame);
}

void test_fragments_respect_configured_bound() {
    const auto fragments = make_fragments(1001, 400);
    assert(fragments.size() == 3);
    assert(fragments[0].offset == 0 && fragments[0].length == 400);
    assert(fragments[1].offset == 400 && fragments[1].length == 400);
    assert(fragments[2].offset == 800 && fragments[2].length == 201);
    for (const auto &fragment : fragments) assert(fragment.length <= 400);
    assert(make_fragments(0, 400).empty());
}

void test_receipt_is_not_publication_acknowledgment() {
    assert(acknowledgment_milestone(MessageKind::Received) == AckMilestone::ReceiverOwnsPayload);
    assert(acknowledgment_milestone(MessageKind::Published) == AckMilestone::ReaderCanDiscoverStep);
    assert(acknowledgment_milestone(MessageKind::Received) != acknowledgment_milestone(MessageKind::Published));
}

void test_acknowledgments_match_full_request() {
    const auto request = valid_envelope();
    auto ack = request;
    ack.kind = MessageKind::Received;
    assert(acknowledgment_matches(request, ack));
    ack.schema_digest[0]++;
    assert(!acknowledgment_matches(request, ack));
    ack = request; ack.field_id++;
    assert(!acknowledgment_matches(request, ack));
    ack = request; ack.fragment_count++;
    assert(!acknowledgment_matches(request, ack));
    ack = request; ack.offsets[0]++;
    assert(!acknowledgment_matches(request, ack));
    ack = request; ack.payload_bytes--;
    assert(!acknowledgment_matches(request, ack));
}

}  // namespace

int main() {
    test_rejects_stale_epoch_before_payload_handling();
    test_rejects_schema_step_and_field_mismatch_before_allocation();
    test_rejects_oversize_and_invalid_mapping_before_payload_handling();
    test_rejects_overflowing_or_invalid_regions();
    test_rejects_duplicate_and_stale_sequences();
    test_rejects_impossible_fragment_drains();
    test_arbitrary_rank_sets_map_deterministically();
    test_wire_envelope_round_trip_and_exact_length();
    test_fragments_respect_configured_bound();
    test_receipt_is_not_publication_acknowledgment();
    test_acknowledgments_match_full_request();
}
