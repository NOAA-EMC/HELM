#include "execution/aggregated_execution.hpp"

#include "execution/role_map.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace amio::detail {

amio_status_t visit_destination_regions(
    const TransportEnvelope &frame, std::size_t element_bytes, std::span<const std::byte> payload,
    const std::function<amio_status_t(const DestinationRegionChunk &, std::span<const std::byte>)> &sink) {
    if (!sink || element_bytes == 0 || frame.dimension_count == 0 || frame.dimension_count > AMIO_MAX_RANK ||
        frame.payload_bytes != payload.size() || frame.fragment_offset > frame.contribution_bytes ||
        frame.payload_bytes > frame.contribution_bytes - frame.fragment_offset ||
        frame.fragment_offset % element_bytes != 0 || frame.payload_bytes % element_bytes != 0)
        return AMIO_ERR_INVALID_INPUT;

    std::uint64_t elements = 1;
    bool empty = false;
    for (std::uint32_t d = 0; d < frame.dimension_count; ++d) {
        const auto extent = frame.extents[d];
        if (extent > std::numeric_limits<std::uint64_t>::max() - frame.offsets[d]) return AMIO_ERR_INVALID_INPUT;
        if (extent == 0) empty = true;
    }
    if (empty) elements = 0;
    else {
        for (std::uint32_t d = 0; d < frame.dimension_count; ++d) {
            const auto extent = frame.extents[d];
            if (extent > std::numeric_limits<std::uint64_t>::max() / elements) return AMIO_ERR_INVALID_INPUT;
            elements *= extent;
        }
    }
    if (elements > std::numeric_limits<std::uint64_t>::max() / element_bytes ||
        elements * element_bytes != frame.contribution_bytes) return AMIO_ERR_INVALID_INPUT;
    if (empty) {
        if (!payload.empty() || frame.fragment_offset != 0) return AMIO_ERR_INVALID_INPUT;
        DestinationRegionChunk region{frame.dimension_count, frame.offsets, frame.extents};
        return sink(region, payload);
    }
    if (payload.empty()) return AMIO_ERR_INVALID_INPUT;

    std::uint64_t linear = frame.fragment_offset / element_bytes;
    std::size_t consumed = 0;
    while (consumed < payload.size()) {
        DestinationRegionChunk region;
        region.dimension_count = frame.dimension_count;
        auto coordinate = linear;
        for (std::uint32_t d = frame.dimension_count; d-- > 0;) {
            region.offsets[d] = frame.offsets[d] + coordinate % frame.extents[d];
            coordinate /= frame.extents[d];
            region.extents[d] = 1;
        }
        const auto last = frame.dimension_count - 1;
        const auto row_remaining = frame.extents[last] - (linear % frame.extents[last]);
        const auto remaining_elements = (payload.size() - consumed) / element_bytes;
        const auto count = std::min<std::uint64_t>(row_remaining, remaining_elements);
        region.extents[last] = count;
        const auto bytes = static_cast<std::size_t>(count) * element_bytes;
        const auto status = sink(region, payload.subspan(consumed, bytes));
        if (status != AMIO_OK) return status;
        consumed += bytes;
        linear += count;
    }
    return AMIO_OK;
}

AggregatedExecution::AggregatedExecution(const ExecutionContext &context, MpiTransport &transport)
    : context_(&context), transport_(&transport), received_sequences_(context.roles().producers.size()) {
    if (context.roles().mode != ExecutionMode::Aggregated) {
        throw std::invalid_argument("aggregated execution requires an aggregated context");
    }
}

amio_status_t AggregatedExecution::forward(const TransportEnvelope &contribution, std::span<const std::byte> bytes,
                                           std::size_t element_bytes) const {
    const auto producer = context_->rank();
    if (context_->role_map().role(producer) != RankRole::Producer || contribution.kind != MessageKind::Chunk ||
        contribution.producer_rank != producer) return AMIO_ERR_INVALID_POLICY;
    const auto writer = context_->role_map().writer_for(producer);
    if (!writer || contribution.writer_rank != *writer) return AMIO_ERR_INVALID_POLICY;
    return transport_->send_payload(*writer, contribution, bytes, element_bytes);
}

amio_status_t AggregatedExecution::receive(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                            const ChunkSink &sink, ProtocolError *protocol_error) const {
    if (protocol_error) *protocol_error = ProtocolError::None;
    if (!sink || context_->role_map().role(context_->rank()) != RankRole::Writer ||
        context_->role_map().writer_for(producer_rank) != context_->rank()) return AMIO_ERR_INVALID_POLICY;

    const auto expected = std::find(validation.producer_ranks.begin(), validation.producer_ranks.end(), producer_rank);
    if (expected == validation.producer_ranks.end()) return AMIO_ERR_INVALID_POLICY;
    const auto mapped_producer = std::find(context_->role_map().producers().begin(), context_->role_map().producers().end(),
                                           static_cast<int>(producer_rank));
    if (mapped_producer == context_->role_map().producers().end() ||
        static_cast<std::size_t>(mapped_producer - context_->role_map().producers().begin()) !=
            static_cast<std::size_t>(expected - validation.producer_ranks.begin())) return AMIO_ERR_INVALID_POLICY;
    std::vector<std::byte> payload;
    TransportEnvelope first;
    std::uint64_t next_offset = 0;
    std::uint32_t expected_index = 0;
    std::uint32_t fragment_count = 0;
    amio_status_t sink_status = AMIO_OK;
    for (;;) {
        TransportEnvelope frame;
        CreditWindow::Ticket credit{};
        ProtocolError error = ProtocolError::None;
        const auto receive_status = transport_->receive_fragment(producer_rank, validation, frame, payload, &credit, &error);
        if (receive_status != AMIO_OK) {
            if (protocol_error) *protocol_error = error;
            return receive_status;
        }

        bool valid = frame.kind == MessageKind::Chunk && frame.producer_rank == producer_rank &&
                     frame.writer_rank == context_->rank();
        if (expected_index == 0) {
            first = frame;
            fragment_count = frame.fragment_count;
            valid = valid && frame.fragment_index == 0 && fragment_count != 0;
            const auto producer_slot = static_cast<std::size_t>(expected - validation.producer_ranks.begin());
            if (valid) {
                SequenceResult sequence_result = SequenceResult::InvalidPeer;
                {
                    std::lock_guard<std::mutex> lock(sequence_mutex_);
                    sequence_result = received_sequences_.accept(producer_slot, frame.sequence);
                }
                valid = sequence_result == SequenceResult::Accepted;
            }
        } else {
            valid = valid && frame.version == first.version && frame.context_epoch == first.context_epoch &&
                    frame.dataset_uuid == first.dataset_uuid && frame.step_id == first.step_id &&
                    frame.schema_digest == first.schema_digest && frame.field_id == first.field_id &&
                    frame.sequence == first.sequence && frame.dimension_count == first.dimension_count &&
                    frame.offsets == first.offsets && frame.extents == first.extents &&
                    frame.contribution_bytes == first.contribution_bytes && frame.fragment_count == fragment_count &&
                    frame.fragment_index == expected_index;
        }
        valid = valid && frame.fragment_offset == next_offset &&
                payload.size() <= std::numeric_limits<std::uint64_t>::max() - next_offset;
        if (valid && sink_status == AMIO_OK) {
            try {
                sink_status = sink(frame, payload);
            } catch (...) {
                sink_status = AMIO_ERR_BACKEND_FAILURE;
            }
        } else if (!valid && sink_status == AMIO_OK) {
            sink_status = AMIO_ERR_INVALID_POLICY;
        }
        next_offset += static_cast<std::uint64_t>(payload.size());
        ++expected_index;
        const bool final_fragment = expected_index >= fragment_count;
        const auto release_status = transport_->release_receiver_credit(credit, payload);
        if (release_status != AMIO_OK) return release_status;
        if (final_fragment) break;
    }
    if (next_offset != first.contribution_bytes && sink_status == AMIO_OK) sink_status = AMIO_ERR_INVALID_POLICY;
    return sink_status;
}

amio_status_t AggregatedExecution::receive_round(const TransportValidationContext &validation,
                                                  const ChunkSink &sink) const {
    if (!sink || context_->role_map().role(context_->rank()) != RankRole::Writer ||
        validation.producer_ranks.size() != context_->role_map().producers().size())
        return AMIO_ERR_INVALID_POLICY;

    for (std::size_t i = 0; i < validation.producer_ranks.size(); ++i) {
        if (validation.producer_ranks[i] != static_cast<std::uint32_t>(context_->role_map().producers()[i]))
            return AMIO_ERR_INVALID_POLICY;
    }

    amio_status_t round_status = AMIO_OK;
    for (const auto producer : validation.producer_ranks) {
        if (context_->role_map().writer_for(producer) != context_->rank()) continue;
        const auto status = receive(producer, validation, sink);
        if (status != AMIO_OK && round_status == AMIO_OK) round_status = status;
    }
    return round_status;
}

amio_status_t AggregatedExecution::receive_regions(std::uint32_t producer_rank,
                                                    const TransportValidationContext &validation,
                                                    std::size_t element_bytes, const RegionSink &sink,
                                                    ProtocolError *protocol_error) const {
    if (!sink || element_bytes == 0) return AMIO_ERR_INVALID_INPUT;
    return receive(producer_rank, validation,
                   [&](const TransportEnvelope &frame, std::span<const std::byte> bytes) {
                       return visit_destination_regions(frame, element_bytes, bytes, sink);
                   }, protocol_error);
}

amio_status_t AggregatedExecution::receive_round_regions(const TransportValidationContext &validation,
                                                          std::size_t element_bytes, const RegionSink &sink) const {
    if (!sink || element_bytes == 0) return AMIO_ERR_INVALID_INPUT;
    return receive_round(validation, [&](const TransportEnvelope &frame, std::span<const std::byte> bytes) {
        return visit_destination_regions(frame, element_bytes, bytes, sink);
    });
}

amio_status_t AggregatedExecution::serve_control(std::uint32_t producer_rank,
                                                  const TransportValidationContext &validation,
                                                  const ServiceHandler &handler, const ServiceHandler &after_ack) const {
    if (!handler || context_->role_map().role(context_->rank()) != RankRole::Writer) return AMIO_ERR_INVALID_POLICY;
    const auto producer = std::find(validation.producer_ranks.begin(), validation.producer_ranks.end(), producer_rank);
    if (producer == validation.producer_ranks.end()) return AMIO_ERR_INVALID_POLICY;
    const auto producer_slot = static_cast<std::size_t>(producer - validation.producer_ranks.begin());
    SequenceLedger control_sequences(validation.producer_ranks.size());
    amio_status_t first_failure = AMIO_OK;
    for (;;) {
        bool finished = false;
        bool invoked = false;
        TransportEnvelope accepted;
        ProtocolError protocol_error = ProtocolError::None;
        const auto status = transport_->receive_control(producer_rank, validation, [&](const TransportEnvelope &frame) -> amio_status_t {
            invoked = true;
            if (control_sequences.accept(producer_slot, frame.sequence) != SequenceResult::Accepted) {
                if (first_failure == AMIO_OK) first_failure = AMIO_ERR_INVALID_POLICY;
                return first_failure;
            }
            finished = frame.kind == MessageKind::Close || frame.kind == MessageKind::EndOfStream;
            if (first_failure != AMIO_OK && !finished) return first_failure;
            amio_status_t handled = AMIO_OK;
            try { handled = handler(frame); }
            catch (...) { handled = AMIO_ERR_BACKEND_FAILURE; }
            if (handled != AMIO_OK && first_failure == AMIO_OK) first_failure = handled;
            if (first_failure != AMIO_OK) return first_failure;
            accepted = frame;
            return AMIO_OK;
        }, &protocol_error);
        if (protocol_error != ProtocolError::None && first_failure == AMIO_OK) first_failure = AMIO_ERR_INVALID_POLICY;
        if (status != AMIO_OK) {
            if (status == AMIO_ERR_COMM_SPLIT_FAILED) return status;
            if (!invoked && protocol_error == ProtocolError::None) return status;
            if (finished) return first_failure != AMIO_OK ? first_failure : status;
            if (protocol_error != ProtocolError::None || first_failure != AMIO_OK) continue;
            return status;
        }
        if (after_ack && !finished) {
            amio_status_t progress_status = AMIO_OK;
            try { progress_status = after_ack(accepted); }
            catch (...) { progress_status = AMIO_ERR_BACKEND_FAILURE; }
            if (progress_status != AMIO_OK && first_failure == AMIO_OK) first_failure = progress_status;
        }
        if (finished) return first_failure;
    }
}

} // namespace amio::detail
