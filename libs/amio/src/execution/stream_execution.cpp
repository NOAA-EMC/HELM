#include "execution/stream_execution.hpp"

#include <algorithm>
#include <limits>

#include "execution/role_map.hpp"

namespace amio::detail {
namespace {
std::string digest_name(const std::array<std::uint8_t, kSchemaDigestBytes> &digest) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result;
    result.resize(digest.size() * 2);
    for (std::size_t i = 0; i < digest.size(); ++i) {
        result[2 * i] = hex[digest[i] >> 4];
        result[2 * i + 1] = hex[digest[i] & 0x0f];
    }
    return result;
}
}  // namespace

StreamExecution::StreamExecution(const ExecutionContext &context, MpiTransport &transport, StreamRetention &retention, std::uint64_t epoch,
                                 std::array<std::uint8_t, kDatasetUuidBytes> dataset_uuid, std::size_t max_step_bytes)
    : context_(&context),
      transport_(&transport),
      retention_(&retention),
      epoch_(epoch),
      dataset_uuid_(dataset_uuid),
      max_step_bytes_(max_step_bytes),
      sent_sequences_(context.size(), 0),
      received_sequences_(context.size()),
      control_sequences_(context.size()),
      release_sequences_(context.size(), 1) {}

amio_status_t StreamExecution::submit(TransportEnvelope frame, std::span<const std::byte> bytes, std::size_t element_bytes) {
    const auto rank = context_->rank();
    const auto destination = context_->role_map().writer_for(rank);
    if (context_->role_map().role(rank) != RankRole::Producer || !destination || frame.producer_rank != rank || frame.context_epoch != epoch_ ||
        frame.dataset_uuid != dataset_uuid_ || frame.kind != MessageKind::Chunk || frame.sequence == 0 || bytes.empty() ||
        bytes.size() > max_step_bytes_ || element_bytes == 0 || frame.dimension_count > AMIO_MAX_RANK)
        return AMIO_ERR_INVALID_POLICY;
    std::size_t elements = 1;
    for (std::size_t i = 0; i < frame.dimension_count; ++i) {
        if (frame.extents[i] == 0 || frame.extents[i] > std::numeric_limits<std::size_t>::max() ||
            elements > std::numeric_limits<std::size_t>::max() / frame.extents[i])
            return AMIO_ERR_PARTITION_INVALID;
        elements *= static_cast<std::size_t>(frame.extents[i]);
    }
    if (elements > std::numeric_limits<std::size_t>::max() / element_bytes || elements * element_bytes != bytes.size())
        return AMIO_ERR_SCHEMA_MISMATCH;
    frame.writer_rank = *destination;
    if (frame.sequence <= sent_sequences_[rank]) return AMIO_ERR_INVALID_POLICY;
    const auto field = "field_" + std::to_string(frame.field_id) + "_producer_" + std::to_string(rank);
    const auto digest = digest_name(frame.schema_digest);
    std::vector<std::uint64_t> shape(frame.extents.begin(), frame.extents.begin() + frame.dimension_count);
    std::vector<std::uint64_t> origin(frame.offsets.begin(), frame.offsets.begin() + frame.dimension_count);
    std::vector<std::byte> copy(bytes.begin(), bytes.end());
    auto retain_status = retention_->append_field(frame.step_id, field, digest, copy, shape, origin);
    if (retain_status == AMIO_ERR_INVALID_HANDLE) {
        RetainedStreamStep retained;
        retained.step_id = frame.step_id;
        retained.schema_digest = digest;
        retained.fields.emplace(field, std::move(copy));
        retained.field_shapes[field] = std::move(shape);
        retained.field_offsets[field] = std::move(origin);
        retained.payload_bytes = bytes.size();
        retain_status = retention_->publish(std::move(retained));
    }
    if (retain_status != AMIO_OK) return retain_status;
    sent_sequences_[rank] = frame.sequence;
    return transport_->send_payload(*destination, frame, bytes, element_bytes);
}

amio_status_t StreamExecution::receive_step(std::uint64_t step_id, const std::array<std::uint8_t, kSchemaDigestBytes> &schema_digest,
                                            const std::vector<std::string> &absent_optional_fields, std::uint32_t fields_per_producer,
                                            std::uint64_t *received_bytes) {
    if (received_bytes) *received_bytes = 0;
    const auto consumer_rank = context_->rank();
    if (context_->role_map().role(consumer_rank) != RankRole::Consumer || fields_per_producer == 0 || fields_per_producer > 65536)
        return AMIO_ERR_INVALID_POLICY;
    const auto fail = [this](amio_status_t status) {
        retention_->finish(status);
        return status;
    };
    TransportValidationContext validation;
    validation.expected_epoch = epoch_;
    validation.expected_dataset_uuid = dataset_uuid_;
    validation.expected_schema_digest = schema_digest;
    validation.expected_step_id = step_id;
    validation.producer_ranks.reserve(context_->roles().producers.size());
    validation.writer_ranks.reserve(context_->roles().consumers.size());
    for (int rank : context_->roles().producers) validation.producer_ranks.push_back(static_cast<std::uint32_t>(rank));
    for (int rank : context_->roles().consumers) validation.writer_ranks.push_back(static_cast<std::uint32_t>(rank));
    validation.max_payload_bytes = max_step_bytes_;
    validation.max_fragment_bytes = transport_->max_fragment_bytes();
    RetainedStreamStep step;
    step.step_id = step_id;
    step.schema_digest = digest_name(schema_digest);
    step.absent_optional_fields.insert(absent_optional_fields.begin(), absent_optional_fields.end());
    for (const auto producer : validation.producer_ranks) {
        const auto destination = context_->role_map().writer_for(producer);
        if (!destination || *destination != consumer_rank) continue;
        for (std::uint32_t field_index = 0; field_index < fields_per_producer; ++field_index) {
            std::vector<std::byte> contribution;
            std::optional<TransportEnvelope> identity;
            for (;;) {
                TransportEnvelope frame;
                CreditWindow::Ticket credit{};
                std::vector<std::byte> fragment;
                const auto status = transport_->receive_fragment(producer, validation, frame, fragment, &credit);
                if (status != AMIO_OK) return fail(status);
                if (!identity) {
                    identity = frame;
                    if (identity->contribution_bytes > max_step_bytes_) {
                        (void)transport_->release_receiver_credit(credit, fragment);
                        return fail(AMIO_ERR_BUDGET_EXCEEDED);
                    }
                    contribution.reserve(static_cast<std::size_t>(identity->contribution_bytes));
                } else if (frame.sequence != identity->sequence || frame.field_id != identity->field_id ||
                           frame.fragment_count != identity->fragment_count || frame.fragment_index != identity->fragment_index + 1 ||
                           frame.fragment_offset != contribution.size()) {
                    (void)transport_->release_receiver_credit(credit, fragment);
                    return fail(AMIO_ERR_INVALID_POLICY);
                }
                if (fragment.size() > max_step_bytes_ - step.payload_bytes - contribution.size()) {
                    (void)transport_->release_receiver_credit(credit, fragment);
                    return fail(AMIO_ERR_BUDGET_EXCEEDED);
                }
                contribution.insert(contribution.end(), fragment.begin(), fragment.end());
                const bool final_fragment = frame.fragment_index + 1 == frame.fragment_count;
                const auto release_status = transport_->release_receiver_credit(credit, fragment);
                if (release_status != AMIO_OK) return fail(release_status);
                if (final_fragment) break;
            }
            if (!identity || contribution.size() != identity->contribution_bytes ||
                received_sequences_.accept(producer, identity->sequence) != SequenceResult::Accepted)
                return fail(AMIO_ERR_INVALID_POLICY);
            const auto key = "field_" + std::to_string(identity->field_id) + "_producer_" + std::to_string(producer);
            step.payload_bytes += contribution.size();
            if (received_bytes) *received_bytes += contribution.size();
            step.fields.emplace(key, std::move(contribution));
            step.field_shapes[key].assign(identity->extents.begin(), identity->extents.begin() + identity->dimension_count);
            step.field_offsets[key].assign(identity->offsets.begin(), identity->offsets.begin() + identity->dimension_count);
        }
    }
    const auto status = retention_->publish(std::move(step));
    return status == AMIO_OK ? status : fail(status);
}

amio_status_t StreamExecution::finish_producer(bool graceful) {
    const auto rank = context_->rank();
    const auto destination = context_->role_map().writer_for(rank);
    if (context_->role_map().role(rank) != RankRole::Producer || !destination || next_control_sequence_ == 0) return AMIO_ERR_INVALID_POLICY;
    // A producer's retained snapshots are the ownership record for accepted
    // stream steps. Do not advertise EOS while any consumer release ACK is
    // still outstanding; callers may retry after the consumer drains them.
    if (retention_->policy() == StreamRetentionPolicy::Reliable && retention_->snapshot().steps != 0) return AMIO_ERR_OPERATION_PENDING;
    TransportEnvelope frame;
    frame.context_epoch = epoch_;
    frame.dataset_uuid = dataset_uuid_;
    frame.kind = graceful ? MessageKind::EndOfStream : MessageKind::Close;
    frame.producer_rank = rank;
    frame.writer_rank = *destination;
    frame.sequence = next_control_sequence_++;
    return transport_->send_control(*destination, frame);
}

amio_status_t StreamExecution::cancel_producer() {
    const auto rank = context_->rank();
    const auto destination = context_->role_map().writer_for(rank);
    if (context_->role_map().role(rank) != RankRole::Producer || !destination || next_control_sequence_ == 0) return AMIO_ERR_INVALID_POLICY;
    if (retention_->policy() == StreamRetentionPolicy::Reliable && retention_->snapshot().steps != 0) return AMIO_ERR_OPERATION_PENDING;
    TransportEnvelope frame;
    frame.context_epoch = epoch_;
    frame.dataset_uuid = dataset_uuid_;
    frame.kind = MessageKind::Cancel;
    frame.producer_rank = rank;
    frame.writer_rank = *destination;
    frame.sequence = next_control_sequence_++;
    return transport_->send_control(*destination, frame);
}

amio_status_t StreamExecution::release_lease(std::uint64_t lease_id) {
    const auto consumer = context_->rank();
    if (context_->role_map().role(consumer) != RankRole::Consumer) return AMIO_ERR_INVALID_POLICY;
    const auto step_id = retention_->lease_step_id(lease_id, 0);
    if (!step_id) return AMIO_ERR_INVALID_HANDLE;
    bool sent_any = false;
    for (int peer : context_->roles().producers) {
        const auto producer = static_cast<std::uint32_t>(peer);
        if (context_->role_map().writer_for(producer) != consumer) continue;
        TransportEnvelope frame;
        frame.context_epoch = epoch_;
        frame.dataset_uuid = dataset_uuid_;
        frame.step_id = *step_id;
        frame.kind = MessageKind::Release;
        frame.producer_rank = producer;
        frame.writer_rank = consumer;
        frame.sequence = release_sequences_[producer];
        if (retention_->policy() == StreamRetentionPolicy::Reliable) {
            const auto status = transport_->send_release(producer, frame);
            if (status != AMIO_OK) return status;
            ++release_sequences_[producer];
        }
        sent_any = true;
    }
    if (!sent_any) return AMIO_ERR_INVALID_POLICY;
    return retention_->release(lease_id);
}

amio_status_t StreamExecution::wait_for_release(std::uint64_t step_id, std::chrono::milliseconds timeout) {
    const auto producer = context_->rank();
    const auto consumer = context_->role_map().writer_for(producer);
    if (context_->role_map().role(producer) != RankRole::Producer || !consumer) return AMIO_ERR_INVALID_POLICY;
    TransportEnvelope frame;
    const auto status = transport_->receive_release(*consumer, epoch_, dataset_uuid_, step_id, frame, timeout);
    if (status != AMIO_OK) return status;
    if (control_sequences_.accept(*consumer, frame.sequence) != SequenceResult::Accepted) return AMIO_ERR_INVALID_POLICY;
    return retention_->acknowledge(step_id, 0);
}

amio_status_t StreamExecution::finish_consumer() {
    const auto consumer_rank = context_->rank();
    if (context_->role_map().role(consumer_rank) != RankRole::Consumer) return AMIO_ERR_INVALID_POLICY;
    TransportValidationContext validation;
    validation.expected_epoch = epoch_;
    validation.expected_dataset_uuid = dataset_uuid_;
    for (int rank : context_->roles().producers) validation.producer_ranks.push_back(static_cast<std::uint32_t>(rank));
    for (int rank : context_->roles().consumers) validation.writer_ranks.push_back(static_cast<std::uint32_t>(rank));
    bool disconnected = false;
    bool cancelled = false;
    for (const auto producer : validation.producer_ranks) {
        const auto destination = context_->role_map().writer_for(producer);
        if (!destination || *destination != consumer_rank) continue;
        const auto status = transport_->receive_control(producer, validation, [&](const TransportEnvelope &frame) {
            if (control_sequences_.accept(producer, frame.sequence) != SequenceResult::Accepted) return AMIO_ERR_INVALID_POLICY;
            if (frame.kind == MessageKind::Close) {
                disconnected = true;
                return AMIO_OK;
            }
            if (frame.kind == MessageKind::Cancel) {
                cancelled = true;
                return AMIO_OK;
            }
            return frame.kind == MessageKind::EndOfStream ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
        });
        if (status != AMIO_OK) {
            retention_->finish(status);
            return status;
        }
    }
    const auto terminal = disconnected ? AMIO_ERR_STREAM_DISCONNECTED : (cancelled ? AMIO_ERR_STEP_FAILED : AMIO_OK);
    retention_->finish(terminal);
    return terminal;
}

}  // namespace amio::detail
