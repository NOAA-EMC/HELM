#include "execution/mpi_transport.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <limits>
#include <stdexcept>
#include <thread>

#include "execution/role_map.hpp"

#ifdef AMIO_HAS_MPI
#include <mpi.h>
#endif

namespace amio::detail {
namespace {
constexpr int kEnvelopeTag = 201;
constexpr int kPayloadTag = 202;
constexpr int kAcknowledgmentTag = 203;
constexpr int kServiceControlTag = 204;
constexpr int kServiceAcknowledgmentTag = 205;

bool is_service_control(MessageKind kind) noexcept {
    return kind == MessageKind::Config || kind == MessageKind::Begin || kind == MessageKind::Contribute || kind == MessageKind::Seal ||
           kind == MessageKind::Cancel || kind == MessageKind::Close || kind == MessageKind::EndOfStream || kind == MessageKind::Release;
}

#ifdef AMIO_HAS_MPI
void acknowledge_error(MPI_Comm communicator, int destination, const TransportEnvelope *received = nullptr) noexcept {
    TransportEnvelope acknowledgment = received == nullptr ? TransportEnvelope{} : *received;
    acknowledgment.kind = MessageKind::Error;
    const auto wire = encode_envelope(acknowledgment);
    (void)MPI_Send(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, destination, kAcknowledgmentTag, communicator);
}
#endif
}  // namespace

MpiTransport::MpiTransport(const ExecutionContext &context, CreditWindow &credits, std::uint64_t max_fragment_bytes)
    : context_(&context), credits_(&credits), max_fragment_bytes_(max_fragment_bytes) {
    if (max_fragment_bytes_ == 0 || max_fragment_bytes_ > static_cast<std::uint64_t>(INT_MAX))
        throw std::invalid_argument("MPI transport fragment limit must be within the positive MPI count range");
#ifndef AMIO_HAS_MPI
    (void)context;
    throw std::invalid_argument("MPI transport is unavailable in a non-MPI build");
#endif
}

amio_status_t MpiTransport::send_fragment(std::uint32_t destination_rank, const TransportEnvelope &envelope,
                                          std::span<const std::byte> payload) const noexcept {
#ifdef AMIO_HAS_MPI
    if (destination_rank >= context_->size() || payload.size() > max_fragment_bytes_ || payload.size() > static_cast<std::size_t>(INT_MAX) ||
        envelope.payload_bytes != payload.size())
        return AMIO_ERR_INVALID_INPUT;
    if (envelope.dimension_count > AMIO_MAX_RANK || envelope.fragment_count == 0 || envelope.fragment_index >= envelope.fragment_count)
        return AMIO_ERR_INVALID_INPUT;
    auto sender_credit = credits_->acquire(destination_rank, CreditOwner::Sender, payload.size(), 1);
    if (!sender_credit) return AMIO_ERR_QUEUE_FULL;
    auto control_credit = credits_->acquire_control(2 * kTransportEnvelopeWireBytes, 2);
    if (!control_credit) {
        (void)credits_->release(*sender_credit);
        return AMIO_ERR_QUEUE_FULL;
    }
    const auto wire = encode_envelope(envelope);
    if (MPI_Send(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, static_cast<int>(destination_rank), kEnvelopeTag,
                 context_->control_communicator()) != MPI_SUCCESS) {
        (void)credits_->release(*sender_credit);
        (void)credits_->release(*control_credit);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    const auto receive_ack = [&](MessageKind expected) noexcept -> amio_status_t {
        std::array<std::uint8_t, kTransportEnvelopeWireBytes> ack_wire{};
        MPI_Status ack_status{};
        int ack_count = 0;
        if (MPI_Recv(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(destination_rank), kAcknowledgmentTag,
                     context_->control_communicator(), &ack_status) != MPI_SUCCESS ||
            MPI_Get_count(&ack_status, MPI_BYTE, &ack_count) != MPI_SUCCESS || ack_count != static_cast<int>(ack_wire.size()))
            return AMIO_ERR_COMM_SPLIT_FAILED;
        TransportEnvelope ack;
        const auto decoded = decode_envelope(ack_wire, ack);
        const bool matches = decoded == ProtocolError::None && acknowledgment_matches(envelope, ack);
        if (!matches) return AMIO_ERR_INVALID_POLICY;
        if (ack.kind == MessageKind::Error) return AMIO_ERR_QUEUE_FULL;
        return ack.kind == expected ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
    };
    // The receiver validates and reserves its buffer before granting permission
    // to send. A rejected envelope therefore cannot leave a stray data message
    // or block a large rendezvous send on a receiver that will never post it.
    auto result = receive_ack(MessageKind::Credit);
    if (result == AMIO_OK) {
        const void *data = payload.empty() ? nullptr : payload.data();
        if (MPI_Send(data, static_cast<int>(payload.size()), MPI_BYTE, static_cast<int>(destination_rank), kPayloadTag,
                     context_->data_communicator()) != MPI_SUCCESS)
            result = AMIO_ERR_COMM_SPLIT_FAILED;
        else
            result = receive_ack(MessageKind::Received);
    }
    (void)credits_->release(*sender_credit);
    (void)credits_->release(*control_credit);
    return result;
#else
    (void)destination_rank;
    (void)envelope;
    (void)payload;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::send_payload(std::uint32_t destination_rank, TransportEnvelope envelope, std::span<const std::byte> payload,
                                         std::size_t element_bytes) const {
    if (element_bytes == 0 || element_bytes > max_fragment_bytes_ || payload.size() % element_bytes != 0) return AMIO_ERR_INVALID_INPUT;
    const auto fragment_limit = max_fragment_bytes_ - max_fragment_bytes_ % element_bytes;
    const auto total_bytes = static_cast<std::uint64_t>(payload.size());
    const auto fragment_count = total_bytes == 0 ? std::uint64_t{1} : total_bytes / fragment_limit + (total_bytes % fragment_limit == 0 ? 0 : 1);
    if (fragment_count > std::numeric_limits<std::uint32_t>::max()) return AMIO_ERR_INVALID_INPUT;
    envelope.fragment_count = static_cast<std::uint32_t>(fragment_count);
    envelope.contribution_bytes = total_bytes;
    std::uint64_t offset = 0;
    for (std::uint64_t index = 0; index < fragment_count; ++index) {
        const auto length = total_bytes == 0 ? std::uint64_t{0} : std::min(fragment_limit, total_bytes - offset);
        auto frame = envelope;
        frame.fragment_index = static_cast<std::uint32_t>(index);
        frame.fragment_offset = offset;
        frame.payload_bytes = length;
        const auto bytes = payload.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(length));
        const auto status = send_fragment(destination_rank, frame, bytes);
        if (status != AMIO_OK) return status;
        offset += length;
    }
    return AMIO_OK;
}

amio_status_t MpiTransport::send_control(std::uint32_t writer_rank, const TransportEnvelope &envelope) const noexcept {
#ifdef AMIO_HAS_MPI
    const auto &roles = context_->roles();
    const auto &destinations = roles.mode == ExecutionMode::Stream ? roles.consumers : roles.writers;
    if (!is_service_control(envelope.kind) || envelope.producer_rank != context_->rank() || envelope.writer_rank != writer_rank ||
        writer_rank >= context_->size() ||
        std::find(roles.producers.begin(), roles.producers.end(), static_cast<int>(context_->rank())) == roles.producers.end() ||
        std::find(destinations.begin(), destinations.end(), static_cast<int>(writer_rank)) == destinations.end() || envelope.dimension_count != 0 ||
        envelope.fragment_count != 1 || envelope.fragment_index != 0 || envelope.fragment_offset != 0 || envelope.contribution_bytes != 0 ||
        envelope.payload_bytes != 0)
        return AMIO_ERR_INVALID_INPUT;
    auto credit = credits_->acquire_control(2 * kTransportEnvelopeWireBytes, 2);
    if (!credit) return AMIO_ERR_QUEUE_FULL;
    const auto wire = encode_envelope(envelope);
    amio_status_t result = AMIO_ERR_COMM_SPLIT_FAILED;
    if (MPI_Send(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, static_cast<int>(writer_rank), kServiceControlTag,
                 context_->control_communicator()) == MPI_SUCCESS) {
        std::array<std::uint8_t, kTransportEnvelopeWireBytes> ack_wire{};
        MPI_Status mpi_status{};
        int count = 0;
        if (MPI_Recv(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(writer_rank), kServiceAcknowledgmentTag,
                     context_->control_communicator(), &mpi_status) == MPI_SUCCESS &&
            MPI_Get_count(&mpi_status, MPI_BYTE, &count) == MPI_SUCCESS && count == static_cast<int>(ack_wire.size())) {
            TransportEnvelope ack;
            const bool matches = decode_envelope(ack_wire, ack) == ProtocolError::None && acknowledgment_matches(envelope, ack);
            if (!matches)
                result = AMIO_ERR_INVALID_POLICY;
            else if (ack.kind == MessageKind::Received)
                result = AMIO_OK;
            else if (ack.kind == MessageKind::Error)
                result = AMIO_ERR_STEP_FAILED;
            else
                result = AMIO_ERR_INVALID_POLICY;
        }
    }
    (void)credits_->release(*credit);
    return result;
#else
    (void)writer_rank;
    (void)envelope;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::send_release(std::uint32_t producer_rank, const TransportEnvelope &envelope) const noexcept {
#ifdef AMIO_HAS_MPI
    const auto &roles = context_->roles();
    const auto consumer = static_cast<int>(context_->rank());
    if (envelope.kind != MessageKind::Release || envelope.producer_rank != producer_rank || envelope.writer_rank != context_->rank() ||
        envelope.sequence == 0 || envelope.dimension_count != 0 || envelope.fragment_count != 1 || envelope.fragment_index != 0 ||
        envelope.fragment_offset != 0 || envelope.contribution_bytes != 0 || envelope.payload_bytes != 0 || producer_rank >= context_->size() ||
        std::find(roles.consumers.begin(), roles.consumers.end(), consumer) == roles.consumers.end() ||
        std::find(roles.producers.begin(), roles.producers.end(), static_cast<int>(producer_rank)) == roles.producers.end() ||
        context_->role_map().writer_for(producer_rank) != context_->rank())
        return AMIO_ERR_INVALID_INPUT;
    auto credit = credits_->acquire_control(2 * kTransportEnvelopeWireBytes, 2);
    if (!credit) return AMIO_ERR_QUEUE_FULL;
    const auto wire = encode_envelope(envelope);
    amio_status_t result = AMIO_ERR_COMM_SPLIT_FAILED;
    if (MPI_Send(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, static_cast<int>(producer_rank), kServiceControlTag,
                 context_->control_communicator()) == MPI_SUCCESS) {
        std::array<std::uint8_t, kTransportEnvelopeWireBytes> ack_wire{};
        MPI_Status status{};
        int count = 0;
        if (MPI_Recv(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(producer_rank), kServiceAcknowledgmentTag,
                     context_->control_communicator(), &status) == MPI_SUCCESS &&
            MPI_Get_count(&status, MPI_BYTE, &count) == MPI_SUCCESS && count == static_cast<int>(ack_wire.size())) {
            TransportEnvelope ack;
            if (decode_envelope(ack_wire, ack) == ProtocolError::None && acknowledgment_matches(envelope, ack))
                result = ack.kind == MessageKind::Received ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
            else
                result = AMIO_ERR_INVALID_POLICY;
        }
    }
    (void)credits_->release(*credit);
    return result;
#else
    (void)producer_rank;
    (void)envelope;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::receive_release(std::uint32_t consumer_rank, std::uint64_t epoch,
                                            const std::array<std::uint8_t, kDatasetUuidBytes> &dataset_uuid, std::uint64_t step_id,
                                            TransportEnvelope &received, std::chrono::milliseconds timeout) const noexcept {
#ifdef AMIO_HAS_MPI
    const auto producer = static_cast<std::uint32_t>(context_->rank());
    const auto &roles = context_->roles();
    if (consumer_rank >= context_->size() || context_->role_map().role(producer) != RankRole::Producer ||
        context_->role_map().writer_for(producer) != consumer_rank)
        return AMIO_ERR_INVALID_POLICY;
    auto credit = credits_->acquire_control(2 * kTransportEnvelopeWireBytes, 2);
    if (!credit) return AMIO_ERR_QUEUE_FULL;
    std::array<std::uint8_t, kTransportEnvelopeWireBytes> wire{};
    const bool infinite = timeout == std::chrono::milliseconds::max();
    const auto deadline = infinite ? std::chrono::steady_clock::time_point::max()
                                   : std::chrono::steady_clock::now() + std::max(timeout, std::chrono::milliseconds::zero());
    int available = 0;
    while (!available) {
        if (MPI_Iprobe(static_cast<int>(consumer_rank), kServiceControlTag, context_->control_communicator(), &available, MPI_STATUS_IGNORE) !=
            MPI_SUCCESS) {
            (void)credits_->release(*credit);
            return AMIO_ERR_COMM_SPLIT_FAILED;
        }
        if (available) break;
        if (!infinite && std::chrono::steady_clock::now() >= deadline) {
            (void)credits_->release(*credit);
            return AMIO_ERR_OPERATION_PENDING;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    MPI_Status status{};
    int count = 0;
    if (MPI_Recv(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, static_cast<int>(consumer_rank), kServiceControlTag,
                 context_->control_communicator(), &status) != MPI_SUCCESS ||
        MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS || count != static_cast<int>(wire.size())) {
        (void)credits_->release(*credit);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    TransportEnvelope frame;
    auto result = decode_envelope(wire, frame) == ProtocolError::None ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
    if (result == AMIO_OK && (frame.kind != MessageKind::Release || frame.context_epoch != epoch || frame.dataset_uuid != dataset_uuid ||
                              frame.step_id != step_id || frame.producer_rank != producer || frame.writer_rank != consumer_rank ||
                              frame.sequence == 0 || frame.dimension_count != 0 || frame.fragment_count != 1 || frame.fragment_index != 0 ||
                              frame.fragment_offset != 0 || frame.contribution_bytes != 0 || frame.payload_bytes != 0 ||
                              std::find(roles.consumers.begin(), roles.consumers.end(), static_cast<int>(consumer_rank)) == roles.consumers.end()))
        result = AMIO_ERR_INVALID_POLICY;
    auto ack = frame;
    ack.kind = result == AMIO_OK ? MessageKind::Received : MessageKind::Error;
    const auto ack_wire = encode_envelope(ack);
    if (MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(consumer_rank), kServiceAcknowledgmentTag,
                 context_->control_communicator()) != MPI_SUCCESS)
        result = AMIO_ERR_COMM_SPLIT_FAILED;
    else if (result == AMIO_OK)
        received = frame;
    (void)credits_->release(*credit);
    return result;
#else
    (void)consumer_rank;
    (void)epoch;
    (void)dataset_uuid;
    (void)step_id;
    (void)received;
    (void)timeout;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::receive_control(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                            const std::function<amio_status_t(const TransportEnvelope &)> &handler,
                                            ProtocolError *protocol_error) const noexcept {
    if (protocol_error) *protocol_error = ProtocolError::None;
    if (!handler) return AMIO_ERR_INVALID_INPUT;
#ifdef AMIO_HAS_MPI
    if (producer_rank >= context_->size()) return AMIO_ERR_INVALID_INPUT;
    auto credit = credits_->acquire_control(2 * kTransportEnvelopeWireBytes, 2);
    if (!credit) return AMIO_ERR_QUEUE_FULL;
    std::array<std::uint8_t, kTransportEnvelopeWireBytes> wire{};
    MPI_Status mpi_status{};
    int count = 0;
    if (MPI_Recv(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, static_cast<int>(producer_rank), kServiceControlTag,
                 context_->control_communicator(), &mpi_status) != MPI_SUCCESS ||
        MPI_Get_count(&mpi_status, MPI_BYTE, &count) != MPI_SUCCESS) {
        (void)credits_->release(*credit);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    TransportEnvelope frame;
    auto error = count == static_cast<int>(wire.size()) ? decode_envelope(wire, frame) : ProtocolError::InvalidWireFrame;
    if (error == ProtocolError::None &&
        (!is_service_control(frame.kind) || frame.dimension_count != 0 || frame.fragment_count != 1 || frame.fragment_index != 0 ||
         frame.fragment_offset != 0 || frame.contribution_bytes != 0 || frame.payload_bytes != 0))
        error = ProtocolError::InvalidKind;
    if (error == ProtocolError::None && (frame.producer_rank != producer_rank || frame.writer_rank != context_->rank()))
        error = ProtocolError::InvalidWriterMapping;
    if (error == ProtocolError::None) error = validate_envelope(frame, validation).error;
    amio_status_t result = error == ProtocolError::None ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
    if (protocol_error) *protocol_error = error;
    if (result == AMIO_OK) {
        try {
            result = handler(frame);
        } catch (...) {
            result = AMIO_ERR_BACKEND_FAILURE;
        }
    }
    auto ack = frame;
    ack.kind = result == AMIO_OK ? MessageKind::Received : MessageKind::Error;
    const auto ack_wire = encode_envelope(ack);
    if (MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(producer_rank), kServiceAcknowledgmentTag,
                 context_->control_communicator()) != MPI_SUCCESS)
        result = AMIO_ERR_COMM_SPLIT_FAILED;
    (void)credits_->release(*credit);
    return result;
#else
    (void)producer_rank;
    (void)validation;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::try_receive_control(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                                const std::function<amio_status_t(const TransportEnvelope &)> &handler,
                                                ProtocolError *protocol_error) const noexcept {
    if (protocol_error) *protocol_error = ProtocolError::None;
    if (!handler || producer_rank >= context_->size()) return AMIO_ERR_INVALID_INPUT;
#ifdef AMIO_HAS_MPI
    int available = 0;
    MPI_Status mpi_status{};
    if (MPI_Iprobe(static_cast<int>(producer_rank), kServiceControlTag, context_->control_communicator(), &available, &mpi_status) != MPI_SUCCESS)
        return AMIO_ERR_COMM_SPLIT_FAILED;
    if (!available) return AMIO_ERR_OPERATION_PENDING;
    return receive_control(producer_rank, validation, handler, protocol_error);
#else
    (void)validation;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::receive_fragment(std::uint32_t source_rank, const TransportValidationContext &validation, TransportEnvelope &envelope,
                                             std::vector<std::byte> &payload, CreditWindow::Ticket *receiver_credit,
                                             ProtocolError *protocol_error) const noexcept {
    if (protocol_error) *protocol_error = ProtocolError::None;
    if (receiver_credit == nullptr) return AMIO_ERR_INVALID_INPUT;
    *receiver_credit = {};
#ifdef AMIO_HAS_MPI
    // A previous receive must not leave uncharged vector capacity behind.
    std::vector<std::byte>().swap(payload);
    if (source_rank >= context_->size()) return AMIO_ERR_INVALID_INPUT;
    auto control_credit = credits_->acquire_control(kTransportEnvelopeWireBytes, 1);
    if (!control_credit) return AMIO_ERR_QUEUE_FULL;
    std::array<std::uint8_t, kTransportEnvelopeWireBytes> wire{};
    MPI_Status status{};
    if (MPI_Recv(wire.data(), static_cast<int>(wire.size()), MPI_BYTE, static_cast<int>(source_rank), kEnvelopeTag, context_->control_communicator(),
                 &status) != MPI_SUCCESS) {
        (void)credits_->release(*control_credit);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    int received = 0;
    if (MPI_Get_count(&status, MPI_BYTE, &received) != MPI_SUCCESS || received != static_cast<int>(wire.size())) {
        if (protocol_error) *protocol_error = ProtocolError::InvalidWireFrame;
        acknowledge_error(context_->control_communicator(), static_cast<int>(source_rank));
        (void)credits_->release(*control_credit);
        return AMIO_ERR_INVALID_POLICY;
    }
    TransportEnvelope decoded;
    auto error = decode_envelope(wire, decoded);
    if (error == ProtocolError::None && (decoded.producer_rank != source_rank || decoded.writer_rank != context_->rank()))
        error = ProtocolError::InvalidWriterMapping;
    auto checked = error == ProtocolError::None ? validate_envelope(decoded, validation) : ProtocolValidationResult{error, 0};
    if (!checked) {
        if (protocol_error) *protocol_error = checked.error;
        TransportEnvelope ack = decoded;
        ack.kind = MessageKind::Error;
        const auto ack_wire = encode_envelope(ack);
        (void)MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(source_rank), kAcknowledgmentTag,
                       context_->control_communicator());
        (void)credits_->release(*control_credit);
        return checked.error == ProtocolError::PayloadTooLarge || checked.error == ProtocolError::FragmentTooLarge ? AMIO_ERR_QUEUE_FULL
                                                                                                                   : AMIO_ERR_INVALID_POLICY;
    }
    if (decoded.payload_bytes > max_fragment_bytes_) {
        if (protocol_error) *protocol_error = ProtocolError::FragmentTooLarge;
        TransportEnvelope ack = decoded;
        ack.kind = MessageKind::Error;
        const auto ack_wire = encode_envelope(ack);
        (void)MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(source_rank), kAcknowledgmentTag,
                       context_->control_communicator());
        (void)credits_->release(*control_credit);
        return AMIO_ERR_QUEUE_FULL;
    }
    auto reserved = credits_->acquire(source_rank, CreditOwner::Receiver, checked.payload_allocation_bytes, 1);
    if (!reserved) {
        TransportEnvelope ack = decoded;
        ack.kind = MessageKind::Error;
        const auto ack_wire = encode_envelope(ack);
        (void)MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(source_rank), kAcknowledgmentTag,
                       context_->control_communicator());
        (void)credits_->release(*control_credit);
        return AMIO_ERR_QUEUE_FULL;
    }
    try {
        payload.resize(checked.payload_allocation_bytes);
    } catch (...) {
        std::vector<std::byte>().swap(payload);
        (void)credits_->release(*reserved);
        TransportEnvelope ack = decoded;
        ack.kind = MessageKind::Error;
        const auto ack_wire = encode_envelope(ack);
        (void)MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(source_rank), kAcknowledgmentTag,
                       context_->control_communicator());
        (void)credits_->release(*control_credit);
        return AMIO_ERR_BACKEND_FAILURE;
    }
    TransportEnvelope ready = decoded;
    ready.kind = MessageKind::Credit;
    const auto ready_wire = encode_envelope(ready);
    if (MPI_Send(ready_wire.data(), static_cast<int>(ready_wire.size()), MPI_BYTE, static_cast<int>(source_rank), kAcknowledgmentTag,
                 context_->control_communicator()) != MPI_SUCCESS) {
        std::vector<std::byte>().swap(payload);
        (void)credits_->release(*reserved);
        (void)credits_->release(*control_credit);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    if (MPI_Recv(payload.empty() ? nullptr : payload.data(), static_cast<int>(payload.size()), MPI_BYTE, static_cast<int>(source_rank), kPayloadTag,
                 context_->data_communicator(), &status) != MPI_SUCCESS) {
        std::vector<std::byte>().swap(payload);
        (void)credits_->release(*reserved);
        (void)credits_->release(*control_credit);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    if (MPI_Get_count(&status, MPI_BYTE, &received) != MPI_SUCCESS || received != static_cast<int>(payload.size())) {
        std::vector<std::byte>().swap(payload);
        acknowledge_error(context_->control_communicator(), static_cast<int>(source_rank), &decoded);
        (void)credits_->release(*reserved);
        (void)credits_->release(*control_credit);
        if (protocol_error) *protocol_error = ProtocolError::InvalidWireFrame;
        return AMIO_ERR_INVALID_POLICY;
    }
    TransportEnvelope ack = decoded;
    ack.kind = MessageKind::Received;
    const auto ack_wire = encode_envelope(ack);
    const auto ack_result = MPI_Send(ack_wire.data(), static_cast<int>(ack_wire.size()), MPI_BYTE, static_cast<int>(source_rank), kAcknowledgmentTag,
                                     context_->control_communicator());
    (void)credits_->release(*control_credit);
    if (ack_result != MPI_SUCCESS) {
        std::vector<std::byte>().swap(payload);
        (void)credits_->release(*reserved);
        return AMIO_ERR_COMM_SPLIT_FAILED;
    }
    envelope = decoded;
    *receiver_credit = *reserved;
    return AMIO_OK;
#else
    (void)source_rank;
    (void)validation;
    (void)envelope;
    (void)payload;
    return AMIO_ERR_UNSUPPORTED;
#endif
}

amio_status_t MpiTransport::release_receiver_credit(const CreditWindow::Ticket &ticket, std::vector<std::byte> &payload) const noexcept {
    if (ticket.owner != CreditOwner::Receiver) return AMIO_ERR_INVALID_POLICY;
    std::vector<std::byte>().swap(payload);
    if (!credits_->release(ticket)) return AMIO_ERR_INVALID_POLICY;
    return AMIO_OK;
}

}  // namespace amio::detail
