#ifndef AMIO_SRC_EXECUTION_AGGREGATED_EXECUTION_HPP
#define AMIO_SRC_EXECUTION_AGGREGATED_EXECUTION_HPP

#include "execution/context.hpp"
#include "execution/mpi_transport.hpp"
#include "execution/transport_protocol.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <mutex>
#include <span>
#include <vector>

namespace amio::detail {

/** One contiguous row of an array contribution in destination coordinates. */
struct DestinationRegionChunk {
    std::uint32_t dimension_count = 0;
    std::array<std::uint64_t, AMIO_MAX_RANK> offsets{};
    std::array<std::uint64_t, AMIO_MAX_RANK> extents{};
};

/** Visit element-aligned fragment bytes as contiguous C-order destination rows.
 *
 * The callback consumes each row before returning. No complete contribution or
 * unbounded list of subregions is allocated. Empty contributions produce one
 * empty region. Invalid sizes, arithmetic overflow, and partial elements are
 * rejected before any callback is invoked.
 */
[[nodiscard]] amio_status_t visit_destination_regions(
    const TransportEnvelope &frame, std::size_t element_bytes, std::span<const std::byte> payload,
    const std::function<amio_status_t(const DestinationRegionChunk &, std::span<const std::byte>)> &sink);

/** Bounded deterministic forwarding for one statically configured writer group. */
class AggregatedExecution {
public:
    /** Consume one validated chunk; the sink must finish using bytes before it returns. */
    using ChunkSink = std::function<amio_status_t(const TransportEnvelope &, std::span<const std::byte>)>;
    /** Consume one contiguous destination row while its receiver credit remains held. */
    using RegionSink = std::function<amio_status_t(const DestinationRegionChunk &, std::span<const std::byte>)>;
    /** Handle a validated service command before its acknowledgment is sent. */
    using ServiceHandler = std::function<amio_status_t(const TransportEnvelope &)>;

    /** Bind the aggregation service to a context and its credit-accounted transport. */
    AggregatedExecution(const ExecutionContext &context, MpiTransport &transport);

    /** Forward a contribution in bounded, element-aligned fragments; use one-byte elements for encoded messages. */
    [[nodiscard]] amio_status_t forward(const TransportEnvelope &contribution, std::span<const std::byte> bytes,
                                        std::size_t element_bytes = 1) const;

    /** Receive all fragments of one producer contribution and pass each bounded chunk to the writer sink.
     *
     * Fragments are validated for stable identity and contiguous byte offsets.
     * Receiver credits remain held through each sink call and are returned only
     * after the sink has consumed the chunk. The sink can write native backend
     * slabs without materializing a complete global field.
     */
    [[nodiscard]] amio_status_t receive(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                         const ChunkSink &sink, ProtocolError *protocol_error = nullptr) const;

    /** Receive one producer's typed contribution as bounded native destination regions. */
    [[nodiscard]] amio_status_t receive_regions(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                                 std::size_t element_bytes, const RegionSink &sink,
                                                 ProtocolError *protocol_error = nullptr) const;

    /** Receive one canonical producer round for this writer.
     *
     * Assigned producers are visited in the configured producer order. Every
     * producer must send exactly one contribution, including an explicit
     * zero-byte contribution when it owns no region. Errors are remembered
     * while the remaining assigned peers are drained, so a bad contribution
     * cannot strand later senders waiting for acknowledgments.
     */
    [[nodiscard]] amio_status_t receive_round(const TransportValidationContext &validation,
                                               const ChunkSink &sink) const;

    /** Receive one canonical writer round as bounded typed destination regions. */
    [[nodiscard]] amio_status_t receive_round_regions(const TransportValidationContext &validation,
                                                       std::size_t element_bytes, const RegionSink &sink) const;

    /** Run a dedicated writer's ordered control loop until close or end-of-stream.
     *
     * The handler completes before the sender receives its acknowledgment.
     * The optional after_ack callback can progress work initiated by the sender
     * after a nonterminal acknowledgment. Close/end cleanup belongs in handler,
     * before its acknowledgment. On failure the loop retains the first cause,
     * rejects subsequent work, and still handles close/end-of-stream for cleanup.
     * Duplicate and stale sequence numbers are rejected before the handler.
     * Blocking MPI calls cannot be forcibly interrupted.
     */
    [[nodiscard]] amio_status_t serve_control(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                               const ServiceHandler &handler, const ServiceHandler &after_ack = {}) const;

private:
    const ExecutionContext *context_;
    MpiTransport *transport_;
    mutable SequenceLedger received_sequences_;
    mutable std::mutex sequence_mutex_;
};

} // namespace amio::detail

#endif
