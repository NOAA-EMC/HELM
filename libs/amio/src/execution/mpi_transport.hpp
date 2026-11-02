#ifndef AMIO_SRC_EXECUTION_MPI_TRANSPORT_HPP
#define AMIO_SRC_EXECUTION_MPI_TRANSPORT_HPP

#include "execution/context.hpp"
#include "execution/credit_window.hpp"
#include "execution/transport_protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace amio::detail {

/** Blocking bounded frame transport over an ExecutionContext's isolated channels.
 *
 * MPI ranks are interpreted in the context communicator ordering. The context
 * must outlive the transport. Receive storage is allocated only after the
 * canonical envelope passes identity, mapping, region, and byte-limit checks.
 */
class MpiTransport {
public:
    /** Create a transport with a finite fragment limit no larger than MPI's count limit. */
    MpiTransport(const ExecutionContext &context, CreditWindow &credits, std::uint64_t max_fragment_bytes);

    /** Send a bounded fragment after the receiver validates its envelope and grants credit.
     *
     * Sender ownership remains reserved through the receiver's receipt
     * acknowledgement. Rejection occurs before the payload enters the data
     * communicator, so it cannot contaminate a later transfer.
     */
    [[nodiscard]] amio_status_t send_fragment(std::uint32_t destination_rank, const TransportEnvelope &envelope,
                                               std::span<const std::byte> payload) const noexcept;

    /** Split and send ordered fragments, each aligned to the given element size.
     * Use an element size of one for opaque byte streams such as encoded GRIB messages.
     */
    [[nodiscard]] amio_status_t send_payload(std::uint32_t destination_rank, TransportEnvelope envelope,
                                              std::span<const std::byte> payload, std::size_t element_bytes = 1) const;

    /** Send a payload-free producer service command using reserved control credits. */
    [[nodiscard]] amio_status_t send_control(std::uint32_t writer_rank, const TransportEnvelope &envelope) const noexcept;

    /** Receive and handle a producer service command using no data credits.
     * The handler is invoked only after identity, routing and size validation.
     */
    [[nodiscard]] amio_status_t receive_control(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                                 const std::function<amio_status_t(const TransportEnvelope &)> &handler,
                                                 ProtocolError *protocol_error = nullptr) const noexcept;

    /** Poll for a control command, returning OPERATION_PENDING when none is ready.
     * Only one receiver may consume service commands for a given peer at a time.
     */
    [[nodiscard]] amio_status_t try_receive_control(std::uint32_t producer_rank, const TransportValidationContext &validation,
                                                     const std::function<amio_status_t(const TransportEnvelope &)> &handler,
                                                     ProtocolError *protocol_error = nullptr) const noexcept;

    /** Receive one fragment, validating the frame before resizing the output payload.
     *
     * The returned credit ticket accounts for receiver ownership. Release it
     * through release_receiver_credit() only after the backend or next owner
     * has finished using the fragment. That operation frees the vector's
     * retained capacity before returning its credit.
     */
    [[nodiscard]] amio_status_t receive_fragment(std::uint32_t source_rank, const TransportValidationContext &validation,
                                                  TransportEnvelope &envelope, std::vector<std::byte> &payload,
                                                  CreditWindow::Ticket *receiver_credit,
                                                  ProtocolError *protocol_error = nullptr) const noexcept;

    /** Free the consumed fragment storage, then release its receiver ticket. */
    [[nodiscard]] amio_status_t release_receiver_credit(const CreditWindow::Ticket &ticket,
                                                         std::vector<std::byte> &payload) const noexcept;

    /** Return the configured per-message fragment ceiling. */
    [[nodiscard]] std::uint64_t max_fragment_bytes() const noexcept { return max_fragment_bytes_; }

private:
    const ExecutionContext *context_;
    CreditWindow *credits_;
    std::uint64_t max_fragment_bytes_;
};

} // namespace amio::detail

#endif
