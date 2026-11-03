#include "execution/context.hpp"
#include "execution/credit_window.hpp"
#include "execution/mpi_transport.hpp"
#include "execution/role_map.hpp"
#include "execution/stream_execution.hpp"
#include "amio/amio_evolution.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

using namespace amio::detail;

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_FUNNELED);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size == 6);

    RoleConfiguration roles;
    roles.mode = ExecutionMode::Stream;
    roles.rank_count = 6;
    roles.producers = {0, 1, 2, 3};
    roles.consumers = {4, 5};
    std::unique_ptr<ExecutionContext> context;
    assert(ExecutionContext::create(roles, ProgressPolicy::Caller, "mpi-stream-contract-v1", MPI_COMM_WORLD, context) == AMIO_OK);
    assert(validate_roles(roles) == AMIO_OK);
    auto invalid_dynamic = roles;
    invalid_dynamic.consumers.push_back(0); // Dynamic attachment cannot overlap static roles.
    assert(validate_roles(invalid_dynamic) == AMIO_ERR_INVALID_POLICY);
    const auto mapped = context->role_map().writer_for(static_cast<std::uint32_t>(rank));
    if (rank < 4) assert(mapped && *mapped == static_cast<std::uint32_t>(4 + rank % 2));

    CreditWindow credits({6, 1024 * 1024, 4, 4 * 1024 * 1024, 24,
                          8 * kTransportEnvelopeWireBytes, 8});
    MpiTransport transport(*context, credits, 256);
    StreamRetention retention(StreamRetentionPolicy::Reliable, 1, 4, 4096);
    std::array<std::uint8_t, kDatasetUuidBytes> uuid{}; uuid.fill(0x52);
    StreamExecution stream(*context, transport, retention, 9, uuid, 4096);

    if (rank >= 4) assert(retention.acquire_next(0, std::chrono::milliseconds(0)).state == StreamAcquireState::Timeout);
    // A full qualification-size run with a deliberately serialized consumer:
    // each producer must retain the source until its consumer releases it.
    for (std::uint64_t step = 0; step < 1000; ++step) {
        std::array<std::uint8_t, kSchemaDigestBytes> schema{};
        schema.fill(static_cast<std::uint8_t>(0x31 + step));
        if (rank < 4) {
            std::array<float, 1> value{static_cast<float>(step * 100 + static_cast<std::uint64_t>(rank))};
            TransportEnvelope frame;
            frame.context_epoch = 9; frame.dataset_uuid = uuid; frame.step_id = step;
            frame.schema_digest = schema; frame.field_id = 7; frame.kind = MessageKind::Chunk;
            frame.producer_rank = static_cast<std::uint32_t>(rank); frame.sequence = step * 2 + 1;
            frame.dimension_count = 1; frame.offsets[0] = static_cast<std::uint64_t>(rank);
            frame.extents[0] = 1;
            if (rank == 0 && step == 0) {
                auto stale = frame;
                stale.context_epoch = 8;
                stale.sequence = 99;
                stale.writer_rank = *mapped;
                assert(transport.send_payload(*mapped, stale, std::as_bytes(std::span(value)), sizeof(float)) == AMIO_ERR_QUEUE_FULL);
            }
            const auto status = stream.submit(frame, std::as_bytes(std::span(value)), sizeof(float));
            assert(status == AMIO_OK);
            if (rank == 0 && step == 0) assert(stream.submit(frame, std::as_bytes(std::span(value)), sizeof(float)) == AMIO_ERR_INVALID_POLICY);
            auto second_field = frame;
            second_field.field_id = 8;
            second_field.sequence += 1;
            std::array<float, 1> second_value{static_cast<float>(step * 100 + static_cast<std::uint64_t>(rank) + 1000)};
            assert(stream.submit(second_field, std::as_bytes(std::span(second_value)), sizeof(float)) == AMIO_OK);
            assert(stream.finish_producer() == AMIO_ERR_OPERATION_PENDING);
            assert(stream.wait_for_release(step) == AMIO_OK);
            assert(retention.snapshot().bytes == 0);
        } else {
            if (rank == 4 && step == 0) {
                TransportValidationContext expected;
                expected.expected_epoch = 9; expected.expected_dataset_uuid = uuid;
                expected.expected_schema_digest = schema; expected.expected_step_id = step;
                expected.producer_ranks = {0, 1, 2, 3}; expected.writer_ranks = {4, 5};
                expected.max_payload_bytes = 4096; expected.max_fragment_bytes = transport.max_fragment_bytes();
                TransportEnvelope rejected;
                CreditWindow::Ticket no_ticket{};
                std::vector<std::byte> no_payload;
                ProtocolError protocol_error = ProtocolError::None;
                assert(transport.receive_fragment(0, expected, rejected, no_payload, &no_ticket, &protocol_error) == AMIO_ERR_INVALID_POLICY);
                assert(protocol_error == ProtocolError::StaleEpoch);
            }
            assert(stream.receive_step(step, schema, {"optional_precipitation"}, 2) == AMIO_OK);
            const auto available = retention.acquire_next(0, std::chrono::milliseconds(0));
            assert(available.state == StreamAcquireState::Available && available.lease);
            assert(available.lease->step->step_id == step);
            assert(available.lease->step->schema_digest.size() == 64);
            const char expected_schema_nibble = "0123456789abcdef"[static_cast<std::uint8_t>(0x31 + step) >> 4];
            assert(available.lease->step->schema_digest.front() == expected_schema_nibble);
            assert(available.lease->step->absent_optional_fields.contains("optional_precipitation"));
            for (const auto producer : roles.producers) {
                if (context->role_map().writer_for(static_cast<std::uint32_t>(producer)) != static_cast<std::uint32_t>(rank)) continue;
                const auto key = "field_7_producer_" + std::to_string(producer);
                const auto found = available.lease->step->fields.find(key);
                assert(found != available.lease->step->fields.end() && found->second.size() == sizeof(float));
                float value = 0;
                std::memcpy(&value, found->second.data(), sizeof(value));
                assert(value == static_cast<float>(step * 100 + producer));
                const auto second = available.lease->step->fields.find("field_8_producer_" + std::to_string(producer));
                assert(second != available.lease->step->fields.end());
                std::memcpy(&value, second->second.data(), sizeof(value));
                assert(value == static_cast<float>(step * 100 + producer + 1000));
            }
            assert(stream.release_lease(available.lease->lease_id) == AMIO_OK);
        }
    }
    if (rank < 4) assert(stream.finish_producer(!(rank == 3)) == AMIO_OK);
    else {
        const auto status = stream.finish_consumer();
        assert(status == (rank == 5 ? AMIO_ERR_STREAM_DISCONNECTED : AMIO_OK));
        const auto final = retention.acquire_next(0, std::chrono::milliseconds(0));
        assert(final.state == (rank == 5 ? StreamAcquireState::Failed : StreamAcquireState::EndOfStream));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    context.reset();

    // Exercise the installed C boundary with the same static 4+2 topology.
    const int32_t producer_ranks[] = {0, 1, 2, 3};
    const int32_t consumer_ranks[] = {4, 5};
    const char configuration[] = "diagnostics:\n  enabled: true\n";
    amio_context_descriptor_t context_descriptor{};
    context_descriptor.struct_size = sizeof(context_descriptor);
    context_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_descriptor.execution_mode = AMIO_EXECUTION_MPI_STREAM;
    context_descriptor.progress_mode = AMIO_PROGRESS_CALLER;
    context_descriptor.rank_count = 6;
    context_descriptor.producer_count = 4;
    context_descriptor.consumer_count = 2;
    context_descriptor.producers = producer_ranks;
    context_descriptor.consumers = consumer_ranks;
    context_descriptor.fortran_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
    context_descriptor.canonical_configuration = configuration;
    context_descriptor.configuration_bytes = sizeof(configuration) - 1;
    amio_context_handle public_context = nullptr;
    assert(amio_init_ex(&context_descriptor, &public_context) == AMIO_OK);

    std::array<std::uint8_t, 32> public_digest{};
    public_digest.fill(0x6c);
    if (rank < 4) {
        const float value = static_cast<float>(5000 + rank);
        amio_stream_contribution_t contribution{};
        contribution.struct_size = sizeof(contribution);
        contribution.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        contribution.step_id = 9001;
        contribution.sequence = 1;
        contribution.field_id = 42;
        contribution.rank = 1;
        contribution.element_bytes = sizeof(value);
        std::copy(public_digest.begin(), public_digest.end(), contribution.schema_digest);
        contribution.offsets[0] = rank;
        contribution.extents[0] = 1;
        assert(amio_stream_submit(public_context, &contribution, &value, sizeof(value)) == AMIO_OK);
        assert(amio_stream_try_wait_release(public_context, 9001, 0) == AMIO_ERR_OPERATION_PENDING);
        assert(amio_stream_finish(public_context, 1) == AMIO_ERR_OPERATION_PENDING);
        assert(amio_stream_wait_release(public_context, 9001) == AMIO_OK);
        assert(amio_stream_finish(public_context, 1) == AMIO_OK);
    } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        amio_stream_receive_descriptor_t receive{};
        receive.struct_size = sizeof(receive);
        receive.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        receive.step_id = 9001;
        receive.fields_per_producer = 1;
        const char absent_name[] = "precipitation";
        const char *absent_names[] = {absent_name};
        const uint64_t absent_name_bytes[] = {sizeof(absent_name) - 1};
        receive.absent_optional_count = 1;
        receive.absent_optional_names = absent_names;
        receive.absent_optional_name_bytes = absent_name_bytes;
        std::copy(public_digest.begin(), public_digest.end(), receive.schema_digest);
        assert(amio_stream_receive_step(public_context, &receive) == AMIO_OK);
        amio_stream_acquire_result_t acquired_public{};
        assert(amio_stream_acquire(public_context, 0, &acquired_public) == AMIO_OK);
        assert(acquired_public.state == AMIO_STREAM_AVAILABLE && acquired_public.lease != nullptr);
        for (const int producer : producer_ranks) {
            if (4 + producer % 2 != rank) continue;
            amio_stream_read_descriptor_t selection{};
            selection.struct_size = sizeof(selection);
            selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            selection.producer_rank = static_cast<std::uint32_t>(producer);
            selection.field_id = 42;
            selection.rank = 1;
            selection.element_bytes = sizeof(float);
            selection.offsets[0] = producer;
            selection.extents[0] = 1;
            float actual = 0;
            size_t read_bytes = 0;
            assert(amio_stream_read(acquired_public.lease, &selection, &actual, sizeof(actual), &read_bytes) == AMIO_OK);
            assert(read_bytes == sizeof(actual) && actual == static_cast<float>(5000 + producer));
            selection.field_id = 99;
            selection.logical_field_name = "precipitation";
            selection.logical_field_name_bytes = sizeof("precipitation") - 1;
            assert(amio_stream_read(acquired_public.lease, &selection, &actual, sizeof(actual), &read_bytes) ==
                   AMIO_ERR_FIELD_NOT_PRESENT);
        }
        assert(amio_stream_release(acquired_public.lease) == AMIO_OK);
        assert(amio_stream_finish(public_context, 1) == AMIO_OK);
        size_t required_diagnostics = 0;
        assert(amio_query_diagnostics(public_context, nullptr, 0, &required_diagnostics) == AMIO_OK);
        std::vector<char> diagnostic_json(required_diagnostics);
        assert(amio_query_diagnostics(public_context, diagnostic_json.data(), diagnostic_json.size(),
                                      &required_diagnostics) == AMIO_OK);
        const std::string diagnostic_text(diagnostic_json.data());
        assert(diagnostic_text.find("transport_received_bytes") != std::string::npos);
        assert(diagnostic_text.find("stream_delivered_bytes") != std::string::npos);
        assert(diagnostic_text.find("stream_released_bytes") != std::string::npos);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    assert(amio_release_context(public_context) == AMIO_OK);

    amio_context_handle cancellation_context = nullptr;
    assert(amio_init_ex(&context_descriptor, &cancellation_context) == AMIO_OK);
    if (rank < 4) assert(amio_stream_cancel(cancellation_context) == AMIO_OK);
    else assert(amio_stream_finish(cancellation_context, 1) == AMIO_ERR_STEP_FAILED);
    MPI_Barrier(MPI_COMM_WORLD);
    assert(amio_release_context(cancellation_context) == AMIO_OK);
    MPI_Finalize();
    return 0;
}
