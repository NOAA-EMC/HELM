#include "execution/context.hpp"
#include "execution/aggregated_execution.hpp"
#include "execution/credit_window.hpp"
#include "execution/mpi_transport.hpp"
#include "execution/role_map.hpp"
#include "workers/comm_split.hpp"
#include "workers/worker_pool.hpp"
#include "drivers/grib2/grib2_steps.hpp"
#ifdef AMIO_TEST_NATIVE_NETCDF
#include "drivers/netcdf/netcdf_driver.hpp"
#include "schema/schema.hpp"
#include "schema/schema_wire_buffer.hpp"
#include <netcdf.h>
#endif

#include <mpi.h>

#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace amio::detail;

namespace {

void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }

void test_role_and_progress_rejections() {
    RoleConfiguration one_rank;
    one_rank.mode = ExecutionMode::Aggregated;
    one_rank.rank_count = 1;
    one_rank.producers = {0};
    require(validate_roles(one_rank) == AMIO_ERR_INVALID_POLICY, "one-rank aggregation must be rejected");
    require(validate_progress_support(ProgressPolicy::Background, 2) == AMIO_ERR_THREADING_UNSUPPORTED,
            "background MPI requires MULTIPLE");
}

void test_destination_region_chunks() {
    TransportEnvelope frame;
    frame.dimension_count = 2;
    frame.offsets[0] = 7;
    frame.offsets[1] = 11;
    frame.extents[0] = 2;
    frame.extents[1] = 3;
    frame.contribution_bytes = 24;
    frame.fragment_offset = 4;
    frame.payload_bytes = 16;
    std::array<std::byte, 16> bytes{};
    std::size_t rows = 0;
    auto sink = [&](const DestinationRegionChunk &region, std::span<const std::byte> part) {
        require(region.dimension_count == 2 && region.extents[0] == 1 && region.extents[1] == 2 && part.size() == 8,
                "fragment is divided into bounded destination rows");
        require(region.offsets[0] == 7 + rows && region.offsets[1] == (rows == 0 ? 12 : 11) &&
                    part.data() == bytes.data() + 8 * rows,
                "each destination row preserves its region and byte offset");
        ++rows;
        return AMIO_OK;
    };
    require(visit_destination_regions(frame, 4, bytes, sink) == AMIO_OK && rows == 2,
            "a fragment crossing a row boundary maps to two native regions");
    frame.fragment_offset = 1;
    rows = 0;
    require(visit_destination_regions(frame, 4, bytes, sink) == AMIO_ERR_INVALID_INPUT && rows == 0,
            "partial-element fragments fail before any destination callback");
    frame.fragment_offset = 4;
    frame.offsets[0] = std::numeric_limits<std::uint64_t>::max();
    require(visit_destination_regions(frame, 4, bytes, sink) == AMIO_ERR_INVALID_INPUT && rows == 0,
            "overflowing destination offsets fail before any destination callback");
    frame.fragment_offset = 0;
    frame.offsets[0] = 7;
    frame.extents[0] = 0;
    frame.contribution_bytes = 0;
    frame.payload_bytes = 0;
    require(visit_destination_regions(frame, 4, {}, [&](const DestinationRegionChunk &region, std::span<const std::byte> part) {
                require(region.extents[0] == 0 && part.empty(), "empty contribution remains an empty destination region");
                return AMIO_OK;
            }) == AMIO_OK,
            "empty contributions participate without inventing payload");
}

} // namespace

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) MPI_Abort(MPI_COMM_WORLD, 1);
    try {
        IOCommunicator default_communicator;
        CommConfig default_config;
        require(split_communicator(default_config, rank, default_communicator) == AMIO_OK &&
                    default_communicator.rank() == rank && default_communicator.size() == size,
                "default I/O communicator reports the parent rank and size");
        MPI_Comm parent_subset = MPI_COMM_NULL;
        require(MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &parent_subset) == MPI_SUCCESS,
                "create caller parent communicator");
        int subset_rank = -1, subset_size = 0;
        require(MPI_Comm_rank(parent_subset, &subset_rank) == MPI_SUCCESS &&
                    MPI_Comm_size(parent_subset, &subset_size) == MPI_SUCCESS,
                "query caller parent communicator");
        g_amio_parent_comm = parent_subset;
        IOCommunicator subset_communicator;
        require(split_communicator(default_config, rank, subset_communicator) == AMIO_OK &&
                    subset_communicator.rank() == subset_rank && subset_communicator.size() == subset_size,
                "default I/O communicator reports custom parent rank and size");
        IOCommunicator copied_communicator = subset_communicator;
        g_amio_parent_comm = MPI_COMM_WORLD;
        require(subset_communicator.handle() == parent_subset,
                "default I/O handle remains bound to its initialization parent");
        require(copied_communicator.handle() == parent_subset && copied_communicator.rank() == subset_rank &&
                    copied_communicator.size() == subset_size,
                "copied I/O communicator retains its parent details");
        require(MPI_Comm_free(&parent_subset) == MPI_SUCCESS, "release caller parent communicator");
        require(subset_communicator.rank() == subset_rank && subset_communicator.size() == subset_size,
                "communicator rank and size remain cached without later MPI calls");
        RoleConfiguration roles;
        roles.mode = ExecutionMode::Aggregated;
        roles.rank_count = 4;
        roles.producers = {0, 1, 2};
        roles.writers = {3};
        RoleMap mapping;
        require(RoleMap::create(roles, mapping) == AMIO_OK, "role map creation");
        test_role_and_progress_rejections();
        test_destination_region_chunks();
        std::unique_ptr<ExecutionContext> context;
        require(ExecutionContext::create(roles, ProgressPolicy::Caller, "aggregation-transport-test", MPI_COMM_WORLD, context) == AMIO_OK,
                "context with isolated control and data communicators");
        require(context->storage_root_rank() == 3, "storage metadata root is selected from the writer group");
        WorkerPoolConfig worker_configuration;
        worker_configuration.thread_configs.emplace_back();
        const auto worker_binding = bind_writer_worker_config(*context, worker_configuration);
        if (rank == 3) {
            int comparison = MPI_UNEQUAL;
            require(worker_binding == AMIO_OK && worker_configuration.io_comm.is_io_rank &&
                        MPI_Comm_compare(worker_configuration.io_comm.handle(), context->writer_communicator(), &comparison) == MPI_SUCCESS &&
                        comparison == MPI_IDENT && worker_configuration.io_comm.size() == 1,
                    "writer workers use the context storage communicator");
        } else {
            require(worker_binding == AMIO_ERR_INVALID_POLICY && !worker_configuration.io_comm.valid,
                    "producer roles cannot bind storage workers");
        }
        std::string metadata(9000, 'x');
        require(compare_context_metadata(*context, metadata, 10000) == AMIO_OK,
                "metadata agreement uses nonzero storage root across chunks");
        if (rank == 0) metadata[5000] = 'y';
        require(compare_context_metadata(*context, metadata, 10000) == AMIO_ERR_SCHEMA_MISMATCH,
                "metadata disagreement is reported to all ranks");
        if (rank == 0) metadata.resize(1);
        require(compare_context_metadata(*context, metadata, 10000) == AMIO_ERR_SCHEMA_MISMATCH,
                "different metadata lengths are reported without out-of-bounds reads");
        require(compare_context_metadata(*context, metadata, 10) == AMIO_ERR_INVALID_INPUT,
                "root metadata beyond the agreed limit is rejected collectively");
        if (rank < 3) {
            int producer_group_size = 0;
            require(context->producer_communicator() != MPI_COMM_NULL &&
                        MPI_Comm_size(context->producer_communicator(), &producer_group_size) == MPI_SUCCESS &&
                        producer_group_size == 3,
                    "producer ranks share a HALO-owned producer communicator");
            require(context->writer_communicator() == MPI_COMM_NULL && !context->can_open_storage(),
                    "producer ranks have no storage communicator or open permission");
            require(context->service_communicator() == MPI_COMM_NULL,
                    "producer ranks are excluded from the service-only communicator");
        } else {
            int writer_group_size = 0;
            int service_group_size = 0;
            require(context->producer_communicator() == MPI_COMM_NULL &&
                        context->writer_communicator() != MPI_COMM_NULL &&
                        MPI_Comm_size(context->writer_communicator(), &writer_group_size) == MPI_SUCCESS &&
                        writer_group_size == 1 && context->can_open_storage() &&
                        context->service_communicator() != MPI_COMM_NULL &&
                        MPI_Comm_size(context->service_communicator(), &service_group_size) == MPI_SUCCESS &&
                        service_group_size == 1,
                    "service writer owns the only storage-opening communicator");
        }

        // Each participant proves control credits remain available while data credits are full.
        CreditWindow credits({1, 8, 1, 16, 2, 64, 2});
        auto sender = credits.acquire(0, CreditOwner::Sender, 8, 1);
        auto receiver = credits.acquire(0, CreditOwner::Receiver, 8, 1);
        require(sender && receiver, "simultaneous sender/receiver credit reservation");
        require(!credits.acquire(0, CreditOwner::Sender, 1, 1), "exhausted sender credits reject admission");
        require(credits.acquire_control(16, 1).has_value(), "reserved control credit survives data exhaustion");

        // A collective admission vote rejects the whole round when one producer
        // cannot reserve its local snapshot; no payload transfer is entered.
        int local_admission = rank == 1 ? AMIO_ERR_QUEUE_FULL : AMIO_OK;
        int round_status = AMIO_OK;
        MPI_Allreduce(&local_admission, &round_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        require(round_status == AMIO_ERR_QUEUE_FULL, "collective local admission rejection");

        // Exercise real AMIO envelope and bounded-fragment transport over the
        // context's HALO-owned control and data communicators.
        CreditWindow transport_credits({4, 16, 4, 64, 16, 4 * kTransportEnvelopeWireBytes, 8});
        MpiTransport transport(*context, transport_credits, 4);
        AggregatedExecution aggregation(*context, transport);
        TransportValidationContext validation;
        validation.expected_epoch = 7;
        validation.expected_dataset_uuid.fill(0x2a);
        validation.expected_schema_digest.emplace();
        validation.expected_schema_digest->fill(0x3b);
        validation.expected_field_id = 0;
        validation.producer_ranks = {0, 1, 2};
        validation.writer_ranks = {3};
        validation.max_payload_bytes = 1024;
        validation.max_fragment_bytes = 4;
        // Canonical two-dataset service round: producer ranks send fragments
        // in deterministic producer/fragment order; producer 2 contributes
        // an explicit empty fragment and still participates in control.
        for (int dataset = 0; dataset < 3; ++dataset) {
            validation.expected_step_id = static_cast<std::uint64_t>(dataset);
            if (rank < 3) {
                const std::array<std::uint8_t, 10> bytes{static_cast<std::uint8_t>(dataset), static_cast<std::uint8_t>(rank), 2, 3, 4,
                                                         5, 6, 7, 8, 9};
                const std::uint32_t fragment_count = rank == 2 ? 1 : 3;
                TransportEnvelope envelope;
                envelope.context_epoch = validation.expected_epoch;
                envelope.dataset_uuid = validation.expected_dataset_uuid;
                envelope.schema_digest = *validation.expected_schema_digest;
                envelope.step_id = static_cast<std::uint64_t>(dataset);
                envelope.kind = MessageKind::Chunk;
                envelope.producer_rank = static_cast<std::uint32_t>(rank);
                envelope.writer_rank = 3;
                envelope.sequence = static_cast<std::uint64_t>(dataset == 2 ? 2 : dataset + 1);
                envelope.fragment_count = fragment_count;
                envelope.dimension_count = 1;
                envelope.offsets[0] = static_cast<std::uint64_t>(rank) * 10;
                envelope.extents[0] = rank == 2 ? 0 : 10;
                const auto payload = rank == 2 ? std::span<const std::byte>{} : std::as_bytes(std::span(bytes));
                require(aggregation.forward(envelope, payload) == AMIO_OK, "forward bounded producer contribution to mapped writer");
            } else {
                std::array<std::uint32_t, 3> observed{};
                std::array<std::uint64_t, 3> observed_bytes{};
                std::array<std::uint8_t, 20> destination{};
                const auto status = aggregation.receive_round(validation,
                    [&](const TransportEnvelope &envelope, std::span<const std::byte> chunk) {
                        const auto producer = envelope.producer_rank;
                        require(producer < observed.size(), "round only contains configured producers");
                        if (dataset == 2) {
                            require(false, "duplicate contribution sequence is rejected before the sink");
                        }
                        require(envelope.step_id == static_cast<std::uint64_t>(dataset) &&
                                    envelope.fragment_index == observed[producer],
                                "dataset and fragment order");
                        require(envelope.fragment_offset == observed_bytes[producer] && chunk.size() <= 4,
                                "writer receives ordered bounded destination chunks");
                        if (producer == 2) require(chunk.empty(), "empty producer sends an explicit empty chunk");
                        else require(chunk.size() == std::min<std::size_t>(4, 10 - observed_bytes[producer]), "chunk payload length");
                        require(visit_destination_regions(envelope, 1, chunk,
                            [&](const DestinationRegionChunk &region, std::span<const std::byte> part) {
                                require(region.dimension_count == 1 && region.offsets[0] <= destination.size() &&
                                            part.size() <= destination.size() - region.offsets[0],
                                        "forwarded region fits the destination");
                                for (std::size_t i = 0; i < part.size(); ++i)
                                    destination[region.offsets[0] + i] = std::to_integer<std::uint8_t>(part[i]);
                                return AMIO_OK;
                            }) == AMIO_OK, "forwarded fragment maps to bounded destination regions");
                        ++observed[producer];
                        observed_bytes[producer] += chunk.size();
                        return AMIO_OK;
                    });
                if (dataset == 2) {
                    require(status == AMIO_ERR_INVALID_POLICY && observed == std::array<std::uint32_t, 3>{},
                            "round drains every duplicate sequence without invoking the sink");
                } else {
                    require(status == AMIO_OK && observed == std::array<std::uint32_t, 3>{3, 3, 1},
                            "writer consumes all canonical contributions before releasing credits");
                    require(observed_bytes == std::array<std::uint64_t, 3>{10, 10, 0}, "contribution byte totals are preserved");
                    for (std::size_t producer = 0; producer < 2; ++producer) {
                        require(destination[producer * 10] == static_cast<std::uint8_t>(dataset) &&
                                    destination[producer * 10 + 1] == producer,
                                "destination values preserve producer and dataset identity");
                        for (std::size_t i = 2; i < 10; ++i)
                            require(destination[producer * 10 + i] == i, "destination values preserve the scientific payload");
                    }
                }
            }
        }
        const auto transport_usage = transport_credits.snapshot();
        require(transport_usage.data_bytes == 0 && transport_usage.data_descriptors == 0 &&
                    transport_usage.control_bytes == 0 && transport_usage.control_descriptors == 0,
                "sender, receiver, and reserved control credits return to zero after completion");
        const auto transport_peak = transport_credits.high_water();
        require(transport_peak.data_bytes <= 64 && transport_peak.data_descriptors <= 16 &&
                    transport_peak.control_bytes <= 4 * kTransportEnvelopeWireBytes &&
                    transport_peak.control_descriptors <= 8 &&
                    (rank == 2 || transport_peak.data_bytes > 0),
                "transport high-water remains bounded after all fragments drain");

        // A full receiver window returns an explicit error acknowledgment so
        // the producer releases its sender ticket without waiting forever.
        validation.expected_step_id = 99;
        if (rank == 3) {
            auto held = transport_credits.acquire(0, CreditOwner::Receiver, 16, 4);
            require(held.has_value(), "hold receiver capacity to exercise admission rejection");
            TransportEnvelope rejected;
            std::vector<std::byte> rejected_payload;
            CreditWindow::Ticket unused{};
            require(transport.receive_fragment(0, validation, rejected, rejected_payload, &unused) == AMIO_ERR_QUEUE_FULL,
                    "receiver rejects a fragment before allocation when credits are exhausted");
            require(transport_credits.release(*held), "release test receiver reservation");
        } else if (rank == 0) {
            TransportEnvelope rejected;
            rejected.context_epoch = validation.expected_epoch;
            rejected.dataset_uuid = validation.expected_dataset_uuid;
            rejected.schema_digest = *validation.expected_schema_digest;
            rejected.step_id = 99;
            rejected.kind = MessageKind::Chunk;
            rejected.producer_rank = 0;
            rejected.writer_rank = 3;
            rejected.sequence = 99;
            rejected.fragment_count = 1;
            rejected.dimension_count = 1;
            rejected.extents[0] = 1;
            rejected.contribution_bytes = 1;
            rejected.payload_bytes = 1;
            const std::byte data{42};
            require(transport.send_fragment(3, rejected, std::span(&data, 1)) == AMIO_ERR_QUEUE_FULL,
                    "producer receives receiver-credit rejection");

            auto held = transport_credits.acquire(3, CreditOwner::Sender, 16, 4);
            require(held.has_value(), "hold sender capacity to exercise local rejection");
            require(transport.send_fragment(3, rejected, std::span(&data, 1)) == AMIO_ERR_QUEUE_FULL,
                    "sender rejects locally before transport when credits are exhausted");
            require(transport_credits.release(*held), "release test sender reservation");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Reject a rendezvous-sized payload before the producer sends data,
        // then use the same channels for a valid fragment. This catches both
        // a blocked sender and a stale rejected payload contaminating the next read.
        constexpr std::size_t large_size = 1024 * 1024;
        CreditWindow large_credits({4, large_size, 2, 2 * large_size, 4, 4 * kTransportEnvelopeWireBytes, 4});
        MpiTransport large_transport(*context, large_credits, large_size);
        auto large_validation = validation;
        large_validation.expected_step_id = 100;
        large_validation.max_payload_bytes = large_size;
        large_validation.max_fragment_bytes = 4;
        if (rank == 0) {
            TransportEnvelope frame;
            frame.context_epoch = large_validation.expected_epoch;
            frame.dataset_uuid = large_validation.expected_dataset_uuid;
            frame.schema_digest = *large_validation.expected_schema_digest;
            frame.step_id = 100;
            frame.kind = MessageKind::Chunk;
            frame.producer_rank = 0;
            frame.writer_rank = 3;
            frame.sequence = 100;
            frame.dimension_count = 1;
            frame.extents[0] = large_size;
            frame.contribution_bytes = large_size;
            frame.payload_bytes = large_size;
            const std::vector<std::byte> large_payload(large_size, std::byte{0x11});
            require(large_transport.send_fragment(3, frame, large_payload) == AMIO_ERR_QUEUE_FULL,
                    "large rejected envelope does not block on a data send");
            frame.sequence = 101;
            frame.extents[0] = 1;
            frame.contribution_bytes = 1;
            frame.payload_bytes = 1;
            const std::byte valid_payload{0x5a};
            require(large_transport.send_fragment(3, frame, std::span(&valid_payload, 1)) == AMIO_OK,
                    "a valid fragment follows rejected admission on the same channels");
        } else if (rank == 3) {
            TransportEnvelope frame;
            std::vector<std::byte> payload;
            CreditWindow::Ticket ticket{};
            require(large_transport.receive_fragment(0, large_validation, frame, payload, &ticket) == AMIO_ERR_QUEUE_FULL &&
                        payload.empty(),
                    "oversized fragment is rejected before receiving data");
            require(large_transport.receive_fragment(0, large_validation, frame, payload, &ticket) == AMIO_OK &&
                        frame.sequence == 101 && payload.size() == 1 && payload[0] == std::byte{0x5a},
                    "valid payload is not contaminated by the rejected fragment");
            require(large_transport.release_receiver_credit(ticket, payload) == AMIO_OK && payload.capacity() == 0 &&
                        large_credits.snapshot().receiver_bytes == 0,
                    "receiver frees retained capacity before releasing its credit");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // A six-byte transport ceiling must not split four-byte model values.
        CreditWindow aligned_credits({4, 16, 4, 64, 16, 4 * kTransportEnvelopeWireBytes, 8});
        MpiTransport aligned_transport(*context, aligned_credits, 6);
        AggregatedExecution aligned_aggregation(*context, aligned_transport);
        auto aligned_validation = validation;
        aligned_validation.expected_step_id = 101;
        aligned_validation.max_fragment_bytes = 6;
        if (rank == 0) {
            TransportEnvelope frame;
            frame.context_epoch = aligned_validation.expected_epoch;
            frame.dataset_uuid = aligned_validation.expected_dataset_uuid;
            frame.schema_digest = *aligned_validation.expected_schema_digest;
            frame.step_id = 101;
            frame.kind = MessageKind::Chunk;
            frame.producer_rank = 0;
            frame.writer_rank = 3;
            frame.sequence = 1;
            frame.dimension_count = 1;
            frame.offsets[0] = 1;
            frame.extents[0] = 2;
            const std::array<std::uint32_t, 2> values{42, 123456};
            const auto bytes = std::as_bytes(std::span(values));
            require(aligned_aggregation.forward(frame, bytes, 8) == AMIO_ERR_INVALID_INPUT,
                    "a fragment ceiling smaller than an element is rejected before transfer");
            require(aligned_aggregation.forward(frame, bytes, sizeof(values[0])) == AMIO_OK,
                    "typed producer values use element-aligned fragments");
        } else if (rank == 3) {
            std::array<std::uint32_t, 4> destination{};
            std::size_t fragments = 0;
            require(aligned_aggregation.receive_regions(0, aligned_validation, sizeof(std::uint32_t),
                [&](const DestinationRegionChunk &region, std::span<const std::byte> part) {
                    require(part.size() == sizeof(std::uint32_t), "fragment contains one complete element");
                    ++fragments;
                    require(region.offsets[0] < destination.size() && region.extents[0] == 1,
                            "typed destination region is in bounds");
                    std::memcpy(&destination[region.offsets[0]], part.data(), part.size());
                    return AMIO_OK;
                }) == AMIO_OK && fragments == 2 && destination[1] == 42 && destination[2] == 123456,
                "writer reconstructs typed values from aligned destination fragments");
        }
        MPI_Barrier(MPI_COMM_WORLD);

#ifdef AMIO_TEST_NATIVE_NETCDF
        // Join the real transport, bounded destination mapping, and native
        // writer in one round; an independent reader checks the stored values.
        const auto native_path = std::filesystem::temp_directory_path() / "amio-aggregate-native-chunks.nc";
        if (rank == 0) {
            std::error_code ec;
            std::filesystem::remove(native_path, ec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        CreditWindow native_credits({4, 16, 4, 64, 16, 4 * kTransportEnvelopeWireBytes, 8});
        MpiTransport native_transport(*context, native_credits, 6);
        AggregatedExecution native_aggregation(*context, native_transport);
        auto native_validation = validation;
        native_validation.expected_step_id = 102;
        native_validation.max_fragment_bytes = 6;
        if (rank == 0) {
            Schema schema;
            Variable variable;
            variable.name = "temperature";
            variable.source_dtype = AMIO_DTYPE_F32;
            variable.stored_dtype = AMIO_DTYPE_F32;
            variable.dimensions.push_back({"x", 4, std::nullopt});
            require(schema.add_variable(std::move(variable)).ok() && schema.seal().ok(), "seal producer schema");
            TransportEnvelope schema_frame;
            schema_frame.context_epoch = native_validation.expected_epoch;
            schema_frame.dataset_uuid = native_validation.expected_dataset_uuid;
            schema_frame.schema_digest = *native_validation.expected_schema_digest;
            schema_frame.step_id = 102;
            schema_frame.kind = MessageKind::Chunk;
            schema_frame.producer_rank = 0;
            schema_frame.writer_rank = 3;
            schema_frame.sequence = 1;
            schema_frame.dimension_count = 1;
            schema_frame.extents[0] = schema.canonical().size();
            require(native_aggregation.forward(schema_frame, std::as_bytes(std::span(schema.canonical()))) == AMIO_OK,
                    "forward canonical schema in bounded chunks");
            TransportEnvelope control;
            control.context_epoch = native_validation.expected_epoch;
            control.dataset_uuid = native_validation.expected_dataset_uuid;
            control.schema_digest = *native_validation.expected_schema_digest;
            control.step_id = 102;
            control.kind = MessageKind::Begin;
            control.producer_rank = 0;
            control.writer_rank = 3;
            control.sequence = 1;
            require(native_transport.send_control(3, control) == AMIO_OK, "writer accepts native step begin");
            TransportEnvelope frame;
            frame.context_epoch = native_validation.expected_epoch;
            frame.dataset_uuid = native_validation.expected_dataset_uuid;
            frame.schema_digest = *native_validation.expected_schema_digest;
            frame.step_id = 102;
            frame.kind = MessageKind::Chunk;
            frame.producer_rank = 0;
            frame.writer_rank = 3;
            frame.sequence = 2;
            frame.dimension_count = 1;
            frame.offsets[0] = 1;
            frame.extents[0] = 2;
            const std::array<float, 2> values{1.25f, 2.5f};
            require(native_aggregation.forward(frame, std::as_bytes(std::span(values)), sizeof(float)) == AMIO_OK,
                    "producer forwards aligned scientific values to native writer");
            control.kind = MessageKind::Close;
            control.sequence = 2;
            require(native_transport.send_control(3, control) == AMIO_OK, "writer closes native step after payload");
        } else if (rank == 3) {
            Schema schema;
            ResourceBudget metadata_budget({2048, 8, 1024, 4});
            SchemaWireBuffer schema_wire(metadata_budget, 1024);
            require(native_aggregation.receive(0, native_validation,
                [&](const TransportEnvelope &frame, std::span<const std::byte> bytes) {
                    if (frame.fragment_index == 0) {
                        if (frame.contribution_bytes > 1024) return static_cast<amio_status_t>(AMIO_ERR_INVALID_INPUT);
                        const auto status = schema_wire.begin(static_cast<std::size_t>(frame.contribution_bytes));
                        if (status != AMIO_OK) return status;
                    }
                    return schema_wire.append(static_cast<std::size_t>(frame.fragment_offset), bytes);
                }) == AMIO_OK && Schema::from_canonical(schema_wire.complete(), 1024, schema).ok(),
                "decode bounded producer schema before native writer open");
            require(metadata_budget.snapshot().control_bytes == schema_wire.complete().size(),
                    "schema wire allocation is charged to the control budget");
            schema_wire.reset();
            require(metadata_budget.snapshot().control_bytes == 0, "schema wire reservation releases after consumption");
            NetCDF_Driver driver;
            driver.set_communicator(context->writer_communicator());
            driver.open_step_write(native_path.string(), schema, 10.0);
            driver.prepare_step_independent_writes("temperature");
            require(native_aggregation.serve_control(0, native_validation,
                [&](const TransportEnvelope &control) {
                    if (control.kind == MessageKind::Begin) return AMIO_OK;
                    if (control.kind == MessageKind::Close) { driver.close_step_write(); return AMIO_OK; }
                    return AMIO_ERR_INVALID_POLICY;
                },
                [&](const TransportEnvelope &control) {
                    if (control.kind != MessageKind::Begin) return static_cast<amio_status_t>(AMIO_OK);
                    return native_aggregation.receive_regions(0, native_validation, sizeof(float),
                        [&](const DestinationRegionChunk &region, std::span<const std::byte> bytes) {
                            RegionSelection selection;
                            selection.offsets.assign(region.offsets.begin(), region.offsets.begin() + region.dimension_count);
                            selection.extents.assign(region.extents.begin(), region.extents.begin() + region.dimension_count);
                            selection.empty = bytes.empty();
                            driver.write_step_region_independent("temperature", selection, bytes.data());
                            return AMIO_OK;
                        });
                }) == AMIO_OK, "writer service stores received regions and closes after payload");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            int ncid = -1, variable = -1;
            std::array<float, 4> values{};
            require(nc_open(native_path.c_str(), NC_NOWRITE, &ncid) == NC_NOERR &&
                        nc_inq_varid(ncid, "temperature", &variable) == NC_NOERR &&
                        nc_get_var_float(ncid, variable, values.data()) == NC_NOERR &&
                        values[1] == 1.25f && values[2] == 2.5f && nc_close(ncid) == NC_NOERR,
                    "independent native reader recovers forwarded values");
            std::error_code ec;
            std::filesystem::remove(native_path, ec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
#endif

        // Producer-encoded GRIB messages cross MPI as bounded opaque bytes;
        // the writer streams them to disk without assembling a global field.
        const auto grib_path = std::filesystem::temp_directory_path() / "amio-aggregate-streamed-grib2.bin";
        if (rank == 0) {
            std::error_code ec;
            std::filesystem::remove(grib_path, ec);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        CreditWindow grib_credits({4, 16, 4, 64, 16, 4 * kTransportEnvelopeWireBytes, 8});
        MpiTransport grib_transport(*context, grib_credits, 4);
        AggregatedExecution grib_aggregation(*context, grib_transport);
        auto grib_validation = validation;
        grib_validation.expected_step_id = 104;
        std::array<unsigned char, 21> encoded{};
        encoded[0] = 'G'; encoded[1] = 'R'; encoded[2] = 'I'; encoded[3] = 'B';
        encoded[7] = 2; encoded[15] = static_cast<unsigned char>(encoded.size()); encoded[16] = 42;
        encoded[17] = encoded[18] = encoded[19] = encoded[20] = '7';
        if (rank == 0) {
            TransportEnvelope frame;
            frame.context_epoch = grib_validation.expected_epoch;
            frame.dataset_uuid = grib_validation.expected_dataset_uuid;
            frame.schema_digest = *grib_validation.expected_schema_digest;
            frame.step_id = 104;
            frame.kind = MessageKind::Chunk;
            frame.producer_rank = 0;
            frame.writer_rank = 3;
            frame.sequence = 1;
            frame.dimension_count = 1;
            frame.extents[0] = 1;
            require(grib_aggregation.forward(frame, std::as_bytes(std::span(encoded))) == AMIO_OK,
                    "producer forwards one complete encoded GRIB message in bounded chunks");
        } else if (rank == 3) {
            Grib2StreamWriter writer(grib_path.string(), 4, 1, 64);
            std::string error;
            require(writer.begin_message(0, 0, encoded.size(), error), "writer opens an indexed complete GRIB message");
            require(grib_aggregation.receive(0, grib_validation,
                [&](const TransportEnvelope &frame, std::span<const std::byte> bytes) {
                    return writer.append_chunk(frame.fragment_offset, bytes, error) ? AMIO_OK : AMIO_ERR_BACKEND_FAILURE;
                }) == AMIO_OK && writer.finish(error) && writer.index().size() == 1,
                "writer validates and publishes the bounded encoded-message stream");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            Grib2FieldRanges field{"temperature", 0, {{0, encoded.size()}}};
            std::vector<std::vector<unsigned char>> messages;
            std::string error;
            require(read_grib2_field_messages(grib_path.string(), field, 64, messages, error) && messages.size() == 1 &&
                        std::equal(encoded.begin(), encoded.end(), messages[0].begin(), messages[0].end()),
                    "indexed GRIB message bytes match the producer encoding");
            std::error_code ec;
            std::filesystem::remove(grib_path, ec);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Payload credits can be completely occupied while bounded lifecycle
        // commands still traverse the reserved service-control channel.
        CreditWindow control_credits({4, 16, 4, 64, 16, 4 * kTransportEnvelopeWireBytes, 8});
        MpiTransport control_transport(*context, control_credits, 4);
        auto control_validation = validation;
        control_validation.expected_step_id = 103;
        const std::array<MessageKind, 4> commands{MessageKind::Begin, MessageKind::Seal, MessageKind::Cancel, MessageKind::Close};
        if (rank == 3) {
            bool invoked = false;
            require(control_transport.try_receive_control(0, control_validation, [&](const TransportEnvelope &) {
                        invoked = true;
                        return AMIO_OK;
                    }) == AMIO_ERR_OPERATION_PENDING && !invoked,
                    "caller-driven control polling returns pending without a command");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            auto held = control_credits.acquire(3, CreditOwner::Sender, 16, 4);
            require(held.has_value(), "occupy producer payload credits during service control");
            TransportEnvelope frame;
            frame.context_epoch = control_validation.expected_epoch;
            frame.dataset_uuid = control_validation.expected_dataset_uuid;
            frame.schema_digest = *control_validation.expected_schema_digest;
            frame.step_id = 103;
            frame.producer_rank = 0;
            frame.writer_rank = 3;
            for (std::size_t i = 0; i < commands.size(); ++i) {
                frame.kind = commands[i];
                frame.sequence = i + 1;
                const auto expected = frame.kind == MessageKind::Cancel ? AMIO_ERR_STEP_FAILED : AMIO_OK;
                require(control_transport.send_control(3, frame) == expected,
                        "service command receives an explicit acceptance or failure acknowledgment");
            }
            frame.kind = MessageKind::Begin;
            frame.sequence = 5;
            ++frame.context_epoch;
            require(control_transport.send_control(3, frame) == AMIO_ERR_STEP_FAILED,
                    "stale control identity receives a failure acknowledgment");
            --frame.context_epoch;
            frame.sequence = 6;
            frame.kind = MessageKind::Close;
            require(control_transport.send_control(3, frame) == AMIO_OK,
                    "valid control follows a rejected stale command");
            require(control_credits.release(*held), "release held sender payload credits");
        } else if (rank == 3) {
            auto held = control_credits.acquire(0, CreditOwner::Receiver, 16, 4);
            require(held.has_value(), "occupy writer payload credits during service control");
            for (std::size_t i = 0; i < commands.size(); ++i) {
                const auto expected = commands[i] == MessageKind::Cancel ? AMIO_ERR_INVALID_POLICY : AMIO_OK;
                require(control_transport.receive_control(0, control_validation, [&](const TransportEnvelope &frame) {
                            require(frame.kind == commands[i] && frame.sequence == i + 1,
                                    "service control retains lifecycle command order");
                            return frame.kind == MessageKind::Cancel ? AMIO_ERR_INVALID_POLICY : AMIO_OK;
                        }) == expected, "writer handles each service command using reserved credits");
            }
            bool invoked = false;
            ProtocolError error = ProtocolError::None;
            require(control_transport.receive_control(0, control_validation, [&](const TransportEnvelope &) {
                        invoked = true;
                        return AMIO_OK;
                    }, &error) == AMIO_ERR_INVALID_POLICY && error == ProtocolError::StaleEpoch && !invoked,
                    "stale control identity is rejected before the service handler");
            require(control_transport.receive_control(0, control_validation, [&](const TransportEnvelope &frame) {
                        return frame.kind == MessageKind::Close && frame.sequence == 6 ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
                    }) == AMIO_OK, "valid control remains usable after stale-command rejection");
            require(control_credits.release(*held), "release held receiver payload credits");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Empty producer joins the control round without sending payload.
        int contributes_bytes = rank == 2 ? 0 : 1;
        int byte_sum = 0;
        MPI_Allreduce(&contributes_bytes, &byte_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        require(byte_sum == 3, "empty producer control participation");

        // Service close is a coordinated control event after accepted rounds.
        int close_vote = 1, close_count = 0;
        MPI_Allreduce(&close_vote, &close_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        require(close_count == 4, "all roles join graceful service close");
        if (rank < 3) require(!mapping.can_open_storage(static_cast<std::uint32_t>(rank)), "producer cannot open storage");
        else require(mapping.can_open_storage(static_cast<std::uint32_t>(rank)), "writer can open storage");
        MPI_Barrier(MPI_COMM_WORLD);

        // A writer with no mapped producer must be able to complete the
        // canonical round without waiting for payload or inventing work.
        context.reset();
        RoleConfiguration sparse_roles;
        sparse_roles.mode = ExecutionMode::Aggregated;
        sparse_roles.rank_count = 4;
        sparse_roles.producers = {0};
        sparse_roles.writers = {1, 2, 3};
        std::unique_ptr<ExecutionContext> sparse_context;
        require(ExecutionContext::create(sparse_roles, ProgressPolicy::Caller, "sparse-writer-round", MPI_COMM_WORLD,
                                         sparse_context) == AMIO_OK,
                "context with zero-count writers");
        WorkerPoolConfig sparse_workers;
        const auto sparse_binding = bind_writer_worker_config(*sparse_context, sparse_workers);
        if (rank == 0) require(sparse_binding == AMIO_ERR_INVALID_POLICY, "producer cannot bind writer workers");
        else require(sparse_binding == AMIO_OK && sparse_workers.io_comm.size() == 3 &&
                         sparse_workers.io_comm.rank() == rank - 1,
                     "all writers, including idle writers, bind the full writer group");
        CreditWindow sparse_credits({2, 16, 2, 64, 4, 4 * kTransportEnvelopeWireBytes, 4});
        MpiTransport sparse_transport(*sparse_context, sparse_credits, 4);
        AggregatedExecution sparse_aggregation(*sparse_context, sparse_transport);
        TransportValidationContext sparse_validation;
        sparse_validation.expected_epoch = 8;
        sparse_validation.expected_dataset_uuid.fill(0x4a);
        sparse_validation.expected_schema_digest.emplace();
        sparse_validation.expected_schema_digest->fill(0x5b);
        sparse_validation.expected_step_id = 0;
        sparse_validation.expected_field_id = 0;
        sparse_validation.producer_ranks = {0};
        sparse_validation.writer_ranks = {1, 2, 3};
        sparse_validation.max_payload_bytes = 16;
        sparse_validation.max_fragment_bytes = 4;
        if (rank == 0) {
            TransportEnvelope empty;
            empty.context_epoch = sparse_validation.expected_epoch;
            empty.dataset_uuid = sparse_validation.expected_dataset_uuid;
            empty.schema_digest = *sparse_validation.expected_schema_digest;
            empty.step_id = 0;
            empty.kind = MessageKind::Chunk;
            empty.producer_rank = 0;
            empty.writer_rank = 1;
            empty.sequence = 1;
            empty.dimension_count = 1;
            empty.extents[0] = 0;
            require(sparse_aggregation.forward(empty, {}) == AMIO_OK, "send explicit empty contribution to mapped writer");
        } else {
            std::uint32_t sink_calls = 0;
            const auto status = sparse_aggregation.receive_round_regions(sparse_validation, 1,
                [&](const DestinationRegionChunk &region, std::span<const std::byte> chunk) {
                    require(rank == 1 && chunk.empty() && region.extents[0] == 0,
                            "only the mapped writer receives the empty destination region");
                    ++sink_calls;
                    return AMIO_OK;
                });
            require(status == AMIO_OK && sink_calls == (rank == 1 ? 1u : 0u),
                    "zero-count writer drains an empty canonical round without waiting");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Lifecycle control reaches every writer, including writers with no
        // mapped producer payload in this round.
        if (rank == 0) {
            TransportEnvelope command;
            command.context_epoch = sparse_validation.expected_epoch;
            command.dataset_uuid = sparse_validation.expected_dataset_uuid;
            command.schema_digest = *sparse_validation.expected_schema_digest;
            command.producer_rank = 0;
            for (const auto kind : {MessageKind::Begin, MessageKind::Close}) {
                command.kind = kind;
                command.sequence = kind == MessageKind::Begin ? 2 : 3;
                for (std::uint32_t writer = 1; writer <= 3; ++writer) {
                    command.writer_rank = writer;
                    require(sparse_transport.send_control(writer, command) == AMIO_OK,
                            "lifecycle control reaches each writer in the static group");
                }
            }
        } else {
            std::size_t commands_handled = 0;
            require(sparse_aggregation.serve_control(0, sparse_validation, [&](const TransportEnvelope &command) {
                        const auto expected = commands_handled++ == 0 ? MessageKind::Begin : MessageKind::Close;
                        return command.kind == expected && command.writer_rank == static_cast<std::uint32_t>(rank)
                                   ? AMIO_OK : AMIO_ERR_INVALID_POLICY;
                    }) == AMIO_OK && commands_handled == 2,
                    "mapped and idle writers leave their service loop after close");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // A post-acknowledgment writer failure must remain observable at close
        // and must still allow the service handler to release backend resources.
        if (rank == 0) {
            TransportEnvelope command;
            command.context_epoch = sparse_validation.expected_epoch;
            command.dataset_uuid = sparse_validation.expected_dataset_uuid;
            command.schema_digest = *sparse_validation.expected_schema_digest;
            command.kind = MessageKind::Begin;
            command.producer_rank = 0;
            command.writer_rank = 1;
            command.sequence = 4;
            require(sparse_transport.send_control(1, command) == AMIO_OK, "writer accepts work before induced failure");
            command.kind = MessageKind::Close;
            command.sequence = 5;
            require(sparse_transport.send_control(1, command) == AMIO_ERR_STEP_FAILED,
                    "writer reports an earlier progress failure at close");
        } else if (rank == 1) {
            bool closed = false;
            const auto status = sparse_aggregation.serve_control(0, sparse_validation,
                [&](const TransportEnvelope &command) {
                    if (command.kind == MessageKind::Close) closed = true;
                    return AMIO_OK;
                },
                [&](const TransportEnvelope &command) {
                    return command.kind == MessageKind::Begin ? AMIO_ERR_BACKEND_FAILURE : AMIO_OK;
                });
            require(status == AMIO_ERR_BACKEND_FAILURE && closed,
                    "service retains its first failure while allowing close cleanup");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            TransportEnvelope command;
            command.context_epoch = sparse_validation.expected_epoch;
            command.dataset_uuid = sparse_validation.expected_dataset_uuid;
            command.schema_digest = *sparse_validation.expected_schema_digest;
            command.kind = MessageKind::Begin;
            command.producer_rank = 0;
            command.writer_rank = 1;
            command.sequence = 6;
            require(sparse_transport.send_control(1, command) == AMIO_OK, "writer accepts the first control sequence");
            require(sparse_transport.send_control(1, command) == AMIO_ERR_STEP_FAILED,
                    "duplicate control sequence is rejected");
            command.kind = MessageKind::Close;
            command.sequence = 7;
            require(sparse_transport.send_control(1, command) == AMIO_ERR_STEP_FAILED,
                    "close reports the earlier duplicate control failure");
        } else if (rank == 1) {
            std::size_t begins = 0;
            bool closed = false;
            const auto status = sparse_aggregation.serve_control(0, sparse_validation, [&](const TransportEnvelope &command) {
                if (command.kind == MessageKind::Begin) ++begins;
                if (command.kind == MessageKind::Close) closed = true;
                return AMIO_OK;
            });
            require(status == AMIO_ERR_INVALID_POLICY && begins == 1 && closed,
                    "duplicate command cannot repeat its handler and close still cleans up");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // A backend failure during a fragmented contribution drains the rest
        // of the accepted bytes and is reported at close rather than stranding
        // the producer in a later fragment send.
        if (rank == 0) {
            TransportEnvelope command;
            command.context_epoch = sparse_validation.expected_epoch;
            command.dataset_uuid = sparse_validation.expected_dataset_uuid;
            command.schema_digest = *sparse_validation.expected_schema_digest;
            command.kind = MessageKind::Begin;
            command.producer_rank = 0;
            command.writer_rank = 1;
            command.sequence = 8;
            require(sparse_transport.send_control(1, command) == AMIO_OK, "writer accepts a step before payload failure");
            auto frame = command;
            frame.kind = MessageKind::Chunk;
            frame.sequence = 2;
            frame.dimension_count = 1;
            frame.extents[0] = 10;
            const std::array<std::byte, 10> payload{};
            require(sparse_aggregation.forward(frame, payload) == AMIO_OK,
                    "sender completes receipt acknowledgments while writer drains failed work");
            command.kind = MessageKind::Close;
            command.sequence = 9;
            require(sparse_transport.send_control(1, command) == AMIO_ERR_STEP_FAILED,
                    "close reports the failed backend contribution");
        } else if (rank == 1) {
            bool closed = false;
            std::size_t writes = 0;
            const auto status = sparse_aggregation.serve_control(0, sparse_validation,
                [&](const TransportEnvelope &command) {
                    if (command.kind == MessageKind::Close) closed = true;
                    return AMIO_OK;
                },
                [&](const TransportEnvelope &command) -> amio_status_t {
                    if (command.kind != MessageKind::Begin) return AMIO_OK;
                    return sparse_aggregation.receive(0, sparse_validation,
                        [&](const TransportEnvelope &, std::span<const std::byte>) {
                            ++writes;
                            return AMIO_ERR_BACKEND_FAILURE;
                        });
                });
            require(status == AMIO_ERR_BACKEND_FAILURE && closed && writes == 1,
                    "service records the first backend failure, drains later chunks, and closes");
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Stream contexts own a distinct HALO consumer group as well as the
        // producer group, without granting persistent storage access.
        context.reset();
        RoleConfiguration stream_roles;
        stream_roles.mode = ExecutionMode::Stream;
        stream_roles.rank_count = 4;
        stream_roles.producers = {0, 1};
        stream_roles.consumers = {2, 3};
        require(ExecutionContext::create(stream_roles, ProgressPolicy::Caller, "stream-role-groups", MPI_COMM_WORLD,
                                         context) == AMIO_OK,
                "stream context creates statically configured producer and consumer groups");
        if (rank < 2) {
            int group_size = 0;
            require(MPI_Comm_size(context->producer_communicator(), &group_size) == MPI_SUCCESS && group_size == 2 &&
                        context->consumer_communicator() == MPI_COMM_NULL && context->service_communicator() == MPI_COMM_NULL &&
                        !context->can_open_storage(),
                    "stream producer receives only its producer communicator");
        } else {
            int group_size = 0;
            int service_group_size = 0;
            require(context->producer_communicator() == MPI_COMM_NULL &&
                        MPI_Comm_size(context->consumer_communicator(), &group_size) == MPI_SUCCESS && group_size == 2 &&
                        context->service_communicator() != MPI_COMM_NULL &&
                        MPI_Comm_size(context->service_communicator(), &service_group_size) == MPI_SUCCESS && service_group_size == 2 &&
                        !context->can_open_storage(),
                    "consumer service receives its consumer communicator and no storage permission");
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } catch (...) {
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Finalize();
    return 0;
}
