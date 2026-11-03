#include "aggregation.hpp"

#include "execution/aggregated_execution.hpp"
#include "execution/context.hpp"
#include "execution/credit_window.hpp"
#include "execution/mpi_transport.hpp"
#include "execution/role_map.hpp"
#include "reference_fields.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace amio::detail;

std::uint64_t parse_positive(const char *text) {
    std::uint64_t value = 0;
    const std::string_view input(text);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() || value == 0)
        throw std::invalid_argument("expected a positive integer");
    return value;
}

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
} // namespace

int run_aggregation_fixture(int argc, char **argv) {
    std::uint64_t writer_count_arg = 0;
    const std::string_view backend = argc > 5 ? argv[5] : "";
    const std::filesystem::path output = argc > 7 ? argv[7] : "";
    if (argc != 8 || std::string_view(argv[2]) != "--writers" || std::string_view(argv[4]) != "--backend" ||
        std::string_view(argv[6]) != "--output") {
        std::cerr << "Usage: amio_qualify aggregate --writers N --backend netcdf4 --output DIR\n";
        return 2;
    }
    try {
        writer_count_arg = parse_positive(argv[3]);
    } catch (const std::exception &error) {
        std::cerr << "invalid aggregate workload: " << error.what() << '\n';
        return 2;
    }
    if (backend != "netcdf4" || output.empty()) return 2;
    constexpr std::uint64_t steps = 4, fields = 3, global_cells = 257;

    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = MPI_THREAD_SINGLE;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED)
            return 1;
    }
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int result = 0;
    try {
        require(size >= 2, "aggregate fixture requires producer and writer ranks");
        RoleConfiguration roles;
        roles.mode = ExecutionMode::Aggregated;
        roles.rank_count = static_cast<std::uint32_t>(size);
        require(writer_count_arg < static_cast<std::uint64_t>(size), "at least one producer rank is required");
        const int writer_count = static_cast<int>(writer_count_arg);
        const int producer_count = size - writer_count;
        for (int peer = 0; peer < producer_count; ++peer) roles.producers.push_back(peer);
        for (int peer = producer_count; peer < size; ++peer) roles.writers.push_back(peer);

        std::unique_ptr<ExecutionContext> context;
        require(ExecutionContext::create(roles, ProgressPolicy::Caller, "amio-qualify-aggregate", MPI_COMM_WORLD, context) == AMIO_OK,
                "create the isolated aggregation context");
        CreditWindow credits({static_cast<std::size_t>(size), 4 * 1024 * 1024, 32,
                              static_cast<std::size_t>(producer_count) * 4 * 1024 * 1024,
                              static_cast<std::size_t>(producer_count) * 32,
                              static_cast<std::size_t>(producer_count) * 4 * kTransportEnvelopeWireBytes, 32});
        MpiTransport transport(*context, credits, 1024 * 1024);
        AggregatedExecution aggregation(*context, transport);
        TransportValidationContext validation;
        validation.expected_epoch = 1;
        validation.expected_dataset_uuid.fill(0x41);
        validation.expected_schema_digest.emplace();
        validation.expected_schema_digest->fill(0x27);
        validation.producer_ranks.reserve(roles.producers.size());
        validation.writer_ranks.reserve(roles.writers.size());
        for (int peer : roles.producers) validation.producer_ranks.push_back(static_cast<std::uint32_t>(peer));
        for (int peer : roles.writers) validation.writer_ranks.push_back(static_cast<std::uint32_t>(peer));
        validation.max_payload_bytes = 4 * 1024 * 1024;
        validation.max_fragment_bytes = 1024 * 1024;

        const auto partition_start = [&](std::uint64_t producer) {
            const auto base = global_cells / static_cast<std::uint64_t>(producer_count);
            const auto remainder = global_cells % static_cast<std::uint64_t>(producer_count);
            return producer * base + std::min(producer, remainder);
        };
        const auto partition_count = [&](std::uint64_t producer) {
            const auto base = global_cells / static_cast<std::uint64_t>(producer_count);
            const auto remainder = global_cells % static_cast<std::uint64_t>(producer_count);
            return base + (producer < remainder ? 1 : 0);
        };

        std::uint64_t sequence = 0;
        for (std::uint64_t step = 0; step < steps; ++step) {
            for (std::uint64_t field = 0; field < fields; ++field) {
                validation.expected_step_id = step;
                validation.expected_field_id = static_cast<std::uint32_t>(field);
                if (rank < producer_count) {
                    const auto local_count = partition_count(static_cast<std::uint64_t>(rank));
                    const auto first_cell = partition_start(static_cast<std::uint64_t>(rank));
                    std::vector<float> values(static_cast<std::size_t>(local_count));
                    for (std::uint64_t cell = 0; cell < local_count; ++cell)
                        values[static_cast<std::size_t>(cell)] = amio::qualification::reference_value(step, field, first_cell + cell);
                    TransportEnvelope frame;
                    frame.context_epoch = validation.expected_epoch;
                    frame.dataset_uuid = validation.expected_dataset_uuid;
                    frame.schema_digest = *validation.expected_schema_digest;
                    frame.step_id = step;
                    frame.field_id = static_cast<std::uint32_t>(field);
                    frame.kind = MessageKind::Chunk;
                    frame.producer_rank = static_cast<std::uint32_t>(rank);
                    frame.writer_rank = *context->role_map().writer_for(static_cast<std::uint32_t>(rank));
                    frame.sequence = ++sequence;
                    frame.dimension_count = 1;
                    frame.offsets[0] = static_cast<std::uint64_t>(rank) == 0 ? 0 : first_cell;
                    frame.extents[0] = local_count;
                    frame.contribution_bytes = values.size() * sizeof(float);
                    frame.fragment_count = 1;
                    const auto forward_status = aggregation.forward(frame, std::as_bytes(std::span(values)), sizeof(float));
                    if (forward_status != AMIO_OK)
                        throw std::runtime_error("forward field chunks to the statically mapped writer returned status " +
                                                 std::to_string(forward_status));
                } else {
                    const auto local_writer = context->rank();
                    const auto status = aggregation.receive_round_regions(validation, sizeof(float),
                        [&](const DestinationRegionChunk &region, std::span<const std::byte> bytes) {
                            require(region.dimension_count == 1 && bytes.size() == region.extents[0] * sizeof(float),
                                    "writer receives bounded contiguous destination rows");
                            const auto *values = reinterpret_cast<const float *>(bytes.data());
                            for (std::uint64_t cell = 0; cell < region.extents[0]; ++cell) {
                                const auto global_cell = region.offsets[0] + cell;
                                std::uint32_t producer = 0;
                                for (; producer < static_cast<std::uint32_t>(producer_count); ++producer) {
                                    const auto begin = partition_start(producer);
                                    if (global_cell >= begin && global_cell < begin + partition_count(producer)) break;
                                }
                                require(producer < static_cast<std::uint32_t>(producer_count),
                                        "every global cell belongs to exactly one producer partition");
                                const auto mapped_writer = context->role_map().writer_for(producer);
                                require(mapped_writer && *mapped_writer == local_writer,
                                        "each global cell is delivered only to its assigned writer");
                                require(values[cell] == amio::qualification::reference_value(step, field, global_cell),
                                        "aggregated destination value matches deterministic scientific reference");
                            }
                            return AMIO_OK;
                        });
                    require(status == AMIO_OK, "writer drains its canonical producer round");
                }
            }
        }
        const auto peak = credits.high_water();
        require(peak.data_bytes <= static_cast<std::size_t>(producer_count) * 4 * 1024 * 1024 &&
                    peak.data_descriptors <= static_cast<std::size_t>(producer_count) * 32,
                "transport credit high-water stays within configured limits");
        MPI_Barrier(MPI_COMM_WORLD);
        int output_status = AMIO_OK;
        if (rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(output, ec);
            if (ec) output_status = AMIO_ERR_BACKEND_FAILURE;
            if (output_status == AMIO_OK) {
                const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
                std::ofstream report(output / ("aggregate-" + std::to_string(stamp) + ".json"), std::ios::out | std::ios::binary);
                if (!report) output_status = AMIO_ERR_BACKEND_FAILURE;
                else report << "{\"schema_version\":1,\"status\":\"PASS\",\"backend_requested\":\"netcdf4\","
                               "\"steps\":4,\"fields\":3,\"global_cells\":257,\"writers\":" << writer_count
                            << ",\"persistent_storage_exercised\":false}\n";
                if (!report) output_status = AMIO_ERR_BACKEND_FAILURE;
            }
        }
        MPI_Bcast(&output_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
        require(output_status == AMIO_OK, "write the qualification result below the requested output directory");
        if (rank == 0) std::cout << "aggregate transport reference verified; report written; persistent storage is not exercised\n";
    } catch (const std::exception &error) {
        std::cerr << "rank " << rank << " aggregate fixture failed: " << error.what() << '\n';
        result = 1;
        MPI_Abort(MPI_COMM_WORLD, result);
    }
    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!initialized && !finalized) MPI_Finalize();
    return global_result;
}
