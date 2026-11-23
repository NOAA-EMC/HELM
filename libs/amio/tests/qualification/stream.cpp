#include "stream.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "amio/amio_evolution.h"
#include "execution/stream_retention.hpp"

namespace {
constexpr std::uint64_t kSteps = 1000;

bool diagnostic_counter(amio_context_handle context, std::string_view name, std::uint64_t expected) {
    size_t required = 0;
    if (amio_query_diagnostics(context, nullptr, 0, &required) != AMIO_OK || required == 0) return false;
    std::vector<char> json(required);
    if (amio_query_diagnostics(context, json.data(), json.size(), &required) != AMIO_OK) return false;
    const std::string_view text(json.data());
    const auto key = text.find(std::string("\"") + std::string(name) + "\":");
    if (key == std::string_view::npos) return false;
    const auto start = key + name.size() + 3;
    std::uint64_t actual = 0;
    const auto result = std::from_chars(text.data() + start, text.data() + text.size(), actual);
    return result.ec == std::errc{} && actual == expected;
}

bool latest_retention_case() {
    using namespace amio::detail;
    StreamRetention retention(StreamRetentionPolicy::Latest, 1, 4, 64);
    for (std::uint64_t id = 0; id < kSteps; ++id) {
        RetainedStreamStep step;
        step.step_id = id;
        step.schema_digest = id % 2 ? "schema-b" : "schema-a";
        step.fields["value"].resize(sizeof(id));
        std::memcpy(step.fields["value"].data(), &id, sizeof(id));
        step.payload_bytes = sizeof(id);
        if (retention.publish(std::move(step)) != AMIO_OK) return false;
    }
    auto next = retention.acquire_next(0, std::chrono::milliseconds(0));
    if (next.state != StreamAcquireState::Skipped || !next.skipped || next.skipped->first != 0 || next.skipped->last != kSteps - 5) return false;
    for (std::uint64_t id = kSteps - 4; id < kSteps; ++id) {
        next = retention.acquire_next(0, std::chrono::milliseconds(0));
        if (next.state != StreamAcquireState::Available || !next.lease || next.lease->step->step_id != id) return false;
        std::uint64_t value = 0;
        std::memcpy(&value, next.lease->step->fields.at("value").data(), sizeof(value));
        if (value != id || retention.release(next.lease->lease_id) != AMIO_OK) return false;
    }
    return retention.snapshot().skipped_steps == kSteps - 4 && retention.close() == AMIO_OK;
}

int reliable_case(int rank, int slow_consumer_ms, bool latest) {
    int32_t producers[4] = {0, 1, 2, 3};
    int32_t consumers[2] = {4, 5};
    static constexpr char reliable_configuration[] = "{}";
    static constexpr char latest_configuration[] = "stream:\n  retention_policy: latest\n";
    const char *configuration = latest ? latest_configuration : reliable_configuration;
    amio_context_descriptor_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    descriptor.execution_mode = AMIO_EXECUTION_MPI_STREAM;
    descriptor.progress_mode = AMIO_PROGRESS_CALLER;
    descriptor.rank_count = 6;
    descriptor.producer_count = 4;
    descriptor.consumer_count = 2;
    descriptor.producers = producers;
    descriptor.consumers = consumers;
    descriptor.fortran_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
    descriptor.canonical_configuration = configuration;
    descriptor.configuration_bytes = std::strlen(configuration);
    amio_context_handle context = nullptr;
    auto status = amio_init_ex(&descriptor, &context);
    if (status != AMIO_OK) return status;

    for (std::uint64_t step = 0; step < kSteps && status == AMIO_OK; ++step) {
        std::array<std::uint8_t, 32> digest{};
        digest[0] = static_cast<std::uint8_t>(step % 2);
        if (rank < 4) {
            const float value = static_cast<float>(step * 100 + static_cast<std::uint64_t>(rank));
            amio_stream_contribution_t contribution{};
            contribution.struct_size = sizeof(contribution);
            contribution.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            contribution.step_id = step;
            contribution.sequence = step + 1;
            contribution.field_id = 1;
            contribution.rank = 1;
            contribution.element_bytes = sizeof(value);
            std::copy(digest.begin(), digest.end(), contribution.schema_digest);
            contribution.offsets[0] = static_cast<std::uint64_t>(rank);
            contribution.extents[0] = 1;
            status = amio_stream_submit(context, &contribution, &value, sizeof(value));
            if (status == AMIO_OK && !latest) status = amio_stream_wait_release(context, step);
        } else {
            amio_stream_receive_descriptor_t receive{};
            receive.struct_size = sizeof(receive);
            receive.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            receive.step_id = step;
            receive.fields_per_producer = 1;
            std::copy(digest.begin(), digest.end(), receive.schema_digest);
            status = amio_stream_receive_step(context, &receive);
            if (status == AMIO_OK && latest) continue;
            amio_stream_acquire_result_t acquired{};
            if (status == AMIO_OK) status = amio_stream_acquire(context, 0, &acquired);
            if (status == AMIO_OK && acquired.state != AMIO_STREAM_AVAILABLE) status = AMIO_ERR_STEP_FAILED;
            for (int producer = 0; status == AMIO_OK && producer < 4; ++producer) {
                if (4 + producer % 2 != rank) continue;
                amio_stream_read_descriptor_t selection{};
                selection.struct_size = sizeof(selection);
                selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                selection.producer_rank = static_cast<std::uint32_t>(producer);
                selection.field_id = 1;
                selection.rank = 1;
                selection.element_bytes = sizeof(float);
                selection.offsets[0] = static_cast<std::uint64_t>(producer);
                selection.extents[0] = 1;
                float actual = 0;
                size_t bytes = 0;
                status = amio_stream_read(acquired.lease, &selection, &actual, sizeof(actual), &bytes);
                const auto expected = static_cast<float>(step * 100 + static_cast<std::uint64_t>(producer));
                if (status == AMIO_OK && (bytes != sizeof(actual) || actual != expected)) status = AMIO_ERR_SCHEMA_MISMATCH;
            }
            if (status == AMIO_OK && slow_consumer_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slow_consumer_ms));
            if (status == AMIO_OK) status = amio_stream_release(acquired.lease);
        }
    }
    if (status == AMIO_OK && latest && rank >= 4) {
        amio_stream_acquire_result_t first{};
        status = amio_stream_acquire(context, 0, &first);
        if (status == AMIO_OK && (first.state != AMIO_STREAM_SKIPPED || first.skipped_first != 0 || first.skipped_last != kSteps - 9))
            status = AMIO_ERR_SCHEMA_MISMATCH;
        for (std::uint64_t step = kSteps - 8; status == AMIO_OK && step < kSteps; ++step) {
            amio_stream_acquire_result_t acquired{};
            status = amio_stream_acquire(context, 0, &acquired);
            if (status == AMIO_OK && (acquired.state != AMIO_STREAM_AVAILABLE || acquired.lease == nullptr)) status = AMIO_ERR_SCHEMA_MISMATCH;
            for (int producer = 0; status == AMIO_OK && producer < 4; ++producer) {
                if (4 + producer % 2 != rank) continue;
                amio_stream_read_descriptor_t selection{};
                selection.struct_size = sizeof(selection);
                selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                selection.producer_rank = static_cast<std::uint32_t>(producer);
                selection.field_id = 1;
                selection.rank = 1;
                selection.element_bytes = sizeof(float);
                selection.offsets[0] = static_cast<std::uint64_t>(producer);
                selection.extents[0] = 1;
                float actual = 0;
                size_t bytes = 0;
                status = amio_stream_read(acquired.lease, &selection, &actual, sizeof(actual), &bytes);
                const auto expected = static_cast<float>(step * 100 + static_cast<std::uint64_t>(producer));
                if (status == AMIO_OK && (bytes != sizeof(actual) || actual != expected)) status = AMIO_ERR_SCHEMA_MISMATCH;
            }
            if (status == AMIO_OK) status = amio_stream_release(acquired.lease);
        }
    }
    if (status == AMIO_OK && rank >= 4) {
        const std::uint64_t delivered = latest ? 64 : kSteps * 8;
        const std::uint64_t skipped = latest ? kSteps - 8 : 0;
        if (!diagnostic_counter(context, "transport_received_bytes", latest ? kSteps * 8 : kSteps * 8) ||
            !diagnostic_counter(context, "stream_delivered_bytes", delivered) || !diagnostic_counter(context, "stream_released_bytes", delivered) ||
            !diagnostic_counter(context, "stream_skips", skipped))
            status = AMIO_ERR_SCHEMA_MISMATCH;
    }
    if (status == AMIO_OK) status = amio_stream_finish(context, 1);
    if (context != nullptr) {
        const auto release_status = amio_release_context(context);
        if (status == AMIO_OK) status = release_status;
    }
    return status;
}
}  // namespace

int run_stream_qualification(int argc, char **argv) {
    std::string_view policy = "reliable";
    int slow_consumer_ms = 0;
    for (int i = 2; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--policy" && i + 1 < argc)
            policy = argv[++i];
        else if (arg == "--slow-consumer-ms" && i + 1 < argc) {
            try {
                slow_consumer_ms = std::stoi(argv[++i]);
            } catch (...) {
                return 2;
            }
            if (slow_consumer_ms < 0 || slow_consumer_ms > 60000) return 2;
        } else
            return 2;
    }
    if ((policy != "reliable" && policy != "latest") || (policy == "latest" && slow_consumer_ms != 0)) return 2;
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided = 0;
        if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED) return 3;
    }
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int status = size == 6 ? 0 : 2;
    if (status == 0 && policy == "latest") status = latest_retention_case() ? 0 : 1;
    if (status == 0 && policy == "reliable") status = reliable_case(rank, slow_consumer_ms, false);
    if (status == 0 && policy == "latest") status = reliable_case(rank, 0, true);
    if (status != 0 && rank == 0) std::cerr << "stream qualification failed, status=" << status << '\n';
    int global_status = 0;
    MPI_Allreduce(&status, &global_status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_status;
}
