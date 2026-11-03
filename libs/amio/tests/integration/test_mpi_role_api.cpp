#include "amio/amio_evolution.h"

#include <mpi.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
} // namespace

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS) return 1;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4 || provided < MPI_THREAD_SERIALIZED) MPI_Abort(MPI_COMM_WORLD, 1);

    try {
        const std::array<int32_t, 3> producers{0, 1, 2};
        const std::array<int32_t, 1> writers{3};
        amio_context_descriptor_t descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        descriptor.execution_mode = AMIO_EXECUTION_PERSISTENT_AGGREGATED;
        descriptor.progress_mode = AMIO_PROGRESS_CALLER;
        descriptor.rank_count = 4;
        descriptor.producer_count = producers.size();
        descriptor.writer_count = writers.size();
        descriptor.producers = producers.data();
        descriptor.writers = writers.data();
        descriptor.fortran_communicator = static_cast<int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
        descriptor.canonical_configuration = "{}";
        descriptor.configuration_bytes = 2;

        auto invalid_descriptor = descriptor;
        if (rank == 1) invalid_descriptor.reserved[0] = 1;
        amio_context_handle context = reinterpret_cast<amio_context_handle>(static_cast<std::uintptr_t>(1));
        require(amio_init_ex(&invalid_descriptor, &context) == AMIO_ERR_INVALID_INPUT && context == nullptr,
                "rank-local descriptor errors are agreed before context communicator creation");
        invalid_descriptor = descriptor;
        if (rank == 2) invalid_descriptor.execution_mode = UINT32_MAX;
        context = reinterpret_cast<amio_context_handle>(static_cast<std::uintptr_t>(1));
        require(amio_init_ex(&invalid_descriptor, &context) == AMIO_ERR_INVALID_POLICY && context == nullptr,
                "rank-local execution mode errors do not strand other members in initialization");
        invalid_descriptor = descriptor;
        if (rank == 3) invalid_descriptor.producers = nullptr;
        context = reinterpret_cast<amio_context_handle>(static_cast<std::uintptr_t>(1));
        require(amio_init_ex(&invalid_descriptor, &context) == AMIO_ERR_INVALID_INPUT && context == nullptr,
                "rank-local missing arrays are rejected collectively before copying membership");
        const std::array<int32_t, 3> reordered_producers{1, 0, 2};
        invalid_descriptor = descriptor;
        if (rank == 1) invalid_descriptor.producers = reordered_producers.data();
        context = reinterpret_cast<amio_context_handle>(static_cast<std::uintptr_t>(1));
        require(amio_init_ex(&invalid_descriptor, &context) == AMIO_ERR_INVALID_POLICY && context == nullptr,
                "different ordered role maps are rejected before communicator splitting");
        context = nullptr;
        require(amio_init_ex(&descriptor, &context) == AMIO_OK && context != nullptr,
                "public API creates the configured producer/writer context");
        uint32_t role = AMIO_ROLE_INACTIVE;
        uint32_t group_rank = UINT32_MAX;
        require(amio_query_context_role(context, &role, &group_rank) == AMIO_OK,
                "public role query succeeds without changing context state");
        require(role == (rank < 3 ? AMIO_ROLE_PRODUCER : AMIO_ROLE_WRITER) &&
                    group_rank == static_cast<uint32_t>(rank < 3 ? rank : 0),
                "public API reports correct role and role-local rank");
        uint32_t progressed = UINT32_MAX;
        require(amio_progress(context, 4, &progressed) == AMIO_OK && progressed == 0,
                "caller-driven progress is exposed through the public context API");
        require(amio_run_service(context) == (rank == 3 ? AMIO_ERR_UNSUPPORTED : AMIO_ERR_INVALID_POLICY),
                "service entry validates writer-only use and reports unavailable dispatch without hanging");
        size_t diagnostics_bytes = 0;
        require(amio_query_diagnostics(context, nullptr, 0, &diagnostics_bytes) == AMIO_OK && diagnostics_bytes > 1,
                "local diagnostic snapshot supports a non-collective size query");
        std::vector<char> diagnostics(diagnostics_bytes);
        require(amio_query_diagnostics(context, diagnostics.data(), diagnostics.size(), &diagnostics_bytes) == AMIO_OK,
                "local diagnostic snapshot fills the caller buffer");
        const std::string diagnostics_json(diagnostics.data());
        require(diagnostics_json.find("persistent_aggregated") != std::string::npos &&
                    diagnostics_json.find("\"rank\":" + std::to_string(rank)) != std::string::npos,
                "diagnostics expose effective execution mode and this rank's topology locally");
        size_t reduced_bytes = 0;
        require(amio_reduce_diagnostics(context, nullptr, 0, &reduced_bytes) == AMIO_OK,
                "reduced diagnostic size query remains local");
        std::vector<char> reduced_diagnostics(reduced_bytes);
        require(amio_reduce_diagnostics(context, reduced_diagnostics.data(), reduced_diagnostics.size(), &reduced_bytes) == AMIO_OK,
                "explicit context reduction succeeds when every context rank calls it");
        require(std::string(reduced_diagnostics.data()).find("\"scope\":\"context\"") != std::string::npos,
                "reduced diagnostics identify their explicit collective scope");
        // Until aggregated dataset integration exists, every role must reject
        // the open without returning a dataset handle.
        constexpr char location[] = "/tmp/amio-unimplemented-aggregated-collection";
        amio_dataset_descriptor_t dataset_descriptor{};
        dataset_descriptor.struct_size = sizeof(dataset_descriptor);
        dataset_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        dataset_descriptor.location = location;
        dataset_descriptor.location_bytes = sizeof(location) - 1;
        dataset_descriptor.format = AMIO_FORMAT_NETCDF4;
        dataset_descriptor.budget_bytes = 4096;
        dataset_descriptor.budget_descriptors = 8;
        dataset_descriptor.control_budget_bytes = 512;
        dataset_descriptor.control_budget_descriptors = 1;
        amio_dataset_handle dataset = reinterpret_cast<amio_dataset_handle>(static_cast<std::uintptr_t>(1));
        require(amio_open_dataset_ex(context, &dataset_descriptor, &dataset) == AMIO_ERR_UNSUPPORTED && dataset == nullptr,
                "unsupported aggregated open is rejected collectively on every role");
        require(amio_release_context(context) == AMIO_OK, "every role can release its context");

        if (provided == MPI_THREAD_MULTIPLE) {
            constexpr char invalid_affinity[] = "{\"worker_pool\":{\"threads\":1,\"cpu_cores\":[-1]}}";
            descriptor.progress_mode = AMIO_PROGRESS_BACKGROUND;
            descriptor.canonical_configuration = invalid_affinity;
            descriptor.configuration_bytes = sizeof(invalid_affinity) - 1;
            context = nullptr;
            require(amio_init_ex(&descriptor, &context) == AMIO_ERR_INVALID_BINDING && context == nullptr,
                    "invalid writer affinity fails collectively before publishing a context");
            constexpr char worker_settings[] = "{\"worker_pool\":{\"threads\":1}}";
            descriptor.canonical_configuration = worker_settings;
            descriptor.configuration_bytes = sizeof(worker_settings) - 1;
            require(amio_init_ex(&descriptor, &context) == AMIO_OK && context != nullptr,
                    "background writer context starts with its writer communicator");
            require(amio_release_context(context) == AMIO_OK, "background writer workers drain before context release");
        }

        const std::array<int32_t, 2> stream_producers{0, 1};
        const std::array<int32_t, 2> consumers{2, 3};
        descriptor.execution_mode = AMIO_EXECUTION_MPI_STREAM;
        descriptor.progress_mode = AMIO_PROGRESS_CALLER;
        descriptor.canonical_configuration = "{}";
        descriptor.configuration_bytes = 2;
        descriptor.producer_count = stream_producers.size();
        descriptor.writer_count = 0;
        descriptor.consumer_count = consumers.size();
        descriptor.producers = stream_producers.data();
        descriptor.writers = nullptr;
        descriptor.consumers = consumers.data();
        context = nullptr;
        require(amio_init_ex(&descriptor, &context) == AMIO_OK && context != nullptr,
                "public API creates the configured producer/consumer context");
        role = AMIO_ROLE_INACTIVE;
        group_rank = UINT32_MAX;
        require(amio_query_context_role(context, &role, &group_rank) == AMIO_OK &&
                    role == (rank < 2 ? AMIO_ROLE_PRODUCER : AMIO_ROLE_CONSUMER) &&
                    group_rank == static_cast<uint32_t>(rank < 2 ? rank : rank - 2),
                "public API reports producer and consumer local ranks");
        require(amio_run_service(context) == AMIO_ERR_INVALID_POLICY,
                "stream participants cannot enter the persistent aggregation service operation");
        require(amio_release_context(context) == AMIO_OK, "producer and consumer roles release their context");
        MPI_Barrier(MPI_COMM_WORLD);
    } catch (...) {
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Finalize();
    return 0;
}
