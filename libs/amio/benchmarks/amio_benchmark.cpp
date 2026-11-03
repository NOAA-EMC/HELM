#include <amio/amio.h>
#include <amio/amio_evolution.h>
#include <mpi.h>
#include <netcdf.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "execution/aggregated_execution.hpp"
#include "execution/context.hpp"
#include "execution/credit_window.hpp"
#include "execution/mpi_transport.hpp"
#include "execution/role_map.hpp"
#include "workload.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}
int aggregated(const amio::benchmark::Workload &w, const std::filesystem::path &output) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    int provided = MPI_THREAD_SINGLE;
    if (!initialized && (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED))
        return 2;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int result = 0;
    double metadata = 0.0, transport_time = 0.0;
    try {
        if (size < 2) throw std::runtime_error("amio-aggregated requires at least two MPI ranks");
        const int writers = size / 2;
        const int producers = size - writers;
        amio::detail::RoleConfiguration roles;
        roles.mode = amio::detail::ExecutionMode::Aggregated;
        roles.rank_count = static_cast<std::uint32_t>(size);
        for (int peer = 0; peer < producers; ++peer) roles.producers.push_back(peer);
        for (int peer = producers; peer < size; ++peer) roles.writers.push_back(peer);
        const auto setup_begin = Clock::now();
        std::unique_ptr<amio::detail::ExecutionContext> context;
        if (amio::detail::ExecutionContext::create(roles, amio::detail::ProgressPolicy::Caller, "amio-benchmark-aggregate", MPI_COMM_WORLD,
                                                   context) != AMIO_OK)
            throw std::runtime_error("cannot create aggregated execution context");
        amio::detail::CreditWindow credits({static_cast<std::size_t>(size), 64 * 1024 * 1024, 64,
                                            static_cast<std::size_t>(producers) * 64 * 1024 * 1024, static_cast<std::size_t>(producers) * 64,
                                            static_cast<std::size_t>(producers) * 8 * amio::detail::kTransportEnvelopeWireBytes, 64});
        amio::detail::MpiTransport transport(*context, credits, 4 * 1024 * 1024);
        amio::detail::AggregatedExecution aggregation(*context, transport);
        amio::detail::TransportValidationContext validation;
        validation.expected_epoch = 1;
        validation.expected_dataset_uuid.fill(0x41);
        validation.expected_schema_digest.emplace();
        validation.expected_schema_digest->fill(0x27);
        for (int peer : roles.producers) validation.producer_ranks.push_back(static_cast<std::uint32_t>(peer));
        for (int peer : roles.writers) validation.writer_ranks.push_back(static_cast<std::uint32_t>(peer));
        validation.max_payload_bytes = 64 * 1024 * 1024;
        validation.max_fragment_bytes = 4 * 1024 * 1024;
        metadata = seconds(Clock::now() - setup_begin);
        const auto part_start = [&](std::uint64_t p) { return p * (w.cells / producers) + std::min<std::uint64_t>(p, w.cells % producers); };
        const auto part_count = [&](std::uint64_t p) { return w.cells / producers + (p < w.cells % producers ? 1 : 0); };
        std::uint64_t sequence = 0;
        const auto transport_begin = Clock::now();
        for (std::uint64_t step = 0; step < w.steps; ++step)
            for (std::uint64_t field = 0; field < w.fields; ++field) {
                validation.expected_step_id = step;
                validation.expected_field_id = static_cast<std::uint32_t>(field);
                if (rank < producers) {
                    const auto count = part_count(static_cast<std::uint64_t>(rank));
                    const auto first = part_start(static_cast<std::uint64_t>(rank));
                    std::vector<float> values(static_cast<std::size_t>(count));
                    for (std::uint64_t i = 0; i < count; ++i)
                        values[static_cast<std::size_t>(i)] = amio::benchmark::value(w.seed, step, field, first + i);
                    amio::detail::TransportEnvelope frame;
                    frame.context_epoch = validation.expected_epoch;
                    frame.dataset_uuid = validation.expected_dataset_uuid;
                    frame.schema_digest = *validation.expected_schema_digest;
                    frame.step_id = step;
                    frame.field_id = static_cast<std::uint32_t>(field);
                    frame.kind = amio::detail::MessageKind::Chunk;
                    frame.producer_rank = static_cast<std::uint32_t>(rank);
                    frame.writer_rank = *context->role_map().writer_for(static_cast<std::uint32_t>(rank));
                    frame.sequence = ++sequence;
                    frame.dimension_count = 1;
                    frame.offsets[0] = first;
                    frame.extents[0] = count;
                    frame.contribution_bytes = values.size() * sizeof(float);
                    frame.fragment_count = 1;
                    const auto status = aggregation.forward(frame, std::as_bytes(std::span(values)), sizeof(float));
                    if (status != AMIO_OK) throw std::runtime_error("aggregation forward failed: " + std::to_string(status));
                } else {
                    const auto status = aggregation.receive_round_regions(
                        validation, sizeof(float), [&](const amio::detail::DestinationRegionChunk &region, std::span<const std::byte> bytes) {
                            if (region.dimension_count != 1 || bytes.size() != region.extents[0] * sizeof(float)) return AMIO_ERR_BACKEND_FAILURE;
                            const auto *values = reinterpret_cast<const float *>(bytes.data());
                            for (std::uint64_t i = 0; i < region.extents[0]; ++i)
                                if (values[i] != amio::benchmark::value(w.seed, step, field, region.offsets[0] + i)) return AMIO_ERR_BACKEND_FAILURE;
                            return AMIO_OK;
                        });
                    if (status != AMIO_OK) throw std::runtime_error("aggregation drain/reference validation failed: " + std::to_string(status));
                }
            }
        transport_time = seconds(Clock::now() - transport_begin);
        double max_metadata = 0, max_transport = 0;
        MPI_Reduce(&metadata, &max_metadata, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&transport_time, &max_transport, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(output, ec);
            if (ec) throw std::runtime_error("cannot create benchmark output directory");
            std::cout << "{\"mode\":\"amio-aggregated\",\"status\":\"PASS\",\"logical_layout\":\"steps-fields-cells-1d\",\"layout\":\"balanced-"
                         "contiguous-1d-regions\","
                      << "\"operators\":\"none\",\"diagnostics\":\"not-applicable\",\"completion\":\"received-and-reference-validated\","
                      << "\"persistent_storage_exercised\":false,\"producers\":" << producers << ",\"writers\":" << writers
                      << ",\"steps\":" << w.steps << ",\"fields\":" << w.fields << ",\"cells\":" << w.cells
                      << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float) << ",\"setup_metadata_seconds\":" << max_metadata
                      << ",\"transport_drain_seconds\":" << max_transport << "}\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "rank " << rank << ": " << e.what() << '\n';
        result = 2;
    }
    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (!initialized) MPI_Finalize();
    return global_result;
}
bool nc_ok(int status, const char *where) {
    if (status == NC_NOERR) return true;
    std::cerr << where << ": " << nc_strerror(status) << '\n';
    return false;
}
int reference(const amio::benchmark::Workload &w, const std::filesystem::path &output) {
    const auto full_begin = Clock::now();
    std::error_code ec;
    std::filesystem::create_directories(output, ec);
    if (ec) return 2;
    double metadata = 0.0;
    double payload_seconds = 0.0;
    std::vector<float> values(static_cast<std::size_t>(w.cells));
    for (std::uint64_t step = 0; step < w.steps; ++step) {
        const auto path = output / ("step-" + std::to_string(step) + ".nc");
        int ncid = -1, cell_dim = -1;
        std::vector<int> vars(static_cast<std::size_t>(w.fields));
        auto metadata_begin = Clock::now();
        if (!nc_ok(nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid), "nc_create")) return 2;
        if (!nc_ok(nc_def_dim(ncid, "cell", static_cast<std::size_t>(w.cells), &cell_dim), "define cell")) return 2;
        for (std::uint64_t f = 0; f < w.fields; ++f) {
            const auto name = "field_" + std::to_string(f);
            if (!nc_ok(nc_def_var(ncid, name.c_str(), NC_FLOAT, 1, &cell_dim, &vars[static_cast<std::size_t>(f)]), "define field")) return 2;
        }
        if (!nc_ok(nc_enddef(ncid), "end metadata")) return 2;
        metadata += seconds(Clock::now() - metadata_begin);
        const auto payload_begin = Clock::now();
        for (std::uint64_t f = 0; f < w.fields; ++f) {
            for (std::uint64_t cell = 0; cell < w.cells; ++cell)
                values[static_cast<std::size_t>(cell)] = amio::benchmark::value(w.seed, step, f, cell);
            if (!nc_ok(nc_put_var_float(ncid, vars[static_cast<std::size_t>(f)], values.data()), "write field")) return 2;
        }
        if (!nc_ok(nc_close(ncid), "close reference step")) return 2;
        payload_seconds += seconds(Clock::now() - payload_begin);
    }
    std::cout << "{\"mode\":\"direct-reference\",\"logical_layout\":\"steps-fields-cells-1d\",\"layout\":\"one-netcdf-file-per-step\",\"operators\":"
                 "\"none\","
              << "\"completion\":\"nc_close\",\"steps\":" << w.steps << ",\"fields\":" << w.fields << ",\"cells\":" << w.cells
              << ",\"metadata_seconds\":" << metadata << ",\"payload_and_close_seconds\":" << payload_seconds
              << ",\"end_to_end_seconds\":" << seconds(Clock::now() - full_begin) << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float)
              << "}\n";
    return 0;
}
int amio_direct(const amio::benchmark::Workload &w, const std::filesystem::path &output, bool diagnostics_enabled) {
    int argc = 0, provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, nullptr, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED) return 2;
    amio_context_handle context = nullptr;
    amio_dataset_handle dataset = nullptr;
    amio_schema_handle schema = nullptr;
    amio_status_t status = AMIO_OK;
    const char *failure_stage = "amio_init_ex";
    const auto full_begin = Clock::now();
    amio_context_descriptor_t context_desc{};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = 1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_CALLER;
    context_desc.rank_count = 1;
    context_desc.fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
    static constexpr char enabled_config[] = "{\"diagnostics\":{\"enabled\":true}}";
    static constexpr char disabled_config[] = "{\"diagnostics\":{\"enabled\":false}}";
    const char *config = diagnostics_enabled ? enabled_config : disabled_config;
    context_desc.canonical_configuration = config;
    context_desc.configuration_bytes = std::char_traits<char>::length(config);
    status = amio_init_ex(&context_desc, &context);
    const auto metadata_begin = Clock::now();
    const auto location = output.string();
    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = 1;
    dataset_desc.location = location.c_str();
    dataset_desc.location_bytes = location.size();
    dataset_desc.format = AMIO_FORMAT_NETCDF4;
    dataset_desc.budget_bytes = std::max<std::uint64_t>(64 * 1024 * 1024, w.fields * w.cells * sizeof(float));
    dataset_desc.budget_descriptors = w.fields * 4 + 64;
    dataset_desc.control_budget_bytes = 1024 * 1024;
    dataset_desc.control_budget_descriptors = w.fields + 8;
    if (status == AMIO_OK) {
        failure_stage = "amio_open_dataset_ex";
        status = amio_open_dataset_ex(context, &dataset_desc, &dataset);
    }
    const char dimension_name[] = "cell";
    amio_dimension_descriptor_t dimension{};
    dimension.struct_size = sizeof(dimension);
    dimension.version = 1;
    dimension.name = dimension_name;
    dimension.name_bytes = sizeof(dimension_name) - 1;
    dimension.extent = w.cells;
    std::vector<std::string> names(static_cast<std::size_t>(w.fields));
    std::vector<amio_variable_descriptor_t> variables(static_cast<std::size_t>(w.fields));
    for (std::uint64_t f = 0; f < w.fields; ++f) {
        names[static_cast<std::size_t>(f)] = "field_" + std::to_string(f);
        auto &v = variables[static_cast<std::size_t>(f)];
        v.struct_size = sizeof(v);
        v.version = 1;
        v.name = names[static_cast<std::size_t>(f)].c_str();
        v.name_bytes = names[static_cast<std::size_t>(f)].size();
        v.source_dtype = AMIO_DTYPE_F32;
        v.stored_dtype = AMIO_DTYPE_F32;
        v.required = 1;
        v.dimension_count = 1;
        v.dimensions = &dimension;
    }
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = 1;
    schema_desc.variable_count = variables.size();
    schema_desc.variables = variables.data();
    if (status == AMIO_OK) {
        failure_stage = "amio_define_schema";
        status = amio_define_schema(dataset, &schema_desc, &schema);
    }
    double metadata = seconds(Clock::now() - metadata_begin);
    std::vector<float> values(static_cast<std::size_t>(w.cells));
    double payload = 0.0;
    for (std::uint64_t step_id = 0; status == AMIO_OK && step_id < w.steps; ++step_id) {
        auto step_metadata_begin = Clock::now();
        amio_step_descriptor_t step_desc{};
        step_desc.struct_size = sizeof(step_desc);
        step_desc.version = 1;
        step_desc.step_id = step_id;
        step_desc.model_time = static_cast<double>(step_id);
        step_desc.schema = schema;
        amio_step_handle step = nullptr;
        failure_stage = "amio_begin_step";
        status = amio_begin_step(dataset, &step_desc, &step);
        metadata += seconds(Clock::now() - step_metadata_begin);
        std::vector<amio_operation_handle> operations;
        for (std::uint64_t f = 0; status == AMIO_OK && f < w.fields; ++f) {
            for (std::uint64_t cell = 0; cell < w.cells; ++cell)
                values[static_cast<std::size_t>(cell)] = amio::benchmark::value(w.seed, step_id, f, cell);
            amio_operation_handle operation = nullptr;
            const auto &name = names[static_cast<std::size_t>(f)];
            const auto payload_begin = Clock::now();
            failure_stage = "amio_put_field";
            status = amio_put_field(step, name.data(), name.size(), values.data(), values.size() * sizeof(float), &operation);
            if (status == AMIO_OK) {
                failure_stage = "wait_source_released";
                status = amio_wait_operation(operation, AMIO_OPERATION_SOURCE_RELEASED, 600000);
            }
            if (status == AMIO_OK) operations.push_back(operation);
            payload += seconds(Clock::now() - payload_begin);
        }
        if (status == AMIO_OK) {
            step_metadata_begin = Clock::now();
            failure_stage = "amio_end_step";
            status = amio_end_step(step);
            metadata += seconds(Clock::now() - step_metadata_begin);
        }
        for (auto operation : operations) {
            if (status == AMIO_OK) {
                const auto drain_begin = Clock::now();
                failure_stage = "amio_wait_operation";
                status = amio_wait_operation(operation, AMIO_OPERATION_BACKEND_COMPLETE, 600000);
                payload += seconds(Clock::now() - drain_begin);
            }
            (void)amio_release_operation(operation);
        }
        if (step) (void)amio_release_step_handle(step);
    }
    if (dataset) {
        const auto close_status = amio_close_dataset_ex(dataset);
        if (status == AMIO_OK) status = close_status;
    }
    if (schema) (void)amio_release_schema(schema);
    if (context) (void)amio_release_context(context);
    if (status == AMIO_OK)
        std::cout << "{\"mode\":\"amio-direct\",\"diagnostics\":\"" << (diagnostics_enabled ? "enabled" : "disabled")
                  << "\",\"logical_layout\":\"steps-fields-cells-1d\",\"layout\":\"AMIO-step-collection\",\"operators\":\"none\","
                  << "\"completion\":\"reader-visible\",\"steps\":" << w.steps << ",\"fields\":" << w.fields << ",\"cells\":" << w.cells
                  << ",\"setup_metadata_seconds\":" << metadata << ",\"payload_drain_seconds\":" << payload
                  << ",\"end_to_end_seconds\":" << seconds(Clock::now() - full_begin) << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float)
                  << "}\n";
    else
        std::cerr << "AMIO direct benchmark failed at " << failure_stage << " with status " << status << '\n';
    MPI_Finalize();
    return status == AMIO_OK ? 0 : 2;
}
}  // namespace

int main(int argc, char **argv) {
    if (argc != 9 || std::string(argv[1]) != "--mode" || std::string(argv[3]) != "--manifest" || std::string(argv[5]) != "--output" ||
        std::string(argv[7]) != "--diagnostics" || (std::string(argv[8]) != "enabled" && std::string(argv[8]) != "disabled")) {
        std::cerr << "Usage: amio_benchmark --mode direct-reference|amio-direct|amio-aggregated --manifest FILE --output PATH --diagnostics "
                     "enabled|disabled\n";
        return 2;
    }
    const std::string mode(argv[2]);
    amio::benchmark::Workload workload;
    std::string error;
    if (!amio::benchmark::load_workload(argv[4], workload, error)) {
        std::cerr << error << '\n';
        return 2;
    }
    const std::filesystem::path output(argv[6]);
    if (mode == "amio-aggregated") {
        return aggregated(workload, output);
    }
    std::error_code ec;
    if (std::filesystem::exists(output, ec)) {
        std::cerr << "output already exists\n";
        return 2;
    }
    if (ec) return 2;
    if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path(), ec);
    if (ec) return 2;
    if (mode == "direct-reference") return reference(workload, output);
    if (mode == "amio-direct") return amio_direct(workload, output, std::string(argv[8]) == "enabled");
    std::cerr << "unknown benchmark mode\n";
    return 2;
}
