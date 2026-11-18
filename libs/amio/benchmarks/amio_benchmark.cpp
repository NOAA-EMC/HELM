#include <amio/amio.h>
#include <amio/amio_evolution.h>
#include <mpi.h>
#include <netcdf.h>
#include <netcdf_par.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
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

void check_amio(amio_status_t status, const char *where) {
    if (status != AMIO_OK) throw std::runtime_error(std::string(where) + ": AMIO status " + std::to_string(static_cast<int>(status)));
}

void check_nc(int status, const char *where) {
    if (status != NC_NOERR) throw std::runtime_error(std::string(where) + ": " + nc_strerror(status));
}

// Step:field variable names shared by the NetCDF reference and AMIO region benchmarks.
std::vector<std::string> build_step_field_names(std::uint64_t steps, std::uint64_t fields) {
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(steps * fields));
    for (std::uint64_t step = 0; step < steps; ++step)
        for (std::uint64_t field = 0; field < fields; ++field) names.push_back("s" + std::to_string(step) + "_f" + std::to_string(field));
    return names;
}

// Reopen the shared NetCDF-4 output collectively (or serially on one rank) and validate every rank's
// hyperslab against the reference generator, outside the timed region.
void validate_netcdf_benchmark_readback(const std::string &path, const std::vector<std::string> &names, const amio::benchmark::Workload &w,
                                        std::uint64_t first, std::uint64_t count, bool parallel_io, const char *mismatch_label,
                                        double &validate_seconds) {
    const auto validate_begin = Clock::now();
    std::vector<float> values(static_cast<std::size_t>(count));
    const std::size_t start_index = static_cast<std::size_t>(first);
    const std::size_t count_size = static_cast<std::size_t>(count);
    int readid = -1;
    if (parallel_io)
        check_nc(nc_open_par(path.c_str(), NC_NOWRITE | NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &readid), "reopen");
    else
        check_nc(nc_open(path.c_str(), NC_NOWRITE, &readid), "reopen");
    for (std::size_t r = 0; r < names.size(); ++r) {
        int varid = -1;
        check_nc(nc_inq_varid(readid, names[r].c_str(), &varid), "inquire variable");
        if (parallel_io) check_nc(nc_var_par_access(readid, varid, NC_COLLECTIVE), "read collective access");
        check_nc(nc_get_vara_float(readid, varid, &start_index, &count_size, values.data()), "read");
        const std::uint64_t step = r / w.fields, field = r % w.fields;
        for (std::uint64_t i = 0; i < count; ++i)
            if (values[static_cast<std::size_t>(i)] != amio::benchmark::value(w.seed, step, field, first + i))
                throw std::runtime_error(mismatch_label);
    }
    check_nc(nc_close(readid), "reopen close");
    validate_seconds = seconds(Clock::now() - validate_begin);
}

// Cross-rank maximum of a per-phase wall time (meaningful on rank 0, where the JSON report prints).
double reduce_max_time(double seconds_value) {
    double max_value = 0.0;
    MPI_Reduce(&seconds_value, &max_value, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return max_value;
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
    const char *progress_setting = std::getenv("AMIO_BENCHMARK_PROGRESS");
    const bool background = !progress_setting || std::string(progress_setting) == "background";
    if (progress_setting && !background && std::string(progress_setting) != "caller") return 2;
    const char *compute_setting = std::getenv("AMIO_BENCHMARK_COMPUTE_MS");
    const int compute_ms = compute_setting ? std::atoi(compute_setting) : 0;
    if (compute_ms < 0 || compute_ms > 60000) return 2;
    int argc = 0, provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, nullptr, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS || provided < MPI_THREAD_MULTIPLE) return 2;
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (std::getenv("AMIO_BENCHMARK_CPU_ONLY")) {
        MPI_Barrier(MPI_COMM_WORLD);
        const auto started = Clock::now();
        volatile std::uint64_t work = 1;
        for (std::uint64_t step = 0; step < w.steps; ++step) {
            const auto until = Clock::now() + std::chrono::milliseconds(compute_ms);
            while (Clock::now() < until) work = work * 1664525 + 1013904223;
        }
        const double total = reduce_max_time(seconds(Clock::now() - started));
        if (rank == 0)
            std::cout << "{\"mode\":\"step-cpu-only\",\"ranks\":" << ranks << ",\"steps\":" << w.steps << ",\"compute_ms_per_step\":" << compute_ms
                      << ",\"end_to_end_seconds\":" << total << "}\n";
        MPI_Finalize();
        return 0;
    }
    const auto count = w.cells / ranks + (static_cast<std::uint64_t>(rank) < w.cells % ranks);
    const auto first = static_cast<std::uint64_t>(rank) * (w.cells / ranks) + std::min<std::uint64_t>(rank, w.cells % ranks);
    std::vector<int32_t> producers(static_cast<std::size_t>(ranks));
    for (int i = 0; i < ranks; ++i) producers[static_cast<std::size_t>(i)] = i;
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
    context_desc.progress_mode = background ? AMIO_PROGRESS_BACKGROUND : AMIO_PROGRESS_CALLER;
    context_desc.rank_count = ranks;
    context_desc.producer_count = ranks;
    context_desc.producers = producers.data();
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
    if (const char *budget_setting = std::getenv("AMIO_BENCHMARK_STEP_BUDGET_BYTES")) {
        char *end = nullptr;
        const auto requested_budget = std::strtoull(budget_setting, &end, 10);
        if (end == budget_setting || *end != '\0' || requested_budget == 0) return 2;
        dataset_desc.budget_bytes = requested_budget;
    }
    dataset_desc.budget_descriptors = w.fields * static_cast<std::uint64_t>(ranks) * 4 + 64;
    dataset_desc.control_budget_bytes = 1024 * 1024;
    dataset_desc.control_budget_descriptors = w.fields * static_cast<std::uint64_t>(ranks) + 8;
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
    std::vector<float> values(static_cast<std::size_t>(count));
    std::vector<amio_field_partition_descriptor_t> partitions(static_cast<std::size_t>(w.fields));
    for (std::size_t i = 0; i < partitions.size(); ++i) {
        partitions[i].struct_size = sizeof(partitions[i]);
        partitions[i].version = 1;
        partitions[i].field = names[i].c_str();
        partitions[i].field_bytes = names[i].size();
        partitions[i].rank = 1;
        partitions[i].offsets[0] = first;
        partitions[i].extents[0] = count;
    }
    double payload = 0.0, submit_time = 0.0, post_wait_time = 0.0, begin_time = 0.0, end_time = 0.0, close_time = 0.0;
    std::uint64_t completed_before_end = 0, submitted = 0;
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
        status = ranks > 1 ? amio_begin_step_partitioned(dataset, &step_desc, partitions.data(), partitions.size(), &step)
                           : amio_begin_step(dataset, &step_desc, &step);
        const auto begin_elapsed = seconds(Clock::now() - step_metadata_begin);
        begin_time += begin_elapsed;
        metadata += begin_elapsed;
        std::vector<amio_operation_handle> operations;
        for (std::uint64_t f = 0; status == AMIO_OK && f < w.fields; ++f) {
            for (std::uint64_t cell = 0; cell < count; ++cell)
                values[static_cast<std::size_t>(cell)] = amio::benchmark::value(w.seed, step_id, f, first + cell);
            amio_operation_handle operation = nullptr;
            const auto &name = names[static_cast<std::size_t>(f)];
            const auto payload_begin = Clock::now();
            failure_stage = "amio_put_field";
            status = amio_put_field(step, name.data(), name.size(), values.data(), values.size() * sizeof(float), &operation);
            if (status == AMIO_OK) {
                failure_stage = "wait_source_released";
                status = amio_wait_operation(operation, AMIO_OPERATION_SOURCE_RELEASED, 600000);
            }
            if (status == AMIO_OK) {
                operations.push_back(operation);
                ++submitted;
            }
            const auto elapsed_put = seconds(Clock::now() - payload_begin);
            payload += elapsed_put;
            submit_time += elapsed_put;
        }
        const auto compute_until = Clock::now() + std::chrono::milliseconds(compute_ms);
        volatile std::uint64_t compute_value = 1;
        while (Clock::now() < compute_until) compute_value = compute_value * 1664525 + 1013904223;
        for (auto operation : operations) {
            amio_operation_status_t state{};
            state.struct_size = sizeof(state);
            state.version = 1;
            if (amio_query_operation(operation, &state) == AMIO_OK && (state.milestones & AMIO_OPERATION_BACKEND_COMPLETE)) ++completed_before_end;
        }
        if (status == AMIO_OK) {
            step_metadata_begin = Clock::now();
            failure_stage = "amio_end_step";
            status = amio_end_step(step);
            const auto end_elapsed = seconds(Clock::now() - step_metadata_begin);
            end_time += end_elapsed;
            metadata += end_elapsed;
        }
        for (auto operation : operations) {
            if (status == AMIO_OK) {
                const auto drain_begin = Clock::now();
                failure_stage = "amio_wait_operation";
                status = amio_wait_operation(operation, AMIO_OPERATION_BACKEND_COMPLETE, 600000);
                const auto elapsed_wait = seconds(Clock::now() - drain_begin);
                payload += elapsed_wait;
                post_wait_time += elapsed_wait;
            }
            (void)amio_release_operation(operation);
        }
        if (step) (void)amio_release_step_handle(step);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto validation_begin = Clock::now();
    if (rank == 0 && status == AMIO_OK) {
        std::vector<float> read_values(static_cast<std::size_t>(w.cells));
        for (std::uint64_t step_id = 0; status == AMIO_OK && step_id < w.steps; ++step_id) {
            const auto digits = std::to_string(step_id);
            const auto path = output / "payloads" / "steps" / (std::string(20 - digits.size(), '0') + digits + ".nc");
            int ncid = -1;
            failure_stage = "readback";
            if (nc_open(path.c_str(), NC_NOWRITE, &ncid) != NC_NOERR) {
                status = AMIO_ERR_BACKEND_FAILURE;
                break;
            }
            for (std::uint64_t field = 0; status == AMIO_OK && field < w.fields; ++field) {
                const auto &name = names[static_cast<std::size_t>(field)];
                int varid = -1;
                if (nc_inq_varid(ncid, name.c_str(), &varid) != NC_NOERR || nc_get_var_float(ncid, varid, read_values.data()) != NC_NOERR)
                    status = AMIO_ERR_BACKEND_FAILURE;
                for (std::uint64_t cell = 0; status == AMIO_OK && cell < w.cells; ++cell)
                    if (read_values[static_cast<std::size_t>(cell)] != amio::benchmark::value(w.seed, step_id, field, cell))
                        status = AMIO_ERR_BACKEND_FAILURE;
                std::size_t required = 0;
                if (status == AMIO_OK) {
                    status = amio_read_step_field(dataset, step_id, name.data(), name.size(), nullptr, read_values.data(),
                                                  read_values.size() * sizeof(float), &required);
                    if (status == AMIO_OK && required != read_values.size() * sizeof(float)) status = AMIO_ERR_BACKEND_FAILURE;
                    for (std::uint64_t cell = 0; status == AMIO_OK && cell < w.cells; ++cell)
                        if (read_values[static_cast<std::size_t>(cell)] != amio::benchmark::value(w.seed, step_id, field, cell))
                            status = AMIO_ERR_BACKEND_FAILURE;
                }
            }
            if (nc_close(ncid) != NC_NOERR && status == AMIO_OK) status = AMIO_ERR_BACKEND_FAILURE;
        }
        if (status == AMIO_OK) {
            amio_validation_request_t request{};
            request.struct_size = sizeof(request);
            request.version = 1;
            request.check_mask = UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1);
            request.max_results = 1;
            amio_validation_result_t result{};
            std::size_t result_count = 0;
            std::uint64_t next = 0;
            failure_stage = "integrity validation";
            status = amio_validate_dataset(dataset, &request, &result, 1, &result_count, &next);
            if (status == AMIO_OK && (result_count != 1 || result.status != AMIO_VALIDATION_PASS)) status = AMIO_ERR_BACKEND_FAILURE;
        }
    }
    int validation_code = static_cast<int>(status);
    MPI_Bcast(&validation_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (status == AMIO_OK) status = static_cast<amio_status_t>(validation_code);
    const double validation_time = seconds(Clock::now() - validation_begin);
    const auto close_begin = Clock::now();
    if (dataset) {
        const auto close_status = amio_close_dataset_ex(dataset);
        if (status == AMIO_OK) status = close_status;
    }
    close_time = seconds(Clock::now() - close_begin);
    if (schema) (void)amio_release_schema(schema);
    if (context) (void)amio_release_context(context);
    const double elapsed = reduce_max_time(seconds(Clock::now() - full_begin) - validation_time);
    metadata = reduce_max_time(metadata);
    payload = reduce_max_time(payload);
    submit_time = reduce_max_time(submit_time);
    post_wait_time = reduce_max_time(post_wait_time);
    begin_time = reduce_max_time(begin_time);
    end_time = reduce_max_time(end_time);
    close_time = reduce_max_time(close_time);
    std::uint64_t total_completed = 0, total_submitted = 0;
    MPI_Reduce(&completed_before_end, &total_completed, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&submitted, &total_submitted, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
    if (status == AMIO_OK && rank == 0)
        std::cout << "{\"mode\":\"amio-direct\",\"diagnostics\":\"" << (diagnostics_enabled ? "enabled" : "disabled")
                  << "\",\"logical_layout\":\"steps-fields-cells-1d\",\"layout\":\"AMIO-step-collection\",\"operators\":\"none\","
                  << "\"completion\":\"reader-visible\",\"steps\":" << w.steps << ",\"fields\":" << w.fields << ",\"cells\":" << w.cells
                  << ",\"ranks\":" << ranks << ",\"progress\":\"" << (background ? "background" : "caller")
                  << "\",\"compute_ms_per_step\":" << compute_ms << ",\"completed_before_end\":" << total_completed
                  << ",\"submitted\":" << total_submitted << ",\"submit_seconds\":" << submit_time << ",\"post_end_wait_seconds\":" << post_wait_time
                  << ",\"begin_seconds\":" << begin_time << ",\"end_seconds\":" << end_time << ",\"validation_seconds\":" << validation_time
                  << ",\"close_seconds\":" << close_time << ",\"budget_bytes\":" << dataset_desc.budget_bytes
                  << ",\"setup_metadata_seconds\":" << metadata << ",\"payload_drain_seconds\":" << payload << ",\"end_to_end_seconds\":" << elapsed
                  << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float) << "}\n";
    else if (status != AMIO_OK)
        std::cerr << "AMIO direct benchmark failed at " << failure_stage << " with status " << status << '\n';
    MPI_Finalize();
    return status == AMIO_OK ? 0 : 2;
}
// Matched NetCDF baseline: multi-rank runs use parallel NetCDF; one-rank runs
// use serial NetCDF to match AMIO's single-rank backend path.
int parallel_netcdf(const amio::benchmark::Workload &w, const std::filesystem::path &output) {
    int initialized = 0;
    MPI_Initialized(&initialized);
    int provided = MPI_THREAD_SINGLE;
    if (!initialized && (MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED))
        return 2;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool parallel_io = size > 1;
    int result = 0;
    double setup_seconds = 0.0, write_seconds = 0.0, validate_seconds = 0.0;
    try {
        const auto ranks = static_cast<std::uint64_t>(size);
        const auto me = static_cast<std::uint64_t>(rank);
        const std::uint64_t first = me * (w.cells / ranks) + std::min<std::uint64_t>(me, w.cells % ranks);
        const std::uint64_t count = w.cells / ranks + (me < w.cells % ranks ? 1 : 0);
        const std::size_t start_index = static_cast<std::size_t>(first);
        const std::size_t count_size = static_cast<std::size_t>(count);

        if (rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(output, ec);
            if (ec) throw std::runtime_error("cannot create benchmark output directory");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const std::string path = (output / "parallel.nc").string();

        const std::vector<std::string> names = build_step_field_names(w.steps, w.fields);

        int ncid = -1, cell_dim = -1;
        std::vector<int> vars(names.size());
        const auto setup_begin = Clock::now();
        if (parallel_io)
            check_nc(nc_create_par(path.c_str(), NC_NETCDF4 | NC_MPIIO | NC_CLOBBER, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid), "create");
        else
            check_nc(nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid), "create");
        check_nc(nc_def_dim(ncid, "cell", static_cast<std::size_t>(w.cells), &cell_dim), "define cell");
        for (std::size_t v = 0; v < names.size(); ++v) {
            check_nc(nc_def_var(ncid, names[v].c_str(), NC_FLOAT, 1, &cell_dim, &vars[v]), "define variable");
            if (parallel_io) check_nc(nc_var_par_access(ncid, vars[v], NC_COLLECTIVE), "set collective access");
        }
        check_nc(nc_enddef(ncid), "enddef");
        setup_seconds = seconds(Clock::now() - setup_begin);

        std::vector<float> values(static_cast<std::size_t>(count));
        MPI_Barrier(MPI_COMM_WORLD);
        const auto write_begin = Clock::now();
        std::size_t v = 0;
        for (std::uint64_t step = 0; step < w.steps; ++step)
            for (std::uint64_t field = 0; field < w.fields; ++field, ++v) {
                for (std::uint64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = amio::benchmark::value(w.seed, step, field, first + i);
                check_nc(nc_put_vara_float(ncid, vars[v], &start_index, &count_size, values.data()), "write");
            }
        check_nc(nc_close(ncid), "close");
        MPI_Barrier(MPI_COMM_WORLD);
        write_seconds = seconds(Clock::now() - write_begin);

        validate_netcdf_benchmark_readback(path, names, w, first, count, parallel_io, "parallel-netcdf reference mismatch", validate_seconds);

        const double max_setup = reduce_max_time(setup_seconds);
        const double max_write = reduce_max_time(write_seconds);
        const double max_validate = reduce_max_time(validate_seconds);
        if (rank == 0) {
            std::cout << "{\"mode\":\"parallel-netcdf\",\"status\":\"PASS\",\"logical_layout\":\"steps-fields-cells-1d\","
                      << "\"layout\":\"one-netcdf4-hdf5-file-" << (parallel_io ? "collective" : "serial")
                      << "-hyperslabs\",\"operators\":\"none\",\"diagnostics\":\"not-applicable\","
                      << "\"completion\":\"nc_close-and-reference-validated\",\"ranks\":" << size << ",\"steps\":" << w.steps
                      << ",\"fields\":" << w.fields << ",\"cells\":" << w.cells << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float)
                      << ",\"setup_metadata_seconds\":" << max_setup << ",\"write_close_seconds\":" << max_write
                      << ",\"validation_seconds\":" << max_validate << "}\n";
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

// Same variables, layout, rank decomposition, and timing boundaries as parallel_netcdf.
// The experimental batch path groups all variable snapshots into one AMIO task.
int amio_region(const amio::benchmark::Workload &w, const std::filesystem::path &output, int worker_threads, bool batch) {
    constexpr int64_t kTimeoutMs = 60000;

    const char *compute_setting = std::getenv("AMIO_BENCHMARK_COMPUTE_MS");
    const int compute_ms = compute_setting ? std::atoi(compute_setting) : 0;
    if (compute_ms < 0 || compute_ms > 60000) return 2;
    auto compute = [compute_ms]() {
        const auto until = Clock::now() + std::chrono::milliseconds(compute_ms);
        volatile std::uint64_t work = 1;
        while (Clock::now() < until) work = work * 1664525 + 1013904223;
    };
    int initialized = 0;
    MPI_Initialized(&initialized);
    int provided = MPI_THREAD_SINGLE;
    if (!initialized)
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    else
        MPI_Query_thread(&provided);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool parallel_io = size > 1;
    int result = 0;
    double setup_seconds = 0.0, write_seconds = 0.0, validate_seconds = 0.0;
    double submit_seconds = 0.0, wait_seconds = 0.0, flush_seconds = 0.0, close_seconds = 0.0;
    try {
        if (provided != MPI_THREAD_MULTIPLE) throw std::runtime_error("amio-region requires MPI_THREAD_MULTIPLE");
        const auto ranks = static_cast<std::uint64_t>(size);
        const auto me = static_cast<std::uint64_t>(rank);
        const std::uint64_t first = me * (w.cells / ranks) + std::min<std::uint64_t>(me, w.cells % ranks);
        const std::uint64_t count = w.cells / ranks + (me < w.cells % ranks ? 1 : 0);
        const std::uint64_t max_count = w.cells / ranks + (w.cells % ranks != 0 ? 1 : 0);
        const std::uint64_t writes = std::max<std::uint64_t>(w.steps * w.fields, 1);
        if (worker_threads < 0 || worker_threads > 128) throw std::runtime_error("amio-region worker count must be between 0 and 128");
        if (writes > 4096) throw std::runtime_error("amio-region supports at most 4096 writes per rank");

        if (rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(output, ec);
            if (ec) throw std::runtime_error("cannot create benchmark output directory");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const std::string path = (output / "amio.nc").string();

        const std::vector<std::string> names = build_step_field_names(w.steps, w.fields);

        // Staging holds every write of a rank at once, so admission never waits on the pool.
        const char *buffer_setting = std::getenv("AMIO_BENCHMARK_STAGING_BUFFERS");
        const auto requested_buffers = buffer_setting ? std::strtoull(buffer_setting, nullptr, 10) : writes;
        if (requested_buffers == 0 || requested_buffers > writes) throw std::runtime_error("invalid benchmark staging buffer count");
        const std::uint64_t buffer_count = batch ? 1 : requested_buffers;
        if (batch && writes > std::numeric_limits<std::uint64_t>::max() / max_count / sizeof(float))
            throw std::runtime_error("batch staging capacity overflows");
        const std::uint64_t buffer_capacity = max_count * (batch ? writes : 1) * sizeof(float);
        const std::string buffers = std::to_string(buffer_count);
        const std::string capacity = std::to_string(buffer_capacity);
        const std::string manifest = "backend: netcdf4\npath: " + path + "\ndata_model: enhanced\nstaging_pool: {buffer_count: " + buffers +
                                     ", buffer_capacity_bytes: " + capacity + ", max_buffer_count: " + buffers +
                                     "}\nworker_pool: {threads: " + std::to_string(worker_threads) + "}\n";

        amio_core_handle core = nullptr;
        amio_dataset_handle dataset = nullptr;
        const auto setup_begin = Clock::now();
        check_amio(amio_init_from_string(manifest.c_str(), "yaml", &core), "init");
        check_amio(amio_open_dataset_from_string(core, manifest.c_str(), "yaml", AMIO_MODE_WRITE, &dataset), "open shared output");
        std::vector<amio_write_variable_t> schema(names.size());
        for (std::size_t i = 0; i < names.size(); ++i) {
            auto &variable = schema[i];
            variable.struct_size = static_cast<std::uint32_t>(sizeof(variable));
            variable.version = AMIO_WRITE_VARIABLE_VERSION;
            variable.name = names[i].c_str();
            variable.dtype = AMIO_DTYPE_F32;
            variable.rank = 1;
            variable.global_extents[0] = static_cast<int64_t>(w.cells);
        }
        check_amio(amio_prepare_write_schema(dataset, schema.data(), schema.size()), "prepare shared output schema");
        setup_seconds = seconds(Clock::now() - setup_begin);

        amio_shape_t shape{};
        shape.rank = 1;
        shape.extents[0] = static_cast<int64_t>(count);
        amio_write_region_t region{};
        region.rank = 1;
        region.global_extents[0] = static_cast<int64_t>(w.cells);
        region.offsets[0] = static_cast<int64_t>(first);

        std::vector<float> values(static_cast<std::size_t>(count));
        std::vector<std::vector<float>> batch_values;
        std::vector<amio_region_write_t> batch_writes;
        if (batch) {
            batch_values.reserve(names.size());
            for (std::size_t i = 0; i < names.size(); ++i) batch_values.emplace_back(static_cast<std::size_t>(count));
            batch_writes.resize(names.size());
        }
        std::vector<amio_io_handle> pending;
        pending.reserve(batch ? 1 : names.size());
        MPI_Barrier(MPI_COMM_WORLD);
        const auto write_begin = Clock::now();
        if (batch) {
            for (std::size_t v = 0; v < names.size(); ++v) {
                const std::uint64_t step = v / w.fields, field = v % w.fields;
                auto &field_values = batch_values[v];
                for (std::uint64_t i = 0; i < count; ++i)
                    field_values[static_cast<std::size_t>(i)] = amio::benchmark::value(w.seed, step, field, first + i);
                batch_writes[v] = {names[v].c_str(), field_values.data(), AMIO_DTYPE_F32, &shape, &region};
            }
            amio_io_handle io = nullptr;
            auto phase = Clock::now();
            check_amio(amio_write_region_batch(dataset, batch_writes.data(), batch_writes.size(), &io), "enqueue region batch");
            submit_seconds += seconds(Clock::now() - phase);
            compute();
            phase = Clock::now();
            check_amio(amio_wait(io, kTimeoutMs), "wait for region batch");
            wait_seconds += seconds(Clock::now() - phase);
        } else {
            std::size_t v = 0;
            for (std::uint64_t step = 0; step < w.steps; ++step)
                for (std::uint64_t field = 0; field < w.fields; ++field, ++v) {
                    for (std::uint64_t i = 0; i < count; ++i)
                        values[static_cast<std::size_t>(i)] = amio::benchmark::value(w.seed, step, field, first + i);
                    amio_io_handle io = nullptr;
                    const auto phase = Clock::now();
                    check_amio(amio_write_region(dataset, names[v].c_str(), values.data(), AMIO_DTYPE_F32, &shape, &region, &io), "enqueue region");
                    submit_seconds += seconds(Clock::now() - phase);
                    pending.push_back(io);
                }
            compute();
            const auto phase = Clock::now();
            for (auto io : pending) check_amio(amio_wait(io, kTimeoutMs), "wait for region");
            wait_seconds += seconds(Clock::now() - phase);
        }
        auto phase = Clock::now();
        if (std::getenv("AMIO_BENCHMARK_SKIP_FLUSH") == nullptr) check_amio(amio_flush(dataset, kTimeoutMs), "flush");
        flush_seconds = seconds(Clock::now() - phase);
        phase = Clock::now();
        check_amio(amio_close_dataset(dataset), "close");
        close_seconds = seconds(Clock::now() - phase);
        MPI_Barrier(MPI_COMM_WORLD);
        write_seconds = seconds(Clock::now() - write_begin);
        check_amio(amio_finalize(core), "finalize");

        validate_netcdf_benchmark_readback(path, names, w, first, count, parallel_io, "amio-region reference mismatch", validate_seconds);

        const double max_setup = reduce_max_time(setup_seconds);
        const double max_write = reduce_max_time(write_seconds);
        const double max_validate = reduce_max_time(validate_seconds);
        const double max_submit = reduce_max_time(submit_seconds);
        const double max_wait = reduce_max_time(wait_seconds);
        const double max_flush = reduce_max_time(flush_seconds);
        const double max_close = reduce_max_time(close_seconds);
        if (rank == 0) {
            std::cout << "{\"mode\":\"" << (batch ? "amio-region-batched" : "amio-region")
                      << "\",\"status\":\"PASS\",\"logical_layout\":\"steps-fields-cells-1d\","
                      << "\"layout\":\"one-netcdf4-file-amio-region-" << (batch ? "batched" : "individual")
                      << "-writes\",\"operators\":\"none\",\"diagnostics\":\"not-applicable\","
                      << "\"completion\":\"flush-close-and-reference-validated\",\"persistent_storage_exercised\":true,\"ranks\":" << size
                      << ",\"compute_ms\":" << compute_ms << ",\"worker_threads\":" << worker_threads << ",\"steps\":" << w.steps
                      << ",\"fields\":" << w.fields << ",\"cells\":" << w.cells << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float)
                      << ",\"setup_metadata_seconds\":" << max_setup << ",\"write_close_seconds\":" << max_write
                      << ",\"validation_seconds\":" << max_validate << ",\"submit_seconds\":" << max_submit << ",\"wait_seconds\":" << max_wait
                      << ",\"flush_seconds\":" << max_flush << ",\"close_seconds\":" << max_close << "}\n";
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
}  // namespace

int main(int argc, char **argv) {
    if ((argc != 9 && argc != 11) || std::string(argv[1]) != "--mode" || std::string(argv[3]) != "--manifest" || std::string(argv[5]) != "--output" ||
        std::string(argv[7]) != "--diagnostics" || (std::string(argv[8]) != "enabled" && std::string(argv[8]) != "disabled")) {
        std::cerr << "Usage: amio_benchmark --mode direct-reference|amio-direct|amio-aggregated|parallel-netcdf|amio-region|amio-region-batched "
                     "--manifest FILE --output PATH "
                     "--diagnostics enabled|disabled [--worker-threads N (AMIO region modes only)]\n";
        return 2;
    }
    const std::string mode(argv[2]);
    int worker_threads = 4;
    if (argc == 11) {
        if ((mode != "amio-region" && mode != "amio-region-batched") || std::string(argv[9]) != "--worker-threads") {
            std::cerr << "--worker-threads is only valid for AMIO region modes\n";
            return 2;
        }
        const std::string value(argv[10]);
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), worker_threads);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || worker_threads < 0 || worker_threads > 128) {
            std::cerr << "Worker thread count must be an integer between 0 and 128\n";
            return 2;
        }
    }
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
    if (mode == "parallel-netcdf") return parallel_netcdf(workload, output);
    if (mode == "amio-region") return amio_region(workload, output, worker_threads, false);
    if (mode == "amio-region-batched") return amio_region(workload, output, worker_threads, true);
    std::cerr << "unknown benchmark mode\n";
    return 2;
}
