#include <amio/amio.h>
#include <amio/amio_evolution.h>
#include <mpi.h>
#include <netcdf.h>
#include <netcdf_par.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
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
        const auto check = [](int status, const char *where) {
            if (status != NC_NOERR) throw std::runtime_error(std::string(where) + ": " + nc_strerror(status));
        };

        if (rank == 0) {
            std::error_code ec;
            std::filesystem::create_directories(output, ec);
            if (ec) throw std::runtime_error("cannot create benchmark output directory");
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const std::string path = (output / "parallel.nc").string();

        std::vector<std::string> names;
        for (std::uint64_t step = 0; step < w.steps; ++step)
            for (std::uint64_t field = 0; field < w.fields; ++field) names.push_back("s" + std::to_string(step) + "_f" + std::to_string(field));

        int ncid = -1, cell_dim = -1;
        std::vector<int> vars(names.size());
        const auto setup_begin = Clock::now();
        if (parallel_io)
            check(nc_create_par(path.c_str(), NC_NETCDF4 | NC_MPIIO | NC_CLOBBER, MPI_COMM_WORLD, MPI_INFO_NULL, &ncid), "create");
        else
            check(nc_create(path.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid), "create");
        check(nc_def_dim(ncid, "cell", static_cast<std::size_t>(w.cells), &cell_dim), "define cell");
        for (std::size_t v = 0; v < names.size(); ++v) {
            check(nc_def_var(ncid, names[v].c_str(), NC_FLOAT, 1, &cell_dim, &vars[v]), "define variable");
            if (parallel_io) check(nc_var_par_access(ncid, vars[v], NC_COLLECTIVE), "set collective access");
        }
        check(nc_enddef(ncid), "enddef");
        setup_seconds = seconds(Clock::now() - setup_begin);

        std::vector<float> values(static_cast<std::size_t>(count));
        MPI_Barrier(MPI_COMM_WORLD);
        const auto write_begin = Clock::now();
        std::size_t v = 0;
        for (std::uint64_t step = 0; step < w.steps; ++step)
            for (std::uint64_t field = 0; field < w.fields; ++field, ++v) {
                for (std::uint64_t i = 0; i < count; ++i)
                    values[static_cast<std::size_t>(i)] = amio::benchmark::value(w.seed, step, field, first + i);
                check(nc_put_vara_float(ncid, vars[v], &start_index, &count_size, values.data()), "write");
            }
        check(nc_close(ncid), "close");
        MPI_Barrier(MPI_COMM_WORLD);
        write_seconds = seconds(Clock::now() - write_begin);

        // Read-back validation happens outside the timed region.
        const auto validate_begin = Clock::now();
        int readid = -1;
        if (parallel_io)
            check(nc_open_par(path.c_str(), NC_NOWRITE | NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &readid), "reopen");
        else
            check(nc_open(path.c_str(), NC_NOWRITE, &readid), "reopen");
        for (std::size_t r = 0; r < names.size(); ++r) {
            int varid = -1;
            check(nc_inq_varid(readid, names[r].c_str(), &varid), "inquire variable");
            if (parallel_io) check(nc_var_par_access(readid, varid, NC_COLLECTIVE), "read collective access");
            check(nc_get_vara_float(readid, varid, &start_index, &count_size, values.data()), "read");
            const std::uint64_t step = r / w.fields, field = r % w.fields;
            for (std::uint64_t i = 0; i < count; ++i)
                if (values[static_cast<std::size_t>(i)] != amio::benchmark::value(w.seed, step, field, first + i))
                    throw std::runtime_error("parallel-netcdf reference mismatch");
        }
        check(nc_close(readid), "reopen close");
        validate_seconds = seconds(Clock::now() - validate_begin);

        double max_setup = 0, max_write = 0, max_validate = 0;
        MPI_Reduce(&setup_seconds, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&write_seconds, &max_write, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&validate_seconds, &max_validate, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
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
void check_amio(amio_status_t status, const char *where) {
    if (status != AMIO_OK) throw std::runtime_error(std::string(where) + ": AMIO status " + std::to_string(static_cast<int>(status)));
}

void check_nc(int status, const char *where) {
    if (status != NC_NOERR) throw std::runtime_error(std::string(where) + ": " + nc_strerror(status));
}

// Same variables, layout, rank decomposition, and timing boundaries as parallel_netcdf.
// The experimental batch path groups all variable snapshots into one AMIO task.
int amio_region(const amio::benchmark::Workload &w, const std::filesystem::path &output, int worker_threads, bool batch) {
    constexpr int64_t kTimeoutMs = 60000;

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
    int result = 0;
    double setup_seconds = 0.0, write_seconds = 0.0, validate_seconds = 0.0;
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

        std::vector<std::string> names;
        for (std::uint64_t step = 0; step < w.steps; ++step)
            for (std::uint64_t field = 0; field < w.fields; ++field) names.push_back("s" + std::to_string(step) + "_f" + std::to_string(field));

        // Staging holds every write of a rank at once, so admission never waits on the pool.
        const std::uint64_t buffer_count = batch ? 1 : writes;
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
            check_amio(amio_write_region_batch(dataset, batch_writes.data(), batch_writes.size(), &io), "enqueue region batch");
            check_amio(amio_wait(io, kTimeoutMs), "wait for region batch");
        } else {
            std::size_t v = 0;
            for (std::uint64_t step = 0; step < w.steps; ++step)
                for (std::uint64_t field = 0; field < w.fields; ++field, ++v) {
                    for (std::uint64_t i = 0; i < count; ++i)
                        values[static_cast<std::size_t>(i)] = amio::benchmark::value(w.seed, step, field, first + i);
                    amio_io_handle io = nullptr;
                    check_amio(amio_write_region(dataset, names[v].c_str(), values.data(), AMIO_DTYPE_F32, &shape, &region, &io), "enqueue region");
                    pending.push_back(io);
                }
            for (auto io : pending) check_amio(amio_wait(io, kTimeoutMs), "wait for region");
        }
        check_amio(amio_flush(dataset, kTimeoutMs), "flush");
        check_amio(amio_close_dataset(dataset), "close");
        MPI_Barrier(MPI_COMM_WORLD);
        write_seconds = seconds(Clock::now() - write_begin);
        check_amio(amio_finalize(core), "finalize");

        // Read-back validation happens outside the timed region.
        const auto validate_begin = Clock::now();
        const std::size_t start_index = static_cast<std::size_t>(first);
        const std::size_t count_size = static_cast<std::size_t>(count);
        int readid = -1;
        check_nc(nc_open_par(path.c_str(), NC_NOWRITE | NC_MPIIO, MPI_COMM_WORLD, MPI_INFO_NULL, &readid), "reopen");
        for (std::size_t r = 0; r < names.size(); ++r) {
            int varid = -1;
            check_nc(nc_inq_varid(readid, names[r].c_str(), &varid), "inquire variable");
            check_nc(nc_var_par_access(readid, varid, NC_COLLECTIVE), "read collective access");
            check_nc(nc_get_vara_float(readid, varid, &start_index, &count_size, values.data()), "read");
            const std::uint64_t step = r / w.fields, field = r % w.fields;
            for (std::uint64_t i = 0; i < count; ++i)
                if (values[static_cast<std::size_t>(i)] != amio::benchmark::value(w.seed, step, field, first + i))
                    throw std::runtime_error("amio-region reference mismatch");
        }
        check_nc(nc_close(readid), "reopen close");
        validate_seconds = seconds(Clock::now() - validate_begin);

        double max_setup = 0, max_write = 0, max_validate = 0;
        MPI_Reduce(&setup_seconds, &max_setup, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&write_seconds, &max_write, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&validate_seconds, &max_validate, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "{\"mode\":\"" << (batch ? "amio-region-batched" : "amio-region")
                      << "\",\"status\":\"PASS\",\"logical_layout\":\"steps-fields-cells-1d\","
                      << "\"layout\":\"one-netcdf4-file-amio-region-" << (batch ? "batched" : "individual")
                      << "-writes\",\"operators\":\"none\",\"diagnostics\":\"not-applicable\","
                      << "\"completion\":\"flush-close-and-reference-validated\",\"persistent_storage_exercised\":true,\"ranks\":" << size
                      << ",\"worker_threads\":" << worker_threads << ",\"steps\":" << w.steps << ",\"fields\":" << w.fields
                      << ",\"cells\":" << w.cells << ",\"bytes\":" << w.steps * w.fields * w.cells * sizeof(float)
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
