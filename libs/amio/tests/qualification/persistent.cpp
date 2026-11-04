#include <amio/amio_evolution.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "native_readers.hpp"
#include "reference_fields.hpp"

namespace {
constexpr std::uint64_t kSteps = 100;
constexpr std::size_t kFields = 20;

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void require_status(amio_status_t status, const char *operation) {
    if (status != AMIO_OK) throw std::runtime_error(std::string(operation) + " returned AMIO status " + std::to_string(status));
}

std::uint64_t cells_for_step(std::uint64_t step) {
    return (step / 25) % 2 == 0 ? 37 : 41;
}

}  // namespace

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS || provided < MPI_THREAD_FUNNELED) return 1;

    amio_context_handle context = nullptr;
    amio_dataset_handle dataset = nullptr;
    amio_schema_handle schema_handle = nullptr;
    amio_step_handle active_step = nullptr;
    std::vector<amio_operation_handle> operations;
    std::filesystem::path root;
    int result = 0;
    int world_rank = 0, world_size = 0;
    try {
        require(
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank) == MPI_SUCCESS && MPI_Comm_size(MPI_COMM_WORLD, &world_size) == MPI_SUCCESS && world_size >= 1,
            "direct qualification requires at least one MPI rank");
        char root_buffer[1024]{};
        if (world_rank == 0) {
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            const auto path = (std::filesystem::temp_directory_path() / ("amio-us1-persistent-" + std::to_string(nonce))).string();
            require(path.size() < sizeof(root_buffer), "temporary qualification path is too long");
            std::copy(path.begin(), path.end(), root_buffer);
        }
        require(MPI_Bcast(root_buffer, sizeof(root_buffer), MPI_CHAR, 0, MPI_COMM_WORLD) == MPI_SUCCESS,
                "broadcasting qualification collection path failed");
        root = root_buffer;
        amio_context_descriptor_t context_desc{};
        context_desc.struct_size = sizeof(context_desc);
        context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
        context_desc.progress_mode = AMIO_PROGRESS_CALLER;
        context_desc.rank_count = static_cast<std::uint32_t>(world_size);
        static constexpr char config[] = "{}";
        context_desc.canonical_configuration = config;
        context_desc.configuration_bytes = sizeof(config) - 1;
        context_desc.fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
        require_status(amio_init_ex(&context_desc, &context), "amio_init_ex");

        const auto root_text = root.string();
        amio_dataset_descriptor_t dataset_desc{};
        dataset_desc.struct_size = sizeof(dataset_desc);
        dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        dataset_desc.location = root_text.c_str();
        dataset_desc.location_bytes = root_text.size();
        dataset_desc.format = AMIO_FORMAT_NETCDF4;
        dataset_desc.budget_bytes = 64 * 1024 * 1024;
        dataset_desc.budget_descriptors = 4096;
        dataset_desc.control_budget_bytes = 1024 * 1024;
        dataset_desc.control_budget_descriptors = 256;
        require_status(amio_open_dataset_ex(context, &dataset_desc, &dataset), "amio_open_dataset_ex");

        std::uint64_t active_cells = 0;
        for (std::uint64_t step_id = 0; step_id < kSteps; ++step_id) {
            const auto cells = cells_for_step(step_id);
            if (step_id == 0 || cells != active_cells) {
                if (schema_handle != nullptr) require_status(amio_release_schema(schema_handle), "amio_release_schema");
                active_cells = cells;
                const char dimension_name[] = "cell";
                amio_dimension_descriptor_t dimension{};
                dimension.struct_size = sizeof(dimension);
                dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                dimension.name = dimension_name;
                dimension.name_bytes = sizeof(dimension_name) - 1;
                dimension.extent = cells;
                std::array<std::string, kFields> names;
                std::array<amio_variable_descriptor_t, kFields> variables{};
                for (std::size_t field = 0; field < kFields; ++field) {
                    names[field] = amio::qualification::field_name(field);
                    auto &variable = variables[field];
                    variable.struct_size = sizeof(variable);
                    variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                    variable.name = names[field].data();
                    variable.name_bytes = names[field].size();
                    variable.source_dtype = AMIO_DTYPE_F32;
                    variable.stored_dtype = AMIO_DTYPE_F32;
                    variable.required = field + 1 == kFields ? 0 : 1;
                    variable.dimension_count = 1;
                    variable.dimensions = &dimension;
                }
                amio_schema_descriptor_t schema_desc{};
                schema_desc.struct_size = sizeof(schema_desc);
                schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                schema_desc.variable_count = variables.size();
                schema_desc.variables = variables.data();
                require_status(amio_define_schema(dataset, &schema_desc, &schema_handle), "amio_define_schema");
            }

            amio_step_descriptor_t step_desc{};
            step_desc.struct_size = sizeof(step_desc);
            step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            step_desc.step_id = step_id;
            step_desc.model_time = static_cast<double>(kSteps - step_id) * 60.0;
            step_desc.schema = schema_handle;
            std::array<std::string, kFields> names;
            std::array<amio_field_partition_descriptor_t, kFields> partitions{};
            const std::uint64_t active_ranks = static_cast<std::uint64_t>(world_size == 1 ? 1 : world_size - 1);
            const std::uint64_t producer_rank = static_cast<std::uint64_t>(world_rank);
            const std::uint64_t quotient = cells / active_ranks;
            const std::uint64_t remainder = cells % active_ranks;
            const std::uint64_t local_cells =
                world_size == 1 ? cells : (producer_rank < active_ranks ? quotient + (producer_rank < remainder ? 1 : 0) : 0);
            const std::uint64_t local_offset = producer_rank < active_ranks ? producer_rank * quotient + std::min(producer_rank, remainder) : cells;
            for (std::size_t field = 0; field < kFields; ++field) {
                names[field] = amio::qualification::field_name(field);
                auto &partition = partitions[field];
                partition.struct_size = sizeof(partition);
                partition.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                partition.field = names[field].data();
                partition.field_bytes = names[field].size();
                partition.rank = 1;
                partition.offsets[0] = static_cast<std::int64_t>(local_offset);
                partition.extents[0] = static_cast<std::int64_t>(local_cells);
            }
            if (world_size == 1)
                require_status(amio_begin_step(dataset, &step_desc, &active_step), "amio_begin_step");
            else
                require_status(amio_begin_step_partitioned(dataset, &step_desc, partitions.data(), partitions.size(), &active_step),
                               "amio_begin_step_partitioned");
            operations.clear();
            const auto cells_now = static_cast<std::size_t>(local_cells);
            for (std::size_t field = 0; field < kFields; ++field) {
                if (local_cells == 0 || (field + 1 == kFields && step_id % 3 == 0)) continue;
                std::vector<float> source(cells_now);
                for (std::uint64_t cell = 0; cell < local_cells; ++cell)
                    source[static_cast<std::size_t>(cell)] = amio::qualification::reference_value(step_id, field, local_offset + cell);
                amio_operation_handle operation = nullptr;
                require_status(
                    amio_put_field(active_step, names[field].data(), names[field].size(), source.data(), source.size() * sizeof(float), &operation),
                    "amio_put_field");
                std::fill(source.begin(), source.end(), -9999.0F);  // prove immediate source reuse after snapshot admission
                require_status(amio_wait_operation(operation, AMIO_OPERATION_SOURCE_RELEASED, 0), "wait source released");
                operations.push_back(operation);
            }
            require_status(amio_end_step(active_step), "amio_end_step");
            for (auto operation : operations) {
                amio_operation_status_t op_status{};
                require_status(amio_query_operation(operation, &op_status), "amio_query_operation");
                require(op_status.outcome == AMIO_OPERATION_SUCCEEDED && (op_status.milestones & AMIO_OPERATION_READER_VISIBLE) != 0,
                        "step did not reach reader-visible success");
                require_status(amio_release_operation(operation), "amio_release_operation");
            }
            operations.clear();
            require_status(amio_release_step_handle(active_step), "amio_release_step_handle");
            active_step = nullptr;

            if (world_rank == 0) {
                amio_step_summary_t step_summary{};
                require_status(amio_describe_step(dataset, step_id, &step_summary), "amio_describe_step");
                const auto expected_present = static_cast<std::uint32_t>(kFields - ((step_id % 3 == 0) ? 1u : 0u));
                require(step_summary.step_id == step_id && step_summary.required_field_count == kFields - 1 &&
                            step_summary.present_field_count == expected_present,
                        "published step required/present-field metadata mismatch");
                const auto expected_time = static_cast<double>(kSteps - step_id) * 60.0;
                for (std::size_t field = 0; field < kFields; ++field) {
                    if (field + 1 == kFields && step_id % 3 == 0) continue;
                    amio::qualification::verify_native_netcdf_field(root, step_id, cells, field, expected_time, step_summary.schema_digest);
                }
            }
        }

        // Collective cancellation leaves no catalog commit and releases the
        // partitioned step reservation on every rank.
        std::array<std::string, kFields> cancel_names;
        std::array<amio_field_partition_descriptor_t, kFields> cancel_partitions{};
        constexpr std::uint64_t cancel_cells = 41;
        const std::uint64_t active_ranks = static_cast<std::uint64_t>(world_size == 1 ? 1 : world_size - 1);
        const std::uint64_t producer_rank = static_cast<std::uint64_t>(world_rank);
        const std::uint64_t quotient = cancel_cells / active_ranks;
        const std::uint64_t remainder = cancel_cells % active_ranks;
        const std::uint64_t local_cells =
            world_size == 1 ? cancel_cells : (producer_rank < active_ranks ? quotient + (producer_rank < remainder ? 1 : 0) : 0);
        const std::uint64_t local_offset =
            producer_rank < active_ranks ? producer_rank * quotient + std::min(producer_rank, remainder) : cancel_cells;
        for (std::size_t field = 0; field < kFields; ++field) {
            cancel_names[field] = amio::qualification::field_name(field);
            auto &part = cancel_partitions[field];
            part.struct_size = sizeof(part);
            part.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            part.field = cancel_names[field].data();
            part.field_bytes = cancel_names[field].size();
            part.rank = 1;
            part.offsets[0] = static_cast<std::int64_t>(local_offset);
            part.extents[0] = static_cast<std::int64_t>(local_cells);
        }
        amio_step_descriptor_t cancel_desc{};
        cancel_desc.struct_size = sizeof(cancel_desc);
        cancel_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        cancel_desc.step_id = kSteps;
        cancel_desc.model_time = 0.0;
        cancel_desc.schema = schema_handle;
        amio_step_handle cancelled_step = nullptr;
        if (world_size == 1)
            require_status(amio_begin_step(dataset, &cancel_desc, &cancelled_step), "begin cancellation step");
        else
            require_status(amio_begin_step_partitioned(dataset, &cancel_desc, cancel_partitions.data(), cancel_partitions.size(), &cancelled_step),
                           "begin collective cancellation step");
        require_status(amio_cancel_step(cancelled_step), "collective amio_cancel_step");
        require_status(amio_release_step_handle(cancelled_step), "release cancelled step handle");

        // The four readers intentionally use a different decomposition from
        // the three writer ranks (writer rank 3 contributed an empty box).
        amio_step_selection_t selection{};
        selection.struct_size = sizeof(selection);
        selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        selection.rank = 1;
        const std::uint64_t reader_ranks = static_cast<std::uint64_t>(world_size);
        const std::uint64_t reader_quotient = cancel_cells / reader_ranks;
        const std::uint64_t reader_remainder = cancel_cells % reader_ranks;
        const std::uint64_t reader_rank = static_cast<std::uint64_t>(world_rank);
        const std::uint64_t reader_cells = reader_quotient + (reader_rank < reader_remainder ? 1 : 0);
        const std::uint64_t reader_offset = reader_rank * reader_quotient + std::min(reader_rank, reader_remainder);
        selection.offsets[0] = static_cast<std::int64_t>(reader_offset);
        selection.extents[0] = static_cast<std::int64_t>(reader_cells);
        selection.strides[0] = 1;
        std::vector<float> selected(static_cast<std::size_t>(reader_cells));
        std::size_t selected_bytes = 0;
        const auto field0 = amio::qualification::field_name(0);
        require_status(amio_read_step_field(dataset, 99, field0.data(), field0.size(), &selection, selected.data(), selected.size() * sizeof(float),
                                            &selected_bytes),
                       "amio_read_step_field");
        require(selected_bytes == selected.size() * sizeof(float), "selected read returned an unexpected byte count");
        for (std::uint64_t i = 0; i < selected.size(); ++i)
            require(selected[static_cast<std::size_t>(i)] == amio::qualification::reference_value(99, 0, reader_offset + i),
                    "AMIO selected-read value mismatch");

        amio_catalog_cursor_t cursor{};
        std::uint64_t listed = 0;
        int has_more = 0;
        do {
            std::array<amio_step_summary_t, 17> page{};
            std::size_t page_size = 0;
            amio_catalog_cursor_t next{};
            require_status(amio_list_steps(dataset, cursor.dataset_uuid[0] == '\0' ? nullptr : &cursor, page.size(), page.data(), page.size(),
                                           &page_size, &next, &has_more),
                           "amio_list_steps");
            listed += page_size;
            cursor = next;
        } while (has_more != 0);
        require(listed == kSteps, "catalog pagination did not enumerate exactly 100 steps");

        amio_validation_request_t validation{};
        validation.struct_size = sizeof(validation);
        validation.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        validation.check_mask = (UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1)) | (UINT64_C(1) << (AMIO_VALIDATE_REQUIRED_FIELDS - 1)) |
                                (UINT64_C(1) << (AMIO_VALIDATE_PAYLOAD_PRESENCE - 1));
        validation.max_results = 3;
        std::array<amio_validation_result_t, 3> checks{};
        std::size_t check_count = 0;
        std::uint64_t check_cursor = 0;
        require_status(amio_validate_dataset(dataset, &validation, checks.data(), checks.size(), &check_count, &check_cursor),
                       "amio_validate_dataset");
        require(check_count == 3, "validation check count differs");
        for (const auto &check : checks) require(check.status == AMIO_VALIDATION_PASS, "structural validation reported failure");

        amio_step_lease_handle lease = nullptr;
        require_status(amio_acquire_step_lease(dataset, 99, &lease), "amio_acquire_step_lease");

        // One producer omits a required field; all ranks must observe the same
        // terminal failure and none may leave the step active or publish it.
        amio_step_descriptor_t failed_desc{};
        failed_desc.struct_size = sizeof(failed_desc);
        failed_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        failed_desc.step_id = kSteps + 1;
        failed_desc.model_time = -60.0;
        failed_desc.schema = schema_handle;
        constexpr std::uint64_t failed_cells = 41;
        const auto producer_ranks = static_cast<std::uint64_t>(world_size == 1 ? 1 : world_size - 1);
        const auto failed_rank = static_cast<std::uint64_t>(world_rank);
        const auto failed_quotient = failed_cells / producer_ranks;
        const auto failed_remainder = failed_cells % producer_ranks;
        const auto failed_local_cells = world_size == 1                ? failed_cells
                                        : failed_rank < producer_ranks ? failed_quotient + (failed_rank < failed_remainder ? 1 : 0)
                                                                       : 0;
        const auto failed_offset = world_size == 1                ? 0
                                   : failed_rank < producer_ranks ? failed_rank * failed_quotient + std::min(failed_rank, failed_remainder)
                                                                  : failed_cells;
        std::array<amio_field_partition_descriptor_t, kFields> failed_partitions{};
        for (std::size_t field = 0; field < kFields; ++field) {
            auto &part = failed_partitions[field];
            part.struct_size = sizeof(part);
            part.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            part.field = cancel_names[field].data();
            part.field_bytes = cancel_names[field].size();
            part.rank = 1;
            part.offsets[0] = static_cast<std::int64_t>(failed_offset);
            part.extents[0] = static_cast<std::int64_t>(failed_local_cells);
        }
        amio_step_handle failed_step = nullptr;
        if (world_size == 1)
            require_status(amio_begin_step(dataset, &failed_desc, &failed_step), "begin failure-propagation step");
        else
            require_status(amio_begin_step_partitioned(dataset, &failed_desc, failed_partitions.data(), failed_partitions.size(), &failed_step),
                           "begin collective failure-propagation step");
        std::vector<amio_operation_handle> failed_operations;
        for (std::size_t field = 0; field < kFields - 1; ++field) {
            if (failed_local_cells == 0 || (world_rank == 0 && field == 0)) continue;
            std::vector<float> source(static_cast<std::size_t>(failed_local_cells));
            for (std::uint64_t cell = 0; cell < failed_local_cells; ++cell)
                source[static_cast<std::size_t>(cell)] = amio::qualification::reference_value(kSteps + 1, field, failed_offset + cell);
            amio_operation_handle operation = nullptr;
            require_status(amio_put_field(failed_step, cancel_names[field].data(), cancel_names[field].size(), source.data(),
                                          source.size() * sizeof(float), &operation),
                           "submit collective failure-propagation field");
            failed_operations.push_back(operation);
        }
        const auto failed_end_status = amio_end_step(failed_step);
        const auto expected_failure_status = world_size <= 2 ? AMIO_ERR_FIELD_NOT_PRESENT : AMIO_ERR_INVALID_POLICY;
        require(failed_end_status == expected_failure_status,
                "incomplete required-field coverage returns the expected failure status (status " + std::to_string(failed_end_status) + ")");
        for (auto operation : failed_operations) {
            amio_operation_status_t operation_status{};
            require_status(amio_query_operation(operation, &operation_status), "query failed collective operation");
            require(operation_status.outcome == AMIO_OPERATION_FAILED, "collective required-field failure propagates to every accepted operation");
            require_status(amio_release_operation(operation), "release failed collective operation");
        }
        require_status(amio_release_step_handle(failed_step), "release collectively failed step handle");
        amio_step_summary_t unpublished{};
        require(amio_describe_step(dataset, kSteps + 1, &unpublished) != AMIO_OK, "collectively failed step never becomes discoverable");

        require_status(amio_release_schema(schema_handle), "amio_release_schema");
        schema_handle = nullptr;
        require(amio_close_dataset_ex(dataset) == AMIO_ERR_INVALID_POLICY, "dataset close must reject an outstanding consumer lease");
        require_status(amio_release_step_lease(lease), "amio_release_step_lease");
        require_status(amio_close_dataset_ex(dataset), "amio_close_dataset_ex");
        dataset = nullptr;
        require_status(amio_release_context(context), "amio_release_context");
        context = nullptr;
    } catch (const std::exception &error) {
        std::cerr << "US1 persistent qualification failed: " << error.what() << '\n';
        result = 1;
    }

    if (active_step != nullptr) (void)amio_release_step_handle(active_step);
    for (auto operation : operations)
        if (operation != nullptr) (void)amio_release_operation(operation);
    if (schema_handle != nullptr) (void)amio_release_schema(schema_handle);
    if (dataset != nullptr) (void)amio_close_dataset_ex(dataset);
    if (context != nullptr) (void)amio_release_context(context);
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) result = 1;
    if (world_rank == 0 && !root.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
    if (MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS) result = 1;
    MPI_Finalize();
    return result;
}
