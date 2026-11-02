#include "amio/amio_evolution.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#ifdef AMIO_TEST_HAS_MPI
#include <mpi.h>
#endif

namespace {
void ok(amio_status_t status) {
    static int call = 0;
    ++call;
    if (status != AMIO_OK) std::fprintf(stderr, "evolution Zarr call %d returned status %d\n", call, static_cast<int>(status));
    assert(status == AMIO_OK);
}
}

int main(int argc, char **argv) {
#ifdef AMIO_TEST_HAS_MPI
    int provided = MPI_THREAD_SINGLE;
    assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_FUNNELED);
    int rank = 0, rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    assert(rank_count == 1 || rank_count == 2 || rank_count == 4);
#else
    (void)argc;
    (void)argv;
    constexpr int rank = 0;
    constexpr int rank_count = 1;
#endif
    const auto parent = std::filesystem::temp_directory_path() / "amio-evolution-zarr-test";
    const auto collection = parent / "collection";
    if (rank == 0) {
        std::filesystem::remove_all(parent);
        std::filesystem::create_directories(parent);
    }
#ifdef AMIO_TEST_HAS_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    const std::string path = collection.string();
    const char config[] = "{}";

    amio_context_descriptor_t context_desc{};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_CALLER;
    context_desc.rank_count = static_cast<std::uint32_t>(rank_count);
    context_desc.canonical_configuration = config;
    context_desc.configuration_bytes = sizeof(config) - 1;
#ifdef AMIO_TEST_HAS_MPI
    context_desc.fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(rank_count == 1 ? MPI_COMM_SELF : MPI_COMM_WORLD));
#endif
    amio_context_handle context = nullptr;
    ok(amio_init_ex(&context_desc, &context));

    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location = path.data();
    dataset_desc.location_bytes = path.size();
    dataset_desc.format = AMIO_FORMAT_ZARR3;
    dataset_desc.budget_bytes = 1024 * 1024;
    dataset_desc.budget_descriptors = 64;
    dataset_desc.control_budget_bytes = 4096;
    dataset_desc.control_budget_descriptors = 8;
    amio_dataset_handle dataset = nullptr;
    ok(amio_open_dataset_ex(context, &dataset_desc, &dataset));

    const char dim_name[] = "x";
    const char required_name[] = "temperature";
    const char optional_name[] = "tendency";
    amio_dimension_descriptor_t dimension{};
    dimension.struct_size = sizeof(dimension);
    dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dimension.name = dim_name;
    dimension.name_bytes = sizeof(dim_name) - 1;
    const int rows = rank_count == 1 ? 4 : rank_count * (rank_count + 1) / 2;
    const int local_offset = rank_count == 1 ? 0 : rank * (rank + 1) / 2;
    const int local_rows = rank_count == 1 ? 4 : rank + 1;
    dimension.extent = rows;
    std::array<amio_variable_descriptor_t, 2> variables{};
    for (auto &variable : variables) {
        variable.struct_size = sizeof(variable);
        variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        variable.source_dtype = AMIO_DTYPE_F32;
        variable.stored_dtype = AMIO_DTYPE_F32;
        variable.dimension_count = 1;
        variable.dimensions = &dimension;
    }
    variables[0].name = required_name;
    variables[0].name_bytes = sizeof(required_name) - 1;
    variables[0].required = 1;
    variables[1].name = optional_name;
    variables[1].name_bytes = sizeof(optional_name) - 1;
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count = variables.size();
    schema_desc.variables = variables.data();
    amio_schema_handle schema = nullptr;
    if (rank_count > 1) {
        const char attribute_name[] = "metadata";
        std::string value(rank == 0 ? 8192 : 1, 'x');
        amio_attribute_descriptor_t attribute{};
        attribute.struct_size = sizeof(attribute);
        attribute.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        attribute.name = attribute_name;
        attribute.name_bytes = sizeof(attribute_name) - 1;
        attribute.type = AMIO_ATTRIBUTE_UTF8;
        attribute.element_count = 1;
        attribute.value = value.data();
        attribute.value_bytes = value.size();
        auto oversized_variables = variables;
        oversized_variables[0].attribute_count = 1;
        oversized_variables[0].attributes = &attribute;
        auto oversized_schema = schema_desc;
        oversized_schema.variables = oversized_variables.data();
        assert(amio_define_schema(dataset, &oversized_schema, &schema) == AMIO_ERR_BUDGET_EXCEEDED && schema == nullptr);
    }
    ok(amio_define_schema(dataset, &schema_desc, &schema));

    amio_step_descriptor_t step_desc{};
    step_desc.struct_size = sizeof(step_desc);
    step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id = 7;
    step_desc.model_time = 42.5;
    step_desc.schema = schema;
    amio_step_handle step = nullptr;
    amio_status_t begin_status = AMIO_OK;
    if (rank_count == 1) {
        begin_status = amio_begin_step(dataset, &step_desc, &step);
    } else {
        std::array<amio_field_partition_descriptor_t, 2> partitions{};
        const std::array<const char *, 2> names{required_name, optional_name};
        for (std::size_t i = 0; i < partitions.size(); ++i) {
            partitions[i].struct_size = sizeof(partitions[i]);
            partitions[i].version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            partitions[i].field = names[i];
            partitions[i].field_bytes = std::strlen(names[i]);
            partitions[i].rank = 1;
            partitions[i].offsets[0] = i == 0 ? local_offset : 0;
            partitions[i].extents[0] = i == 0 ? local_rows : 0;
        }
        begin_status = amio_begin_step_partitioned(dataset, &step_desc, partitions.data(), partitions.size(), &step);
    }
    ok(begin_status);
    std::vector<float> values(static_cast<std::size_t>(rows));
    for (int i = 0; i < rows; ++i) values[static_cast<std::size_t>(i)] = static_cast<float>((i + 1) * 10);
    amio_operation_handle operation = nullptr;
    if (rank_count == 1) {
        ok(amio_put_field(step, required_name, sizeof(required_name) - 1, values.data(), values.size() * sizeof(float), &operation));
    } else {
        std::vector<float> local_values(static_cast<std::size_t>(local_rows));
        for (int i = 0; i < local_rows; ++i)
            local_values[static_cast<std::size_t>(i)] = static_cast<float>((local_offset + i + 1) * 10);
        ok(amio_put_field(step, required_name, sizeof(required_name) - 1, local_values.data(),
                          local_values.size() * sizeof(float), &operation));
    }
    ok(amio_end_step(step));
    if (rank_count == 1) assert(amio_wait_operation(operation, AMIO_OPERATION_DURABLE, 0) == AMIO_ERR_UNSUPPORTED);

    if (rank == 0) {
        amio_step_summary_t summary{};
        ok(amio_describe_step(dataset, 7, &summary));
        assert(summary.step_id == 7 && summary.present_field_count == 1);
        std::vector<float> readback(static_cast<std::size_t>(rows));
        std::size_t required_bytes = 0;
        ok(amio_read_step_field(dataset, 7, required_name, sizeof(required_name) - 1, nullptr,
                                readback.data(), readback.size() * sizeof(float), &required_bytes));
        assert(required_bytes == values.size() * sizeof(float) && readback == values);
        amio_step_selection_t selection{};
        selection.struct_size = sizeof(selection);
        selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        selection.rank = 1;
        selection.offsets[0] = 1;
        selection.extents[0] = 2;
        selection.strides[0] = 1;
        std::array<float, 2> selected{};
        ok(amio_read_step_field(dataset, 7, required_name, sizeof(required_name) - 1, &selection,
                                selected.data(), sizeof(selected), &required_bytes));
        assert(required_bytes == sizeof(selected) && selected[0] == 20.f && selected[1] == 30.f);
        assert(amio_read_step_field(dataset, 7, optional_name, sizeof(optional_name) - 1, nullptr,
                                    nullptr, 0, &required_bytes) == AMIO_ERR_FIELD_NOT_PRESENT);
    }

    amio_step_lease_handle lease = nullptr;
    ok(amio_acquire_step_lease(dataset, 7, &lease));
    assert(amio_release_step_handle(reinterpret_cast<amio_step_handle>(lease)) == AMIO_ERR_INVALID_HANDLE);

    ok(amio_release_operation(operation));
    ok(amio_release_step_handle(step));
    ok(amio_release_schema(schema));
    assert(amio_close_dataset_ex(dataset) == AMIO_ERR_INVALID_POLICY);
    ok(amio_release_step_lease(lease));
    assert(amio_release_step_lease(lease) == AMIO_ERR_INVALID_HANDLE);
    ok(amio_close_dataset_ex(dataset));
    ok(amio_release_context(context));
    if (rank == 0) std::filesystem::remove_all(parent);
#ifdef AMIO_TEST_HAS_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    assert(MPI_Finalize() == MPI_SUCCESS);
#endif
}
