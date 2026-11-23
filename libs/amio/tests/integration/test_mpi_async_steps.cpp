#include <mpi.h>

#include <array>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "amio/amio_evolution.h"
#include "c_boundary/transfer_test_hook.hpp"

static amio_status_t fail_transfer(void *) {
    return AMIO_ERR_BACKEND_FAILURE;
}
struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool released = false;
};
static amio_status_t hold_transfer(void *context) {
    auto &gate = *static_cast<Gate *>(context);
    std::unique_lock<std::mutex> lock(gate.mutex);
    gate.entered = true;
    gate.changed.notify_all();
    gate.changed.wait(lock, [&] { return gate.released; });
    return AMIO_OK;
}

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    assert(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS);
    assert(provided == MPI_THREAD_MULTIPLE);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    assert(size >= 2);
    const int contributors = size >= 3 ? size - 1 : size;
    const bool empty = rank >= contributors;
    std::vector<int32_t> producers(static_cast<std::size_t>(size));
    for (int i = 0; i < size; ++i) producers[static_cast<std::size_t>(i)] = i;
    constexpr char config[] = "{}";
    amio_context_descriptor_t context_desc{};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_BACKGROUND;
    context_desc.rank_count = static_cast<uint32_t>(size);
    context_desc.producer_count = static_cast<uint32_t>(size);
    context_desc.producers = producers.data();
    context_desc.fortran_communicator = static_cast<int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
    context_desc.canonical_configuration = config;
    context_desc.configuration_bytes = sizeof(config) - 1;
    amio_context_handle context = nullptr;
    assert(amio_init_ex(&context_desc, &context) == AMIO_OK);

    const auto root = std::filesystem::temp_directory_path() / "amio-mpi-async-steps";
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto location = root.string();
    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location = location.data();
    dataset_desc.location_bytes = location.size();
    dataset_desc.format = AMIO_FORMAT_NETCDF4;
    dataset_desc.budget_bytes = 1024 * 1024;
    dataset_desc.budget_descriptors = 64;
    dataset_desc.control_budget_bytes = 64 * 1024;
    dataset_desc.control_budget_descriptors = 16;
    amio_dataset_handle dataset = nullptr;
    assert(amio_open_dataset_ex(context, &dataset_desc, &dataset) == AMIO_OK);

    constexpr char dim[] = "x";
    constexpr std::array<const char *, 2> names{"a", "b"};
    amio_dimension_descriptor_t dimension{};
    dimension.struct_size = sizeof(dimension);
    dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dimension.name = dim;
    dimension.name_bytes = 1;
    dimension.extent = static_cast<uint64_t>(contributors);
    std::array<amio_variable_descriptor_t, 2> variables{};
    for (std::size_t i = 0; i < variables.size(); ++i) {
        variables[i].struct_size = sizeof(variables[i]);
        variables[i].version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        variables[i].name = names[i];
        variables[i].name_bytes = 1;
        variables[i].source_dtype = AMIO_DTYPE_F32;
        variables[i].stored_dtype = AMIO_DTYPE_F32;
        variables[i].required = i == 0 ? 1 : 0;
        variables[i].dimension_count = 1;
        variables[i].dimensions = &dimension;
    }
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count = variables.size();
    schema_desc.variables = variables.data();
    amio_schema_handle schema = nullptr;
    assert(amio_define_schema(dataset, &schema_desc, &schema) == AMIO_OK);
    amio_step_descriptor_t step_desc{};
    step_desc.struct_size = sizeof(step_desc);
    step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id = 1;
    step_desc.model_time = 1;
    step_desc.schema = schema;
    std::array<amio_field_partition_descriptor_t, 2> partitions{};
    for (std::size_t i = 0; i < partitions.size(); ++i) {
        partitions[i].struct_size = sizeof(partitions[i]);
        partitions[i].version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        partitions[i].field = names[i];
        partitions[i].field_bytes = 1;
        partitions[i].rank = 1;
        partitions[i].offsets[0] = empty ? 0 : rank;
        partitions[i].extents[0] = empty ? 0 : 1;
    }
    amio_step_handle step = nullptr;
    assert(amio_begin_step_partitioned(dataset, &step_desc, partitions.data(), partitions.size(), &step) == AMIO_OK);
    std::array<amio_operation_handle, 2> operations{};
    if (!empty) {
        const int first = rank % 2;
        const int second = 1 - first;
        float first_value = static_cast<float>(rank + 1 + first * 10);
        assert(amio_put_field(step, names[first], 1, &first_value, sizeof(first_value), &operations[first]) == AMIO_OK);
        first_value = -99;
        // Ranks wait for different fields before submitting their second field.
        assert(amio_wait_operation(operations[first], AMIO_OPERATION_BACKEND_COMPLETE, 10000) == AMIO_OK);
        float second_value = static_cast<float>(rank + 1 + second * 10);
        assert(amio_put_field(step, names[second], 1, &second_value, sizeof(second_value), &operations[second]) == AMIO_OK);
        second_value = -99;
        assert(amio_wait_operation(operations[second], AMIO_OPERATION_BACKEND_COMPLETE, 10000) == AMIO_OK);
    }
    assert(amio_end_step(step) == AMIO_OK);
    assert(amio_end_step(step) == AMIO_ERR_INVALID_POLICY);
    assert(amio_cancel_step(step) == AMIO_ERR_INVALID_POLICY);
    if (!empty) {
        assert(amio_release_operation(operations[0]) == AMIO_OK);
        assert(amio_release_operation(operations[1]) == AMIO_OK);
    }
    assert(amio_release_step_handle(step) == AMIO_OK);
    if (rank == 0) {
        for (int field = 0; field < 2; ++field) {
            std::vector<float> values(static_cast<std::size_t>(contributors));
            size_t bytes = 0;
            assert(amio_read_step_field(dataset, 1, names[field], 1, nullptr, values.data(), values.size() * sizeof(float), &bytes) == AMIO_OK);
            for (int i = 0; i < contributors; ++i) assert(values[static_cast<std::size_t>(i)] == float(i + 1 + field * 10));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // An optional field can be absent everywhere without delaying the
    // required field's independent transfer.
    step_desc.step_id = 2;
    step = nullptr;
    auto optional_absent = partitions;
    optional_absent[1].offsets[0] = 0;
    optional_absent[1].extents[0] = 0;
    assert(amio_begin_step_partitioned(dataset, &step_desc, optional_absent.data(), optional_absent.size(), &step) == AMIO_OK);
    amio_operation_handle required_only = nullptr;
    if (!empty) {
        float value = static_cast<float>(rank + 1);
        assert(amio_put_field(step, names[0], 1, &value, sizeof(value), &required_only) == AMIO_OK);
        assert(amio_wait_operation(required_only, AMIO_OPERATION_BACKEND_COMPLETE, 10000) == AMIO_OK);
    }
    assert(amio_end_step(step) == AMIO_OK);
    if (!empty) assert(amio_release_operation(required_only) == AMIO_OK);
    assert(amio_release_step_handle(step) == AMIO_OK);
    if (rank == 0) {
        amio_step_summary_t summary{};
        assert(amio_describe_step(dataset, 2, &summary) == AMIO_OK && summary.present_field_count == 1);
        float unused = 0;
        size_t bytes = 0;
        assert(amio_read_step_field(dataset, 2, names[1], 1, nullptr, &unused, sizeof(unused), &bytes) == AMIO_ERR_FIELD_NOT_PRESENT);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Cancellation drains a blocked worker before releasing its snapshot.
    step_desc.step_id = 3;
    step = nullptr;
    assert(amio_begin_step_partitioned(dataset, &step_desc, partitions.data(), partitions.size(), &step) == AMIO_OK);
    Gate gate;
    std::thread release_gate;
    if (rank == 0) {
        amio_set_transfer_test_hook(hold_transfer, &gate);
        release_gate = std::thread([&] {
            std::unique_lock<std::mutex> lock(gate.mutex);
            gate.changed.wait(lock, [&] { return gate.entered; });
            gate.released = true;
            gate.changed.notify_all();
        });
    }
    amio_operation_handle cancelled = nullptr;
    if (!empty) {
        float value = static_cast<float>(rank + 1);
        assert(amio_put_field(step, names[0], 1, &value, sizeof(value), &cancelled) == AMIO_OK);
    }
    assert(amio_cancel_step(step) == AMIO_OK);
    if (rank == 0) {
        release_gate.join();
        amio_set_transfer_test_hook(nullptr, nullptr);
    }
    if (!empty) {
        amio_operation_status_t status{};
        assert(amio_query_operation(cancelled, &status) == AMIO_OK && status.outcome == AMIO_OPERATION_FAILED);
        assert(amio_release_operation(cancelled) == AMIO_OK);
    }
    assert(amio_release_step_handle(step) == AMIO_OK);
    MPI_Barrier(MPI_COMM_WORLD);

    // A backend error on one rank must become terminal everywhere without
    // publishing the failed step or invalidating the previous one.
    step_desc.step_id = 4;
    step = nullptr;
    assert(amio_begin_step_partitioned(dataset, &step_desc, partitions.data(), partitions.size(), &step) == AMIO_OK);
    if (rank == 0) amio_set_transfer_test_hook(fail_transfer, nullptr);
    operations = {};
    if (!empty) {
        for (int field = 0; field < 2; ++field) {
            float value = static_cast<float>(rank + 1 + field * 10);
            assert(amio_put_field(step, names[field], 1, &value, sizeof(value), &operations[field]) == AMIO_OK);
        }
    }
    assert(amio_end_step(step) == AMIO_ERR_BACKEND_FAILURE);
    if (rank == 0) amio_set_transfer_test_hook(nullptr, nullptr);
    if (!empty) {
        for (auto operation : operations) {
            amio_operation_status_t status{};
            assert(amio_query_operation(operation, &status) == AMIO_OK);
            assert(status.outcome == AMIO_OPERATION_FAILED);
            assert(amio_release_operation(operation) == AMIO_OK);
        }
    }
    assert(amio_release_step_handle(step) == AMIO_OK);
    if (rank == 0) {
        amio_step_summary_t summary{};
        assert(amio_describe_step(dataset, 3, &summary) != AMIO_OK);
        assert(amio_describe_step(dataset, 4, &summary) != AMIO_OK);
        assert(amio_describe_step(dataset, 2, &summary) == AMIO_OK);
        assert(amio_describe_step(dataset, 1, &summary) == AMIO_OK);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    assert(amio_release_schema(schema) == AMIO_OK);
    assert(amio_close_dataset_ex(dataset) == AMIO_OK);
    assert(amio_release_context(context) == AMIO_OK);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
    MPI_Finalize();
}
