#include <mpi.h>
#include <netcdf.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "amio/amio_evolution.h"

#define check(value)                                                                                    \
    do {                                                                                                \
        if (!(value)) {                                                                                 \
            std::fprintf(stderr, "operator roundtrip check failed at line %d: %s\n", __LINE__, #value); \
            std::abort();                                                                               \
        }                                                                                               \
    } while (false)

int main(int argc, char **argv) {
    int provided = MPI_THREAD_SINGLE;
    check(MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS);
    check(provided == MPI_THREAD_MULTIPLE);
    int rank = -1, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    check(size == 2);
    const std::array<int32_t, 2> producer{0, 1};
    constexpr char config[] = "{}";
    amio_context_descriptor_t context_desc{};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_CALLER;
    context_desc.rank_count = size;
    context_desc.producer_count = size;
    context_desc.producers = producer.data();
    context_desc.fortran_communicator = static_cast<int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
    context_desc.canonical_configuration = config;
    context_desc.configuration_bytes = sizeof(config) - 1;
    amio_context_handle context = nullptr;
    check(amio_init_ex(&context_desc, &context) == AMIO_OK);

    const auto root = std::filesystem::temp_directory_path() / "amio-operator-roundtrip";
    std::error_code ec;
    if (rank == 0) std::filesystem::remove_all(root, ec);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto root_text = root.string();
    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location = root_text.data();
    dataset_desc.location_bytes = root_text.size();
    dataset_desc.format = AMIO_FORMAT_NETCDF4;
    dataset_desc.budget_bytes = 1024 * 1024;
    dataset_desc.budget_descriptors = 64;
    dataset_desc.control_budget_bytes = 64 * 1024;
    dataset_desc.control_budget_descriptors = 16;
    amio_dataset_handle dataset = nullptr;
    check(amio_open_dataset_ex(context, &dataset_desc, &dataset) == AMIO_OK);

    constexpr char dim_name[] = "x", field_name[] = "temperature";
    amio_dimension_descriptor_t dimension{};
    dimension.struct_size = sizeof(dimension);
    dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dimension.name = dim_name;
    dimension.name_bytes = sizeof(dim_name) - 1;
    dimension.extent = 4;
    constexpr char precision_key[] = "amio.precision", precision_value[] = "f64_to_f32_absolute";
    constexpr char tolerance_key[] = "amio.tolerance", field_auth_key[] = "amio.authorize_field_loss";
    constexpr char restart_auth_key[] = "amio.authorize_lossy_restart";
    constexpr char codec_key[] = "amio.codec", codec_value[] = "deflate";
    constexpr double tolerance = 1e-5;
    constexpr uint8_t yes = 1;
    std::array<amio_attribute_descriptor_t, 5> attrs{};
    auto attr = [&](std::size_t i, const char *name, std::size_t n, uint32_t type, const void *value, uint64_t bytes) {
        attrs[i].struct_size = sizeof(amio_attribute_descriptor_t);
        attrs[i].version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        attrs[i].name = name;
        attrs[i].name_bytes = n;
        attrs[i].type = type;
        attrs[i].element_count = 1;
        attrs[i].value = value;
        attrs[i].value_bytes = bytes;
    };
    attr(0, precision_key, sizeof(precision_key) - 1, AMIO_ATTRIBUTE_UTF8, precision_value, sizeof(precision_value) - 1);
    attr(1, tolerance_key, sizeof(tolerance_key) - 1, AMIO_ATTRIBUTE_F64, &tolerance, sizeof(tolerance));
    attr(2, field_auth_key, sizeof(field_auth_key) - 1, AMIO_ATTRIBUTE_BOOL, &yes, sizeof(yes));
    attr(3, restart_auth_key, sizeof(restart_auth_key) - 1, AMIO_ATTRIBUTE_BOOL, &yes, sizeof(yes));
    attr(4, codec_key, sizeof(codec_key) - 1, AMIO_ATTRIBUTE_UTF8, codec_value, sizeof(codec_value) - 1);
    amio_variable_descriptor_t variable{};
    variable.struct_size = sizeof(variable);
    variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    variable.name = field_name;
    variable.name_bytes = sizeof(field_name) - 1;
    variable.source_dtype = AMIO_DTYPE_F64;
    variable.stored_dtype = AMIO_DTYPE_F32;
    variable.required = 1;
    variable.dimension_count = 1;
    variable.dimensions = &dimension;
    variable.attribute_count = attrs.size();
    variable.attributes = attrs.data();
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count = 1;
    schema_desc.variables = &variable;
    constexpr uint8_t no = 0;
    attrs[3].value = &no;
    amio_schema_handle unauthorized_schema = nullptr;
    check(amio_define_schema(dataset, &schema_desc, &unauthorized_schema) == AMIO_OK);
    amio_step_descriptor_t rejected_step_desc{};
    rejected_step_desc.struct_size = sizeof(rejected_step_desc);
    rejected_step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    rejected_step_desc.step_id = 0;
    rejected_step_desc.model_time = 1.0;
    rejected_step_desc.schema = unauthorized_schema;
    amio_step_handle rejected_step = nullptr;
    amio_field_partition_descriptor_t rejected_partition{};
    rejected_partition.struct_size = sizeof(rejected_partition);
    rejected_partition.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    rejected_partition.field = field_name;
    rejected_partition.field_bytes = sizeof(field_name) - 1;
    rejected_partition.rank = 1;
    rejected_partition.offsets[0] = rank * 2;
    rejected_partition.extents[0] = 2;
    check(amio_begin_step_partitioned(dataset, &rejected_step_desc, &rejected_partition, 1, &rejected_step) == AMIO_ERR_INVALID_INPUT &&
          rejected_step == nullptr);
    check(amio_release_schema(unauthorized_schema) == AMIO_OK);
    attrs[3].value = &yes;
    amio_schema_handle schema = nullptr;
    check(amio_define_schema(dataset, &schema_desc, &schema) == AMIO_OK);

    amio_step_descriptor_t step_desc{};
    step_desc.struct_size = sizeof(step_desc);
    step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id = 0;
    step_desc.model_time = 1.0;
    step_desc.schema = schema;
    amio_step_handle step = nullptr;
    amio_field_partition_descriptor_t partition{};
    partition.struct_size = sizeof(partition);
    partition.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    partition.field = field_name;
    partition.field_bytes = sizeof(field_name) - 1;
    partition.rank = 1;
    partition.offsets[0] = rank * 2;
    partition.extents[0] = 2;
    check(amio_begin_step_partitioned(dataset, &step_desc, &partition, 1, &step) == AMIO_OK);
    const std::array<double, 2> source = rank == 0 ? std::array<double, 2>{1.1, -2.2} : std::array<double, 2>{0.0, -0.0};
    amio_operation_handle operation = nullptr;
    check(amio_put_field(step, field_name, sizeof(field_name) - 1, source.data(), sizeof(source), &operation) == AMIO_OK);
    const auto end_status = amio_end_step(step);
    if (end_status != AMIO_OK) std::fprintf(stderr, "end status=%d\n", end_status);
    check(end_status == AMIO_OK);
    std::array<float, 4> readback{};
    std::size_t read_bytes = 0;
    if (rank == 0) {
        check(amio_read_step_field(dataset, 0, field_name, sizeof(field_name) - 1, nullptr, readback.data(), sizeof(readback), &read_bytes) ==
              AMIO_OK);
        check(read_bytes == sizeof(readback) && readback[0] == 1.1f && readback[2] == 0.0f && std::signbit(readback[3]));
    }

    amio_validation_request_t request{};
    request.struct_size = sizeof(request);
    request.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    request.check_mask = UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1);
    request.max_results = 1;
    amio_validation_result_t result{};
    std::size_t count = 0;
    uint64_t cursor = 0;
    if (rank == 0) {
        check(amio_validate_dataset(dataset, &request, &result, 1, &count, &cursor) == AMIO_OK);
        check(count == 1 && result.status == AMIO_VALIDATION_PASS);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const auto payload = root / "payloads/steps/00000000000000000000.nc";
    int ncid = -1, varid = -1;
    std::array<float, 4> native{};
    if (rank == 0) {
        check(nc_open(payload.c_str(), NC_NOWRITE, &ncid) == NC_NOERR);
        check(nc_inq_varid(ncid, field_name, &varid) == NC_NOERR && nc_get_var_float(ncid, varid, native.data()) == NC_NOERR);
        int shuffle = 0, deflate = 0, level = 0;
        check(nc_inq_var_deflate(ncid, varid, &shuffle, &deflate, &level) == NC_NOERR && deflate == 1 && level == 1);
        check(nc_close(ncid) == NC_NOERR);
        check(native == readback);
        check(nc_open(payload.c_str(), NC_WRITE, &ncid) == NC_NOERR);
        check(nc_inq_varid(ncid, field_name, &varid) == NC_NOERR);
        native[0] += 1.0f;
        check(nc_put_var_float(ncid, varid, native.data()) == NC_NOERR && nc_close(ncid) == NC_NOERR);
        check(amio_validate_dataset(dataset, &request, &result, 1, &count, &cursor) == AMIO_OK);
        check(count == 1 && result.status == AMIO_VALIDATION_FAIL);
    }

    check(amio_release_operation(operation) == AMIO_OK);
    check(amio_release_step_handle(step) == AMIO_OK);
    check(amio_release_schema(schema) == AMIO_OK);
    check(amio_close_dataset_ex(dataset) == AMIO_OK);
    check(amio_release_context(context) == AMIO_OK);
    if (rank == 0) std::filesystem::remove_all(root, ec);
    MPI_Finalize();
}
