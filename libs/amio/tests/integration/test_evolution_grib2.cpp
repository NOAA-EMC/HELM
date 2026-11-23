#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include "amio/amio_evolution.h"

#ifdef AMIO_TEST_HAS_MPI
#include <mpi.h>
#endif

namespace {
void ok(amio_status_t status) {
    static unsigned call = 0;
    ++call;
    if (status != AMIO_OK) std::fprintf(stderr, "evolution GRIB2 call %u returned %d\n", call, static_cast<int>(status));
    assert(status == AMIO_OK);
}
}  // namespace

int main() {
#ifdef AMIO_TEST_HAS_MPI
    int provided = MPI_THREAD_SINGLE;
    assert(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided) == MPI_SUCCESS);
    assert(provided >= MPI_THREAD_FUNNELED);
#endif
    const auto root = std::filesystem::temp_directory_path() / "amio-evolution-grib2-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto collection = root / "collection";
    const std::string location = collection.string();
    const char configuration[] =
        "drt: libaec\n"
        "completion:\n"
        "  level: directory_sync\n"
        "grib2:\n"
        "  center: 7\n"
        "  parameter_category: 2\n"
        "  parameter_number: 3\n"
        "  scaled_value_first_surface: 85000\n"
        "  lat_first: 10000000\n"
        "  lon_first: 0\n"
        "  lat_last: 0\n"
        "  lon_last: 20000000\n";

    amio_context_descriptor_t context_desc{};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_CALLER;
    context_desc.rank_count = 1;
    context_desc.canonical_configuration = configuration;
    context_desc.configuration_bytes = sizeof(configuration) - 1;
#ifdef AMIO_TEST_HAS_MPI
    context_desc.fortran_communicator = static_cast<int64_t>(MPI_Comm_c2f(MPI_COMM_SELF));
#endif
    amio_context_handle context = nullptr;
    ok(amio_init_ex(&context_desc, &context));

    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location = location.data();
    dataset_desc.location_bytes = location.size();
    dataset_desc.format = AMIO_FORMAT_GRIB2;
    dataset_desc.budget_bytes = 1024 * 1024;
    dataset_desc.budget_descriptors = 32;
    dataset_desc.control_budget_bytes = 4096;
    dataset_desc.control_budget_descriptors = 8;
    amio_dataset_handle dataset = nullptr;
    ok(amio_open_dataset_ex(context, &dataset_desc, &dataset));

    const char field_name[] = "air_temperature";
    const char y_name[] = "latitude";
    const char x_name[] = "longitude";
    amio_dimension_descriptor_t dimensions[2]{};
    for (auto &dimension : dimensions) {
        dimension.struct_size = sizeof(dimension);
        dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    }
    dimensions[0].name = y_name;
    dimensions[0].name_bytes = sizeof(y_name) - 1;
    dimensions[0].extent = 2;
    dimensions[1].name = x_name;
    dimensions[1].name_bytes = sizeof(x_name) - 1;
    dimensions[1].extent = 3;

    const int64_t category = 0;
    const int64_t parameter = 0;
    const int64_t surface_type = 100;
    const int64_t level = 50000;
    const char *attribute_names[] = {"grib2.parameter_category", "grib2.parameter_number", "grib2.type_of_first_fixed_surface",
                                     "grib2.scaled_value_first_surface"};
    const int64_t attribute_values[] = {category, parameter, surface_type, level};
    amio_attribute_descriptor_t attributes[4]{};
    for (std::size_t i = 0; i < 4; ++i) {
        attributes[i].struct_size = sizeof(attributes[i]);
        attributes[i].version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        attributes[i].name = attribute_names[i];
        attributes[i].name_bytes = std::char_traits<char>::length(attribute_names[i]);
        attributes[i].type = AMIO_ATTRIBUTE_I64;
        attributes[i].element_count = 1;
        attributes[i].value = &attribute_values[i];
        attributes[i].value_bytes = sizeof(int64_t);
    }
    amio_variable_descriptor_t variable{};
    variable.struct_size = sizeof(variable);
    variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    variable.name = field_name;
    variable.name_bytes = sizeof(field_name) - 1;
    variable.source_dtype = AMIO_DTYPE_F32;
    variable.stored_dtype = AMIO_DTYPE_F32;
    variable.required = 1;
    variable.dimension_count = 2;
    variable.dimensions = dimensions;
    variable.attribute_count = 4;
    variable.attributes = attributes;
    auto duplicate_variable = variable;
    const char duplicate_name[] = "air_temperature_duplicate";
    duplicate_variable.name = duplicate_name;
    duplicate_variable.name_bytes = sizeof(duplicate_name) - 1;
    amio_variable_descriptor_t duplicate_variables[] = {variable, duplicate_variable};
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count = 2;
    schema_desc.variables = duplicate_variables;
    amio_schema_handle schema = nullptr;
    assert(amio_define_schema(dataset, &schema_desc, &schema) == AMIO_ERR_INVALID_INPUT);
    schema_desc.variable_count = 1;
    schema_desc.variables = &variable;
    ok(amio_define_schema(dataset, &schema_desc, &schema));

    amio_step_descriptor_t step_desc{};
    step_desc.struct_size = sizeof(step_desc);
    step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id = 3;
    step_desc.model_time = 12.0;
    step_desc.schema = schema;
    amio_step_handle step = nullptr;
    ok(amio_begin_step(dataset, &step_desc, &step));
    const std::array<float, 6> values{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    amio_operation_handle operation = nullptr;
    ok(amio_put_field(step, field_name, sizeof(field_name) - 1, values.data(), sizeof(values), &operation));
    ok(amio_end_step(step));
    ok(amio_wait_operation(operation, AMIO_OPERATION_DURABLE, 0));

    amio_step_summary_t summary{};
    ok(amio_describe_step(dataset, 3, &summary));
    assert(summary.present_field_count == 1);
    std::array<float, 6> readback{};
    std::size_t required = 0;
    ok(amio_read_step_field(dataset, 3, field_name, sizeof(field_name) - 1, nullptr, readback.data(), sizeof(readback), &required));
    assert(required == sizeof(values));
    assert(readback == values);

    ok(amio_release_operation(operation));
    ok(amio_release_step_handle(step));
    ok(amio_release_schema(schema));
    ok(amio_close_dataset_ex(dataset));
    ok(amio_release_context(context));
    std::filesystem::remove_all(root);
#ifdef AMIO_TEST_HAS_MPI
    assert(MPI_Finalize() == MPI_SUCCESS);
#endif
}
