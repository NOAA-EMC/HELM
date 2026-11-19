#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "tool_common.hpp"
#include "utility_fixtures.hpp"
#ifdef AMIO_CLI_TEST_HAS_MPI
#include <mpi.h>
#endif

int main() {
#ifdef AMIO_CLI_TEST_HAS_MPI
    int provided = 0;
    assert(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided) == MPI_SUCCESS && provided >= MPI_THREAD_MULTIPLE);
#endif
    const char *fixture_root = std::getenv("AMIO_UTILITY_FIXTURE_DIR");
    const bool preserve_fixture = fixture_root && *fixture_root;
    const auto root = preserve_fixture ? std::filesystem::path(fixture_root)
                                       : std::filesystem::temp_directory_path() /
                                             ("amio-cli-copy-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto source = root / "source";
    const auto destination = root / "destination";
    std::filesystem::create_directories(root);
    assert(write_amio_utility_fixtures(root));
    const std::string config = "{}";
    amio_context_descriptor_t context_desc{};
    context_desc.struct_size = sizeof(context_desc);
    context_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    context_desc.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    context_desc.progress_mode = AMIO_PROGRESS_CALLER;
#ifdef AMIO_CLI_TEST_HAS_MPI
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    context_desc.rank_count = static_cast<std::uint32_t>(size);
    context_desc.fortran_communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
#else
    context_desc.rank_count = 1;
#endif
    context_desc.canonical_configuration = config.data();
    context_desc.configuration_bytes = config.size();
    amio_context_handle context = nullptr;
    assert(amio_init_ex(&context_desc, &context) == AMIO_OK);
    const auto source_text = source.string();
    amio_dataset_descriptor_t dataset_desc{};
    dataset_desc.struct_size = sizeof(dataset_desc);
    dataset_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dataset_desc.location = source_text.data();
    dataset_desc.location_bytes = source_text.size();
    dataset_desc.format = AMIO_FORMAT_NETCDF4;
    dataset_desc.budget_bytes = 64u * 1024u * 1024u;
    dataset_desc.budget_descriptors = 4096;
    dataset_desc.control_budget_bytes = 16u * 1024u * 1024u;
    dataset_desc.control_budget_descriptors = 1024;
    amio_dataset_handle dataset = nullptr;
    assert(amio_open_dataset_ex(context, &dataset_desc, &dataset) == AMIO_OK);
    const char field_name[] = "temperature", attribute_name[] = "units", unit_value[] = "K";
    amio_attribute_descriptor_t attr{};
    attr.struct_size = sizeof(attr);
    attr.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    attr.name = attribute_name;
    attr.name_bytes = sizeof(attribute_name) - 1;
    attr.type = AMIO_ATTRIBUTE_UTF8;
    attr.element_count = 1;
    attr.value = unit_value;
    attr.value_bytes = sizeof(unit_value) - 1;
    amio_variable_descriptor_t variable{};
    variable.struct_size = sizeof(variable);
    variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    variable.name = field_name;
    variable.name_bytes = sizeof(field_name) - 1;
    variable.source_dtype = AMIO_DTYPE_F32;
    variable.stored_dtype = AMIO_DTYPE_F32;
    const char dimension_name[] = "sample";
    amio_dimension_descriptor_t dimension{};
    dimension.struct_size = sizeof(dimension);
    dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    dimension.name = dimension_name;
    dimension.name_bytes = sizeof(dimension_name) - 1;
    dimension.extent = 1;
    variable.required = 1;
    variable.dimension_count = 1;
    variable.dimensions = &dimension;
    variable.attribute_count = 1;
    variable.attributes = &attr;
    amio_schema_descriptor_t schema_desc{};
    schema_desc.struct_size = sizeof(schema_desc);
    schema_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    schema_desc.variable_count = 1;
    schema_desc.variables = &variable;
    amio_schema_handle schema = nullptr;
    assert(amio_define_schema(dataset, &schema_desc, &schema) == AMIO_OK);
    amio_step_descriptor_t step_desc{};
    step_desc.struct_size = sizeof(step_desc);
    step_desc.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    step_desc.step_id = 0;
    step_desc.model_time = 42.5;
    step_desc.schema = schema;
    amio_step_handle step = nullptr;
    const auto begin_status = amio_begin_step(dataset, &step_desc, &step);
    if (begin_status != AMIO_OK) std::fprintf(stderr, "source begin status=%d\n", begin_status);
    assert(begin_status == AMIO_OK);
    const float expected = 273.15f;
    amio_operation_handle operation = nullptr;
    assert(amio_put_field(step, field_name, sizeof(field_name) - 1, &expected, sizeof(expected), &operation) == AMIO_OK);
    assert(amio_end_step(step) == AMIO_OK);
    assert(amio_release_operation(operation) == AMIO_OK);
    assert(amio_release_step_handle(step) == AMIO_OK);
    assert(amio_release_schema(schema) == AMIO_OK);
    assert(amio_close_dataset_ex(dataset) == AMIO_OK);
    assert(amio_release_context(context) == AMIO_OK);

    const auto config_path = root / "config.json";
    {
        std::ofstream out(config_path);
        out << config;
    }
    amio::tools::Options copy;
    copy.command = "copy";
    copy.json = true;
    copy.positional = {source.string(), destination.string()};
    copy.values["--config"] = config_path.string();
    copy.values["--steps"] = "0:0";
    copy.values["--fields"] = field_name;
    const auto copied = amio::tools::run_copy(copy);
    if (copied.exit_code != 0) {
        std::fprintf(stderr, "copy code=%d summary=%s error=%s\n", copied.exit_code, copied.summary.c_str(),
                     copied.errors.empty() ? "" : copied.errors.front().c_str());
    }
    assert(copied.exit_code == 0);
    assert(std::filesystem::exists(destination / "head.json"));
    amio::tools::Options readback_options;
    readback_options.values["--config"] = config_path.string();
    amio::tools::CliRuntime readback_runtime;
    assert(amio::tools::open_cli_dataset(destination, readback_options, readback_runtime) == AMIO_OK);
    float actual = 0;
    std::size_t actual_bytes = 0;
    assert(amio_read_step_field(readback_runtime.dataset, 0, field_name, sizeof(field_name) - 1, nullptr, &actual, sizeof(actual), &actual_bytes) ==
           AMIO_OK);
    assert(actual_bytes == sizeof(actual) && actual == expected);
    amio::tools::Options validate;
    validate.command = "validate";
    validate.positional = {destination.string()};
    const auto validated = amio::tools::run_validate(validate);
    assert(validated.exit_code == 0);
    assert(validated.summary_json.find("\"complete\":false") != std::string::npos);
    amio::tools::Options inspect;
    inspect.command = "inspect";
    inspect.positional = {destination.string()};
    const auto inspected = amio::tools::run_inspect(inspect);
    assert(inspected.exit_code == 0);
    assert(inspected.summary_json.find("\"format\":\"netcdf4\"") != std::string::npos);
    assert(inspected.summary_json.find("\"model_time\":42.500000") != std::string::npos);
    assert(!inspected.results_json.empty() && inspected.results_json.front().find("\"model_time\":42.500000") != std::string::npos);
    auto wrong_uuid = inspect;
    wrong_uuid.values["--cursor"] = "wrong-uuid|generation|commits/0.json";
    assert(amio::tools::run_inspect(wrong_uuid).exit_code == 2);
    amio::tools::CollectionInfo source_info(source, "");
    assert(amio::tools::open_collection(source, source_info) == AMIO_OK);
    auto stale_generation = inspect;
    stale_generation.values["--cursor"] = source_info.uuid + "|stale-generation|commits/0.json";
    assert(amio::tools::run_inspect(stale_generation).exit_code == 2);
    amio::tools::Options verify_integrity;
    verify_integrity.command = "validate";
    verify_integrity.positional = {destination.string()};
    verify_integrity.flags = {"--integrity"};
    const auto integrity = amio::tools::run_validate(verify_integrity);
    assert(integrity.exit_code == 0);
    const auto existing = amio::tools::run_copy(copy);
    assert(existing.exit_code == 2);
    assert(std::filesystem::exists(destination / "head.json"));
    auto missing_field = copy;
    missing_field.positional[1] = (root / "must-not-create").string();
    missing_field.values["--fields"] = "missing";
    assert(amio::tools::run_copy(missing_field).exit_code == 2);
    assert(!std::filesystem::exists(root / "must-not-create"));
    amio::tools::Options diagnostics;
    diagnostics.command = "diagnostics";
    diagnostics.values["--input"] = (root / "stalled-run.json").string();
    const auto diagnostic_report = amio::tools::run_diagnostics(diagnostics);
    assert(diagnostic_report.exit_code == 0);
    assert(diagnostic_report.summary_json.find("bytes_written") != std::string::npos);
    assert(diagnostic_report.summary_json.find("must-not-be-echoed") == std::string::npos);
    assert(std::filesystem::exists(root / "zarr-copy.yaml"));
    if (!preserve_fixture) {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        assert(!ec);
    }
#ifdef AMIO_CLI_TEST_HAS_MPI
    assert(MPI_Finalize() == MPI_SUCCESS);
#endif
    return 0;
}
