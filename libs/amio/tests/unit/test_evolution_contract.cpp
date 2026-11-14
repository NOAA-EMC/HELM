#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

#include "amio/amio_evolution.h"
#include "c_boundary/amio_tool_api.hpp"

#ifdef AMIO_TEST_HAS_MPI
#include <mpi.h>
#endif
#ifdef AMIO_TEST_HAS_NETCDF
#include <netcdf.h>
#endif

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool read_json_uint64_array_item(const char *json, const char *key, std::size_t item_index, std::uint64_t &out) {
    if (json == nullptr || key == nullptr) return false;
    const std::string marker = std::string("\"") + key + "\":[";
    const char *value = std::strstr(json, marker.c_str());
    if (value == nullptr) return false;
    value += marker.size();
    for (std::size_t item = 0; item <= item_index; ++item) {
        char *end = nullptr;
        const auto parsed = std::strtoull(value, &end, 10);
        if (end == value) return false;
        if (item == item_index) {
            out = parsed;
            return true;
        }
        if (*end != ',') return false;
        value = end + 1;
    }
    return false;
}

void test_additive_status_values() {
    static_assert(AMIO_ERR_INVALID_INPUT == 17);
    static_assert(AMIO_ERR_UNSUPPORTED == 18);
    static_assert(AMIO_ERR_BUFFER_TOO_SMALL == 29);
    static_assert(AMIO_VALIDATION_PASS == 0);
    static_assert(AMIO_VALIDATION_FAIL == 1);
    static_assert(AMIO_VALIDATION_UNSUPPORTED == 2);
    expect(sizeof(amio_descriptor_header_t) == 2 * sizeof(uint32_t), "descriptor header must be two fixed-width words");
    static_assert(sizeof(amio_field_partition_descriptor_t) == 160);
}

void test_validation_descriptor_rejection_and_size_query() {
    amio_validation_request_t request{};
    request.struct_size = sizeof(request);
    request.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    request.check_mask = UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1);
    request.max_results = 8;

    size_t count = 99;
    uint64_t next_cursor = 99;
    expect(amio_validate_dataset(nullptr, &request, nullptr, 0, &count, &next_cursor) == AMIO_ERR_NULL_HANDLE,
           "null dataset must be rejected before validation");
    expect(count == 0 && next_cursor == 0, "rejected validation must clear output count and cursor");

    request.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 + 1;
    expect(amio_validate_dataset(nullptr, &request, nullptr, 0, &count, &next_cursor) == AMIO_ERR_INVALID_INPUT,
           "unknown descriptor version must be rejected before reading version-specific fields");
    request.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    request.struct_size = static_cast<uint32_t>(offsetof(amio_validation_request_t, reserved));
    expect(amio_validate_dataset(nullptr, &request, nullptr, 0, &count, &next_cursor) == AMIO_ERR_INVALID_INPUT,
           "truncated descriptor must be rejected");

    request.struct_size = sizeof(request);
    request.check_mask |= UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1);
    auto bogus = reinterpret_cast<amio_dataset_handle>(static_cast<uintptr_t>(0x1010));
    expect(amio_validate_dataset(bogus, &request, nullptr, 0, &count, &next_cursor) == AMIO_ERR_INVALID_HANDLE,
           "validation rejects a forged dataset token without dereferencing it");
}

void test_typed_release_names_reject_invalid_handles() {
    auto bogus_step = reinterpret_cast<amio_step_handle>(static_cast<uintptr_t>(UINT64_C(0x12345678)));
    auto bogus_lease = reinterpret_cast<amio_step_lease_handle>(static_cast<uintptr_t>(UINT64_C(0x87654321)));
    expect(amio_release_step_handle(nullptr) == AMIO_ERR_NULL_HANDLE, "null producer step handle must be rejected");
    expect(amio_release_step_lease(nullptr) == AMIO_ERR_NULL_HANDLE, "null consumer lease must be rejected");
    expect(amio_release_step_handle(bogus_step) == AMIO_ERR_INVALID_HANDLE, "invalid producer step handle must not be dereferenced");
    expect(amio_release_step_lease(bogus_lease) == AMIO_ERR_INVALID_HANDLE, "invalid consumer lease must not be dereferenced");
}

void test_context_initialization_outputs_and_capability_query() {
    amio_context_handle context = reinterpret_cast<amio_context_handle>(static_cast<uintptr_t>(0x1234));
    amio_context_descriptor_t descriptor{};
    descriptor.struct_size = sizeof(descriptor);
#ifdef AMIO_TEST_HAS_MPI
    int rank_count = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    descriptor.rank_count = 1;
    descriptor.fortran_communicator = static_cast<int64_t>(MPI_Comm_c2f(MPI_COMM_SELF));
#else
    descriptor.rank_count = 1;
#endif
    descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1 + 1;
    expect(amio_init_ex(&descriptor, &context) == AMIO_ERR_INVALID_INPUT, "unknown context descriptor version is rejected");
    expect(context == nullptr, "rejected context initialization clears its output handle");

    size_t required = 19;
    expect(amio_query_capabilities(nullptr, nullptr, 0, &required) == AMIO_ERR_NULL_HANDLE, "capability query rejects a null context");
    expect(required == 0, "rejected capability query clears its required length");

    descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
    descriptor.execution_mode = AMIO_EXECUTION_PERSISTENT_DIRECT;
    descriptor.progress_mode = AMIO_PROGRESS_CALLER;
#ifdef AMIO_TEST_HAS_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    descriptor.rank_count = static_cast<uint32_t>(rank_count);
    const auto caller_communicator = rank_count == 1 ? MPI_COMM_SELF : MPI_COMM_WORLD;
    descriptor.fortran_communicator = static_cast<int64_t>(MPI_Comm_c2f(caller_communicator));
#else
    descriptor.rank_count = 1;
#endif
    constexpr char canonical[] = "{}";
    descriptor.canonical_configuration = canonical;
    descriptor.configuration_bytes = sizeof(canonical) - 1;
    expect(amio_init_ex(&descriptor, &context) == AMIO_OK, "valid direct context initializes");
    expect(context != nullptr, "successful initialization returns a context handle");
    if (context != nullptr) {
        uint32_t local_role = AMIO_ROLE_INACTIVE;
        uint32_t local_group_rank = UINT32_MAX;
        expect(amio_query_context_role(context, &local_role, &local_group_rank) == AMIO_OK && local_role == AMIO_ROLE_PRODUCER &&
                   local_group_rank == static_cast<uint32_t>(
#ifdef AMIO_TEST_HAS_MPI
                                           rank
#else
                                           0
#endif
                                           ),
               "role query reports the local producer role and producer-group rank");
        required = 0;
        expect(amio_query_capabilities(context, nullptr, 0, &required) == AMIO_OK && required > 1, "capability query supports size discovery");
        std::vector<char> report(required);
        expect(amio_query_capabilities(context, report.data(), report.size(), &required) == AMIO_OK, "capability query fills a caller buffer");
        expect(std::strstr(report.data(), "borrowed_buffer\":\"unsupported") != nullptr,
               "capability report rejects unimplemented borrowed buffers explicitly");
        expect(std::strstr(report.data(), "f64_f32_precision\":\"compiled") != nullptr &&
                   std::strstr(report.data(), "logical_integrity\":\"compiled") != nullptr &&
                   std::strstr(report.data(), "native_lossless_codec_selection\":\"destination-negotiated") != nullptr &&
                   std::strstr(report.data(), "native_lossy_codec\":\"unsupported") != nullptr,
               "capability report identifies compiled precision/integrity and unsupported per-field lossy codecs");
        size_t diagnostic_bytes = 0;
        expect(amio_query_diagnostics(context, nullptr, 0, &diagnostic_bytes) == AMIO_OK && diagnostic_bytes > 1,
               "local diagnostics support a bounded size query");
        std::vector<char> diagnostic_report(diagnostic_bytes);
        expect(amio_query_diagnostics(context, diagnostic_report.data(), diagnostic_report.size(), &diagnostic_bytes) == AMIO_OK,
               "local diagnostics fill a caller-owned buffer");
        expect(std::strstr(diagnostic_report.data(), "\"api_version\":2") != nullptr &&
                   std::strstr(diagnostic_report.data(), "\"topology\":{\"execution_mode\":\"persistent_direct\",\"progress_mode\":\"caller\"") !=
                       nullptr &&
                   std::strstr(diagnostic_report.data(), "\"rank_count\":") != nullptr &&
                   std::strstr(diagnostic_report.data(), "\"dropped_sample_count\":0") != nullptr,
               "diagnostics expose a versioned snapshot with effective topology");
        char too_small[1]{};
        expect(amio_query_diagnostics(context, too_small, sizeof(too_small), &diagnostic_bytes) == AMIO_ERR_BUFFER_TOO_SMALL,
               "diagnostics reject undersized output buffers and report the required size");
        size_t reduced_bytes = 0;
        expect(amio_reduce_diagnostics(context, nullptr, 0, &reduced_bytes) == AMIO_OK && reduced_bytes > 1,
               "collective diagnostic reduction has a local size query");
        expect(amio_reduce_diagnostics(context, too_small, sizeof(too_small), &reduced_bytes) == AMIO_ERR_BUFFER_TOO_SMALL,
               "diagnostic reduction agrees on undersized buffers before entering counter reductions");
        std::vector<char> reduced_report(reduced_bytes);
        expect(amio_reduce_diagnostics(context, reduced_report.data(), reduced_report.size(), &reduced_bytes) == AMIO_OK,
               "all context members can collectively reduce cumulative counters");
        expect(std::strstr(reduced_report.data(), "\"api_version\":2") != nullptr &&
                   std::strstr(reduced_report.data(), "\"scope\":\"context\"") != nullptr &&
                   std::strstr(reduced_report.data(), "\"accepted\":\"00000000000000000000\"") != nullptr &&
                   std::strstr(reduced_report.data(), "\"operator_input_bytes\":") != nullptr &&
                   std::strstr(reduced_report.data(), "\"operator_output_bytes\":") != nullptr &&
                   std::strlen(reduced_report.data()) + 1 == reduced_bytes,
               "reduction returns every fixed-width counter and reports its exact buffer size");
        const auto root = std::filesystem::temp_directory_path() / "amio-evolution-contract-collection";
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
        const auto location = root.string();
        amio_dataset_descriptor_t dataset_descriptor{};
        dataset_descriptor.struct_size = sizeof(dataset_descriptor);
        dataset_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        dataset_descriptor.location = location.data();
        dataset_descriptor.location_bytes = location.size();
        dataset_descriptor.format = AMIO_FORMAT_NETCDF4;
        dataset_descriptor.budget_bytes = 1024 * 1024;
        dataset_descriptor.budget_descriptors = 128;
        dataset_descriptor.control_budget_bytes = 4096;
        dataset_descriptor.control_budget_descriptors = 8;

        amio_dataset_handle dataset = reinterpret_cast<amio_dataset_handle>(static_cast<uintptr_t>(0x1234));
#ifndef AMIO_HAS_TENSORSTORE
        auto unsupported = dataset_descriptor;
        unsupported.format = AMIO_FORMAT_ZARR3;
        expect(amio_open_dataset_ex(context, &unsupported, &dataset) == AMIO_ERR_UNSUPPORTED, "unavailable evolution backend is rejected explicitly");
        expect(dataset == nullptr, "rejected evolution dataset open clears its output handle");
#endif
        const auto dataset_open_status = amio_open_dataset_ex(context, &dataset_descriptor, &dataset);
        if (dataset_open_status != AMIO_OK)
            std::fprintf(stderr, "dataset open status: %d, root=%d, marker=%d\n", dataset_open_status, std::filesystem::exists(root),
                         std::filesystem::exists(root / "amio.json"));
        expect(dataset_open_status == AMIO_OK, "valid NetCDF evolution dataset claims a new exclusive collection");
        expect(dataset != nullptr, "successful evolution dataset open returns a typed dataset handle");
        if (dataset != nullptr) {
            amio_catalog_cursor_t empty_cursor{};
            amio_step_summary_t unused_summary{};
            size_t empty_count = 99;
            int empty_more = 1;
            expect(amio_list_steps(dataset, nullptr, 1, &unused_summary, 1, &empty_count, &empty_cursor, &empty_more) == AMIO_OK &&
                       empty_count == 0 && empty_more == 0 && empty_cursor.dataset_uuid[0] != '\0',
                   "empty collection listing succeeds with an initialized dataset cursor");
        }
        expect(amio_release_context(context) == AMIO_ERR_INVALID_POLICY, "context cannot release its communicator while a dataset borrows it");

        amio_validation_request_t validation{};
        validation.struct_size = sizeof(validation);
        validation.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        validation.check_mask = (UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1)) | (UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1));
        validation.max_results = 2;
        size_t result_count = 0;
        uint64_t next_cursor = 0;
        expect(amio_validate_dataset(dataset, &validation, nullptr, 0, &result_count, &next_cursor) == AMIO_OK && result_count == 2,
               "evolution datasets support bounded validation page size queries");
        amio_validation_result_t validation_results[2]{};
        expect(amio_validate_dataset(dataset, &validation, validation_results, 2, &result_count, &next_cursor) == AMIO_OK,
               "evolution dataset validation returns a bounded page");
        expect(validation_results[0].status == AMIO_VALIDATION_PASS && validation_results[1].status == AMIO_VALIDATION_UNSUPPORTED,
               "structural completion passes while an empty collection has no integrity evidence");

        const char dimension_name[] = "x";
        amio_dimension_descriptor_t dimension{};
        dimension.struct_size = sizeof(dimension);
        dimension.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        dimension.name = dimension_name;
        dimension.name_bytes = sizeof(dimension_name) - 1;
        dimension.extent = 4;

        const char variable_name[] = "temperature";
        const char attribute_name[] = "units";
        const char attribute_value[] = "K";
        amio_attribute_descriptor_t attribute{};
        attribute.struct_size = sizeof(attribute);
        attribute.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        attribute.name = attribute_name;
        attribute.name_bytes = sizeof(attribute_name) - 1;
        attribute.type = AMIO_ATTRIBUTE_UTF8;
        attribute.element_count = 1;
        attribute.value = attribute_value;
        attribute.value_bytes = sizeof(attribute_value) - 1;

        amio_variable_descriptor_t variable{};
        variable.struct_size = sizeof(variable);
        variable.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        variable.name = variable_name;
        variable.name_bytes = sizeof(variable_name) - 1;
        variable.source_dtype = AMIO_DTYPE_F32;
        variable.stored_dtype = AMIO_DTYPE_F32;
        variable.required = 1;
        variable.dimension_count = 1;
        variable.dimensions = &dimension;
        variable.attribute_count = 1;
        variable.attributes = &attribute;

        const char optional_variable_name[] = "optional_temperature";
        amio_variable_descriptor_t optional_variable = variable;
        optional_variable.name = optional_variable_name;
        optional_variable.name_bytes = sizeof(optional_variable_name) - 1;
        optional_variable.required = 0;
        optional_variable.attribute_count = 0;
        optional_variable.attributes = nullptr;
        const char precision_name[] = "amio.precision";
        const char precision_value[] = "f64_to_f32_absolute";
        const char tolerance_name[] = "amio.tolerance";
        const char field_authorization_name[] = "amio.authorize_field_loss";
        const char restart_authorization_name[] = "amio.authorize_lossy_restart";
        const double conversion_tolerance = 0.00001;
        const uint8_t authorized = 1;
        amio_attribute_descriptor_t precision_attributes[4]{};
        precision_attributes[0] = {sizeof(amio_attribute_descriptor_t),
                                   AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
                                   precision_name,
                                   sizeof(precision_name) - 1,
                                   AMIO_ATTRIBUTE_UTF8,
                                   0,
                                   1,
                                   precision_value,
                                   sizeof(precision_value) - 1,
                                   {0, 0}};
        precision_attributes[1] = {sizeof(amio_attribute_descriptor_t),
                                   AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
                                   tolerance_name,
                                   sizeof(tolerance_name) - 1,
                                   AMIO_ATTRIBUTE_F64,
                                   0,
                                   1,
                                   &conversion_tolerance,
                                   sizeof(conversion_tolerance),
                                   {0, 0}};
        precision_attributes[2] = {sizeof(amio_attribute_descriptor_t),
                                   AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
                                   field_authorization_name,
                                   sizeof(field_authorization_name) - 1,
                                   AMIO_ATTRIBUTE_BOOL,
                                   0,
                                   1,
                                   &authorized,
                                   sizeof(authorized),
                                   {0, 0}};
        precision_attributes[3] = {sizeof(amio_attribute_descriptor_t),
                                   AMIO_EVOLUTION_DESCRIPTOR_VERSION_1,
                                   restart_authorization_name,
                                   sizeof(restart_authorization_name) - 1,
                                   AMIO_ATTRIBUTE_BOOL,
                                   0,
                                   1,
                                   &authorized,
                                   sizeof(authorized),
                                   {0, 0}};
        const char low_precision_name[] = "low_precision";
        auto low_precision_variable = variable;
        low_precision_variable.name = low_precision_name;
        low_precision_variable.name_bytes = sizeof(low_precision_name) - 1;
        low_precision_variable.source_dtype = AMIO_DTYPE_F64;
        low_precision_variable.stored_dtype = AMIO_DTYPE_F32;
        low_precision_variable.required = 0;
        low_precision_variable.attribute_count = 4;
        low_precision_variable.attributes = precision_attributes;
        const amio_variable_descriptor_t variables[] = {variable, optional_variable, low_precision_variable};

        amio_schema_descriptor_t schema_descriptor{};
        schema_descriptor.struct_size = sizeof(schema_descriptor);
        schema_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
        schema_descriptor.variable_count = 3;
        schema_descriptor.variables = variables;
        amio_schema_handle schema = nullptr;
        expect(amio_define_schema(dataset, &schema_descriptor, &schema) == AMIO_OK,
               "typed schema is copied, sealed, and registered before the first step");
        expect(schema != nullptr, "successful schema registration returns a typed schema handle");
        // Repeated definitions must share the dataset's retained immutable
        // schema even while callers keep older observation handles alive.
        amio_schema_handle repeated_schemas[8]{};
        for (auto &repeated : repeated_schemas) {
            expect(amio_define_schema(dataset, &schema_descriptor, &repeated) == AMIO_OK && repeated != nullptr,
                   "equivalent schema definitions remain usable with existing handles");
        }
        for (auto repeated : repeated_schemas) expect(amio_release_schema(repeated) == AMIO_OK, "release repeated schema observation");
        {
            std::string oversized_value(8192, 'x');
            auto oversized_attribute = attribute;
            oversized_attribute.value = oversized_value.data();
            oversized_attribute.value_bytes = oversized_value.size();
            auto oversized_variable = variable;
            oversized_variable.attributes = &oversized_attribute;
            auto oversized_schema = schema_descriptor;
            oversized_schema.variable_count = 1;
            oversized_schema.variables = &oversized_variable;
            amio_schema_handle rejected = reinterpret_cast<amio_schema_handle>(static_cast<uintptr_t>(1));
            expect(amio_define_schema(dataset, &oversized_schema, &rejected) == AMIO_ERR_BUDGET_EXCEEDED && rejected == nullptr,
                   "schema input exceeding the control budget is rejected before copying its value");
        }
        {
            const auto tiny_location_path = root / "tiny-metadata-budget";
            const auto tiny_location = tiny_location_path.string();
            auto tiny_budget_descriptor = dataset_descriptor;
            tiny_budget_descriptor.location = tiny_location.data();
            tiny_budget_descriptor.location_bytes = tiny_location.size();
            tiny_budget_descriptor.control_budget_bytes = 1;
            tiny_budget_descriptor.control_budget_descriptors = 1;
            amio_dataset_handle tiny_dataset = nullptr;
            expect(amio_open_dataset_ex(context, &tiny_budget_descriptor, &tiny_dataset) == AMIO_OK && tiny_dataset != nullptr,
                   "dataset accepts a finite control budget before schema metadata is admitted");
            if (tiny_dataset != nullptr) {
                amio_schema_handle rejected_schema = reinterpret_cast<amio_schema_handle>(static_cast<uintptr_t>(0x1234));
                expect(
                    amio_define_schema(tiny_dataset, &schema_descriptor, &rejected_schema) == AMIO_ERR_BUDGET_EXCEEDED && rejected_schema == nullptr,
                    "schema retention is rejected when its metadata exceeds the configured control budget");
                expect(amio_close_dataset_ex(tiny_dataset) == AMIO_OK, "dataset with rejected schema metadata closes without a retained handle");
            }
        }
        if (schema != nullptr) {
            amio_step_descriptor_t step_descriptor{};
            step_descriptor.struct_size = sizeof(step_descriptor);
            step_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            step_descriptor.step_id = 0;
            step_descriptor.model_time = 12.5;
            step_descriptor.schema = schema;
            amio_step_handle step = nullptr;
            expect(amio_begin_step(dataset, &step_descriptor, &step) == AMIO_OK && step != nullptr,
                   "public lifecycle begins a step with a registered schema");
            if (step != nullptr) {
                const float expected[] = {1.25f, 2.5f, 3.75f, 5.0f};
                amio_memory_descriptor_t host_memory{};
                host_memory.struct_size = sizeof(host_memory);
                host_memory.version = 1;
                host_memory.memory_kind = 0;
                host_memory.ownership = 0;
                host_memory.rank = 2;
                host_memory.producer_writes_synchronized = 1;
                host_memory.element_bytes = sizeof(float);
                host_memory.extents[0] = 2;
                host_memory.extents[1] = 2;
                host_memory.strides[0] = 2 * sizeof(float);
                host_memory.strides[1] = sizeof(float);
                amio_operation_handle operation = nullptr;
                expect(amio_put_field_memory(step, variable_name, sizeof(variable_name) - 1, &host_memory, expected, sizeof(expected), &operation) ==
                               AMIO_OK &&
                           operation != nullptr,
                       "typed host-memory submission snapshots a complete field and returns an operation handle");
                const double low_precision_values[] = {1.1, 2.2, 3.3, -0.0};
                amio_operation_handle precision_operation = nullptr;
                expect(amio_put_field(step, low_precision_name, sizeof(low_precision_name) - 1, low_precision_values, sizeof(low_precision_values),
                                      &precision_operation) == AMIO_OK &&
                           precision_operation != nullptr,
                       "authorized F64 input is admitted for bounded F32 storage conversion");
                amio_operation_status_t operation_status{};
                expect(amio_query_operation(operation, &operation_status) == AMIO_OK && operation_status.outcome == AMIO_OPERATION_PENDING &&
                           (operation_status.milestones & AMIO_OPERATION_SOURCE_RELEASED) != 0,
                       "operation status distinguishes source reuse from publication");
                expect(amio_wait_operation(operation, AMIO_OPERATION_SOURCE_RELEASED, 0) == AMIO_OK, "wait can observe source release independently");
                expect(amio_wait_operation(operation, AMIO_OPERATION_BACKEND_COMPLETE, 0) == AMIO_ERR_TIMEOUT,
                       "a timed-out completion wait leaves accepted snapshot work pending");
                expect(amio_close_dataset_ex(dataset) == AMIO_ERR_INVALID_POLICY,
                       "dataset close cannot release accepted work owned by an active step");
                expect(amio_query_operation(operation, &operation_status) == AMIO_OK && operation_status.outcome == AMIO_OPERATION_PENDING &&
                           (operation_status.milestones & AMIO_OPERATION_SOURCE_RELEASED) != 0,
                       "rejected close preserves the accepted operation and its source-release state");
                expect(amio_release_step_handle(step) == AMIO_ERR_INVALID_POLICY,
                       "an active synchronous step cannot be released and stranded before sealing");
                const auto end_status = amio_end_step(step);
                if (end_status != AMIO_OK) std::fprintf(stderr, "end step status=%d\n", end_status);
                expect(end_status == AMIO_OK, "end publishes the closed native payload and completion record");
                diagnostic_bytes = 0;
                expect(amio_query_diagnostics(context, nullptr, 0, &diagnostic_bytes) == AMIO_OK && diagnostic_bytes > 1,
                       "diagnostics remain locally queryable after step publication");
                diagnostic_report.assign(diagnostic_bytes, '\0');
                expect(amio_query_diagnostics(context, diagnostic_report.data(), diagnostic_report.size(), &diagnostic_bytes) == AMIO_OK,
                       "published-step diagnostics fill the caller buffer");
                expect(std::strstr(diagnostic_report.data(), "\"bytes_accepted\":48") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"bytes_published\":48") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"step_id\":0") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"cumulative_histograms\":[") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"histogram_names\":[") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"resource_names\":[") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"cumulative_resource_high_water\":[0,48,") != nullptr &&
                           std::strstr(diagnostic_report.data(), "\"resource_high_water\":[0,48,") != nullptr,
                       "diagnostics report step counters and lifetime timing/resource aggregates after scope eviction");
                std::uint64_t event_history_high_water = 0;
                expect(read_json_uint64_array_item(diagnostic_report.data(), "cumulative_resource_high_water", 10, event_history_high_water) &&
                           event_history_high_water > 0,
                       "public diagnostics JSON reports bounded event-history occupancy");
                expect(amio_cancel_step(step) == AMIO_ERR_INVALID_POLICY, "a completed step cannot be rolled back by cancellation");
                expect(amio_query_operation(operation, &operation_status) == AMIO_OK && operation_status.outcome == AMIO_OPERATION_SUCCEEDED &&
                           (operation_status.milestones & AMIO_OPERATION_READER_VISIBLE) != 0,
                       "successful end reports reader visibility separately from source release");
                expect(amio_wait_operation(operation, AMIO_OPERATION_READER_VISIBLE, 0) == AMIO_OK,
                       "wait can observe publication visibility independently");
                expect(amio_wait_operation(operation, AMIO_OPERATION_DURABLE, 0) == AMIO_ERR_UNSUPPORTED,
                       "visibility does not falsely imply stable-storage durability");

                amio_step_summary_t summary{};
                expect(amio_describe_step(dataset, 0, &summary) == AMIO_OK && summary.step_id == 0 && summary.present_field_count == 2 &&
                           summary.required_field_count == 1,
                       "step description returns committed field-presence metadata");
                amio_catalog_cursor_t cursor{};
                amio_step_summary_t listed[1]{};
                size_t listed_count = 0;
                int has_more = 0;
                expect(amio_list_steps(dataset, nullptr, 1, listed, 1, &listed_count, &cursor, &has_more) == AMIO_OK && listed_count == 1 &&
                           listed[0].step_id == 0 && has_more == 0 && cursor.dataset_uuid[0] != '\0',
                       "bounded list returns step summaries and a dataset-bound cursor");
                float readback[4]{};
                size_t read_bytes = 0;
                expect(amio_read_step_field(dataset, 0, variable_name, sizeof(variable_name) - 1, nullptr, nullptr, 0, &read_bytes) == AMIO_OK &&
                           read_bytes == sizeof(readback),
                       "read supports a bounded size query before allocation");
                expect(amio_read_step_field(dataset, 0, variable_name, sizeof(variable_name) - 1, nullptr, readback, sizeof(readback), &read_bytes) ==
                           AMIO_OK,
                       "read returns the complete committed field");
                for (std::size_t i = 0; i < 4; ++i) expect(readback[i] == expected[i], "public read returns the native field value");
                float converted_readback[4]{};
                expect(amio_read_step_field(dataset, 0, low_precision_name, sizeof(low_precision_name) - 1, nullptr, converted_readback,
                                            sizeof(converted_readback), &read_bytes) == AMIO_OK &&
                           read_bytes == sizeof(converted_readback) && converted_readback[0] == static_cast<float>(1.1) &&
                           std::signbit(converted_readback[3]),
                       "stored F32 conversion is visible through the public reader with signed zero preserved");
                float cached_readback[4]{};
                expect(amio_read_step_field(dataset, 0, variable_name, sizeof(variable_name) - 1, nullptr, cached_readback, sizeof(cached_readback),
                                            &read_bytes) == AMIO_OK &&
                           std::equal(std::begin(readback), std::end(readback), std::begin(cached_readback)),
                       "repeated full-field reads preserve values through the step-aware cache");
                amio_step_selection_t selection{};
                selection.struct_size = sizeof(selection);
                selection.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                selection.rank = 1;
                selection.offsets[0] = 1;
                selection.extents[0] = 2;
                selection.strides[0] = 1;
                float selected[2]{};
                expect(amio_read_step_field(dataset, 0, variable_name, sizeof(variable_name) - 1, &selection, selected, sizeof(selected),
                                            &read_bytes) == AMIO_OK &&
                           selected[0] == expected[1] && selected[1] == expected[2],
                       "selected read returns the requested contiguous region");
                float cached_selected[2]{};
                expect(amio_read_step_field(dataset, 0, variable_name, sizeof(variable_name) - 1, &selection, cached_selected,
                                            sizeof(cached_selected), &read_bytes) == AMIO_OK &&
                           cached_selected[0] == selected[0] && cached_selected[1] == selected[1],
                       "repeated selected reads reuse the selection-specific cache entry");
                diagnostic_bytes = 0;
                expect(amio_query_diagnostics(context, nullptr, 0, &diagnostic_bytes) == AMIO_OK && diagnostic_bytes > 1,
                       "diagnostics remain queryable after cached reads");
                diagnostic_report.assign(diagnostic_bytes, '\0');
                expect(amio_query_diagnostics(context, diagnostic_report.data(), diagnostic_report.size(), &diagnostic_bytes) == AMIO_OK,
                       "cached-read diagnostics fill the caller buffer");
                std::uint64_t cache_high_water = 0;
                expect(std::strstr(diagnostic_report.data(), "\"prefetch_bytes\"") != nullptr &&
                           read_json_uint64_array_item(diagnostic_report.data(), "cumulative_resource_high_water", 6, cache_high_water) &&
                           cache_high_water == sizeof(readback) + sizeof(converted_readback) + sizeof(selected),
                       "diagnostics report bounded step-read cache occupancy in the prefetch high-water dimension");
                expect(amio_read_step_field(dataset, 0, optional_variable_name, sizeof(optional_variable_name) - 1, nullptr, nullptr, 0,
                                            &read_bytes) == AMIO_ERR_FIELD_NOT_PRESENT,
                       "an absent optional field is reported explicitly");

                validation.check_mask = (UINT64_C(1) << (AMIO_VALIDATE_REQUIRED_FIELDS - 1)) | (UINT64_C(1) << (AMIO_VALIDATE_PAYLOAD_PRESENCE - 1));
                expect(amio_validate_dataset(dataset, &validation, validation_results, 2, &result_count, &next_cursor) == AMIO_OK &&
                           validation_results[0].status == AMIO_VALIDATION_PASS && validation_results[1].status == AMIO_VALIDATION_PASS,
                       "bounded structural validation confirms required fields and payload presence");
#ifdef AMIO_TEST_HAS_NETCDF
                const auto payload = root / "payloads" / "steps" / "00000000000000000000.nc";
                int reader = -1;
                expect(nc_open(payload.c_str(), NC_NOWRITE, &reader) == NC_NOERR, "published step is independently openable by native NetCDF-C");
                if (reader >= 0) {
                    int variable_id = -1;
                    float actual[4]{};
                    expect(nc_inq_varid(reader, variable_name, &variable_id) == NC_NOERR && nc_get_var_float(reader, variable_id, actual) == NC_NOERR,
                           "native reader finds and reads the named step field");
                    for (std::size_t i = 0; i < 4; ++i) expect(actual[i] == expected[i], "native reader returns the submitted field value");
                    int precision_varid = -1;
                    float native_converted[4]{};
                    expect(nc_inq_varid(reader, low_precision_name, &precision_varid) == NC_NOERR &&
                               nc_get_var_float(reader, precision_varid, native_converted) == NC_NOERR &&
                               native_converted[0] == static_cast<float>(1.1) && std::signbit(native_converted[3]),
                           "independent NetCDF reader verifies converted values and signed zero");
                    nc_close(reader);
                }
                expect(std::filesystem::exists(root / "head.json") && std::filesystem::exists(root / "commits" / "0.json"),
                       "completion record and authoritative head exist after successful publication");
#endif
                expect(amio_release_operation(operation) == AMIO_OK, "operation observation has a distinct release operation");
                expect(amio_release_operation(precision_operation) == AMIO_OK, "precision conversion operation observation can be released");
                expect(amio_release_step_handle(step) == AMIO_OK, "producer step handle has a distinct release operation");

#ifdef AMIO_TEST_HAS_NETCDF
                std::atomic<bool> model_time_read_started{false};
                std::atomic<bool> second_step_write_finished{false};
                std::atomic<bool> model_time_read_failed{false};
                std::thread model_time_reader([&] {
                    model_time_read_started.store(true, std::memory_order_release);
                    do {
                        double model_time = 0.0;
                        if (amio_get_step_model_time(dataset, 0, &model_time) != AMIO_OK || model_time != 12.5) {
                            model_time_read_failed.store(true, std::memory_order_release);
                            break;
                        }
                    } while (!second_step_write_finished.load(std::memory_order_acquire));
                });
                while (!model_time_read_started.load(std::memory_order_acquire)) std::this_thread::yield();
#endif
                step_descriptor.step_id = 1;
                amio_step_handle second_step = nullptr;
                expect(amio_begin_step(dataset, &step_descriptor, &second_step) == AMIO_OK && second_step != nullptr,
                       "a later step can reuse the registered immutable schema");
                if (second_step != nullptr) {
                    amio_operation_handle second_operation = nullptr;
                    expect(amio_put_field(second_step, variable_name, sizeof(variable_name) - 1, expected, sizeof(expected), &second_operation) ==
                               AMIO_OK,
                           "later step accepts its required field");
                    expect(amio_end_step(second_step) == AMIO_OK, "later step publishes after linking the prior commit");
                    expect(amio_release_operation(second_operation) == AMIO_OK, "later operation observation is released");
                    expect(amio_release_step_handle(second_step) == AMIO_OK, "later producer handle is released");
                }
#ifdef AMIO_TEST_HAS_NETCDF
                second_step_write_finished.store(true, std::memory_order_release);
                model_time_reader.join();
                expect(!model_time_read_failed.load(std::memory_order_acquire),
                       "model-time reads remain valid while a later NetCDF step is being written");
#endif
                amio_catalog_cursor_t first_page_cursor{};
                amio_step_summary_t first_page[1]{};
                size_t first_page_count = 0;
                int first_page_more = 0;
                expect(amio_list_steps(dataset, nullptr, 1, first_page, 1, &first_page_count, &first_page_cursor, &first_page_more) == AMIO_OK &&
                           first_page_count == 1 && first_page[0].step_id == 1 && first_page_more == 1,
                       "bounded listing returns the head page and continuation cursor");
                amio_catalog_cursor_t final_cursor{};
                amio_step_summary_t second_page[1]{};
                size_t second_page_count = 0;
                int second_page_more = 1;
                expect(amio_list_steps(dataset, &first_page_cursor, 1, second_page, 1, &second_page_count, &final_cursor, &second_page_more) ==
                               AMIO_OK &&
                           second_page_count == 1 && second_page[0].step_id == 0 && second_page_more == 0,
                       "continuation cursor returns the older committed step exactly once");

                amio_step_descriptor_t cancelled_descriptor{};
                cancelled_descriptor.struct_size = sizeof(cancelled_descriptor);
                cancelled_descriptor.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
                cancelled_descriptor.step_id = 2;
                cancelled_descriptor.model_time = 13.0;
                cancelled_descriptor.schema = schema;
                amio_step_handle cancelled_step = nullptr;
                expect(amio_begin_step(dataset, &cancelled_descriptor, &cancelled_step) == AMIO_OK && cancelled_step != nullptr,
                       "a later logical step can begin after publication");
                if (cancelled_step != nullptr) {
                    amio_operation_handle cancelled_operation = nullptr;
                    expect(amio_put_field(cancelled_step, variable_name, sizeof(variable_name) - 1, expected, sizeof(expected),
                                          &cancelled_operation) == AMIO_OK,
                           "cancel path admits and snapshots a field before cancellation");
                    expect(amio_cancel_step(cancelled_step) == AMIO_OK, "an open public step can be explicitly cancelled before publication");
                    amio_operation_status_t cancelled_status{};
                    expect(
                        amio_query_operation(cancelled_operation, &cancelled_status) == AMIO_OK && cancelled_status.outcome == AMIO_OPERATION_FAILED,
                        "cancelling a step fails its accepted operation explicitly");
                    expect(amio_release_operation(cancelled_operation) == AMIO_OK, "a cancelled operation can release its observer handle");
                    expect(amio_release_step_handle(cancelled_step) == AMIO_OK, "a cancelled producer step can release its handle");
                }
                amio_step_handle reused_cancelled_id = nullptr;
                expect(amio_begin_step(dataset, &cancelled_descriptor, &reused_cancelled_id) == AMIO_ERR_INVALID_POLICY &&
                           reused_cancelled_id == nullptr,
                       "a cancelled step ID cannot be reused");
                amio_catalog_cursor_t after_cancel_cursor{};
                amio_step_summary_t after_cancel_steps[3]{};
                size_t after_cancel_count = 0;
                int after_cancel_more = 0;
                expect(amio_list_steps(dataset, nullptr, 3, after_cancel_steps, 3, &after_cancel_count, &after_cancel_cursor, &after_cancel_more) ==
                               AMIO_OK &&
                           after_cancel_count == 2 && after_cancel_steps[0].step_id == 1 && after_cancel_steps[1].step_id == 0,
                       "a cancelled step remains absent from complete-step discovery");

                amio_step_descriptor_t incomplete_descriptor = cancelled_descriptor;
                incomplete_descriptor.step_id = 3;
                incomplete_descriptor.model_time = 14.0;
                amio_step_handle incomplete_step = nullptr;
                expect(amio_begin_step(dataset, &incomplete_descriptor, &incomplete_step) == AMIO_OK && incomplete_step != nullptr,
                       "a step can begin before discovering a required contribution is missing");
                if (incomplete_step != nullptr) {
                    expect(amio_end_step(incomplete_step) == AMIO_ERR_FIELD_NOT_PRESENT, "an incomplete step cannot be implicitly sealed");
                    expect(amio_release_step_handle(incomplete_step) == AMIO_OK, "a failed terminal step releases its producer handle");
                }
                amio_step_handle reused_failed_id = nullptr;
                expect(amio_begin_step(dataset, &incomplete_descriptor, &reused_failed_id) == AMIO_ERR_INVALID_POLICY && reused_failed_id == nullptr,
                       "a failed step ID cannot be reused");
                amio_step_summary_t after_failure_steps[3]{};
                size_t after_failure_count = 0;
                amio_catalog_cursor_t after_failure_cursor{};
                int after_failure_more = 0;
                expect(amio_list_steps(dataset, nullptr, 3, after_failure_steps, 3, &after_failure_count, &after_failure_cursor,
                                       &after_failure_more) == AMIO_OK &&
                           after_failure_count == 2 && after_failure_steps[0].step_id == 1 && after_failure_steps[1].step_id == 0,
                       "an incomplete failed step remains absent from complete-step discovery");
            }
        }
        expect(amio_release_schema(schema) == AMIO_OK, "caller can release its schema observation handle");
#ifdef AMIO_TEST_HAS_NETCDF
        validation.check_mask = UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1);
        expect(amio_validate_dataset(dataset, &validation, validation_results, 2, &result_count, &next_cursor) == AMIO_OK && result_count == 1 &&
                   validation_results[0].status == AMIO_VALIDATION_PASS,
               "integrity validation verifies canonical logical bytes for every committed payload");
        const auto corrupted_payload = root / "payloads" / "steps" / "00000000000000000000.nc";
        int corrupt_ncid = -1, corrupt_varid = -1;
        float corrupted_values[4]{};
        bool corrupted = nc_open(corrupted_payload.c_str(), NC_WRITE, &corrupt_ncid) == NC_NOERR;
        if (corrupted) corrupted = nc_inq_varid(corrupt_ncid, variable_name, &corrupt_varid) == NC_NOERR;
        if (corrupted) corrupted = nc_get_var_float(corrupt_ncid, corrupt_varid, corrupted_values) == NC_NOERR;
        if (corrupted) {
            corrupted_values[0] += 1.0f;
            corrupted = nc_put_var_float(corrupt_ncid, corrupt_varid, corrupted_values) == NC_NOERR;
        }
        if (corrupt_ncid >= 0) nc_close(corrupt_ncid);
        expect(corrupted, "test fixture can mutate a committed native payload");
        expect(amio_validate_dataset(dataset, &validation, validation_results, 2, &result_count, &next_cursor) == AMIO_OK && result_count == 1 &&
                   validation_results[0].status == AMIO_VALIDATION_FAIL,
               "integrity validation rejects a corrupted stored field");
#endif
        amio_dataset_handle reader = nullptr;
        const auto reader_status = amio_open_dataset_reader(context, location.data(), location.size(), &reader);
        if (reader_status != AMIO_OK) std::fprintf(stderr, "reader open status: %d\n", reader_status);
        expect(reader_status == AMIO_OK && reader != nullptr, "read-only utility handle reopens an existing collection and its schemas");
        if (reader != nullptr) {
            float reopened_values[4]{};
            size_t reopened_bytes = 0;
            expect(amio_read_step_field(reader, 0, variable_name, sizeof(variable_name) - 1, nullptr, reopened_values, sizeof(reopened_values),
                                        &reopened_bytes) == AMIO_OK &&
                       reopened_bytes == sizeof(reopened_values) && reopened_values[0] == 2.25f,
                   "reopened collection supports public bounded native field reads");
            amio_validation_request_t reopened_validation{};
            reopened_validation.struct_size = sizeof(reopened_validation);
            reopened_validation.version = AMIO_EVOLUTION_DESCRIPTOR_VERSION_1;
            reopened_validation.check_mask = (UINT64_C(1) << (AMIO_VALIDATE_COMPLETION_CHAIN - 1)) |
                                             (UINT64_C(1) << (AMIO_VALIDATE_REQUIRED_FIELDS - 1)) |
                                             (UINT64_C(1) << (AMIO_VALIDATE_PAYLOAD_PRESENCE - 1));
            reopened_validation.max_results = 3;
            amio_validation_result_t reopened_results[3]{};
            size_t reopened_count = 0;
            uint64_t reopened_cursor = 0;
            expect(amio_validate_dataset(reader, &reopened_validation, reopened_results, 3, &reopened_count, &reopened_cursor) == AMIO_OK &&
                       reopened_count == 3 && reopened_results[0].status == AMIO_VALIDATION_PASS && reopened_cursor == 0,
                   "reopened collection exposes the public structural validation report");
            reopened_validation.check_mask = UINT64_C(1) << (AMIO_VALIDATE_INTEGRITY - 1);
            reopened_validation.max_results = 1;
            expect(amio_validate_dataset(reader, &reopened_validation, reopened_results, 3, &reopened_count, &reopened_cursor) == AMIO_OK &&
                       reopened_count == 1 && reopened_results[0].status == AMIO_VALIDATION_FAIL,
                   "reopened collection integrity validation detects changed committed bytes");
            amio_schema_handle forbidden_schema = reinterpret_cast<amio_schema_handle>(static_cast<uintptr_t>(1));
            expect(amio_define_schema(reader, &schema_descriptor, &forbidden_schema) == AMIO_ERR_INVALID_POLICY && forbidden_schema == nullptr,
                   "read-only utility handle rejects schema mutation");
            expect(amio_close_dataset_ex(reader) == AMIO_OK, "read-only utility handle closes cleanly");
        }
        amio_step_lease_handle persistent_lease = nullptr;
        expect(amio_acquire_step_lease(dataset, 0, &persistent_lease) == AMIO_OK && persistent_lease != nullptr,
               "published step can be held by a persistent consumer lease");
        if (persistent_lease != nullptr) {
            size_t stream_read_bytes = 0;
            expect(amio_stream_read(persistent_lease, nullptr, nullptr, 0, &stream_read_bytes) == AMIO_ERR_INVALID_HANDLE,
                   "persistent lease is rejected by stream reads");
            expect(amio_stream_release(persistent_lease) == AMIO_ERR_INVALID_HANDLE, "persistent lease is rejected by stream release");
            expect(amio_release_step_lease(persistent_lease) == AMIO_OK, "persistent lease releases through its matching API");
        }
        expect(amio_close_dataset_ex(dataset) == AMIO_OK, "evolution dataset closes after schema registration");
        expect(amio_release_context(context) == AMIO_OK, "context closes after its dataset is released");
        constexpr char diagnostics_off[] = "{\"diagnostics\":{\"enabled\":false}}";
        amio_context_descriptor_t disabled_descriptor = descriptor;
        disabled_descriptor.canonical_configuration = diagnostics_off;
        disabled_descriptor.configuration_bytes = sizeof(diagnostics_off) - 1;
        amio_context_handle disabled_context = nullptr;
        expect(amio_init_ex(&disabled_descriptor, &disabled_context) == AMIO_OK, "context initializes with standard diagnostics disabled");
        size_t disabled_bytes = 0;
        expect(amio_query_diagnostics(disabled_context, nullptr, 0, &disabled_bytes) == AMIO_ERR_UNSUPPORTED,
               "diagnostics query reports unsupported when collection is disabled");
        expect(amio_release_context(disabled_context) == AMIO_OK, "diagnostics-disabled context releases cleanly");
        expect(std::filesystem::exists(root / "amio.json"), "exclusive collection identity is persisted");
        std::filesystem::remove_all(root, ec);
    }
}

}  // namespace

int main() {
#ifdef AMIO_TEST_HAS_MPI
    int provided = MPI_THREAD_SINGLE;
    int argc = 0;
    char **argv = nullptr;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#endif
    test_additive_status_values();
    test_validation_descriptor_rejection_and_size_query();
    test_typed_release_names_reject_invalid_handles();
    test_context_initialization_outputs_and_capability_query();
#ifdef AMIO_TEST_HAS_MPI
    MPI_Finalize();
#endif
    std::fprintf(stdout, "test_evolution_contract: %s\n", failures == 0 ? "passed" : "failed");
    return failures == 0 ? 0 : 1;
}
