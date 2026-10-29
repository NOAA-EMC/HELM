#include "catalog/collection_reader.hpp"
#include "catalog/posix_catalog.hpp"
#include "execution/direct_execution.hpp"
#include "schema/schema.hpp"
#include "staging/resource_budget.hpp"
#include "steps/step_coordinator.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace amio::detail;

namespace {

std::shared_ptr<Schema> make_schema(bool optional_temperature = false) {
    auto schema = std::make_shared<Schema>();
    Variable temperature;
    temperature.name = "temperature";
    temperature.source_dtype = AMIO_DTYPE_F64;
    temperature.stored_dtype = AMIO_DTYPE_F64;
    temperature.required = !optional_temperature;
    temperature.dimensions.push_back({"y", 4, std::nullopt});
    assert(schema->add_variable(std::move(temperature)).ok());
    Variable pressure;
    pressure.name = "pressure";
    pressure.source_dtype = AMIO_DTYPE_F32;
    pressure.stored_dtype = AMIO_DTYPE_F32;
    pressure.dimensions.push_back({"y", 4, std::nullopt});
    assert(schema->add_variable(std::move(pressure)).ok());
    assert(schema->seal().ok());
    return schema;
}

void register_schema(StepCoordinator &coordinator, std::shared_ptr<Schema> schema) {
    assert(coordinator.register_schema(std::move(schema)) == AMIO_OK);
}

StepBeginRequest begin_request(std::uint64_t id, const Schema &schema, std::size_t bytes = 128) {
    StepBeginRequest request;
    request.step_id = id;
    request.model_time = 12.5;
    request.schema_digest = schema.digest();
    request.snapshot_bytes = bytes;
    request.snapshot_descriptors = 2;
    for (const auto &[name, variable] : schema.variables()) {
        const bool absent = !variable.required;
        const std::size_t element_size = variable.source_dtype == AMIO_DTYPE_F64 ? 8 : 4;
        ContributionDeclaration declaration{name, absent ? 0 : element_size * 4, absent};
        if (absent) {
            declaration.region.empty = true;
        } else {
            for (const auto &dimension : variable.dimensions) {
                declaration.region.offsets.push_back(0);
                declaration.region.extents.push_back(dimension.extent);
            }
        }
        request.contributions.push_back(std::move(declaration));
    }
    return request;
}

void test_step_lifecycle_and_source_release_are_distinct_from_publication() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-0001", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(1, *schema)) == AMIO_OK);
    assert(coordinator.state() == StepState::Open);

    std::vector<double> values(4, 1.0);
    auto operation = coordinator.put("temperature", values.data(), values.size() * sizeof(double));
    assert(operation.status == AMIO_OK);
    assert(operation.source_released);
    assert(!operation.payload_complete);
    std::vector<float> pressure(4, 1.0f);
    assert(coordinator.put("pressure", pressure.data(), pressure.size() * sizeof(float)).status == AMIO_OK);
    assert(coordinator.end_step() == AMIO_OK);
    assert(coordinator.state() == StepState::Sealed);
    assert(coordinator.operation_snapshot(operation.handle).source_released);
    assert(!coordinator.operation_snapshot(operation.handle).reader_visible);
    assert(coordinator.mark_payload_complete() == AMIO_OK);
    assert(coordinator.state() == StepState::PayloadReady);
    assert(coordinator.mark_publishing() == AMIO_OK);
    assert(coordinator.mark_publication_visible(true) == AMIO_OK);
    assert(coordinator.state() == StepState::Complete);
    assert(coordinator.operation_snapshot(operation.handle).storage_synced);
}

void test_schema_changes_only_between_steps_and_ids_are_never_reused() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-0001", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(7, *schema)) == AMIO_OK);
    auto replacement = make_schema(true);
    assert(coordinator.register_schema(replacement) == AMIO_ERR_INVALID_POLICY);
    assert(coordinator.cancel_step() == AMIO_OK);
    assert(coordinator.state() == StepState::Failed);
    assert(coordinator.begin_step(begin_request(7, *schema)) == AMIO_ERR_INVALID_POLICY);
    assert(coordinator.begin_step(begin_request(8, *schema)) == AMIO_OK);
    assert(coordinator.register_schema(replacement) == AMIO_ERR_INVALID_POLICY);
    assert(coordinator.cancel_step() == AMIO_OK);
    assert(coordinator.register_schema(replacement) == AMIO_OK);
}

void test_required_and_optional_fields_and_duplicate_contributions() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema(true);
    StepCoordinator coordinator("dataset-uuid-0001", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(1, *schema)) == AMIO_OK);
    std::vector<float> values(4, 2.0f);
    assert(coordinator.put("pressure", values.data(), values.size() * sizeof(float)).status == AMIO_OK);
    assert(coordinator.put("not-in-schema", values.data(), values.size() * sizeof(float)).status == AMIO_ERR_INVALID_INPUT);
    assert(coordinator.end_step() == AMIO_OK); // optional temperature may be absent
    assert(coordinator.mark_payload_complete() == AMIO_OK);
    assert(coordinator.mark_publishing() == AMIO_OK);
    assert(coordinator.mark_publication_visible() == AMIO_OK);

    assert(coordinator.begin_step(begin_request(2, *schema)) == AMIO_OK);
    assert(coordinator.put("pressure", values.data(), values.size() * sizeof(float)).status == AMIO_OK);
    assert(coordinator.put("pressure", values.data(), values.size() * sizeof(float)).status == AMIO_ERR_INVALID_POLICY);
    assert(coordinator.end_step() == AMIO_OK);
    assert(coordinator.mark_payload_complete() == AMIO_OK);
    assert(coordinator.mark_publishing() == AMIO_OK);
    assert(coordinator.mark_publication_visible() == AMIO_OK);
    assert(coordinator.state() == StepState::Complete);
}

void test_budget_rejection_timeout_and_cancellation_preserve_state() {
    ResourceBudget budget({256, 8, 64, 2});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-0001", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(1, *schema, 512)) == AMIO_ERR_BUDGET_EXCEEDED);
    assert(coordinator.state() == StepState::Failed);
    assert(budget.snapshot().data_bytes == 0);

    assert(coordinator.begin_step(begin_request(2, *schema, 64)) == AMIO_OK);
    assert(coordinator.wait_for_publication(0) == AMIO_ERR_TIMEOUT);
    assert(coordinator.state() == StepState::Open);
    assert(coordinator.cancel_step() == AMIO_OK);
    assert(coordinator.state() == StepState::Failed);
    assert(coordinator.cancel_step() == AMIO_ERR_STEP_FAILED);
    assert(budget.snapshot().data_bytes == 0);
    assert(coordinator.begin_step(begin_request(3, *schema, 32)) == AMIO_ERR_BUDGET_EXCEEDED);
    assert(budget.snapshot().data_bytes == 0);
    auto empty = begin_request(4, *schema, 0);
    empty.empty_contributor = true;
    empty.snapshot_descriptors = 1;
    for (auto &declaration : empty.contributions) {
        declaration.absent = true;
        declaration.snapshot_bytes = 0;
        declaration.region.empty = true;
    }
    assert(coordinator.begin_step(empty) == AMIO_OK);
    assert(coordinator.end_step() == AMIO_OK);
    assert(coordinator.mark_payload_complete() == AMIO_OK);
    assert(coordinator.mark_publishing() == AMIO_OK);
    assert(coordinator.mark_publication_visible() == AMIO_OK);
}

void test_missing_required_field_fails_without_publication() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-0001", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(1, *schema)) == AMIO_OK);
    assert(coordinator.end_step() == AMIO_ERR_FIELD_NOT_PRESENT);
    assert(coordinator.state() == StepState::Failed);
    assert(coordinator.mark_publication_visible() == AMIO_ERR_STEP_FAILED);
    assert(budget.snapshot().data_bytes == 0);
}

void test_distributed_required_fields_are_deferred_to_global_partition_check() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-distributed-required", budget);
    register_schema(coordinator, schema);
    auto request = begin_request(1, *schema, 16);
    request.defer_required_field_validation = true;
    for (auto &declaration : request.contributions) {
        if (declaration.variable == "temperature") {
            declaration.absent = true;
            declaration.snapshot_bytes = 0;
            declaration.region = {};
            declaration.region.empty = true;
        }
    }
    assert(coordinator.begin_step(request) == AMIO_OK);
    std::vector<float> pressure(4, 1000.0f);
    assert(coordinator.put("pressure", pressure.data(), pressure.size() * sizeof(float)).status == AMIO_OK);
    // The API layer proves that this rank's box is complemented by other
    // ranks before allowing payload publication.
    assert(coordinator.end_step() == AMIO_OK);
    assert(coordinator.mark_payload_complete() == AMIO_OK);
    assert(coordinator.mark_publishing() == AMIO_OK);
    assert(coordinator.mark_publication_visible() == AMIO_OK);
    assert(coordinator.state() == StepState::Complete);
}

void test_region_selection_bounds_and_snapshot_size() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-region", budget);
    register_schema(coordinator, schema);
    auto request = begin_request(1, *schema, 32);
    for (auto &declaration : request.contributions) {
        if (declaration.variable == "temperature") {
            declaration.snapshot_bytes = 2 * sizeof(double);
            declaration.region.offsets = {1};
            declaration.region.extents = {2};
        } else {
            declaration.snapshot_bytes = 4 * sizeof(float);
            declaration.region.offsets = {0};
            declaration.region.extents = {4};
        }
    }
    assert(coordinator.begin_step(request) == AMIO_OK);
    std::vector<double> partial(2, 3.0);
    std::vector<float> whole(4, 4.0f);
    assert(coordinator.put("temperature", partial.data(), partial.size() * sizeof(double)).status == AMIO_OK);
    assert(coordinator.put("pressure", whole.data(), whole.size() * sizeof(float)).status == AMIO_OK);
    const auto &stored_region = coordinator.contributions().at("temperature").region;
    assert(stored_region.offsets == std::vector<std::uint64_t>{1});
    assert(stored_region.extents == std::vector<std::uint64_t>{2});
    assert(coordinator.cancel_step() == AMIO_OK);

    auto invalid = begin_request(2, *schema, 32);
    for (auto &declaration : invalid.contributions) {
        declaration.region.offsets = {3};
        declaration.region.extents = {2};
        declaration.snapshot_bytes = declaration.variable == "temperature" ? 16 : 8;
    }
    assert(coordinator.begin_step(invalid) == AMIO_ERR_INVALID_INPUT);
}

void test_one_producer_owner_per_dataset_identity() {
    ResourceBudget budget({4096, 32, 128, 4});
    StepCoordinator owner("dataset-uuid-owner", budget);
    bool rejected = false;
    try {
        StepCoordinator duplicate("dataset-uuid-owner", budget);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    assert(rejected);
}

void test_reconciliation_and_dataset_poison_are_terminally_distinct() {
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-reconcile", budget);
    register_schema(coordinator, schema);
    auto empty = begin_request(1, *schema, 0);
    empty.empty_contributor = true;
    empty.snapshot_descriptors = 1;
    for (auto &declaration : empty.contributions) {
        declaration.absent = true;
        declaration.snapshot_bytes = 0;
        declaration.region.empty = true;
    }
    assert(coordinator.begin_step(empty) == AMIO_OK);
    assert(coordinator.end_step() == AMIO_OK);
    assert(coordinator.mark_payload_complete() == AMIO_OK);
    assert(coordinator.mark_publishing() == AMIO_OK);
    assert(coordinator.mark_reconciling() == AMIO_OK);
    assert(coordinator.state() == StepState::Reconciling);
    assert(coordinator.cancel_step() == AMIO_ERR_STEP_FAILED);
    assert(coordinator.mark_publication_visible() == AMIO_OK);
    assert(coordinator.state() == StepState::Complete);
    assert(coordinator.fail_publication(AMIO_ERR_BACKEND_FAILURE) == AMIO_ERR_INVALID_POLICY);

    auto second = begin_request(2, *schema, 0);
    second.empty_contributor = true;
    second.snapshot_descriptors = 1;
    for (auto &declaration : second.contributions) {
        declaration.absent = true;
        declaration.snapshot_bytes = 0;
        declaration.region.empty = true;
    }
    assert(coordinator.begin_step(second) == AMIO_OK);
    assert(coordinator.poison_dataset(AMIO_ERR_BACKEND_FAILURE) == AMIO_ERR_BACKEND_FAILURE);
    assert(coordinator.dataset_poisoned());
    assert(coordinator.begin_step(begin_request(3, *schema)) == AMIO_ERR_STEP_FAILED);
}

void test_direct_execution_votes_after_payloads_and_publishes_authoritative_head() {
    const auto root = std::filesystem::temp_directory_path() / "amio-direct-execution-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);

    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-0001", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(1, *schema)) == AMIO_OK);
    std::vector<double> temperature(4, 7.5);
    std::vector<float> pressure(4, 1000.0f);
    assert(coordinator.put("temperature", temperature.data(), temperature.size() * sizeof(double)).status == AMIO_OK);
    assert(coordinator.put("pressure", pressure.data(), pressure.size() * sizeof(float)).status == AMIO_OK);
    assert(coordinator.end_step() == AMIO_OK);

    PosixCatalog catalog(root, "dataset-uuid-0001", "owner-1");
    assert(catalog.create_exclusive() == AMIO_OK);
    DirectExecution direct(catalog);
    CompletionRecord record;
    record.required_fields = {"pressure", "temperature"};
    record.present_fields = record.required_fields;
    bool voted = false;
    const auto result = direct.publish(
        coordinator, std::move(record),
        [&](const auto &contributions, CompletionRecord &completion) -> amio_status_t {
            for (const auto &[field, contribution] : contributions) {
                const auto path = std::string("payloads/steps/1/") + encode_field_path(field) + ".bin";
                const auto status = catalog.write_payload(path, contribution.bytes.data(), contribution.bytes.size());
                if (status != AMIO_OK) return status;
                completion.payloads.push_back({field, path, contribution.bytes.size(), 0,
                                               contribution.region.offsets, contribution.region.extents});
            }
            return AMIO_OK;
        },
        [&](amio_status_t local_status) {
            voted = true;
            return local_status;
        });

    assert(voted);
    assert(result.status == AMIO_OK && result.state == PublicationState::Complete);
    assert(coordinator.state() == StepState::Complete);
    CollectionReader reader(root, "dataset-uuid-0001");
    CatalogStep described;
    assert(reader.describe_step(1, described) == AMIO_OK);
    assert(described.completion.present_fields.size() == 2);

    assert(coordinator.begin_step(begin_request(2, *schema)) == AMIO_OK);
    assert(coordinator.put("temperature", temperature.data(), temperature.size() * sizeof(double)).status == AMIO_OK);
    assert(coordinator.put("pressure", pressure.data(), pressure.size() * sizeof(float)).status == AMIO_OK);
    assert(coordinator.end_step() == AMIO_OK);
    CompletionRecord second_record;
    second_record.required_fields = {"pressure", "temperature"};
    second_record.present_fields = second_record.required_fields;
    const auto second_result = direct.publish(
        coordinator, std::move(second_record),
        [&](const auto &contributions, CompletionRecord &completion) -> amio_status_t {
            for (const auto &[field, contribution] : contributions) {
                const auto path = std::string("payloads/steps/2/") + encode_field_path(field) + ".bin";
                const auto write_status = catalog.write_payload(path, contribution.bytes.data(), contribution.bytes.size());
                if (write_status != AMIO_OK) return write_status;
                completion.payloads.push_back({field, path, contribution.bytes.size(), 0,
                                               contribution.region.offsets, contribution.region.extents});
            }
            return AMIO_OK;
        },
        [](amio_status_t local_status) { return local_status; });
    assert(second_result.status == AMIO_OK && second_result.state == PublicationState::Complete);
    auto page = reader.list_steps({}, 2);
    assert(page.status == AMIO_OK && page.steps.size() == 2);
    assert(page.steps[0].completion.step_id == 2);
    assert(page.steps[0].completion.previous_commit == result.commit_path);
    std::filesystem::remove_all(root, ec);
}

void test_direct_execution_fails_without_publishing_when_a_writer_vote_fails() {
    const auto root = std::filesystem::temp_directory_path() / "amio-direct-execution-failure-test";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    ResourceBudget budget({4096, 32, 128, 4});
    auto schema = make_schema();
    StepCoordinator coordinator("dataset-uuid-0002", budget);
    register_schema(coordinator, schema);
    assert(coordinator.begin_step(begin_request(1, *schema)) == AMIO_OK);
    std::vector<double> temperature(4, 7.5);
    std::vector<float> pressure(4, 1000.0f);
    assert(coordinator.put("temperature", temperature.data(), temperature.size() * sizeof(double)).status == AMIO_OK);
    assert(coordinator.put("pressure", pressure.data(), pressure.size() * sizeof(float)).status == AMIO_OK);
    assert(coordinator.end_step() == AMIO_OK);

    PosixCatalog catalog(root, "dataset-uuid-0002", "owner-2");
    assert(catalog.create_exclusive() == AMIO_OK);
    DirectExecution direct(catalog);
    CompletionRecord record;
    amio_status_t voted_status = AMIO_OK;
    const auto result = direct.publish(
        coordinator, std::move(record),
        [](const auto &, CompletionRecord &) -> amio_status_t { return AMIO_ERR_BACKEND_FAILURE; },
        [&](amio_status_t local_status) {
            voted_status = local_status;
            return local_status;
        });
    assert(voted_status == AMIO_ERR_BACKEND_FAILURE);
    assert(result.status == AMIO_ERR_BACKEND_FAILURE && result.state == PublicationState::Failed);
    assert(coordinator.state() == StepState::Failed && coordinator.dataset_poisoned());
    assert(!std::filesystem::exists(root / "head.json"));
    std::filesystem::remove_all(root, ec);
}

} // namespace

int main() {
    test_step_lifecycle_and_source_release_are_distinct_from_publication();
    test_schema_changes_only_between_steps_and_ids_are_never_reused();
    test_required_and_optional_fields_and_duplicate_contributions();
    test_budget_rejection_timeout_and_cancellation_preserve_state();
    test_missing_required_field_fails_without_publication();
    test_distributed_required_fields_are_deferred_to_global_partition_check();
    test_region_selection_bounds_and_snapshot_size();
    test_one_producer_owner_per_dataset_identity();
    test_reconciliation_and_dataset_poison_are_terminally_distinct();
    test_direct_execution_votes_after_payloads_and_publishes_authoritative_head();
    test_direct_execution_fails_without_publishing_when_a_writer_vote_fails();
    std::cout << "test_step_coordinator: passed\n";
}
