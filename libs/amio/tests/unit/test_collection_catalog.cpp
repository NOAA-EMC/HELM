#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "catalog/collection_catalog.hpp"

using namespace amio::detail;

namespace {

CompletionRecord record(std::uint64_t step_id, const std::string &previous = {}) {
    CompletionRecord value;
    value.dataset_uuid = "dataset-uuid-0001";
    value.step_id = step_id;
    value.schema_digest = "fnv1a64-v1:0123456789abcdef";
    value.required_fields = {"temperature"};
    value.present_fields = {"temperature"};
    value.previous_commit = previous;
    return value;
}

void test_payloads_commit_head_and_discovery_publish_in_order() {
    auto store = std::make_shared<MemoryCatalogStore>();
    CollectionCatalog catalog(store, "/collections/model", "dataset-uuid-0001", "owner-1");
    assert(catalog.create_exclusive() == AMIO_OK);
    auto publication = catalog.publish(record(1));
    assert(publication.status == AMIO_OK);
    assert(publication.state == PublicationState::Complete);
    assert(catalog.read_head().step_id == 1);
    assert(catalog.list_steps().size() == 1);
    assert(store->write_order() == std::vector<std::string>({"payload", "commit", "head"}));
}

void test_commit_write_failure_never_advances_head() {
    auto store = std::make_shared<MemoryCatalogStore>();
    CollectionCatalog catalog(store, "/collections/fault", "dataset-uuid-0001", "owner-1");
    assert(catalog.create_exclusive() == AMIO_OK);
    store->fail_next(CatalogWrite::Commit, FailureMoment::BeforeWrite);
    const auto result = catalog.publish(record(1));
    assert(result.status == AMIO_ERR_BACKEND_FAILURE);
    assert(catalog.read_head().empty());
    assert(catalog.list_orphans().size() == 1);
}

void test_ambiguous_head_ack_and_stale_reads_require_reconciliation() {
    auto store = std::make_shared<MemoryCatalogStore>();
    CollectionCatalog catalog(store, "/collections/ambiguous", "dataset-uuid-0001", "owner-1");
    assert(catalog.create_exclusive() == AMIO_OK);
    store->fail_next(CatalogWrite::Head, FailureMoment::AfterWrite);
    auto result = catalog.publish(record(1));
    assert(result.status == AMIO_ERR_PUBLICATION_PENDING);
    assert(result.state == PublicationState::Reconciling);
    store->serve_stale_reads(true);
    assert(catalog.reconcile(result.commit_id).state == PublicationState::Reconciling);
    store->serve_stale_reads(false);
    assert(catalog.reconcile(result.commit_id).state == PublicationState::Complete);
}

void test_traversal_references_duplicate_owners_and_unreferenced_payloads_are_rejected_or_reported() {
    auto store = std::make_shared<MemoryCatalogStore>();
    CollectionCatalog catalog(store, "/collections/safe", "dataset-uuid-0001", "owner-1");
    assert(catalog.create_exclusive() == AMIO_OK);
    assert(catalog.publish(record(1, "../../outside")).status == AMIO_ERR_INVALID_INPUT);
    CollectionCatalog second_owner(store, "/collections/safe", "dataset-uuid-0001", "owner-2");
    assert(second_owner.create_exclusive() == AMIO_ERR_INVALID_POLICY);
    store->inject_unreferenced_payload("payloads/failed-attempt/temperature.bin");
    assert(catalog.list_orphans().size() == 1);
}

void test_catalog_persists_grib2_message_ranges_and_rejects_invalid_ranges() {
    auto value = record(7);
    PayloadDescriptor payload;
    payload.field = "temperature";
    payload.relative_path = "payloads/steps/7.grib2";
    payload.bytes = 12;
    payload.producer = 3;
    payload.message_ranges = {{0, 4}, {4, 8}};
    payload.operator_pipeline = "f64_to_f32_absolute|sha256-logical-v1|native";
    payload.integrity_regions.push_back({{0}, {2}, "sha256-logical-v1:f766ba187f3c364260c2514311bda61b82bed83c6c86ee4edd3ad18b9b0e0569"});
    payload.integrity_regions.push_back({{2}, {2}, "sha256-logical-v1:f766ba187f3c364260c2514311bda61b82bed83c6c86ee4edd3ad18b9b0e0569"});
    value.payloads.push_back(payload);

    std::string encoded;
    assert(encode_commit_document(value, encoded) == AMIO_OK);
    CompletionRecord decoded;
    assert(parse_commit_document(encoded, decoded) == AMIO_OK);
    assert(decoded.payloads.size() == 1);
    assert(decoded.payloads[0].message_ranges.size() == 2);
    assert(decoded.payloads[0].message_ranges[0].offset == 0);
    assert(decoded.payloads[0].message_ranges[0].length == 4);
    assert(decoded.payloads[0].message_ranges[1].offset == 4);
    assert(decoded.payloads[0].message_ranges[1].length == 8);
    assert(decoded.payloads[0].integrity_regions.size() == 2);
    assert(decoded.payloads[0].integrity_regions[1].offsets == std::vector<std::uint64_t>{2});
    assert(decoded.payloads[0].integrity_regions[1].extents == std::vector<std::uint64_t>{2});
    assert(decoded.payloads[0].operator_pipeline == payload.operator_pipeline);

    value.payloads[0].message_ranges[1] = {8, 8};
    assert(encode_commit_document(value, encoded) == AMIO_ERR_INVALID_INPUT);
}

}  // namespace

int main() {
    test_payloads_commit_head_and_discovery_publish_in_order();
    test_commit_write_failure_never_advances_head();
    test_ambiguous_head_ack_and_stale_reads_require_reconciliation();
    test_traversal_references_duplicate_owners_and_unreferenced_payloads_are_rejected_or_reported();
    test_catalog_persists_grib2_message_ranges_and_rejects_invalid_ranges();
    std::cout << "test_collection_catalog: passed\n";
}
