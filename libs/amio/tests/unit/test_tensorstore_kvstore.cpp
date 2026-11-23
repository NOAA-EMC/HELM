#include <barrier>
#include <cassert>
#include <filesystem>
#include <string>
#include <thread>

#include "catalog/tensorstore_kvstore.hpp"

int main() {
    using namespace amio::detail;
    const auto root = (std::filesystem::temp_directory_path() / "amio-tensorstore-kvstore-test").string();
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);

    auto opened = tensorstore::kvstore::Open({{"driver", "file"}, {"path", root + "/"}}).result();
    assert(opened.ok());
    TensorStoreKvStore store(std::move(opened).value());

    // No guarantee is inferred from the generic TensorStore API. A catalog
    // must reject publication until a driver-specific qualification supplies
    // the required capability record.
    KvStoreCatalog uncertified(store, "collection", "dataset-uuid", "owner", store.capabilities());
    assert(uncertified.certify(PosixSyncLevel::Visibility) == AMIO_ERR_UNSUPPORTED);

    assert(store.write("collection/key", "first", KvWriteCondition::CreateOnly) == KvWriteOutcome::Written);
    const auto duplicate = store.write("collection/key", "second", KvWriteCondition::CreateOnly);
    assert(duplicate == KvWriteOutcome::AlreadyExists);
    std::string value;

    // The create-only condition must remain atomic when independent callers
    // race to publish the same immutable key.
    std::barrier start(3);
    KvWriteOutcome race_a = KvWriteOutcome::Unknown;
    KvWriteOutcome race_b = KvWriteOutcome::Unknown;
    std::thread writer_a([&] {
        start.arrive_and_wait();
        race_a = store.write("collection/race", "writer-a", KvWriteCondition::CreateOnly);
    });
    std::thread writer_b([&] {
        start.arrive_and_wait();
        race_b = store.write("collection/race", "writer-b", KvWriteCondition::CreateOnly);
    });
    start.arrive_and_wait();
    writer_a.join();
    writer_b.join();
    assert((race_a == KvWriteOutcome::Written && race_b == KvWriteOutcome::AlreadyExists) ||
           (race_b == KvWriteOutcome::Written && race_a == KvWriteOutcome::AlreadyExists));
    assert(store.read("collection/race", value, true) == KvReadOutcome::Found);
    assert(value == "writer-a" || value == "writer-b");

    assert(store.read("collection/key", value, true) == KvReadOutcome::Found);
    assert(value == "first");
    std::uint64_t bytes = 0;
    assert(store.size("collection/key", bytes, true) == KvReadOutcome::Found);
    assert(bytes == value.size());

    assert(store.write("collection/key", "replacement", KvWriteCondition::Replace) == KvWriteOutcome::Written);
    assert(store.read("collection/key", value, true) == KvReadOutcome::Found);
    assert(value == "replacement");
    assert(store.read("missing", value, true) == KvReadOutcome::Missing);
    assert(!store.sync("collection/key"));

    // Exercise the complete generic catalog against the real local TensorStore
    // KvStore after the adapter's key semantics above have been observed.
    const KvStoreCapabilities measured_file_capabilities{true, true, true, false};
    KvStoreCatalog catalog(store, "catalog", "dataset-uuid", "owner-token", measured_file_capabilities);
    assert(catalog.certify(PosixSyncLevel::Visibility) == AMIO_OK);
    assert(catalog.create_exclusive() == AMIO_OK);
    assert(store.write("catalog/payloads/1/temperature.bin", "data", KvWriteCondition::CreateOnly) == KvWriteOutcome::Written);
    CompletionRecord record;
    record.dataset_uuid = "dataset-uuid";
    record.step_id = 1;
    record.schema_digest = "schema-digest";
    record.required_fields = {"temperature"};
    record.present_fields = record.required_fields;
    record.payloads.push_back({"temperature", "payloads/1/temperature.bin", 4, 0, {0}, {1}});
    const auto published = catalog.publish(record);
    assert(published.status == AMIO_OK && published.state == PublicationState::Complete);
    assert(published.commit_path == "commits/1.json");
    assert(store.read("catalog/head.json", value, true) == KvReadOutcome::Found);
    assert(value.find("commits/1.json") != std::string::npos);
    assert(store.read("catalog/commits/1.json", value, true) == KvReadOutcome::Found);
    CompletionRecord decoded;
    assert(parse_commit_document(value, decoded) == AMIO_OK);
    assert(decoded.dataset_uuid == record.dataset_uuid && decoded.step_id == record.step_id);
    std::filesystem::remove_all(root);
}
