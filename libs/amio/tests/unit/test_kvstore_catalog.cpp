#include "catalog/kvstore_catalog.hpp"

#include <cassert>
#include <map>
#include <string>
#include <string_view>

using namespace amio::detail;

namespace {

class MemoryKvStore final : public KvStore {
public:
    KvWriteOutcome write(std::string_view key, std::string_view value, KvWriteCondition condition) override {
        auto found = values.find(std::string(key));
        if (condition == KvWriteCondition::CreateOnly && found != values.end()) return KvWriteOutcome::AlreadyExists;
        values[std::string(key)] = std::string(value);
        if (unknown_after_write == key) {
            unknown_after_write.clear();
            return KvWriteOutcome::Unknown;
        }
        return KvWriteOutcome::Written;
    }

    KvReadOutcome read(std::string_view key, std::string &value, bool require_fresh) override {
        if (!require_fresh) return KvReadOutcome::Unknown;
        const auto found = values.find(std::string(key));
        if (found == values.end()) return KvReadOutcome::Missing;
        value = found->second;
        return KvReadOutcome::Found;
    }

    KvReadOutcome size(std::string_view key, std::uint64_t &bytes, bool require_fresh) override {
        std::string value;
        const auto result = read(key, value, require_fresh);
        if (result == KvReadOutcome::Found) bytes = value.size();
        return result;
    }

    bool sync(std::string_view) override { return false; }

    std::map<std::string, std::string, std::less<>> values;
    std::string unknown_after_write;
};

KvStoreCapabilities visible_capabilities(bool create_only = true) {
    return {true, create_only, true, false};
}

CompletionRecord make_record() {
    CompletionRecord record;
    record.dataset_uuid = "dataset-uuid-0001";
    record.step_id = 3;
    record.schema_digest = "fnv1a64-v1:0123456789abcdef";
    record.required_fields = {"temperature"};
    record.present_fields = {"temperature"};
    record.payloads.push_back({"temperature", "payloads/3/temperature.bin", 4, 0, {0}, {1}});
    return record;
}

void test_uncertified_capabilities_and_external_owner_gate() {
    MemoryKvStore store;
    KvStoreCatalog uncertified(store, "collections/model", "dataset-uuid-0001", "owner-1", {});
    assert(uncertified.certify(PosixSyncLevel::Visibility) == AMIO_ERR_UNSUPPORTED);

    const auto no_create = visible_capabilities(false);
    KvStoreCatalog no_owner(store, "collections/model", "dataset-uuid-0001", "owner-1", no_create);
    assert(no_owner.certify(PosixSyncLevel::Visibility) == AMIO_ERR_INVALID_POLICY);

    KvStoreCatalog externally_owned(store, "collections/model", "dataset-uuid-0001", "owner-1", no_create, true);
    assert(externally_owned.certify(PosixSyncLevel::Visibility) == AMIO_OK);
    assert(externally_owned.create_exclusive() == AMIO_OK);
    assert(store.write("collections/model/payloads/3/temperature.bin", "data", KvWriteCondition::CreateOnly) == KvWriteOutcome::Written);
    assert(externally_owned.publish(make_record()).status == AMIO_OK);
}

void test_ambiguous_head_ack_is_reconciled_by_fresh_read() {
    MemoryKvStore store;
    KvStoreCatalog catalog(store, "collections/model", "dataset-uuid-0001", "owner-1", visible_capabilities());
    assert(catalog.create_exclusive() == AMIO_OK);
    const std::string payload("data", 4);
    assert(store.write("collections/model/payloads/3/temperature.bin", payload, KvWriteCondition::CreateOnly) == KvWriteOutcome::Written);

    store.unknown_after_write = "collections/model/head.json";
    const auto result = catalog.publish(make_record());
    assert(result.status == AMIO_OK);
    assert(result.state == PublicationState::Complete);
    assert(result.commit_path == "commits/3.json");
}

void test_ambiguous_create_and_commit_acknowledgments_are_reconciled() {
    MemoryKvStore store;
    KvStoreCatalog catalog(store, "collections/model", "dataset-uuid-0001", "owner-1", visible_capabilities());
    store.unknown_after_write = "collections/model/amio.json";
    assert(catalog.create_exclusive() == AMIO_OK);

    const std::string payload("data", 4);
    assert(store.write("collections/model/payloads/3/temperature.bin", payload, KvWriteCondition::CreateOnly) == KvWriteOutcome::Written);
    store.unknown_after_write = "collections/model/commits/3.json";
    const auto result = catalog.publish(make_record());
    assert(result.status == AMIO_OK);
    assert(result.state == PublicationState::Complete);
    assert(result.commit_path == "commits/3.json");
}

void test_requested_sync_is_not_inferred_from_generic_store() {
    MemoryKvStore store;
    KvStoreCatalog catalog(store, "collections/model", "dataset-uuid-0001", "owner-1", visible_capabilities());
    assert(catalog.certify(PosixSyncLevel::FileSync) == AMIO_ERR_UNSUPPORTED);
    assert(catalog.certify(PosixSyncLevel::DirectorySync) == AMIO_ERR_UNSUPPORTED);
}

} // namespace

int main() {
    test_uncertified_capabilities_and_external_owner_gate();
    test_ambiguous_head_ack_is_reconciled_by_fresh_read();
    test_ambiguous_create_and_commit_acknowledgments_are_reconciled();
    test_requested_sync_is_not_inferred_from_generic_store();
}
