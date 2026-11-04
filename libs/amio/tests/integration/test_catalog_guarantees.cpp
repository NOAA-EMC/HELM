#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>

#include "catalog/collection_reader.hpp"
#include "catalog/posix_catalog.hpp"

using namespace amio::detail;

namespace {

std::filesystem::path make_root(const char *name) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() / (std::string("amio-catalog-guarantees-") + name + "-" + std::to_string(stamp));
}

CompletionRecord step(std::uint64_t id, std::string previous, const std::string &payload_path, std::string guarantee) {
    CompletionRecord record;
    record.dataset_uuid = "catalog-guarantees-test";
    record.step_id = id;
    record.schema_digest = "fnv1a64-v1:0123456789abcdef";
    record.required_fields = {"temperature"};
    record.present_fields = {"temperature"};
    record.previous_commit = std::move(previous);
    record.payloads.push_back({"temperature", payload_path, 4, 0, {0}, {1}});
    record.achieved_guarantee = std::move(guarantee);
    return record;
}

void test_visibility_and_requested_sync_levels() {
    const auto root = make_root("success");
    {
        PosixCatalog catalog(root, "catalog-guarantees-test", "owner-success");
        assert(catalog.create_exclusive() == AMIO_OK);
        const auto schema_path = "payloads/schemas/fnv1a64-v1:0123456789abcdef.json";
        const std::string schema_bytes = "schema";
        assert(catalog.write_payload(schema_path, schema_bytes.data(), schema_bytes.size()) == AMIO_OK);
        std::string previous;
        const PosixSyncLevel levels[] = {PosixSyncLevel::Visibility, PosixSyncLevel::FileSync, PosixSyncLevel::DirectorySync};
        const char *names[] = {"visibility", "file_sync", "directory_sync"};
        for (std::uint64_t id = 1; id <= 3; ++id) {
            const auto relative = "payloads/" + std::to_string(id) + "/temperature.bin";
            const std::string bytes = "data";
            assert(catalog.write_payload(relative, bytes.data(), bytes.size()) == AMIO_OK);
            const auto result = catalog.publish(step(id, previous, relative, "visibility"), levels[id - 1]);
            assert(result.status == AMIO_OK && result.state == PublicationState::Complete);
            CollectionReader reader(root, "catalog-guarantees-test");
            CatalogStep visible;
            assert(reader.describe_step(id, visible) == AMIO_OK);
            assert(visible.completion.achieved_guarantee == names[id - 1]);
            assert(std::filesystem::is_regular_file(root / "head.json"));
            assert(std::filesystem::is_regular_file(root / result.commit_path));
            previous = result.commit_path;
        }
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    assert(!ec);
}

void test_unreadable_head_stays_reconciling_after_payload_and_commit() {
    const auto root = make_root("ambiguous-head");
    {
        PosixCatalog catalog(root, "catalog-guarantees-test", "owner-ambiguous");
        assert(catalog.create_exclusive() == AMIO_OK);
        std::error_code ec;
        std::filesystem::create_directory(root / "head.json", ec);
        assert(!ec);
        const auto relative = std::string("payloads/1/temperature.bin");
        const std::string bytes = "data";
        assert(catalog.write_payload(relative, bytes.data(), bytes.size()) == AMIO_OK);
        const auto result = catalog.publish(step(1, {}, relative, "visibility"));
        assert(result.status == AMIO_ERR_PUBLICATION_PENDING);
        assert(result.state == PublicationState::Reconciling);
        assert(std::filesystem::is_regular_file(root / result.commit_path));
        assert(std::filesystem::is_directory(root / "head.json"));
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    assert(!ec);
}

void test_concurrent_readers_only_traverse_complete_head_generations() {
    const auto root = make_root("concurrent-readers");
    {
        PosixCatalog catalog(root, "catalog-guarantees-test", "owner-concurrent");
        assert(catalog.create_exclusive() == AMIO_OK);
        std::string previous;
        auto publish_step = [&](std::uint64_t id) {
            const auto relative = "payloads/" + std::to_string(id) + "/temperature.bin";
            const std::string bytes = "data";
            assert(catalog.write_payload(relative, bytes.data(), bytes.size()) == AMIO_OK);
            const auto result = catalog.publish(step(id, previous, relative, "visibility"));
            assert(result.status == AMIO_OK && result.state == PublicationState::Complete);
            previous = result.commit_path;
        };
        publish_step(0);

        std::atomic<bool> stop{false};
        std::atomic<bool> readers_ok{true};
        std::atomic<std::uint64_t> observations{0};
        std::thread reader([&] {
            while (!stop.load(std::memory_order_acquire)) {
                CollectionReader collection(root, "catalog-guarantees-test");
                const auto page = collection.list_steps({}, 32);
                if (page.status != AMIO_OK || page.steps.empty()) {
                    readers_ok.store(false, std::memory_order_release);
                    break;
                }
                observations.fetch_add(1, std::memory_order_relaxed);
            }
        });
        while (observations.load(std::memory_order_relaxed) == 0 && readers_ok.load(std::memory_order_acquire)) std::this_thread::yield();
        for (std::uint64_t id = 1; id <= 12; ++id) publish_step(id);
        stop.store(true, std::memory_order_release);
        reader.join();
        assert(readers_ok.load(std::memory_order_acquire));
        assert(observations.load(std::memory_order_relaxed) > 0);
    }
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    assert(!ec);
}

}  // namespace

int main() {
    test_visibility_and_requested_sync_levels();
    test_unreadable_head_stays_reconciling_after_payload_and_commit();
    test_concurrent_readers_only_traverse_complete_head_generations();
}
