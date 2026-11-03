#include <unistd.h>

#include <cassert>
#include <filesystem>
#include <string>

#include "catalog/collection_reader.hpp"
#include "catalog/posix_catalog.hpp"

using namespace amio::detail;

int main() {
    const auto root = std::filesystem::temp_directory_path() / ("amio-reader-" + std::to_string(::getpid()));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    PosixCatalog writer(root, "reader-dataset", "test-owner");
    assert(writer.create_exclusive() == AMIO_OK);
    const std::string payload = "12345678";
    CompletionRecord first;
    first.dataset_uuid = "reader-dataset";
    first.step_id = 1;
    first.schema_digest = "schema-a";
    first.required_fields = {"temperature"};
    first.present_fields = {"temperature"};
    first.payloads.push_back({"temperature", "payloads/1/temperature.nc", payload.size(), 0, {0}, {1}});
    assert(writer.write_payload(first.payloads.front().relative_path, payload.data(), payload.size()) == AMIO_OK);
    auto published = writer.publish(first);
    assert(published.status == AMIO_OK);

    CompletionRecord second = first;
    second.step_id = 2;
    second.previous_commit = published.commit_path;
    second.schema_digest = "schema-b";
    auto published_second = writer.publish(second);
    assert(published_second.status == AMIO_OK);

    CollectionReader reader(root, "reader-dataset");
    const auto page = reader.list_steps({}, 1);
    assert(page.status == AMIO_OK && page.steps.size() == 1 && page.has_more);
    assert(page.steps.front().completion.step_id == 2);
    const auto older = reader.list_steps(page.next, 1);
    assert(older.status == AMIO_OK && older.steps.size() == 1 && !older.has_more);
    assert(older.steps.front().completion.step_id == 1);
    CatalogStep described;
    assert(reader.describe_step(1, described) == AMIO_OK);
    assert(described.completion.schema_digest == "schema-a");

    auto stale = page.next;
    stale.generation = "commits/999.json";
    assert(reader.list_steps(stale, 1).status == AMIO_ERR_INVALID_INPUT);
    std::filesystem::remove_all(root, ec);
}
