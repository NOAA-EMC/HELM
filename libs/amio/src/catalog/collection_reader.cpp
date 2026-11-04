#include "catalog/collection_reader.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <set>

namespace amio::detail {
namespace {

bool payload_size_matches(const std::filesystem::path &root, const PayloadDescriptor &payload) {
    if (!catalog_path_is_safe(payload.relative_path) || !payload.relative_path.starts_with("payloads/")) return false;
    const int fd = ::open((root / payload.relative_path).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st{};
    const bool stat_ok = ::fstat(fd, &st) == 0;
    const bool ok =
        stat_ok && (payload.array_container ? S_ISDIR(st.st_mode)
                                            : (S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) == payload.bytes));
    return ::close(fd) == 0 && ok;
}

bool valid_record(const CompletionRecord &record, const std::string &uuid) {
    if (record.dataset_uuid != uuid || record.schema_digest.empty()) return false;
    std::set<std::string, std::less<>> required(record.required_fields.begin(), record.required_fields.end());
    std::set<std::string, std::less<>> present(record.present_fields.begin(), record.present_fields.end());
    if (required.size() != record.required_fields.size() || present.size() != record.present_fields.size()) return false;
    for (const auto &field : required)
        if (!present.contains(field)) return false;
    std::set<std::string, std::less<>> payload_fields;
    for (const auto &payload : record.payloads) {
        if (!present.contains(payload.field) || !payload_fields.insert(payload.field).second || payload.offsets.size() != payload.extents.size())
            return false;
    }
    return true;
}

}  // namespace

CollectionReader::CollectionReader(std::filesystem::path root, std::string dataset_uuid, CatalogLimits limits)
    : root_(std::move(root)), uuid_(std::move(dataset_uuid)), limits_(limits) {}

CatalogPage CollectionReader::list_steps(const CatalogCursor &cursor, std::size_t max_results) const {
    CatalogPage page;
    if (uuid_.empty() || max_results == 0 || max_results > limits_.payloads || (!cursor.dataset_uuid.empty() && cursor.dataset_uuid != uuid_)) {
        page.status = AMIO_ERR_INVALID_INPUT;
        return page;
    }
    std::string head;
    bool missing = false;
    if (!read_bounded_catalog_file(root_ / "head.json", limits_.string_bytes + 512, head, missing)) {
        page.status = AMIO_ERR_STEP_FAILED;
        return page;
    }
    std::uint64_t head_step = 0;
    std::string generation;
    if (!parse_catalog_head(head, uuid_, head_step, generation)) {
        page.status = AMIO_ERR_BACKEND_FAILURE;
        return page;
    }
    if ((!cursor.generation.empty() && cursor.generation != generation) || (!cursor.generation.empty() && cursor.dataset_uuid != uuid_)) {
        page.status = AMIO_ERR_INVALID_INPUT;
        return page;
    }

    std::string current = cursor.next_commit.empty() ? generation : cursor.next_commit;
    if (!catalog_path_is_safe(current) || !current.starts_with("commits/")) {
        page.status = AMIO_ERR_INVALID_INPUT;
        return page;
    }
    std::set<std::string, std::less<>> visited;
    while (!current.empty() && page.steps.size() < max_results) {
        if (!visited.insert(current).second) {
            page.status = AMIO_ERR_BACKEND_FAILURE;
            page.steps.clear();
            return page;
        }
        std::string bytes;
        if (!read_bounded_catalog_file(root_ / current, limits_.document_bytes, bytes, missing)) {
            page.status = AMIO_ERR_BACKEND_FAILURE;
            page.steps.clear();
            return page;
        }
        CompletionRecord record;
        if (parse_commit_document(bytes, record, limits_) != AMIO_OK || !valid_record(record, uuid_) || record.step_id > head_step) {
            page.status = AMIO_ERR_BACKEND_FAILURE;
            page.steps.clear();
            return page;
        }
        for (const auto &payload : record.payloads)
            if (!payload_size_matches(root_, payload)) {
                page.status = AMIO_ERR_BACKEND_FAILURE;
                page.steps.clear();
                return page;
            }
        current = record.previous_commit;
        page.steps.push_back({std::move(record), false});
    }
    page.has_more = !current.empty();
    page.next = {uuid_, generation, page.has_more ? current : std::string{}};
    return page;
}

amio_status_t CollectionReader::describe_step(std::uint64_t step_id, CatalogStep &result) const {
    result = {};
    CatalogCursor cursor;
    for (;;) {
        auto page = list_steps(cursor, 64);
        if (page.status != AMIO_OK) return page.status;
        for (const auto &step : page.steps)
            if (step.completion.step_id == step_id) {
                result = step;
                return AMIO_OK;
            }
        if (!page.has_more) return AMIO_ERR_STEP_FAILED;
        cursor = std::move(page.next);
    }
}

}  // namespace amio::detail
