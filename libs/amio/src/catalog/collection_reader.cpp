#include "catalog/collection_reader.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <fcntl.h>
#include <set>
#include <sys/stat.h>
#include <unistd.h>

namespace amio::detail {
namespace {

bool safe_relative(std::string_view path) {
    if (path.empty() || path.front() == '/' || path.find('\\') != std::string_view::npos || path.find('\0') != std::string_view::npos) return false;
    std::size_t at = 0;
    while (at <= path.size()) {
        const auto slash = path.find('/', at);
        const auto component = path.substr(at, slash == std::string_view::npos ? path.size() - at : slash - at);
        if (component.empty() || component == "." || component == "..") return false;
        if (slash == std::string_view::npos) break;
        at = slash + 1;
    }
    return true;
}

bool read_bounded(const std::filesystem::path &path, std::size_t limit, std::string &out) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st {};
    bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) <= limit;
    if (ok) {
        out.assign(static_cast<std::size_t>(st.st_size), '\0');
        std::size_t offset = 0;
        while (offset < out.size()) {
            const auto n = ::read(fd, out.data() + offset, out.size() - offset);
            if (n < 0) { if (errno == EINTR) continue; ok = false; break; }
            if (n == 0) { ok = false; break; }
            offset += static_cast<std::size_t>(n);
        }
    }
    return ::close(fd) == 0 && ok;
}

bool payload_size_matches(const std::filesystem::path &root, const PayloadDescriptor &payload) {
    if (!safe_relative(payload.relative_path) || !payload.relative_path.starts_with("payloads/")) return false;
    const int fd = ::open((root / payload.relative_path).c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st {};
    const bool stat_ok = ::fstat(fd, &st) == 0;
    const bool ok = stat_ok && (payload.array_container ? S_ISDIR(st.st_mode) :
        (S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) == payload.bytes));
    return ::close(fd) == 0 && ok;
}

bool parse_head(std::string_view text, std::string_view uuid, std::uint64_t &step, std::string &commit) {
    const auto prefix = std::string("{\"format_version\":1,\"dataset_uuid\":\"") + std::string(uuid) + "\",\"step_id\":";
    constexpr std::string_view marker = ",\"commit\":\"";
    if (!text.starts_with(prefix)) return false;
    const auto marker_at = text.find(marker, prefix.size());
    if (marker_at == std::string_view::npos || marker_at == prefix.size() || text.size() < marker_at + marker.size() + 2 || text.substr(text.size() - 2) != "\"}") return false;
    const auto parsed = std::from_chars(text.data() + prefix.size(), text.data() + marker_at, step);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + marker_at) return false;
    commit.assign(text.substr(marker_at + marker.size(), text.size() - marker_at - marker.size() - 2));
    return safe_relative(commit) && commit.starts_with("commits/");
}

bool valid_record(const CompletionRecord &record, const std::string &uuid) {
    if (record.dataset_uuid != uuid || record.schema_digest.empty()) return false;
    std::set<std::string, std::less<>> required(record.required_fields.begin(), record.required_fields.end());
    std::set<std::string, std::less<>> present(record.present_fields.begin(), record.present_fields.end());
    if (required.size() != record.required_fields.size() || present.size() != record.present_fields.size()) return false;
    for (const auto &field : required) if (!present.contains(field)) return false;
    std::set<std::string, std::less<>> payload_fields;
    for (const auto &payload : record.payloads) {
        if (!present.contains(payload.field) || !payload_fields.insert(payload.field).second ||
            payload.offsets.size() != payload.extents.size()) return false;
    }
    return true;
}

} // namespace

CollectionReader::CollectionReader(std::filesystem::path root, std::string dataset_uuid, CatalogLimits limits)
    : root_(std::move(root)), uuid_(std::move(dataset_uuid)), limits_(limits) {}

CatalogPage CollectionReader::list_steps(const CatalogCursor &cursor, std::size_t max_results) const {
    CatalogPage page;
    if (uuid_.empty() || max_results == 0 || max_results > limits_.payloads ||
        (!cursor.dataset_uuid.empty() && cursor.dataset_uuid != uuid_)) {
        page.status = AMIO_ERR_INVALID_INPUT;
        return page;
    }
    std::string head;
    if (!read_bounded(root_ / "head.json", limits_.string_bytes + 512, head)) {
        page.status = AMIO_ERR_STEP_FAILED;
        return page;
    }
    std::uint64_t head_step = 0;
    std::string generation;
    if (!parse_head(head, uuid_, head_step, generation)) {
        page.status = AMIO_ERR_BACKEND_FAILURE;
        return page;
    }
    if ((!cursor.generation.empty() && cursor.generation != generation) ||
        (!cursor.generation.empty() && cursor.dataset_uuid != uuid_)) {
        page.status = AMIO_ERR_INVALID_INPUT;
        return page;
    }

    std::string current = cursor.next_commit.empty() ? generation : cursor.next_commit;
    if (!safe_relative(current) || !current.starts_with("commits/")) {
        page.status = AMIO_ERR_INVALID_INPUT;
        return page;
    }
    std::set<std::string, std::less<>> visited;
    while (!current.empty() && page.steps.size() < max_results) {
        if (!visited.insert(current).second) { page.status = AMIO_ERR_BACKEND_FAILURE; page.steps.clear(); return page; }
        std::string bytes;
        if (!read_bounded(root_ / current, limits_.document_bytes, bytes)) { page.status = AMIO_ERR_BACKEND_FAILURE; page.steps.clear(); return page; }
        CompletionRecord record;
        if (parse_commit_document(bytes, record, limits_) != AMIO_OK || !valid_record(record, uuid_) || record.step_id > head_step) {
            page.status = AMIO_ERR_BACKEND_FAILURE; page.steps.clear(); return page;
        }
        for (const auto &payload : record.payloads) if (!payload_size_matches(root_, payload)) {
            page.status = AMIO_ERR_BACKEND_FAILURE; page.steps.clear(); return page;
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
        for (const auto &step : page.steps) if (step.completion.step_id == step_id) { result = step; return AMIO_OK; }
        if (!page.has_more) return AMIO_ERR_STEP_FAILED;
        cursor = std::move(page.next);
    }
}

} // namespace amio::detail
