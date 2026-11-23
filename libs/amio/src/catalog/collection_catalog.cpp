#include "catalog/collection_catalog.hpp"

#include <algorithm>
#include <mutex>
#include <set>

namespace amio::detail {
namespace {
std::mutex owners_mutex;
std::set<std::string, std::less<>> owners;

bool safe_path(const std::string &path) {
    if (path.empty() || path.front() == '/' || path.find('\\') != std::string::npos) return false;
    std::size_t start = 0;
    while (start <= path.size()) {
        const auto end = path.find('/', start);
        const auto component = path.substr(start, end == std::string::npos ? end : end - start);
        if (component.empty() || component == "." || component == "..") return false;
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return true;
}
}  // namespace

void MemoryCatalogStore::fail_next(CatalogWrite write, FailureMoment moment) {
    failures_.push_back({write, moment});
}
void MemoryCatalogStore::inject_unreferenced_payload(std::string path) {
    payloads_.push_back(std::move(path));
}

CollectionCatalog::CollectionCatalog(std::shared_ptr<MemoryCatalogStore> store, std::string root, std::string uuid, std::string owner)
    : store_(std::move(store)), root_(std::move(root)), uuid_(std::move(uuid)), owner_(std::move(owner)) {}

bool CollectionCatalog::fail(CatalogWrite write, FailureMoment moment) {
    const auto found =
        std::find_if(store_->failures_.begin(), store_->failures_.end(), [=](const auto &f) { return f.write == write && f.moment == moment; });
    if (found == store_->failures_.end()) return false;
    store_->failures_.erase(found);
    return true;
}

amio_status_t CollectionCatalog::create_exclusive() {
    if (!store_ || root_.empty() || uuid_.empty() || owner_.empty()) return AMIO_ERR_INVALID_INPUT;
    if (created_) return AMIO_OK;
    std::lock_guard lock(owners_mutex);
    if (!owners.insert(root_).second) return AMIO_ERR_INVALID_POLICY;
    created_ = true;
    return AMIO_OK;
}

CollectionPublicationResult CollectionCatalog::publish(CompletionRecord record) {
    if (!created_ || record.dataset_uuid != uuid_) return {AMIO_ERR_INVALID_POLICY, PublicationState::Failed, {}};
    for (const auto &payload : record.payloads)
        if (!safe_path(payload.relative_path) || !payload.relative_path.starts_with("payloads/"))
            return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, {}};
    if (!record.previous_commit.empty() && (!safe_path(record.previous_commit) || !record.previous_commit.starts_with("commits/")))
        return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, {}};

    store_->write_order_.push_back("payload");
    if (fail(CatalogWrite::Payload, FailureMoment::BeforeWrite)) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, {}};
    if (record.payloads.empty())
        store_->payloads_.push_back("payloads/step-" + std::to_string(record.step_id) + "/metadata");
    else
        for (const auto &payload : record.payloads) store_->payloads_.push_back(payload.relative_path);
    if (fail(CatalogWrite::Payload, FailureMoment::AfterWrite)) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, {}};

    const auto id = "commits/" + std::to_string(record.step_id) + ".json";
    std::string encoded;
    const auto status = encode_commit_document(record, encoded);
    if (status != AMIO_OK) return {status, PublicationState::Failed, id};
    store_->write_order_.push_back("commit");
    if (fail(CatalogWrite::Commit, FailureMoment::BeforeWrite)) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, id};
    store_->commits_[id] = encoded;
    if (fail(CatalogWrite::Commit, FailureMoment::AfterWrite)) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, id};

    store_->write_order_.push_back("head");
    if (fail(CatalogWrite::Head, FailureMoment::BeforeWrite)) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, id};
    store_->head_ = id;
    if (fail(CatalogWrite::Head, FailureMoment::AfterWrite)) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, id};
    return {AMIO_OK, PublicationState::Complete, id};
}

CollectionPublicationResult CollectionCatalog::reconcile(const std::string &commit_id) const {
    if (!created_ || !safe_path(commit_id) || !commit_id.starts_with("commits/"))
        return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, commit_id};
    if (store_->stale_reads_) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, commit_id};
    std::string current = store_->head_;
    std::set<std::string> seen;
    while (!current.empty() && seen.insert(current).second) {
        if (current == commit_id) return {AMIO_OK, PublicationState::Complete, commit_id};
        const auto found = store_->commits_.find(current);
        if (found == store_->commits_.end()) break;
        CompletionRecord record;
        if (parse_commit_document(found->second, record) != AMIO_OK || record.dataset_uuid != uuid_) break;
        current = record.previous_commit;
    }
    return {AMIO_ERR_STEP_FAILED, PublicationState::Failed, commit_id};
}

CompletionRecord CollectionCatalog::read_head() const {
    CompletionRecord record;
    if (!store_ || store_->head_.empty() || store_->stale_reads_) return record;
    const auto found = store_->commits_.find(store_->head_);
    if (found != store_->commits_.end()) (void)parse_commit_document(found->second, record);
    return record;
}

std::vector<CompletionRecord> CollectionCatalog::list_steps() const {
    std::vector<CompletionRecord> records;
    if (!store_ || store_->stale_reads_) return records;
    std::string current = store_->head_;
    std::set<std::string> seen;
    while (!current.empty() && seen.insert(current).second) {
        const auto found = store_->commits_.find(current);
        if (found == store_->commits_.end()) break;
        CompletionRecord record;
        if (parse_commit_document(found->second, record) != AMIO_OK || record.dataset_uuid != uuid_) break;
        records.push_back(record);
        current = record.previous_commit;
    }
    std::reverse(records.begin(), records.end());
    return records;
}

std::vector<std::string> CollectionCatalog::list_orphans() const {
    std::set<std::string> referenced;
    for (const auto &[id, json] : store_->commits_) {
        (void)id;
        CompletionRecord record;
        if (parse_commit_document(json, record) == AMIO_OK && record.dataset_uuid == uuid_) {
            for (const auto &payload : record.payloads) referenced.insert(payload.relative_path);
            referenced.insert("payloads/step-" + std::to_string(record.step_id) + "/metadata");
        }
    }
    std::vector<std::string> orphaned;
    for (const auto &payload : store_->payloads_)
        if (!referenced.contains(payload)) orphaned.push_back(payload);
    return orphaned;
}

}  // namespace amio::detail
