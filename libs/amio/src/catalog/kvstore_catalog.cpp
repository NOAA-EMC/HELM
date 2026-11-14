#include "catalog/kvstore_catalog.hpp"

#include <algorithm>
#include <charconv>
#include <set>
#include <utility>

namespace amio::detail {
namespace {

bool safe_key(std::string_view value) {
    if (value.empty() || value.front() == '/' || value.find('\\') != std::string_view::npos || value.find('\0') != std::string_view::npos)
        return false;
    std::size_t at = 0;
    while (at <= value.size()) {
        const auto slash = value.find('/', at);
        const auto component = value.substr(at, slash == std::string_view::npos ? value.size() - at : slash - at);
        if (component.empty() || component == "." || component == "..") return false;
        if (slash == std::string_view::npos) break;
        at = slash + 1;
    }
    return true;
}

bool safe_identity(std::string_view value) {
    if (value.empty() || value.size() > 128) return false;
    for (const unsigned char c : value)
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
    return true;
}

std::string identity(std::string_view uuid, std::string_view owner) {
    return std::string("{\"format_version\":1,\"layout\":\"step_collection\",\"dataset_uuid\":\"") + std::string(uuid) + "\",\"owner_token\":\"" +
           std::string(owner) + "\"}";
}

std::string make_head(std::string_view uuid, std::uint64_t step, std::string_view commit) {
    return std::string("{\"format_version\":1,\"dataset_uuid\":\"") + std::string(uuid) + "\",\"step_id\":" + std::to_string(step) +
           ",\"commit\":\"" + std::string(commit) + "\"}";
}

bool parse_head(std::string_view text, std::string_view uuid, std::string &commit) {
    const auto prefix = std::string("{\"format_version\":1,\"dataset_uuid\":\"") + std::string(uuid) + "\",\"step_id\":";
    constexpr std::string_view marker = ",\"commit\":\"";
    if (!text.starts_with(prefix)) return false;
    const auto marker_at = text.find(marker, prefix.size());
    if (marker_at == std::string_view::npos || marker_at == prefix.size()) return false;
    std::uint64_t step = 0;
    const auto parsed = std::from_chars(text.data() + prefix.size(), text.data() + marker_at, step);
    (void)step;
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + marker_at || text.size() < marker_at + marker.size() + 2 ||
        text.substr(text.size() - 2) != "\"}")
        return false;
    commit.assign(text.substr(marker_at + marker.size(), text.size() - marker_at - marker.size() - 2));
    return safe_key(commit) && commit.rfind("commits/", 0) == 0;
}

}  // namespace

KvStoreCatalog::KvStoreCatalog(KvStore &store, std::string prefix, std::string dataset_uuid, std::string owner_token,
                               KvStoreCapabilities capabilities, bool external_exclusive_owner, CatalogLimits limits)
    : store_(&store),
      prefix_(std::move(prefix)),
      uuid_(std::move(dataset_uuid)),
      owner_(std::move(owner_token)),
      capabilities_(capabilities),
      external_exclusive_owner_(external_exclusive_owner),
      limits_(limits) {}

std::string KvStoreCatalog::key(std::string_view relative) const {
    return prefix_.empty() ? std::string(relative) : prefix_ + "/" + std::string(relative);
}

amio_status_t KvStoreCatalog::certify(PosixSyncLevel requested) const {
    if (!store_ || !safe_key(prefix_.empty() ? std::string_view("collection") : std::string_view(prefix_)) || !safe_identity(uuid_) ||
        !safe_identity(owner_)) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (!capabilities_.atomic_single_key_replace || !capabilities_.fresh_read_after_write) return AMIO_ERR_UNSUPPORTED;
    if (!capabilities_.atomic_create_if_absent && !external_exclusive_owner_) return AMIO_ERR_INVALID_POLICY;
    if (requested == PosixSyncLevel::DirectorySync) return AMIO_ERR_UNSUPPORTED;
    if (requested != PosixSyncLevel::Visibility && !capabilities_.stable_sync) return AMIO_ERR_UNSUPPORTED;
    return AMIO_OK;
}

amio_status_t KvStoreCatalog::create_exclusive() {
    const auto certified = certify(PosixSyncLevel::Visibility);
    if (certified != AMIO_OK) return certified;
    const auto marker = key("amio.json");
    const auto contents = identity(uuid_, owner_);
    if (!capabilities_.atomic_create_if_absent) {
        std::string existing;
        const auto read = store_->read(marker, existing, true);
        if (read == KvReadOutcome::Found) {
            if (existing == contents) {
                created_ = true;
                return AMIO_OK;
            }
            return AMIO_ERR_INVALID_POLICY;
        }
        if (read != KvReadOutcome::Missing) return AMIO_ERR_PUBLICATION_PENDING;
    }
    const auto outcome =
        store_->write(marker, contents, capabilities_.atomic_create_if_absent ? KvWriteCondition::CreateOnly : KvWriteCondition::Replace);
    if (outcome == KvWriteOutcome::Written) {
        created_ = true;
        return AMIO_OK;
    }
    std::string observed;
    const auto read = store_->read(marker, observed, true);
    if (outcome == KvWriteOutcome::Unknown && read == KvReadOutcome::Found && observed == contents) {
        // In the no-CAS case this is safe only under the caller's declared external owner precondition.
        created_ = true;
        return AMIO_OK;
    }
    return outcome == KvWriteOutcome::Unknown || read == KvReadOutcome::Unknown ? AMIO_ERR_PUBLICATION_PENDING : AMIO_ERR_INVALID_POLICY;
}

PublicationResult KvStoreCatalog::publish(CompletionRecord record, PosixSyncLevel requested) {
    const auto certified = certify(requested);
    if (certified != AMIO_OK) return {certified, PublicationState::Failed, {}};
    if (!created_ || record.dataset_uuid != uuid_) return {AMIO_ERR_INVALID_POLICY, PublicationState::Failed, {}};
    for (const auto &payload : record.payloads) {
        if (!safe_key(payload.relative_path) || payload.relative_path.rfind("payloads/", 0) != 0)
            return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, {}};
        std::uint64_t payload_bytes = 0;
        const auto read = store_->size(key(payload.relative_path), payload_bytes, true);
        if (read == KvReadOutcome::Unknown) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, {}};
        if (read != KvReadOutcome::Found || payload_bytes != payload.bytes) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, {}};
        if (requested != PosixSyncLevel::Visibility && !store_->sync(key(payload.relative_path))) {
            return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, {}};
        }
    }
    record.achieved_guarantee =
        requested == PosixSyncLevel::DirectorySync ? "directory_sync" : (requested == PosixSyncLevel::FileSync ? "file_sync" : "visibility");
    std::string encoded;
    const auto encode_status = encode_commit_document(record, encoded, limits_);
    if (encode_status != AMIO_OK) return {encode_status, PublicationState::Failed, {}};
    const auto commit = std::string("commits/") + std::to_string(record.step_id) + ".json";
    if (!capabilities_.atomic_create_if_absent) {
        std::string existing;
        const auto read = store_->read(key(commit), existing, true);
        if (read == KvReadOutcome::Found && existing != encoded) return {AMIO_ERR_INVALID_POLICY, PublicationState::Failed, commit};
        if (read == KvReadOutcome::Unknown || read == KvReadOutcome::Failed)
            return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, commit};
    }
    const auto commit_outcome =
        store_->write(key(commit), encoded, capabilities_.atomic_create_if_absent ? KvWriteCondition::CreateOnly : KvWriteCondition::Replace);
    if (commit_outcome != KvWriteOutcome::Written) {
        std::string observed;
        const auto read = store_->read(key(commit), observed, true);
        if (read == KvReadOutcome::Unknown || commit_outcome == KvWriteOutcome::Unknown) {
            if (read != KvReadOutcome::Found) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, commit};
        }
        if (read != KvReadOutcome::Found || observed != encoded) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, commit};
    }
    if (requested != PosixSyncLevel::Visibility && !store_->sync(key(commit))) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, commit};
    const auto head_value = make_head(uuid_, record.step_id, commit);
    const auto head_outcome = store_->write(key("head.json"), head_value, KvWriteCondition::Replace);
    if (head_outcome == KvWriteOutcome::Written) return {AMIO_OK, PublicationState::Complete, commit};
    const auto reconciled = reconcile(commit);
    if (reconciled.state == PublicationState::Complete) return reconciled;
    if (head_outcome == KvWriteOutcome::Unknown || reconciled.status == AMIO_ERR_PUBLICATION_PENDING) {
        return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, commit};
    }
    return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, commit};
}

PublicationResult KvStoreCatalog::reconcile(std::string_view relative_commit) const {
    if (!safe_key(relative_commit) || relative_commit.rfind("commits/", 0) != 0) return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, {}};
    std::string head_value;
    const auto head_read = store_->read(key("head.json"), head_value, true);
    if (head_read == KvReadOutcome::Unknown || head_read == KvReadOutcome::Failed)
        return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(relative_commit)};
    if (head_read == KvReadOutcome::Missing) return {AMIO_ERR_STEP_FAILED, PublicationState::Failed, std::string(relative_commit)};
    std::string current;
    if (!parse_head(head_value, uuid_, current)) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(relative_commit)};
    std::set<std::string> visited;
    const auto max_steps = std::min<std::size_t>(limits_.payloads, 100000);
    for (std::size_t index = 0; index < max_steps; ++index) {
        if (current == relative_commit) return {AMIO_OK, PublicationState::Complete, std::string(relative_commit)};
        if (!visited.insert(current).second) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, std::string(relative_commit)};
        std::string commit_json;
        const auto read = store_->read(key(current), commit_json, true);
        if (read != KvReadOutcome::Found) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(relative_commit)};
        CompletionRecord record;
        if (parse_commit_document(commit_json, record, limits_) != AMIO_OK || record.dataset_uuid != uuid_) {
            return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(relative_commit)};
        }
        current = record.previous_commit;
        if (current.empty()) return {AMIO_ERR_STEP_FAILED, PublicationState::Failed, std::string(relative_commit)};
    }
    return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(relative_commit)};
}

}  // namespace amio::detail
