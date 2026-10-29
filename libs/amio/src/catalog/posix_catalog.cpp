#include "catalog/posix_catalog.hpp"

#include <atomic>
#include <charconv>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <set>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace amio::detail {
namespace {

std::mutex owner_mutex;
std::map<std::string, std::string, std::less<>> owners;
std::atomic<std::uint64_t> temp_sequence{1};

bool safe_relative_path(std::string_view path) {
    if (path.empty() || path.front() == '/' || path.find('\\') != std::string_view::npos || path.find('\0') != std::string_view::npos) return false;
    std::size_t at = 0;
    while (at <= path.size()) {
        const auto slash = path.find('/', at);
        const auto part = path.substr(at, slash == std::string_view::npos ? path.size() - at : slash - at);
        if (part.empty() || part == "." || part == "..") return false;
        if (slash == std::string_view::npos) break;
        at = slash + 1;
    }
    return true;
}

bool sync_requested(PosixSyncLevel level) { return level != PosixSyncLevel::Visibility; }
bool sync_directory_requested(PosixSyncLevel level) { return level == PosixSyncLevel::DirectorySync; }

PosixSyncLevel sync_level(std::string_view value, bool &valid) {
    valid = true;
    if (value == "visibility") return PosixSyncLevel::Visibility;
    if (value == "file_sync") return PosixSyncLevel::FileSync;
    if (value == "directory_sync") return PosixSyncLevel::DirectorySync;
    valid = false;
    return PosixSyncLevel::Visibility;
}

bool sync_dir(const std::filesystem::path &path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) return false;
    const bool ok = ::fsync(fd) == 0;
    const int close_status = ::close(fd);
    return ok && close_status == 0;
}

bool sync_dir_chain(std::filesystem::path path, const std::filesystem::path &root) {
    path = path.lexically_normal();
    const auto normalized_root = root.lexically_normal();
    while (path == normalized_root || path.string().starts_with(normalized_root.string() + "/")) {
        if (!sync_dir(path)) return false;
        if (path == normalized_root) return true;
        const auto parent = path.parent_path();
        if (parent == path || parent.empty()) return false;
        path = parent;
    }
    return false;
}

bool write_all(int fd, std::string_view bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto written = ::write(fd, bytes.data() + offset, bytes.size() - offset);
        if (written < 0) { if (errno == EINTR) continue; return false; }
        if (written == 0) return false;
        offset += static_cast<std::size_t>(written);
    }
    return true;
}

bool file_equals(const std::filesystem::path &path, std::string_view expected) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return false;
    struct stat st {};
    bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) == expected.size();
    std::string actual;
    if (ok) {
        actual.resize(expected.size());
        std::size_t offset = 0;
        while (offset < actual.size()) {
            const auto count = ::read(fd, actual.data() + offset, actual.size() - offset);
            if (count < 0) { if (errno == EINTR) continue; ok = false; break; }
            if (count == 0) { ok = false; break; }
            offset += static_cast<std::size_t>(count);
        }
    }
    const int close_status = ::close(fd);
    return ok && close_status == 0 && actual == expected;
}

std::string head_json(std::string_view uuid, std::uint64_t step, std::string_view commit) {
    // UUID and commit are validated to a restricted path-safe character set by callers.
    return std::string("{\"format_version\":1,\"dataset_uuid\":\"") + std::string(uuid) + "\",\"step_id\":" +
           std::to_string(step) + ",\"commit\":\"" + std::string(commit) + "\"}";
}

bool read_bounded(const std::filesystem::path &path, std::size_t max_bytes, std::string &bytes, bool &missing) {
    missing = false;
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) { missing = errno == ENOENT; return false; }
    struct stat st {};
    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 || static_cast<std::uint64_t>(st.st_size) > max_bytes) {
        ::close(fd); return false;
    }
    bytes.assign(static_cast<std::size_t>(st.st_size), '\0');
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto count = ::read(fd, bytes.data() + offset, bytes.size() - offset);
        if (count < 0) { if (errno == EINTR) continue; ::close(fd); return false; }
        if (count == 0) { ::close(fd); return false; }
        offset += static_cast<std::size_t>(count);
    }
    return ::close(fd) == 0;
}

bool parse_head(std::string_view text, std::string_view uuid, std::uint64_t &step, std::string &commit) {
    const auto prefix = std::string("{\"format_version\":1,\"dataset_uuid\":\"") + std::string(uuid) + "\",\"step_id\":";
    constexpr std::string_view marker = ",\"commit\":\"";
    if (!text.starts_with(prefix)) return false;
    const auto step_begin = prefix.size();
    const auto marker_at = text.find(marker, step_begin);
    if (marker_at == std::string_view::npos || marker_at == step_begin) return false;
    const auto parsed = std::from_chars(text.data() + step_begin, text.data() + marker_at, step);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + marker_at) return false;
    const auto commit_begin = marker_at + marker.size();
    if (text.size() < commit_begin + 2 || text.substr(text.size() - 2) != "\"}") return false;
    commit.assign(text.substr(commit_begin, text.size() - commit_begin - 2));
    return safe_relative_path(commit) && commit.rfind("commits/", 0) == 0;
}

bool valid_uuid(std::string_view uuid) {
    if (uuid.empty() || uuid.size() > 128) return false;
    for (const unsigned char c : uuid) if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_')) return false;
    return true;
}

} // namespace

struct PosixCatalog::Impl {
    std::filesystem::path root;
    std::string uuid;
    std::string owner;
    CatalogLimits limits;
    std::string owner_key;
    bool claimed = false;
    bool created = false;

    Impl(std::filesystem::path path, std::string id, std::string token, CatalogLimits configured)
        : root(std::move(path)), uuid(std::move(id)), owner(std::move(token)), limits(configured) {}

    void release_owner() {
        if (!claimed) return;
        std::lock_guard lock(owner_mutex);
        const auto it = owners.find(owner_key);
        if (it != owners.end() && it->second == owner) owners.erase(it);
        claimed = false;
    }

    bool claim_owner() {
        std::error_code ec;
        auto absolute = std::filesystem::absolute(root, ec).lexically_normal();
        if (ec) return false;
        owner_key = absolute.string();
        std::lock_guard lock(owner_mutex);
        if (owners.contains(owner_key)) return false;
        owners.emplace(owner_key, owner);
        claimed = true;
        return true;
    }

    bool resolve(std::string_view relative, std::filesystem::path &path, bool create_parent) const {
        if (!safe_relative_path(relative)) return false;
        path = root / std::filesystem::path(relative);
        std::error_code ec;
        const auto parent = path.parent_path();
        if (create_parent) std::filesystem::create_directories(parent, ec);
        if (ec) return false;
        const auto root_real = std::filesystem::weakly_canonical(root, ec);
        if (ec) return false;
        const auto parent_real = std::filesystem::weakly_canonical(parent, ec);
        if (ec) return false;
        auto root_it = root_real.begin();
        auto parent_it = parent_real.begin();
        for (; root_it != root_real.end() && parent_it != parent_real.end(); ++root_it, ++parent_it) if (*root_it != *parent_it) return false;
        return root_it == root_real.end();
    }

    bool atomic_write(std::string_view relative, std::string_view bytes, bool replace, PosixSyncLevel sync) const {
        std::filesystem::path destination;
        if (!resolve(relative, destination, true)) return false;
        const auto temp = destination.string() + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(temp_sequence.fetch_add(1));
        const int fd = ::open(temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0644);
        if (fd < 0) return false;
        bool ok = write_all(fd, bytes);
        if (ok && sync_requested(sync)) ok = ::fsync(fd) == 0;
        const int close_status = ::close(fd);
        ok = ok && close_status == 0;
        if (ok) {
            if (replace) ok = ::rename(temp.c_str(), destination.c_str()) == 0;
            else {
                // link() provides create-exclusive publication without replacing an existing immutable record.
                ok = ::link(temp.c_str(), destination.c_str()) == 0;
                if (ok) ::unlink(temp.c_str());
            }
        }
        if (!ok) ::unlink(temp.c_str());
        if (ok && sync_directory_requested(sync)) ok = sync_dir_chain(destination.parent_path(), root.parent_path());
        return ok;
    }

    bool sync_payload(std::string_view relative, std::uint64_t expected_size, PosixSyncLevel sync) const {
        std::filesystem::path path;
        if (!resolve(relative, path, false)) return false;
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return false;
        struct stat st {};
        bool ok = ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0 && static_cast<std::uint64_t>(st.st_size) == expected_size;
        if (ok && sync_requested(sync)) ok = ::fsync(fd) == 0;
        const int close_status = ::close(fd);
        ok = ok && close_status == 0;
        if (ok && sync_directory_requested(sync)) ok = sync_dir_chain(path.parent_path(), root.parent_path());
        return ok;
    }

    bool visible_array_container(std::string_view relative) const {
        std::filesystem::path path;
        if (!resolve(relative, path, false)) return false;
        const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (fd < 0) return false;
        struct stat st {};
        const bool ok = ::fstat(fd, &st) == 0 && S_ISDIR(st.st_mode);
        return ::close(fd) == 0 && ok;
    }
};

PosixCatalog::PosixCatalog(std::filesystem::path root, std::string dataset_uuid, std::string owner_token, CatalogLimits limits)
    : impl_(std::make_unique<Impl>(std::move(root), std::move(dataset_uuid), std::move(owner_token), limits)) {
    std::error_code ec;
    auto absolute = std::filesystem::absolute(impl_->root, ec);
    if (!ec) impl_->root = absolute.lexically_normal();
}
PosixCatalog::~PosixCatalog() { if (impl_) impl_->release_owner(); }
PosixCatalog::PosixCatalog(PosixCatalog &&) noexcept = default;
PosixCatalog &PosixCatalog::operator=(PosixCatalog &&) noexcept = default;

amio_status_t PosixCatalog::create_exclusive() {
    if (!impl_ || !valid_uuid(impl_->uuid) || impl_->owner.empty() || impl_->owner.size() > 128 || impl_->root.empty()) return AMIO_ERR_INVALID_INPUT;
    if (impl_->created) return AMIO_OK;
    if (!impl_->claim_owner()) return AMIO_ERR_INVALID_POLICY;
    std::error_code ec;
    std::filesystem::create_directories(impl_->root.parent_path(), ec);
    if (ec || !std::filesystem::create_directory(impl_->root, ec) || ec) { impl_->release_owner(); return AMIO_ERR_INVALID_POLICY; }
    impl_->created = true;
    const std::string identity = std::string("{\"format_version\":1,\"layout\":\"step_collection\",\"dataset_uuid\":\"") + impl_->uuid + "\"}";
    if (!impl_->atomic_write("amio.json", identity, false, PosixSyncLevel::Visibility)) {
        impl_->release_owner(); impl_->created = false; return AMIO_ERR_BACKEND_FAILURE;
    }
    return AMIO_OK;
}

amio_status_t PosixCatalog::write_payload(std::string_view relative_path, const void *data, std::size_t bytes, PosixSyncLevel sync) {
    if (!impl_ || !impl_->created || !impl_->claimed || (bytes && !data) || !safe_relative_path(relative_path) || relative_path.rfind("payloads/", 0) != 0) {
        return AMIO_ERR_INVALID_INPUT;
    }
    const auto contents = bytes == 0 ? std::string_view{} : std::string_view(static_cast<const char *>(data), bytes);
    return impl_->atomic_write(relative_path, contents, false, sync) ? AMIO_OK : AMIO_ERR_BACKEND_FAILURE;
}

PublicationResult PosixCatalog::publish(CompletionRecord record, PosixSyncLevel sync) {
    if (!impl_ || !impl_->created || !impl_->claimed || record.dataset_uuid != impl_->uuid) return {AMIO_ERR_INVALID_POLICY, PublicationState::Failed, {}};
    if (sync_requested(sync)) {
        const std::string schema_path = "payloads/schemas/" + record.schema_digest + ".json";
        const std::string metadata_paths[] = {"amio.json", schema_path};
        for (const auto &metadata_path : metadata_paths) {
            std::error_code ec;
            const auto bytes = std::filesystem::file_size(impl_->root / metadata_path, ec);
            if (ec || !impl_->sync_payload(metadata_path, bytes, sync))
                return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, {}};
        }
    }
    for (const auto &payload : record.payloads) {
        if (payload.array_container) {
            // TensorStore has completed its array write futures, but AMIO has
            // no generic recursive stable-media sync contract for this tree.
            if (sync != PosixSyncLevel::Visibility || !impl_->visible_array_container(payload.relative_path))
                return {AMIO_ERR_UNSUPPORTED, PublicationState::Failed, {}};
        } else if (!impl_->sync_payload(payload.relative_path, payload.bytes, sync)) {
            return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, {}};
        }
    }
    record.achieved_guarantee = sync == PosixSyncLevel::DirectorySync ? "directory_sync" : (sync == PosixSyncLevel::FileSync ? "file_sync" : "visibility");
    std::string encoded;
    const auto encoded_status = encode_commit_document(record, encoded, impl_->limits);
    if (encoded_status != AMIO_OK) return {encoded_status, PublicationState::Failed, {}};
    const auto commit = std::string("commits/") + std::to_string(record.step_id) + ".json";
    const auto created = impl_->atomic_write(commit, encoded, false, sync);
    if (!created) {
        if (!file_equals(impl_->root / commit, encoded) || !impl_->sync_payload(commit, encoded.size(), sync)) {
            return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, commit};
        }
    }
    const auto head = head_json(record.dataset_uuid, record.step_id, commit);
    if (impl_->atomic_write("head.json", head, true, sync)) return {AMIO_OK, PublicationState::Complete, commit};
    auto outcome = reconcile(commit);
    if (outcome.state == PublicationState::Complete) return outcome;
    if (outcome.status == AMIO_ERR_PUBLICATION_PENDING) return outcome;
    return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, commit};
}

PublicationResult PosixCatalog::reconcile(std::string_view commit_path) const {
    if (!impl_ || !safe_relative_path(commit_path) || commit_path.rfind("commits/", 0) != 0) return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, {}};
    std::filesystem::path path;
    if (!impl_->resolve(commit_path, path, false)) return {AMIO_ERR_INVALID_INPUT, PublicationState::Failed, std::string(commit_path)};
    std::string bytes; bool missing = false;
    if (!read_bounded(path, impl_->limits.document_bytes, bytes, missing)) {
        return {missing ? AMIO_ERR_STEP_FAILED : AMIO_ERR_PUBLICATION_PENDING,
                missing ? PublicationState::Failed : PublicationState::Reconciling, std::string(commit_path)};
    }
    CompletionRecord record;
    if (parse_commit_document(bytes, record, impl_->limits) != AMIO_OK || record.dataset_uuid != impl_->uuid) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, std::string(commit_path)};
    std::filesystem::path head_path;
    if (!impl_->resolve("head.json", head_path, false)) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
    std::string head_bytes;
    if (!read_bounded(head_path, impl_->limits.string_bytes + 512, head_bytes, missing)) {
        return {missing ? AMIO_ERR_STEP_FAILED : AMIO_ERR_PUBLICATION_PENDING,
                missing ? PublicationState::Failed : PublicationState::Reconciling, std::string(commit_path)};
    }
    std::uint64_t head_step = 0; std::string current;
    if (!parse_head(head_bytes, impl_->uuid, head_step, current)) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
    bool guarantee_valid = false;
    const auto guarantee = sync_level(record.achieved_guarantee, guarantee_valid);
    if (!guarantee_valid) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, std::string(commit_path)};
    if (sync_requested(guarantee)) {
        for (const auto &payload : record.payloads) {
            if (!impl_->sync_payload(payload.relative_path, payload.bytes, guarantee)) {
                return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
            }
        }
        if (!impl_->sync_payload(commit_path, bytes.size(), guarantee) || !impl_->sync_payload("head.json", head_bytes.size(), guarantee)) {
            return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
        }
    }
    std::set<std::string> seen;
    const auto traversal_limit = std::min<std::size_t>(impl_->limits.payloads, 100000);
    for (std::size_t traversed = 0; traversed < traversal_limit; ++traversed) {
        if (current == commit_path) return {AMIO_OK, PublicationState::Complete, std::string(commit_path)};
        if (!seen.insert(current).second) return {AMIO_ERR_BACKEND_FAILURE, PublicationState::Failed, std::string(commit_path)};
        std::filesystem::path current_path;
        if (!impl_->resolve(current, current_path, false)) return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
        std::string current_bytes;
        if (!read_bounded(current_path, impl_->limits.document_bytes, current_bytes, missing)) {
            return {missing ? AMIO_ERR_PUBLICATION_PENDING : AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
        }
        CompletionRecord current_record;
        if (parse_commit_document(current_bytes, current_record, impl_->limits) != AMIO_OK || current_record.dataset_uuid != impl_->uuid || current_record.step_id > head_step) {
            return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
        }
        current = current_record.previous_commit;
        if (current.empty()) return {AMIO_ERR_STEP_FAILED, PublicationState::Failed, std::string(commit_path)};
    }
    return {AMIO_ERR_PUBLICATION_PENDING, PublicationState::Reconciling, std::string(commit_path)};
}

const std::filesystem::path &PosixCatalog::root() const noexcept {
    static const std::filesystem::path empty;
    return impl_ ? impl_->root : empty;
}

} // namespace amio::detail
