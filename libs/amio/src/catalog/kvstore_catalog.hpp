#ifndef AMIO_SRC_CATALOG_KVSTORE_CATALOG_HPP
#define AMIO_SRC_CATALOG_KVSTORE_CATALOG_HPP

#include <cstdint>
#include <string>
#include <string_view>

#include "catalog/posix_catalog.hpp"

namespace amio::detail {

enum class KvWriteCondition { CreateOnly, Replace };
enum class KvWriteOutcome { Written, AlreadyExists, Unknown, Failed };
enum class KvReadOutcome { Found, Missing, Unknown, Failed };

struct KvStoreCapabilities {
    bool atomic_single_key_replace = false;
    bool atomic_create_if_absent = false;
    bool fresh_read_after_write = false;
    bool stable_sync = false;
};

class KvStore {
   public:
    virtual ~KvStore() = default;
    [[nodiscard]] virtual KvWriteOutcome write(std::string_view key, std::string_view value, KvWriteCondition condition) = 0;
    [[nodiscard]] virtual KvReadOutcome read(std::string_view key, std::string &value, bool require_fresh) = 0;
    [[nodiscard]] virtual KvReadOutcome size(std::string_view key, std::uint64_t &bytes, bool require_fresh) = 0;
    [[nodiscard]] virtual bool sync(std::string_view key) = 0;
};

class KvStoreCatalog {
   public:
    KvStoreCatalog(KvStore &store, std::string prefix, std::string dataset_uuid, std::string owner_token, KvStoreCapabilities capabilities,
                   bool external_exclusive_owner = false, CatalogLimits limits = {});
    [[nodiscard]] amio_status_t certify(PosixSyncLevel requested) const;
    [[nodiscard]] amio_status_t create_exclusive();
    [[nodiscard]] PublicationResult publish(CompletionRecord record, PosixSyncLevel requested = PosixSyncLevel::Visibility);
    [[nodiscard]] PublicationResult reconcile(std::string_view relative_commit) const;

   private:
    [[nodiscard]] std::string key(std::string_view relative) const;
    KvStore *store_;
    std::string prefix_;
    std::string uuid_;
    std::string owner_;
    KvStoreCapabilities capabilities_;
    bool external_exclusive_owner_;
    bool created_ = false;
    CatalogLimits limits_;
};

}  // namespace amio::detail

#endif  // AMIO_SRC_CATALOG_KVSTORE_CATALOG_HPP
