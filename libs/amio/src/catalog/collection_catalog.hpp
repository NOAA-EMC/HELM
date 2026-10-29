#ifndef AMIO_SRC_CATALOG_COLLECTION_CATALOG_HPP
#define AMIO_SRC_CATALOG_COLLECTION_CATALOG_HPP

#include "catalog/catalog_format.hpp"
#include "catalog/posix_catalog.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace amio::detail {

enum class CatalogWrite { Payload, Commit, Head };
enum class FailureMoment { BeforeWrite, AfterWrite };

/** Small deterministic store used to exercise publication ordering and ambiguous acknowledgments. */
class MemoryCatalogStore {
public:
    void fail_next(CatalogWrite write, FailureMoment moment);
    void serve_stale_reads(bool stale) noexcept { stale_reads_ = stale; }
    void inject_unreferenced_payload(std::string path);
    [[nodiscard]] const std::vector<std::string> &write_order() const noexcept { return write_order_; }
private:
    friend class CollectionCatalog;
    struct Failure { CatalogWrite write; FailureMoment moment; };
    std::vector<Failure> failures_;
    std::vector<std::string> write_order_;
    std::map<std::string, std::string, std::less<>> commits_;
    std::vector<std::string> payloads_;
    std::string head_;
    bool stale_reads_ = false;
};

struct CollectionPublicationResult {
    amio_status_t status = AMIO_ERR_BACKEND_FAILURE;
    PublicationState state = PublicationState::Failed;
    std::string commit_id;
};

class CollectionCatalog {
public:
    CollectionCatalog(std::shared_ptr<MemoryCatalogStore> store, std::string root, std::string uuid, std::string owner);
    [[nodiscard]] amio_status_t create_exclusive();
    [[nodiscard]] CollectionPublicationResult publish(CompletionRecord record);
    [[nodiscard]] CollectionPublicationResult reconcile(const std::string &commit_id) const;
    [[nodiscard]] CompletionRecord read_head() const;
    [[nodiscard]] std::vector<CompletionRecord> list_steps() const;
    [[nodiscard]] std::vector<std::string> list_orphans() const;
private:
    [[nodiscard]] bool fail(CatalogWrite write, FailureMoment moment);
    std::shared_ptr<MemoryCatalogStore> store_;
    std::string root_, uuid_, owner_;
    bool created_ = false;
};

} // namespace amio::detail

#endif
