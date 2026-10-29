#ifndef AMIO_SRC_CATALOG_COLLECTION_READER_HPP
#define AMIO_SRC_CATALOG_COLLECTION_READER_HPP

#include "catalog/catalog_format.hpp"
#include "amio/amio_errors.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace amio::detail {

struct CatalogCursor {
    std::string dataset_uuid;
    std::string generation;
    std::string next_commit;
};

struct CatalogStep {
    CompletionRecord completion;
    bool legacy_unverified = false;
};

struct CatalogPage {
    amio_status_t status = AMIO_OK;
    std::vector<CatalogStep> steps;
    CatalogCursor next;
    bool has_more = false;
};

/** Read-only bounded discovery by following immutable commit links from head.json. */
class CollectionReader {
public:
    CollectionReader(std::filesystem::path root, std::string dataset_uuid, CatalogLimits limits = {});
    [[nodiscard]] CatalogPage list_steps(const CatalogCursor &cursor, std::size_t max_results) const;
    [[nodiscard]] amio_status_t describe_step(std::uint64_t step_id, CatalogStep &result) const;
    [[nodiscard]] const std::string &dataset_uuid() const noexcept { return uuid_; }
private:
    std::filesystem::path root_;
    std::string uuid_;
    CatalogLimits limits_;
};

} // namespace amio::detail

#endif
