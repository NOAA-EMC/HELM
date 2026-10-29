#ifndef AMIO_SRC_CATALOG_POSIX_CATALOG_HPP
#define AMIO_SRC_CATALOG_POSIX_CATALOG_HPP

#include "catalog/catalog_format.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace amio::detail {

enum class PublicationState { Failed, Reconciling, Complete };
enum class PosixSyncLevel { Visibility, FileSync, DirectorySync };

struct PublicationResult {
    amio_status_t status = AMIO_ERR_BACKEND_FAILURE;
    PublicationState state = PublicationState::Failed;
    std::string commit_path;
};

class PosixCatalog {
   public:
    PosixCatalog(std::filesystem::path root, std::string dataset_uuid, std::string owner_token,
                 CatalogLimits limits = {});
    ~PosixCatalog();
    PosixCatalog(const PosixCatalog &) = delete;
    PosixCatalog &operator=(const PosixCatalog &) = delete;
    PosixCatalog(PosixCatalog &&) noexcept;
    PosixCatalog &operator=(PosixCatalog &&) noexcept;

    [[nodiscard]] amio_status_t create_exclusive();
    [[nodiscard]] amio_status_t write_payload(std::string_view relative_path, const void *data, std::size_t bytes,
                                              PosixSyncLevel sync = PosixSyncLevel::Visibility);
    [[nodiscard]] PublicationResult publish(CompletionRecord record, PosixSyncLevel sync = PosixSyncLevel::Visibility);
    [[nodiscard]] PublicationResult reconcile(std::string_view commit_path) const;
    [[nodiscard]] const std::filesystem::path &root() const noexcept;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace amio::detail

#endif // AMIO_SRC_CATALOG_POSIX_CATALOG_HPP
