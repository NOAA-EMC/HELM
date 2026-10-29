#ifndef AMIO_SRC_CATALOG_TENSORSTORE_KVSTORE_HPP
#define AMIO_SRC_CATALOG_TENSORSTORE_KVSTORE_HPP

// Concrete bridge between TensorStore's generic KvStore API and the private
// AMIO catalog interface.  This header intentionally has no TensorStore
// dependency in builds which do not define AMIO_HAS_TENSORSTORE.
#include "catalog/kvstore_catalog.hpp"

#ifdef AMIO_HAS_TENSORSTORE

#include <tensorstore/kvstore/kvstore.h>
#include <tensorstore/kvstore/operations.h>

#include <absl/time/clock.h>

#include <limits>
#include <utility>

namespace amio::detail {

// These flags are evidence supplied by a store-specific qualification run,
// not inferred from TensorStore's generic API.  The default is deliberately
// uncertified: driver-specific atomicity, visibility and durability differ.
struct TensorStoreKvStoreCertification {
    KvStoreCapabilities capabilities{};
};

class TensorStoreKvStore final : public KvStore {
   public:
    explicit TensorStoreKvStore(tensorstore::kvstore::KvStore store,
                                 TensorStoreKvStoreCertification certification = {})
        : store_(std::move(store)), certification_(certification) {}

    [[nodiscard]] const KvStoreCapabilities &capabilities() const noexcept {
        return certification_.capabilities;
    }

    [[nodiscard]] KvWriteOutcome write(std::string_view key, std::string_view value,
                                       KvWriteCondition condition) override {
        if (!store_.valid() || !valid_key(key)) return KvWriteOutcome::Failed;
        tensorstore::kvstore::WriteOptions options;
        if (condition == KvWriteCondition::CreateOnly) {
            // TensorStore expresses create-if-absent as a conditional write.
            options.generation_conditions.if_equal = tensorstore::StorageGeneration::NoValue();
        }
        auto result = tensorstore::kvstore::Write(
                          store_, key, absl::Cord(value), std::move(options))
                          .result();
        if (!result) return KvWriteOutcome::Unknown;  // Could have reached the destination.
        if (!tensorstore::StorageGeneration::IsUnknown(result->generation)) return KvWriteOutcome::Written;
        return condition == KvWriteCondition::CreateOnly ? KvWriteOutcome::AlreadyExists
                                                         : KvWriteOutcome::Unknown;
    }

    [[nodiscard]] KvReadOutcome read(std::string_view key, std::string &value,
                                     bool require_fresh) override {
        if (!store_.valid() || !valid_key(key)) return KvReadOutcome::Failed;
        tensorstore::kvstore::ReadOptions options;
        if (require_fresh) options.staleness_bound = absl::Now();
        auto result = tensorstore::kvstore::Read(store_, key, options).result();
        if (!result) return KvReadOutcome::Unknown;
        if (result->not_found()) {
            value.clear();
            return KvReadOutcome::Missing;
        }
        if (!result->has_value()) return KvReadOutcome::Unknown;
        value = result->value.Flatten();
        return KvReadOutcome::Found;
    }

    [[nodiscard]] KvReadOutcome size(std::string_view key, std::uint64_t &bytes,
                                     bool require_fresh) override {
        // TensorStore has no portable metadata-only size call. ReadResult keeps
        // the value as a Cord, so querying its size avoids an additional copy;
        // stores with expensive full-value reads should not enable catalog
        // publication until they provide a driver-specific size path.
        if (!store_.valid() || !valid_key(key)) return KvReadOutcome::Failed;
        tensorstore::kvstore::ReadOptions options;
        if (require_fresh) options.staleness_bound = absl::Now();
        auto result = tensorstore::kvstore::Read(store_, key, options).result();
        if (!result) return KvReadOutcome::Unknown;
        if (result->not_found()) return KvReadOutcome::Missing;
        if (!result->has_value()) return KvReadOutcome::Unknown;
        const auto value_size = result->value.size();
        if (value_size > std::numeric_limits<std::uint64_t>::max()) return KvReadOutcome::Failed;
        bytes = static_cast<std::uint64_t>(value_size);
        return KvReadOutcome::Found;
    }

    [[nodiscard]] bool sync(std::string_view key) override {
        // TensorStore write completion follows the driver's durability
        // contract. There is no generic additional flush/sync primitive.
        (void)key;
        return false;
    }

   private:
    static bool valid_key(std::string_view key) {
        return !key.empty() && key.front() != '/' && key.find('\\') == std::string_view::npos &&
               key.find('\0') == std::string_view::npos;
    }

    tensorstore::kvstore::KvStore store_;
    TensorStoreKvStoreCertification certification_;
};

}  // namespace amio::detail

#endif  // AMIO_HAS_TENSORSTORE

#endif  // AMIO_SRC_CATALOG_TENSORSTORE_KVSTORE_HPP
