#ifndef AMIO_SRC_CATALOG_CATALOG_FORMAT_HPP
#define AMIO_SRC_CATALOG_CATALOG_FORMAT_HPP

#include "amio/amio_errors.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace amio::detail {

struct CatalogLimits {
    std::size_t document_bytes = 16 * 1024 * 1024;
    std::size_t string_bytes = 1024 * 1024;
    std::size_t fields = 10000;
    std::size_t payloads = 100000;
    std::size_t dimensions = 7;
};

struct SchemaDocument {
    std::uint32_t format_version = 1;
    std::string dataset_uuid;
    std::string digest;
    std::string canonical_schema;
    std::vector<std::string> required_fields;
};

struct PayloadDescriptor {
    std::string field;
    std::string relative_path;
    std::uint64_t bytes = 0;
    std::uint32_t producer = 0;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> extents;
    bool array_container = false;
    struct MessageRange {
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
    };
    // Optional ordered ranges for complete messages in a GRIB2 payload.
    // Empty for array-oriented backends and legacy catalog records.
    std::vector<MessageRange> message_ranges;
};

struct CompletionRecord {
    std::uint32_t format_version = 1;
    std::string dataset_uuid;
    std::uint64_t step_id = 0;
    std::string schema_digest;
    std::vector<std::string> required_fields;
    std::vector<std::string> present_fields;
    std::vector<PayloadDescriptor> payloads;
    std::string achieved_guarantee = "visibility";
    std::string previous_commit;

    [[nodiscard]] bool empty() const noexcept { return dataset_uuid.empty(); }
};

[[nodiscard]] std::string encode_field_path(std::string_view field);
[[nodiscard]] bool decode_field_path(std::string_view component, std::string &field, std::size_t max_bytes = 1024 * 1024);
[[nodiscard]] std::string encode_typed_double(double value);
[[nodiscard]] amio_status_t encode_schema_document(const SchemaDocument &schema, std::string &json,
                                                   const CatalogLimits &limits = {});
[[nodiscard]] amio_status_t parse_schema_document(std::string_view json, SchemaDocument &schema,
                                                  const CatalogLimits &limits = {});
[[nodiscard]] amio_status_t encode_commit_document(const CompletionRecord &record, std::string &json,
                                                   const CatalogLimits &limits = {});
[[nodiscard]] amio_status_t parse_commit_document(std::string_view json, CompletionRecord &record,
                                                  const CatalogLimits &limits = {});

} // namespace amio::detail

#endif // AMIO_SRC_CATALOG_CATALOG_FORMAT_HPP
