#ifndef AMIO_SRC_DRIVERS_ZARR_ZARR_STEPS_HPP
#define AMIO_SRC_DRIVERS_ZARR_ZARR_STEPS_HPP

#include "catalog/catalog_format.hpp"
#include "schema/schema.hpp"

#include <cstdint>
#include <map>
#include <span>
#include <cstddef>
#include <string>
#include <vector>

namespace amio::detail {

// Names and immutable metadata used by the TensorStore step adapter. Each
// field is a native Zarr v3 array below a unique step payload directory.
struct ZarrStepLayout {
    std::string step_root;
    std::string schema_path;
    std::vector<std::string> array_paths;
};

struct ZarrRegionView {
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> extents;
    std::span<const std::byte> data;
};

using ZarrShardShapes = std::map<std::string, std::vector<std::uint64_t>, std::less<>>;

[[nodiscard]] bool make_zarr_step_layout(const std::string &collection_root,
                                         std::uint64_t step_id,
                                         const Schema &schema,
                                         ZarrStepLayout &layout,
                                         std::string &error);

// Check all extents with overflow protection before any arrays are created.
[[nodiscard]] bool validate_zarr_step_schema(const Schema &schema, std::string &error);

// Validate a per-step set of present arrays. Missing optional fields are
// allowed; required omissions, undeclared fields, and byte/shape mismatches
// fail before any TensorStore array is created.
[[nodiscard]] bool validate_zarr_step_payloads(
    const Schema &schema,
    const std::map<std::string, std::vector<std::byte>, std::less<>> &payloads,
    std::string &error);
[[nodiscard]] bool validate_zarr_step_payloads(
    const Schema &schema,
    const std::map<std::string, std::span<const std::byte>, std::less<>> &payloads,
    std::string &error);

// Derive a regular shard grid compatible with every producer rectangle. The
// returned shard shape divides all global extents, offsets, and region sizes.
[[nodiscard]] bool derive_zarr_shard_shapes(
    const Schema &schema,
    const std::map<std::string, std::vector<ZarrRegionView>, std::less<>> &regions,
    std::size_t max_shard_bytes, ZarrShardShapes &shard_shapes, std::string &error);

// Persist/read complete field payloads as independent native Zarr v3 arrays.
// The schema is stored in a uint8 Zarr array at layout.schema_path. Calls
// block on every TensorStore future before returning. These APIs are
// unavailable in NCZarr builds and return false with a diagnostic there.
[[nodiscard]] bool write_zarr_step(const std::string &collection_root, const std::string &dataset_uuid,
                                   std::uint64_t step_id, const Schema &schema,
                                   const std::map<std::string, std::vector<std::byte>, std::less<>> &payloads,
                                   std::string &error);
[[nodiscard]] bool write_zarr_step(const std::string &collection_root, const std::string &dataset_uuid,
                                   std::uint64_t step_id, const Schema &schema,
                                   const std::map<std::string, std::span<const std::byte>, std::less<>> &payloads,
                                   std::string &error);
[[nodiscard]] bool create_zarr_step(const std::string &collection_root, const std::string &dataset_uuid,
                                    std::uint64_t step_id, const Schema &schema,
                                    const ZarrShardShapes &shard_shapes, std::string &error);
[[nodiscard]] bool write_zarr_step_regions(const std::string &collection_root, const std::string &dataset_uuid,
                                           std::uint64_t step_id, const Schema &schema,
                                           const ZarrShardShapes &shard_shapes,
                                           const std::map<std::string, ZarrRegionView, std::less<>> &regions,
                                           std::string &error);
[[nodiscard]] bool read_zarr_step_field(const std::string &collection_root, const std::string &dataset_uuid,
                                        std::uint64_t step_id, const Schema &expected_schema,
                                        const std::string &field, std::vector<std::byte> &payload,
                                        std::string &error);
[[nodiscard]] bool read_zarr_step_field(const std::string &collection_root, const std::string &dataset_uuid,
                                        std::uint64_t step_id, const Schema &expected_schema,
                                        const std::string &field, std::span<const std::uint64_t> offsets,
                                        std::span<const std::uint64_t> extents, std::vector<std::byte> &payload,
                                        std::string &error);

} // namespace amio::detail

#endif
