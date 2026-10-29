#include "drivers/zarr/zarr_steps.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <limits>
#include <optional>
#include <stdexcept>

#ifdef AMIO_HAS_TENSORSTORE
#include <tensorstore/context.h>
#include <tensorstore/open.h>
#include <tensorstore/spec.h>
#include <tensorstore/tensorstore.h>
#include <tensorstore/index_space/dim_expression.h>
#include <tensorstore/util/result.h>
#include <nlohmann/json.hpp>
#endif

namespace amio::detail {

namespace {
constexpr std::uint64_t kMaxZarrShardsPerField = 1u << 20;

template <class PayloadMap>
bool validate_payload_map(const Schema &schema, const PayloadMap &payloads, std::string &error) {
    if (!validate_zarr_step_schema(schema, error)) return false;
    for (const auto &[name, variable] : schema.variables()) {
        const auto payload = payloads.find(name);
        if (payload == payloads.end()) {
            if (variable.required) {
                error = "Missing required Zarr step field: " + name;
                return false;
            }
            continue;
        }
        std::size_t width = 0;
        switch (variable.stored_dtype) {
            case AMIO_DTYPE_I8: case AMIO_DTYPE_U8: width = 1; break;
            case AMIO_DTYPE_I16: case AMIO_DTYPE_U16: width = 2; break;
            case AMIO_DTYPE_I32: case AMIO_DTYPE_U32: case AMIO_DTYPE_F32: width = 4; break;
            case AMIO_DTYPE_I64: case AMIO_DTYPE_U64: case AMIO_DTYPE_F64: width = 8; break;
            default: error = "Unsupported Zarr step dtype: " + name; return false;
        }
        std::size_t elements = 1;
        for (const auto &dimension : variable.dimensions) {
            if (dimension.extent > std::numeric_limits<std::size_t>::max() / elements) {
                error = "Zarr step field byte size overflows: " + name;
                return false;
            }
            elements *= static_cast<std::size_t>(dimension.extent);
        }
        if (width > std::numeric_limits<std::size_t>::max() / elements ||
            payload->second.size() != width * elements) {
            error = "Zarr step payload size does not match schema for field: " + name;
            return false;
        }
    }
    for (const auto &[name, payload] : payloads) {
        (void)payload;
        if (!schema.variables().contains(name)) {
            error = "Undeclared Zarr step field: " + name;
            return false;
        }
    }
    return true;
}
} // namespace

bool validate_zarr_step_schema(const Schema &schema, std::string &error) {
    if (!schema.sealed()) {
        error = "Zarr step schema must be sealed before payload creation";
        return false;
    }
    for (const auto &[name, variable] : schema.variables()) {
        if (name.empty() || variable.dimensions.empty()) {
            error = "Zarr step fields require a name and at least one dimension";
            return false;
        }
        std::uint64_t elements = 1;
        for (const auto &dimension : variable.dimensions) {
            if (dimension.extent == 0 || elements > std::numeric_limits<std::uint64_t>::max() / dimension.extent) {
                error = "Zarr step field has a zero or overflowing extent: " + name;
                return false;
            }
            elements *= dimension.extent;
        }
    }
    return true;
}

bool validate_zarr_step_payloads(
    const Schema &schema,
    const std::map<std::string, std::vector<std::byte>, std::less<>> &payloads,
    std::string &error) {
    return validate_payload_map(schema, payloads, error);
}

bool derive_zarr_shard_shapes(
    const Schema &schema,
    const std::map<std::string, std::vector<ZarrRegionView>, std::less<>> &regions,
    std::size_t max_shard_bytes, ZarrShardShapes &shard_shapes, std::string &error) {
    if (max_shard_bytes == 0 || !validate_zarr_step_schema(schema, error)) return false;
    shard_shapes.clear();
    for (const auto &[name, variable] : schema.variables()) {
        const auto found = regions.find(name);
        if (found == regions.end()) { error = "Missing Zarr producer regions for field: " + name; return false; }
        std::vector<std::uint64_t> shape;
        shape.reserve(variable.dimensions.size());
        for (const auto &dimension : variable.dimensions) shape.push_back(dimension.extent);
        for (const auto &region : found->second) {
            if (region.offsets.size() != shape.size() || region.extents.size() != shape.size()) {
                error = "Zarr producer region rank does not match field: " + name;
                return false;
            }
            for (std::size_t d = 0; d < shape.size(); ++d) {
                const auto offset = region.offsets[d], extent = region.extents[d];
                const auto global_extent = variable.dimensions[d].extent;
                if (extent == 0 || offset > global_extent || extent > global_extent - offset) {
                    error = "Invalid Zarr producer region for field: " + name;
                    return false;
                }
                shape[d] = std::gcd(shape[d], offset);
                shape[d] = std::gcd(shape[d], extent);
            }
        }
        if (std::any_of(shape.begin(), shape.end(), [](std::uint64_t d) { return d == 0; })) {
            error = "Could not derive a non-zero Zarr shard shape for field: " + name;
            return false;
        }
        std::size_t width = 0;
        switch (variable.stored_dtype) {
            case AMIO_DTYPE_I8: case AMIO_DTYPE_U8: width = 1; break;
            case AMIO_DTYPE_I16: case AMIO_DTYPE_U16: width = 2; break;
            case AMIO_DTYPE_I32: case AMIO_DTYPE_U32: case AMIO_DTYPE_F32: width = 4; break;
            case AMIO_DTYPE_I64: case AMIO_DTYPE_U64: case AMIO_DTYPE_F64: width = 8; break;
            default: error = "Unsupported Zarr field dtype: " + name; return false;
        }
        auto shard_bytes = [&]() -> std::optional<std::size_t> {
            std::size_t bytes = width;
            for (const auto extent : shape) {
                if (extent > std::numeric_limits<std::size_t>::max() / bytes) return std::nullopt;
                bytes *= static_cast<std::size_t>(extent);
            }
            return bytes;
        };
        auto bytes = shard_bytes();
        while (!bytes || *bytes > max_shard_bytes) {
            auto largest = std::max_element(shape.begin(), shape.end());
            if (largest == shape.end() || *largest <= 1) {
                error = "Zarr shard exceeds the configured byte budget for field: " + name;
                return false;
            }
            // Replacing a dimension with 1 always preserves grid alignment;
            // halving an even extent retains larger bounded shards cheaply.
            *largest = (*largest % 2 == 0) ? *largest / 2 : 1;
            bytes = shard_bytes();
        }
        std::uint64_t shard_count = 1;
        for (std::size_t d = 0; d < shape.size(); ++d) {
            const auto global = variable.dimensions[d].extent;
            const auto count = global / shape[d] + (global % shape[d] == 0 ? 0 : 1);
            if (count == 0 || count > kMaxZarrShardsPerField / shard_count) {
                error = "Zarr shard grid exceeds the per-field shard-count limit: " + name;
                return false;
            }
            shard_count *= count;
        }
        shard_shapes.emplace(name, std::move(shape));
    }
    return true;
}

bool validate_zarr_step_payloads(
    const Schema &schema,
    const std::map<std::string, std::span<const std::byte>, std::less<>> &payloads,
    std::string &error) {
    return validate_payload_map(schema, payloads, error);
}

bool make_zarr_step_layout(const std::string &collection_root, std::uint64_t step_id,
                           const Schema &schema, ZarrStepLayout &layout, std::string &error) {
    if (collection_root.empty() || !validate_zarr_step_schema(schema, error)) return false;
    // Catalog paths are treated as opaque relative keys by TensorStore. The
    // zero-padded numeric component is lexically ordered and collision-free.
    auto digits = std::to_string(step_id);
    std::string id(20 - (digits.size() < 20 ? digits.size() : 20), '0');
    id += digits;
    layout = {};
    layout.step_root = collection_root + "/payloads/steps/" + id;
    layout.schema_path = layout.step_root + "/schema.json";
    layout.array_paths.reserve(schema.variables().size());
    for (const auto &[name, variable] : schema.variables()) {
        (void)variable;
        layout.array_paths.push_back(layout.step_root + "/fields/" + encode_field_path(name));
    }
    return true;
}

bool write_zarr_step(const std::string &collection_root, const std::string &dataset_uuid,
                     std::uint64_t step_id, const Schema &schema,
                     const std::map<std::string, std::span<const std::byte>, std::less<>> &payloads,
                     std::string &error) {
#ifdef AMIO_HAS_TENSORSTORE
    if (!validate_zarr_step_payloads(schema, payloads, error)) return false;
    std::map<std::string, std::vector<ZarrRegionView>, std::less<>> all_regions;
    for (const auto &[name, variable] : schema.variables()) {
        (void)variable;
        const auto payload = payloads.find(name);
        if (payload == payloads.end()) { all_regions.emplace(name, std::vector<ZarrRegionView>{}); continue; }
        ZarrRegionView region;
        for (const auto &dimension : schema.variables().at(name).dimensions) {
            region.offsets.push_back(0);
            region.extents.push_back(dimension.extent);
        }
        region.data = payload->second;
        all_regions.emplace(name, std::vector<ZarrRegionView>{std::move(region)});
    }
    ZarrShardShapes shard_shapes;
    if (!derive_zarr_shard_shapes(schema, all_regions, 64u * 1024u * 1024u, shard_shapes, error) ||
        !create_zarr_step(collection_root, dataset_uuid, step_id, schema, shard_shapes, error)) return false;
    std::map<std::string, ZarrRegionView, std::less<>> regions;
    for (const auto &[name, values] : all_regions) if (!values.empty()) regions.emplace(name, values.front());
    return write_zarr_step_regions(collection_root, dataset_uuid, step_id, schema, shard_shapes, regions, error);
#else
    (void)collection_root; (void)dataset_uuid; (void)step_id; (void)schema; (void)payloads;
    error = "Native Zarr v3 step I/O requires AMIO_HAS_TENSORSTORE; NCZarr mode is unsupported";
    return false;
#endif
}

bool create_zarr_step(const std::string &collection_root, const std::string &dataset_uuid,
                      std::uint64_t step_id, const Schema &schema, const ZarrShardShapes &shard_shapes,
                      std::string &error) {
#ifdef AMIO_HAS_TENSORSTORE
    ZarrStepLayout layout;
    if (dataset_uuid.empty() || !make_zarr_step_layout(collection_root, step_id, schema, layout, error) ||
        shard_shapes.size() != schema.variables().size()) return false;
    auto dtype_name = [](amio_dtype_t type) -> const char * {
        switch (type) {
            case AMIO_DTYPE_F32: return "float32"; case AMIO_DTYPE_F64: return "float64";
            case AMIO_DTYPE_I8: return "int8"; case AMIO_DTYPE_I16: return "int16";
            case AMIO_DTYPE_I32: return "int32"; case AMIO_DTYPE_I64: return "int64";
            case AMIO_DTYPE_U8: return "uint8"; case AMIO_DTYPE_U16: return "uint16";
            case AMIO_DTYPE_U32: return "uint32"; case AMIO_DTYPE_U64: return "uint64";
            default: return nullptr;
        }
    };
    auto create_array = [&](const std::string &path, const std::vector<std::uint64_t> &shape,
                            const char *dtype, const std::vector<std::uint64_t> *shard_shape) {
        nlohmann::json spec = {{"driver", "zarr3"}, {"kvstore", {{"driver", "file"}, {"path", path}}}};
        nlohmann::json codecs;
        if (shard_shape == nullptr) {
            spec["metadata"] = {{"shape", shape}, {"data_type", dtype},
                                {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", shape}}}}},
                                {"chunk_key_encoding", {{"name", "default"}}},
                                {"codecs", nlohmann::json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}})}};
        } else {
            std::vector<std::uint64_t> inner_chunks;
            inner_chunks.reserve(shard_shape->size());
            for (const auto extent : *shard_shape) {
                std::uint64_t chunk = std::min<std::uint64_t>(extent, 64);
                while (chunk > 1 && extent % chunk != 0) --chunk;
                inner_chunks.push_back(chunk);
            }
            codecs = nlohmann::json::array({{{"name", "sharding_indexed"},
                {"configuration", {{"chunk_shape", inner_chunks},
                    {"codecs", nlohmann::json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}})},
                    {"index_codecs", nlohmann::json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}})},
                    {"index_location", "end"}}}}});
            spec["metadata"] = {{"shape", shape}, {"data_type", dtype},
                                {"chunk_grid", {{"name", "regular"}, {"configuration", {{"chunk_shape", *shard_shape}}}}},
                                {"chunk_key_encoding", {{"name", "default"}}}, {"codecs", codecs}};
        }
        auto parsed = tensorstore::Spec::FromJson(spec);
        if (!parsed.ok()) throw std::runtime_error(std::string(parsed.status().message()));
        auto opened = tensorstore::Open(parsed.value(), tensorstore::Context::Default(), tensorstore::OpenMode::create,
                                        tensorstore::ReadWriteMode::read_write).result();
        if (!opened.ok()) throw std::runtime_error(std::string(opened.status().message()));
    };
    try {
        SchemaDocument document;
        document.dataset_uuid = dataset_uuid;
        document.digest = schema.digest();
        document.canonical_schema = schema.canonical();
        for (const auto &[name, variable] : schema.variables()) if (variable.required) document.required_fields.push_back(name);
        std::string json;
        if (encode_schema_document(document, json) != AMIO_OK) { error = "Unable to encode step schema metadata"; return false; }
        std::vector<std::byte> schema_bytes(json.size());
        std::memcpy(schema_bytes.data(), json.data(), json.size());
        create_array(layout.schema_path, {static_cast<std::uint64_t>(schema_bytes.size())}, "uint8", nullptr);
        auto schema_spec = tensorstore::Spec::FromJson(nlohmann::json{{"driver", "zarr3"}, {"kvstore", {{"driver", "file"}, {"path", layout.schema_path}}}});
        if (!schema_spec.ok()) throw std::runtime_error(std::string(schema_spec.status().message()));
        auto schema_open = tensorstore::Open(schema_spec.value(), tensorstore::Context::Default(), tensorstore::OpenMode::open,
                                             tensorstore::ReadWriteMode::read_write).result();
        if (!schema_open.ok()) throw std::runtime_error(std::string(schema_open.status().message()));
        auto schema_store = std::move(schema_open).value();
        auto schema_array = tensorstore::UnownedToShared(tensorstore::Array(
            tensorstore::ElementPointer<const void>(static_cast<const void *>(schema_bytes.data()), schema_store.dtype()),
            std::vector<tensorstore::Index>{static_cast<tensorstore::Index>(schema_bytes.size())}, tensorstore::c_order));
        auto schema_write = tensorstore::Write(schema_array, schema_store).result();
        if (!schema_write.ok()) throw std::runtime_error(std::string(schema_write.status().message()));

        std::size_t path_index = 0;
        for (const auto &[name, variable] : schema.variables()) {
            const auto *dtype = dtype_name(variable.stored_dtype);
            const auto shard = shard_shapes.find(name);
            if (dtype == nullptr || shard == shard_shapes.end() || shard->second.size() != variable.dimensions.size())
                throw std::runtime_error("Invalid Zarr dtype or shard shape for field: " + name);
            std::vector<std::uint64_t> shape;
            for (const auto &dimension : variable.dimensions) shape.push_back(dimension.extent);
            create_array(layout.array_paths.at(path_index++), shape, dtype, &shard->second);
        }
    } catch (const std::exception &e) { error = e.what(); return false; }
    return true;
#else
    (void)collection_root; (void)dataset_uuid; (void)step_id; (void)schema; (void)shard_shapes;
    error = "Native Zarr v3 step I/O requires AMIO_HAS_TENSORSTORE; NCZarr mode is unsupported";
    return false;
#endif
}

bool write_zarr_step_regions(const std::string &collection_root, const std::string &dataset_uuid,
                             std::uint64_t step_id, const Schema &schema, const ZarrShardShapes &shard_shapes,
                             const std::map<std::string, ZarrRegionView, std::less<>> &regions,
                             std::string &error) {
#ifdef AMIO_HAS_TENSORSTORE
    ZarrStepLayout layout;
    if (dataset_uuid.empty() || !make_zarr_step_layout(collection_root, step_id, schema, layout, error)) return false;
    try {
        std::size_t path_index = 0;
        for (const auto &[field, variable] : schema.variables()) {
            const auto path = layout.array_paths.at(path_index++);
            const auto region_it = regions.find(field);
            if (region_it == regions.end()) continue;
            const auto &region = region_it->second;
            const auto shard_it = shard_shapes.find(field);
            if (shard_it == shard_shapes.end() || region.offsets.size() != variable.dimensions.size() ||
                region.extents.size() != variable.dimensions.size() || shard_it->second.size() != variable.dimensions.size())
                throw std::runtime_error("Invalid Zarr region or shard shape for field: " + field);
            std::size_t width = 0;
            switch (variable.stored_dtype) {
                case AMIO_DTYPE_I8: case AMIO_DTYPE_U8: width = 1; break;
                case AMIO_DTYPE_I16: case AMIO_DTYPE_U16: width = 2; break;
                case AMIO_DTYPE_I32: case AMIO_DTYPE_U32: case AMIO_DTYPE_F32: width = 4; break;
                case AMIO_DTYPE_I64: case AMIO_DTYPE_U64: case AMIO_DTYPE_F64: width = 8; break;
                default: throw std::runtime_error("Unsupported Zarr field dtype: " + field);
            }
            std::size_t elements = 1;
            std::vector<tensorstore::Index> offsets, extents, dims;
            for (std::size_t d = 0; d < variable.dimensions.size(); ++d) {
                const auto global = variable.dimensions[d].extent;
                if (region.offsets[d] > global || region.extents[d] > global - region.offsets[d] ||
                    region.extents[d] == 0 || region.offsets[d] % shard_it->second[d] != 0 ||
                    (region.offsets[d] + region.extents[d] != global && region.extents[d] % shard_it->second[d] != 0))
                    throw std::runtime_error("Zarr region is not aligned to complete shards: " + field);
                if (region.extents[d] > std::numeric_limits<std::size_t>::max() / elements)
                    throw std::runtime_error("Zarr region size overflows: " + field);
                elements *= static_cast<std::size_t>(region.extents[d]);
                if (region.offsets[d] > static_cast<std::uint64_t>(std::numeric_limits<tensorstore::Index>::max()) ||
                    region.extents[d] > static_cast<std::uint64_t>(std::numeric_limits<tensorstore::Index>::max()))
                    throw std::runtime_error("Zarr region exceeds TensorStore index range: " + field);
                offsets.push_back(static_cast<tensorstore::Index>(region.offsets[d]));
                extents.push_back(static_cast<tensorstore::Index>(region.extents[d]));
                dims.push_back(static_cast<tensorstore::Index>(region.extents[d]));
            }
            if (elements > std::numeric_limits<std::size_t>::max() / width || region.data.size() != elements * width)
                throw std::runtime_error("Zarr region payload size mismatch: " + field);
            auto spec = tensorstore::Spec::FromJson(nlohmann::json{{"driver", "zarr3"}, {"kvstore", {{"driver", "file"}, {"path", path}}}});
            if (!spec.ok()) throw std::runtime_error(std::string(spec.status().message()));
            auto opened = tensorstore::Open(spec.value(), tensorstore::Context::Default(), tensorstore::OpenMode::open,
                                            tensorstore::ReadWriteMode::read_write).result();
            if (!opened.ok()) throw std::runtime_error(std::string(opened.status().message()));
            auto store = std::move(opened).value();
            auto array = tensorstore::UnownedToShared(tensorstore::Array(
                tensorstore::ElementPointer<const void>(static_cast<const void *>(region.data.data()), store.dtype()), dims,
                tensorstore::c_order));
            auto result = tensorstore::Write(array, store | tensorstore::AllDims().SizedInterval(offsets, extents)).result();
            if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
        }
        for (const auto &[field, region] : regions) {
            (void)region;
            if (!schema.variables().contains(field)) throw std::runtime_error("Undeclared Zarr region field: " + field);
        }
    } catch (const std::exception &e) { error = e.what(); return false; }
    return true;
#else
    (void)collection_root; (void)dataset_uuid; (void)step_id; (void)schema; (void)shard_shapes; (void)regions;
    error = "Native Zarr v3 step I/O requires AMIO_HAS_TENSORSTORE; NCZarr mode is unsupported";
    return false;
#endif
}

bool write_zarr_step(const std::string &collection_root, const std::string &dataset_uuid,
                     std::uint64_t step_id, const Schema &schema,
                     const std::map<std::string, std::vector<std::byte>, std::less<>> &payloads,
                     std::string &error) {
#ifdef AMIO_HAS_TENSORSTORE
    std::map<std::string, std::span<const std::byte>, std::less<>> views;
    for (const auto &[field, bytes] : payloads) views.emplace(field, std::span<const std::byte>(bytes));
    return write_zarr_step(collection_root, dataset_uuid, step_id, schema, views, error);
#else
    (void)collection_root; (void)dataset_uuid; (void)step_id; (void)schema; (void)payloads;
    error = "Native Zarr v3 step I/O requires AMIO_HAS_TENSORSTORE; NCZarr mode is unsupported";
    return false;
#endif
}

bool read_zarr_step_field(const std::string &collection_root, const std::string &dataset_uuid,
                          std::uint64_t step_id, const Schema &expected_schema, const std::string &field,
                          std::vector<std::byte> &payload, std::string &error) {
    return read_zarr_step_field(collection_root, dataset_uuid, step_id, expected_schema, field, {}, {}, payload, error);
}

bool read_zarr_step_field(const std::string &collection_root, const std::string &dataset_uuid,
                          std::uint64_t step_id, const Schema &expected_schema, const std::string &field,
                          std::span<const std::uint64_t> offsets, std::span<const std::uint64_t> extents,
                          std::vector<std::byte> &payload, std::string &error) {
#ifdef AMIO_HAS_TENSORSTORE
    if (offsets.size() != extents.size()) {
        error = "Invalid Zarr step selection rank";
        return false;
    }
    ZarrStepLayout layout;
    if (dataset_uuid.empty() || !make_zarr_step_layout(collection_root, step_id, expected_schema, layout, error)) return false;
    const auto variable_it = expected_schema.variables().find(field);
    if (variable_it == expected_schema.variables().end()) { error = "Field is absent from step schema: " + field; return false; }
    try {
        auto open_array = [](const std::string &path) {
            nlohmann::json spec = {{"driver", "zarr3"}, {"kvstore", {{"driver", "file"}, {"path", path}}}};
            auto parsed = tensorstore::Spec::FromJson(spec);
            if (!parsed.ok()) throw std::runtime_error(std::string(parsed.status().message()));
            auto result = tensorstore::Open(parsed.value(), tensorstore::Context::Default(), tensorstore::OpenMode::open,
                                            tensorstore::ReadWriteMode::read).result();
            if (!result.ok()) throw std::runtime_error(std::string(result.status().message()));
            return std::move(result).value();
        };
        auto schema_store = open_array(layout.schema_path);
        const auto schema_shape = schema_store.domain().shape();
        if (schema_shape.size() != 1 || schema_shape[0] <= 0 || schema_shape[0] > Schema::kMaxCanonicalBytes)
            throw std::runtime_error("Invalid step schema array shape");
        std::string json(static_cast<std::size_t>(schema_shape[0]), '\0');
        auto schema_array = tensorstore::UnownedToShared(tensorstore::Array(
            tensorstore::ElementPointer<void>(static_cast<void *>(json.data()), tensorstore::dtype_v<std::uint8_t>),
            std::vector<tensorstore::Index>{schema_shape[0]}, tensorstore::c_order));
        auto schema_result = tensorstore::Read(schema_store, schema_array).result();
        if (!schema_result.ok()) throw std::runtime_error(std::string(schema_result.status().message()));
        SchemaDocument doc;
        if (parse_schema_document(json, doc) != AMIO_OK || doc.dataset_uuid != dataset_uuid || doc.digest != expected_schema.digest())
            throw std::runtime_error("Step schema metadata does not match requested dataset/schema");

        std::size_t index = 0;
        for (auto it = expected_schema.variables().begin(); it != variable_it; ++it) ++index;
        auto store = open_array(layout.array_paths.at(index));
        auto expected_dtype = [&]() -> tensorstore::DataType {
            switch (variable_it->second.stored_dtype) {
                case AMIO_DTYPE_F32: return tensorstore::dtype_v<float>; case AMIO_DTYPE_F64: return tensorstore::dtype_v<double>;
                case AMIO_DTYPE_I8: return tensorstore::dtype_v<std::int8_t>; case AMIO_DTYPE_I16: return tensorstore::dtype_v<std::int16_t>;
                case AMIO_DTYPE_I32: return tensorstore::dtype_v<std::int32_t>; case AMIO_DTYPE_I64: return tensorstore::dtype_v<std::int64_t>;
                case AMIO_DTYPE_U8: return tensorstore::dtype_v<std::uint8_t>; case AMIO_DTYPE_U16: return tensorstore::dtype_v<std::uint16_t>;
                case AMIO_DTYPE_U32: return tensorstore::dtype_v<std::uint32_t>; case AMIO_DTYPE_U64: return tensorstore::dtype_v<std::uint64_t>;
                default: return tensorstore::DataType{};
            }
        }();
        if (!expected_dtype.valid() || store.dtype() != expected_dtype ||
            store.rank() != static_cast<tensorstore::DimensionIndex>(variable_it->second.dimensions.size()))
            throw std::runtime_error("Step array rank does not match schema");
        for (std::size_t d = 0; d < variable_it->second.dimensions.size(); ++d)
            if (store.domain().shape()[d] != static_cast<tensorstore::Index>(variable_it->second.dimensions[d].extent))
                throw std::runtime_error("Step array shape does not match schema");
        if (!offsets.empty() && offsets.size() != variable_it->second.dimensions.size()) {
            error = "Zarr selection rank does not match field rank";
            return false;
        }
        std::vector<std::uint64_t> read_shape;
        std::vector<tensorstore::Index> read_offsets, read_shape_index;
        read_shape.reserve(variable_it->second.dimensions.size());
        read_offsets.reserve(variable_it->second.dimensions.size());
        read_shape_index.reserve(variable_it->second.dimensions.size());
        for (std::size_t d = 0; d < variable_it->second.dimensions.size(); ++d) {
            const auto global = variable_it->second.dimensions[d].extent;
            const auto offset = offsets.empty() ? 0 : offsets[d];
            const auto extent = extents.empty() ? global : extents[d];
            if (extent == 0 || offset > global || extent > global - offset ||
                offset > static_cast<std::uint64_t>(std::numeric_limits<tensorstore::Index>::max()) ||
                extent > static_cast<std::uint64_t>(std::numeric_limits<tensorstore::Index>::max())) {
                error = "Zarr selection is outside the stored field";
                return false;
            }
            read_shape.push_back(extent);
            read_offsets.push_back(static_cast<tensorstore::Index>(offset));
            read_shape_index.push_back(static_cast<tensorstore::Index>(extent));
        }
        std::uint64_t count = 1;
        for (const auto extent : read_shape) {
            if (extent > std::numeric_limits<std::uint64_t>::max() / count) {
                error = "Zarr selection element count overflows";
                return false;
            }
            count *= extent;
        }
        const auto width = variable_it->second.stored_dtype == AMIO_DTYPE_I8 || variable_it->second.stored_dtype == AMIO_DTYPE_U8 ? 1 :
            (variable_it->second.stored_dtype == AMIO_DTYPE_I16 || variable_it->second.stored_dtype == AMIO_DTYPE_U16 ? 2 :
             (variable_it->second.stored_dtype == AMIO_DTYPE_I32 || variable_it->second.stored_dtype == AMIO_DTYPE_U32 || variable_it->second.stored_dtype == AMIO_DTYPE_F32 ? 4 : 8));
        if (count > std::numeric_limits<std::size_t>::max() / width) throw std::runtime_error("Step array byte count overflows");
        payload.resize(static_cast<std::size_t>(count) * width);
        auto array = tensorstore::UnownedToShared(tensorstore::Array(
            tensorstore::ElementPointer<void>(static_cast<void *>(payload.data()), store.dtype()),
            read_shape_index, tensorstore::c_order));
        tensorstore::Result<void> read_result;
        if (offsets.empty()) {
            read_result = tensorstore::Read(store, array).result();
        } else {
            read_result = tensorstore::Read(store | tensorstore::AllDims().SizedInterval(read_offsets, read_shape_index), array).result();
        }
        if (!read_result.ok()) throw std::runtime_error(std::string(read_result.status().message()));
    } catch (const std::exception &e) { error = e.what(); payload.clear(); return false; }
    return true;
#else
    (void)collection_root; (void)dataset_uuid; (void)step_id; (void)expected_schema; (void)field; (void)payload;
    error = "Native Zarr v3 step I/O requires AMIO_HAS_TENSORSTORE; NCZarr mode is unsupported";
    return false;
#endif
}

} // namespace amio::detail
