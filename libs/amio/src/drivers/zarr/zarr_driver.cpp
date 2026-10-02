// zarr_driver.cpp -- AMIO Zarr_Driver implementation (TensorStore mode).
//
// This translation unit implements the Zarr_Driver for Zarr v3 datasets.
// When AMIO_HAS_TENSORSTORE is defined, the driver uses Google TensorStore
// configured for Zarr v3.  When not defined, the driver compiles but
// open_write/open_read throw indicating TensorStore is unavailable (the
// NCZarr fallback in zarr_nczarr_fallback.cpp handles that case).
//
// Registration: BackendRegistrar<Zarr_Driver>("zarr3") at static init.
//
// Validates: R8.1, R8.2, R8.3, R8.4, R8.5, R8.9, R8.10

#include "drivers/zarr/zarr_driver.hpp"

#include <algorithm>
#include <conf/config.hpp>
#include <conf/error.hpp>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "drivers/common/var_attributes.hpp"
#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"

#ifdef AMIO_HAS_TENSORSTORE
#include <tensorstore/context.h>
#include <tensorstore/index_space/dim_expression.h>
#include <tensorstore/open.h>
#include <tensorstore/spec.h>
#include <tensorstore/tensorstore.h>
#include <tensorstore/util/result.h>
#include <tensorstore/util/status.h>

#include <nlohmann/json.hpp>
#endif

namespace amio::detail {

// ===================================================================
// Static registration with BackendFactory under key "zarr3".
// ===================================================================

namespace {
BackendRegistrar<Zarr_Driver> reg_zarr3("zarr3");
}  // anonymous namespace

// ===================================================================
// Helper: determine if a URI is a cloud URI.
// ===================================================================

bool Zarr_Driver::is_cloud_uri(const std::string &uri) {
    return uri.rfind("s3://", 0) == 0 || uri.rfind("gs://", 0) == 0 || uri.rfind("https://", 0) == 0;
}

// ===================================================================
// Helper: categorize network/auth errors (R8.9).
// ===================================================================

std::string Zarr_Driver::categorize_error(const std::string &message) {
    // Heuristic categorization based on error message content.
    std::string lower_msg = message;
    std::transform(lower_msg.begin(), lower_msg.end(), lower_msg.begin(), [](unsigned char c) { return std::tolower(c); });

    if (lower_msg.find("auth") != std::string::npos || lower_msg.find("credential") != std::string::npos ||
        lower_msg.find("permission") != std::string::npos || lower_msg.find("forbidden") != std::string::npos ||
        lower_msg.find("401") != std::string::npos || lower_msg.find("403") != std::string::npos) {
        return "authentication/authorization error";
    }
    if (lower_msg.find("network") != std::string::npos || lower_msg.find("connect") != std::string::npos ||
        lower_msg.find("timeout") != std::string::npos || lower_msg.find("dns") != std::string::npos ||
        lower_msg.find("unreachable") != std::string::npos) {
        return "network error";
    }
    if (lower_msg.find("scheme") != std::string::npos || lower_msg.find("unsupported") != std::string::npos) {
        return "unsupported URI scheme";
    }
    return "I/O error";
}

// ===================================================================
// Helper: convert amio_dtype_t to TensorStore dtype string.
// ===================================================================

std::string Zarr_Driver::dtype_to_string(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return "float32";
        case AMIO_DTYPE_F64:
            return "float64";
        case AMIO_DTYPE_I8:
            return "int8";
        case AMIO_DTYPE_I16:
            return "int16";
        case AMIO_DTYPE_I32:
            return "int32";
        case AMIO_DTYPE_I64:
            return "int64";
        case AMIO_DTYPE_U8:
            return "uint8";
        case AMIO_DTYPE_U16:
            return "uint16";
        case AMIO_DTYPE_U32:
            return "uint32";
        case AMIO_DTYPE_U64:
            return "uint64";
        default:
            return "float32";
    }
}

// ===================================================================
// Helper: convert amio_dtype_t to element size in bytes.
// ===================================================================

std::size_t Zarr_Driver::dtype_size(amio_dtype_t dtype) {
    // Delegate to the shared element_size helper (backend_driver.hpp).  Zarr's
    // existing contract falls back to a 4-byte (float32) width for an
    // unrecognized dtype, so preserve that default here.
    const std::size_t size = element_size(dtype);
    return size == 0 ? 4 : size;
}

// ===================================================================
// parse_zarr_config -- extract and validate Zarr config from CONF.
// ===================================================================

ZarrConfig Zarr_Driver::parse_zarr_config(const conf::Config &config) {
    ZarrConfig cfg;

    // Collect missing required fields (R8.10).
    std::vector<std::string> missing_fields;

    // URI / path (required).
    if (!config.has("uri")) {
        missing_fields.push_back("uri");
    } else {
        cfg.uri = config.get_string("uri");
    }

    // Chunk shape (required).
    if (!config.has("chunk_shape")) {
        missing_fields.push_back("chunk_shape");
    } else {
        auto chunks = config.get_int_list("chunk_shape");
        cfg.chunk_shape.assign(chunks.begin(), chunks.end());
    }

    // Shard shape (required).
    if (!config.has("shard_shape")) {
        missing_fields.push_back("shard_shape");
    } else {
        auto shards = config.get_int_list("shard_shape");
        cfg.shard_shape.assign(shards.begin(), shards.end());
    }

    // Array shape (required).
    if (!config.has("array_shape")) {
        missing_fields.push_back("array_shape");
    } else {
        auto shape = config.get_int_list("array_shape");
        cfg.array_shape.assign(shape.begin(), shape.end());
    }

    // Codec (required).
    if (!config.has("codec")) {
        missing_fields.push_back("codec");
    } else {
        cfg.codec = config.get_string("codec");
    }

    // Data type (optional, derived from VarMeta at write time).
    if (config.has("dtype")) {
        cfg.dtype_str = config.get_string("dtype");
    }

    // Report all missing fields in a single error (R8.10).
    if (!missing_fields.empty()) {
        std::ostringstream oss;
        oss << "Zarr_Driver: missing required configuration fields: ";
        for (std::size_t i = 0; i < missing_fields.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << missing_fields[i];
        }
        throw std::runtime_error(oss.str());
    }

    return cfg;
}

// ===================================================================
// validate_sharding -- chunk dims must divide shard dims (R8.3).
// All dimensions must be positive integers.
// ===================================================================

void Zarr_Driver::validate_sharding(const ZarrConfig &cfg) {
    if (cfg.chunk_shape.size() != cfg.shard_shape.size()) {
        std::string msg =
            "Zarr_Driver: chunk_shape and shard_shape must have the same "
            "number of dimensions (got " +
            std::to_string(cfg.chunk_shape.size()) + " vs " + std::to_string(cfg.shard_shape.size()) + ")";
        throw std::runtime_error(msg);
    }

    for (std::size_t i = 0; i < cfg.chunk_shape.size(); ++i) {
        if (cfg.chunk_shape[i] <= 0) {
            std::string msg =
                "Zarr_Driver: chunk_shape[" + std::to_string(i) + "] must be a positive integer (got " + std::to_string(cfg.chunk_shape[i]) + ")";
            throw std::runtime_error(msg);
        }
        if (cfg.shard_shape[i] <= 0) {
            std::string msg =
                "Zarr_Driver: shard_shape[" + std::to_string(i) + "] must be a positive integer (got " + std::to_string(cfg.shard_shape[i]) + ")";
            throw std::runtime_error(msg);
        }
        if (cfg.shard_shape[i] % cfg.chunk_shape[i] != 0) {
            std::string msg = "Zarr_Driver: chunk_shape[" + std::to_string(i) + "] (" + std::to_string(cfg.chunk_shape[i]) +
                              ") must evenly divide shard_shape[" + std::to_string(i) + "] (" + std::to_string(cfg.shard_shape[i]) + ")";
            throw std::runtime_error(msg);
        }
    }
}

// ===================================================================
// validate_codec -- must be one of {blosc, zstandard} (R8.4).
// ===================================================================

void Zarr_Driver::validate_codec(const ZarrConfig &cfg) {
    if (cfg.codec != "blosc" && cfg.codec != "zstandard") {
        std::string msg = "Zarr_Driver: codec must be one of {blosc, zstandard} (got '" + cfg.codec + "')";
        throw std::runtime_error(msg);
    }
}

// ===================================================================
// TensorStore-specific helpers (only compiled with TensorStore).
// ===================================================================

#ifdef AMIO_HAS_TENSORSTORE

namespace {

// Build the KvStore spec JSON for a given URI (R8.2).
// Cloud URIs (s3://, gs://, https://) route through TensorStore
// KvStore HTTP REST transport.
nlohmann::json build_kvstore_spec(const std::string &uri) {
    nlohmann::json kvstore;

    if (uri.rfind("s3://", 0) == 0) {
        // S3 bucket: s3://bucket/path
        kvstore["driver"] = "s3";
        std::string path = uri.substr(5);  // Remove "s3://"
        auto slash_pos = path.find('/');
        if (slash_pos != std::string::npos) {
            kvstore["bucket"] = path.substr(0, slash_pos);
            kvstore["path"] = path.substr(slash_pos + 1);
        } else {
            kvstore["bucket"] = path;
            kvstore["path"] = "";
        }
    } else if (uri.rfind("gs://", 0) == 0) {
        // Google Cloud Storage: gs://bucket/path
        kvstore["driver"] = "gcs";
        std::string path = uri.substr(5);  // Remove "gs://"
        auto slash_pos = path.find('/');
        if (slash_pos != std::string::npos) {
            kvstore["bucket"] = path.substr(0, slash_pos);
            kvstore["path"] = path.substr(slash_pos + 1);
        } else {
            kvstore["bucket"] = path;
            kvstore["path"] = "";
        }
    } else if (uri.rfind("https://", 0) == 0) {
        // HTTPS endpoint: route through http KvStore.
        kvstore["driver"] = "http";
        kvstore["base_url"] = uri;
    } else {
        // Local filesystem path.
        kvstore["driver"] = "file";
        kvstore["path"] = uri;
    }

    return kvstore;
}

// Map an amio_dtype_t to the corresponding TensorStore DataType.
//
// Mirrors the dtype case set in Zarr_Driver::dtype_to_string so the
// read/write array views carry the variable's real element type rather
// than a hardcoded float32 (Req 11.1, 11.2).  Returns a default-
// constructed (invalid) DataType for an unsupported dtype tag; callers
// check DataType::valid() and throw so the cordon maps it to
// AMIO_ERR_INVALID_INPUT (Req 11.4).
tensorstore::DataType to_ts_dtype(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return tensorstore::dtype_v<float>;
        case AMIO_DTYPE_F64:
            return tensorstore::dtype_v<double>;
        case AMIO_DTYPE_I8:
            return tensorstore::dtype_v<std::int8_t>;
        case AMIO_DTYPE_I16:
            return tensorstore::dtype_v<std::int16_t>;
        case AMIO_DTYPE_I32:
            return tensorstore::dtype_v<std::int32_t>;
        case AMIO_DTYPE_I64:
            return tensorstore::dtype_v<std::int64_t>;
        case AMIO_DTYPE_U8:
            return tensorstore::dtype_v<std::uint8_t>;
        case AMIO_DTYPE_U16:
            return tensorstore::dtype_v<std::uint16_t>;
        case AMIO_DTYPE_U32:
            return tensorstore::dtype_v<std::uint32_t>;
        case AMIO_DTYPE_U64:
            return tensorstore::dtype_v<std::uint64_t>;
        default:
            // Unsupported dtype -> invalid DataType; caller throws.
            return tensorstore::DataType{};
    }
}

// Map a TensorStore DataType back to an amio_dtype_t (inverse of
// to_ts_dtype).  Used by describe_variable to report the store's
// element type from ts_store_.dtype().  Returns true and writes *out
// on success; returns false for an element type with no AMIO dtype
// mapping (e.g. bool, complex, string), in which case the variable
// cannot be described robustly (describe_variable -> found = false).
bool to_amio_dtype(tensorstore::DataType dt, amio_dtype_t &out) {
    if (dt == tensorstore::dtype_v<float>) {
        out = AMIO_DTYPE_F32;
        return true;
    }
    if (dt == tensorstore::dtype_v<double>) {
        out = AMIO_DTYPE_F64;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::int8_t>) {
        out = AMIO_DTYPE_I8;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::int16_t>) {
        out = AMIO_DTYPE_I16;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::int32_t>) {
        out = AMIO_DTYPE_I32;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::int64_t>) {
        out = AMIO_DTYPE_I64;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::uint8_t>) {
        out = AMIO_DTYPE_U8;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::uint16_t>) {
        out = AMIO_DTYPE_U16;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::uint32_t>) {
        out = AMIO_DTYPE_U32;
        return true;
    }
    if (dt == tensorstore::dtype_v<std::uint64_t>) {
        out = AMIO_DTYPE_U64;
        return true;
    }
    return false;
}

}  // anonymous namespace

#endif  // AMIO_HAS_TENSORSTORE

// ===================================================================
// open_write -- validate config, configure TensorStore for writing.
//
// This implementation is used only in TensorStore mode.  When
// AMIO_NCZARR_FALLBACK is defined, the NCZarr fallback file
// (zarr_nczarr_fallback.cpp) provides this method instead.
// ===================================================================

#ifndef AMIO_NCZARR_FALLBACK

void Zarr_Driver::open_write(const conf::Config &config) {
    if (is_open_) {
        throw std::runtime_error("Zarr_Driver: already open");
    }

    // Parse and validate configuration.
    config_ = parse_zarr_config(config);
    validate_sharding(config_);
    validate_codec(config_);

    // Parse CF/UGRID convention metadata + per-variable attributes
    // from the manifest.  These are injected into the Zarr v3 group /
    // array `attributes` metadata below.
    attributes_ = parse_dataset_attributes(config);

#ifdef AMIO_HAS_TENSORSTORE
    // Build TensorStore JSON spec for Zarr v3 with sharding.
    nlohmann::json spec;
    spec["driver"] = "zarr3";
    spec["kvstore"] = build_kvstore_spec(config_.uri);

    // Configure metadata: dtype, shape, codecs.
    auto &metadata = spec["metadata"];
    metadata["shape"] = config_.array_shape;
    metadata["data_type"] = config_.dtype_str.empty() ? "float32" : config_.dtype_str;

    // Inject CF/UGRID attributes into the Zarr v3 `attributes` map.
    // Zarr v3 stores user metadata under metadata.attributes, which
    // TensorStore persists in zarr.json (the CF/Zarr equivalent of
    // netCDF attributes).  The global `Conventions` string plus any
    // global attributes are written here; per-variable attributes for
    // this array are merged in as well (one array per driver instance).
    {
        nlohmann::json attrs_json = nlohmann::json::object();
        attrs_json["Conventions"] = attributes_.conventions;
        for (const auto &kv : attributes_.global.items) {
            if (kv.second.is_numeric) {
                attrs_json[kv.first] = kv.second.number;
            } else {
                attrs_json[kv.first] = kv.second.text;
            }
        }
        metadata["attributes"] = std::move(attrs_json);
    }

    // Configure zarr3_sharding_indexed codec chain (R8.3, R8.4):
    //   outer: bytes → sharding_indexed
    //   inner (per-chunk): bytes → byte-shuffle → compression
    nlohmann::json codecs = nlohmann::json::array();

    // Bytes codec (required for Zarr v3 outer level).
    nlohmann::json bytes_codec;
    bytes_codec["name"] = "bytes";
    bytes_codec["configuration"]["endian"] = "little";
    codecs.push_back(bytes_codec);

    // Sharding codec with inner codecs (R8.3).
    nlohmann::json sharding;
    sharding["name"] = "sharding_indexed";
    auto &shard_cfg = sharding["configuration"];
    shard_cfg["chunk_shape"] = config_.chunk_shape;

    // Inner codecs: bytes → byte-shuffle → compression.
    nlohmann::json inner_codecs = nlohmann::json::array();

    // Inner bytes codec.
    nlohmann::json inner_bytes;
    inner_bytes["name"] = "bytes";
    inner_bytes["configuration"]["endian"] = "little";
    inner_codecs.push_back(inner_bytes);

    // Byte-Shuffle filter (R8.4).
    nlohmann::json byte_shuffle;
    byte_shuffle["name"] = "transpose";
    byte_shuffle["configuration"]["order"] = "C";
    inner_codecs.push_back(byte_shuffle);

    // Compression codec: exactly one of {Blosc, Zstandard} (R8.4).
    nlohmann::json compression;
    if (config_.codec == "blosc") {
        compression["name"] = "blosc";
        compression["configuration"]["cname"] = "lz4";
        compression["configuration"]["clevel"] = 5;
        compression["configuration"]["shuffle"] = "bitshuffle";
        compression["configuration"]["typesize"] = 4;
        compression["configuration"]["blocksize"] = 0;
    } else {  // zstandard
        compression["name"] = "zstd";
        compression["configuration"]["level"] = 3;
    }
    inner_codecs.push_back(compression);

    shard_cfg["codecs"] = inner_codecs;
    codecs.push_back(sharding);

    metadata["codecs"] = codecs;
    metadata["chunk_grid"]["name"] = "regular";
    metadata["chunk_grid"]["configuration"]["chunk_shape"] = config_.shard_shape;

    // Create mode for writing.
    spec["create"] = true;
    spec["delete_existing"] = true;

    // Open TensorStore.
    ts_context_ = tensorstore::Context::Default();

    auto open_result =
        tensorstore::Open(tensorstore::Spec::FromJson(spec).value(), ts_context_,
                          tensorstore::OpenMode::create | tensorstore::OpenMode::delete_existing, tensorstore::ReadWriteMode::read_write)
            .result();

    if (!open_result.ok()) {
        // Network/auth errors abort operation, leave target unchanged,
        // return categorized error (R8.9).
        std::string err_msg = std::string(open_result.status().message());
        std::string category = categorize_error(err_msg);
        std::string full_msg = "Zarr_Driver: failed to open TensorStore for writing (" + category + "): " + err_msg;
        throw std::runtime_error(full_msg);
    }

    ts_store_ = std::move(open_result).value();

#else
    // TensorStore not available at compile time.
    // This path is reached only if NCZarr fallback (task 7.3) is not
    // handling the request.
    std::string msg =
        "Zarr_Driver: TensorStore is not available in this build. "
        "Rebuild with AMIO_HAS_TENSORSTORE=ON or use NCZarr fallback "
        "mode.";
    throw std::runtime_error(msg);
#endif  // AMIO_HAS_TENSORSTORE

    is_open_ = true;
    is_write_mode_ = true;
}

// ===================================================================
// open_read -- validate config, open TensorStore for reading.
// ===================================================================

void Zarr_Driver::open_read(const conf::Config &config) {
    if (is_open_) {
        throw std::runtime_error("Zarr_Driver: already open");
    }

    // Parse and validate configuration.
    config_ = parse_zarr_config(config);
    validate_sharding(config_);
    validate_codec(config_);

#ifdef AMIO_HAS_TENSORSTORE
    // Build TensorStore JSON spec for reading.
    nlohmann::json spec;
    spec["driver"] = "zarr3";
    spec["kvstore"] = build_kvstore_spec(config_.uri);
    spec["open"] = true;

    ts_context_ = tensorstore::Context::Default();

    auto open_result =
        tensorstore::Open(tensorstore::Spec::FromJson(spec).value(), ts_context_, tensorstore::OpenMode::open, tensorstore::ReadWriteMode::read)
            .result();

    if (!open_result.ok()) {
        // Network/auth errors abort, leave target unchanged (R8.9).
        std::string err_msg = std::string(open_result.status().message());
        std::string category = categorize_error(err_msg);
        std::string full_msg = "Zarr_Driver: failed to open TensorStore for reading (" + category + "): " + err_msg;
        throw std::runtime_error(full_msg);
    }

    ts_store_ = std::move(open_result).value();

#else
    // TensorStore not available at compile time.
    std::string msg =
        "Zarr_Driver: TensorStore is not available in this build. "
        "Rebuild with AMIO_HAS_TENSORSTORE=ON or use NCZarr fallback "
        "mode.";
    throw std::runtime_error(msg);
#endif  // AMIO_HAS_TENSORSTORE

    is_open_ = true;
    is_write_mode_ = false;
}

// ===================================================================
// write -- serialize StagingBuffer through TensorStore (R8.1, R8.4).
//
// The Byte-Shuffle filter and compression codec are applied by
// TensorStore automatically based on the codec chain configured in
// open_write.  Network/auth errors abort the operation and leave the
// target unchanged (R8.9).
// ===================================================================

void Zarr_Driver::write(const StagingBuffer &src, const VarMeta &meta) {
    if (!is_open_ || !is_write_mode_) {
        std::string msg = "Zarr_Driver: write called on driver not open for writing";
        throw std::runtime_error(msg);
    }

#ifdef AMIO_HAS_TENSORSTORE
    // Determine the element size and total element count.
    const std::size_t elem_size = dtype_size(meta.dtype);
    const std::size_t total_elements = src.used_bytes / elem_size;
    (void)total_elements;  // Used implicitly via shape.

    // Build the domain from the variable shape.
    std::vector<tensorstore::Index> shape_vec;
    for (int d = 0; d < meta.shape.rank; ++d) {
        shape_vec.push_back(static_cast<tensorstore::Index>(meta.shape.extents[d]));
    }

    // Create a shared array view over the staging buffer data.
    // The non-owning shared_ptr ensures TensorStore does not free
    // the staging buffer (ownership remains with Staging_Pool).
    //
    // Resolve the element type from the variable's dtype so the write
    // array view carries the real element type rather than a hardcoded
    // float32; the same dispatch is applied on the read side so
    // write/read round-trips are byte-equal (Req 11.2, 11.3).
    tensorstore::DataType dt = to_ts_dtype(meta.dtype);
    if (!dt.valid()) {
        std::string msg = "Zarr_Driver: unsupported dtype";  // -> AMIO_ERR_INVALID_INPUT (Req 11.4)
        throw std::runtime_error(msg);
    }
    auto array = tensorstore::Array(tensorstore::ElementPointer<const void>(src.data, dt), shape_vec, tensorstore::c_order);

    // Write the data to TensorStore.  The codec chain configured in
    // open_write applies Byte-Shuffle + compression automatically.
    auto write_future = tensorstore::Write(array, ts_store_);
    auto write_result = write_future.result();

    if (!write_result.ok()) {
        // Network/auth errors abort, leave target unchanged (R8.9).
        std::string err_msg = std::string(write_result.status().message());
        std::string category = categorize_error(err_msg);
        std::string full_msg = "Zarr_Driver: write failed (" + category + "): " + err_msg;
        throw std::runtime_error(full_msg);
    }

#else
    (void)src;
    (void)meta;
    std::string msg = "Zarr_Driver: TensorStore is not available in this build.";
    throw std::runtime_error(msg);
#endif  // AMIO_HAS_TENSORSTORE
}

// ===================================================================
// read -- read from TensorStore into StagingBuffer (R8.1, R8.5).
// ===================================================================

void Zarr_Driver::read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) {
    if (!is_open_ || is_write_mode_) {
        std::string msg = "Zarr_Driver: read called on driver not open for reading";
        throw std::runtime_error(msg);
    }

#ifdef AMIO_HAS_TENSORSTORE
    (void)timestep;  // Timestep is encoded in the array path/index.

    // Determine the element size.
    const std::size_t elem_size = dtype_size(meta.dtype);

    // Determine the selected extents per dimension.  With a bounding box
    // the selection is the box extents (the number of elements chosen
    // per dimension); without one it is the full variable shape (Req
    // 10.1, 10.4).  The array view passed to tensorstore::Read and the
    // reported used_bytes are both sized from these selected extents so
    // a subset read reports the sub-region size, not the full array
    // (Req 10.3).
    std::vector<tensorstore::Index> shape_vec;
    if (bbox.has_value()) {
        shape_vec.reserve(static_cast<std::size_t>(bbox->rank));
        for (int d = 0; d < bbox->rank; ++d) {
            shape_vec.push_back(static_cast<tensorstore::Index>(bbox->extents[d]));
        }
    } else {
        shape_vec.reserve(static_cast<std::size_t>(meta.shape.rank));
        for (int d = 0; d < meta.shape.rank; ++d) {
            shape_vec.push_back(static_cast<tensorstore::Index>(meta.shape.extents[d]));
        }
    }

    // Calculate total bytes for the selected region.
    std::size_t total_elements = 1;
    for (const tensorstore::Index ext : shape_vec) {
        total_elements *= static_cast<std::size_t>(ext);
    }
    std::size_t total_bytes = total_elements * elem_size;

    if (total_bytes > dst.capacity_bytes) {
        std::string msg = "Zarr_Driver: staging buffer capacity (" + std::to_string(dst.capacity_bytes) + " bytes) insufficient for read payload (" +
                          std::to_string(total_bytes) + " bytes)";
        throw std::runtime_error(msg);
    }

    // Create a target array view over the staging buffer, sized to the
    // selected region.
    //
    // Resolve the element type from the variable's dtype so the read
    // transfers the bytes for the real element type rather than
    // reinterpreting them as float32 (Req 11.1, 11.2).  An unsupported
    // dtype yields an invalid DataType; throw so the cordon maps it to
    // AMIO_ERR_INVALID_INPUT (Req 11.4).
    tensorstore::DataType dt = to_ts_dtype(meta.dtype);
    if (!dt.valid()) {
        std::string msg = "Zarr_Driver: unsupported dtype";  // -> AMIO_ERR_INVALID_INPUT (Req 11.4)
        throw std::runtime_error(msg);
    }
    auto array = tensorstore::Array(tensorstore::ElementPointer<void>(dst.data, dt), shape_vec, tensorstore::c_order);

    // Restrict the read domain to the bounding box, if any.  For each
    // dimension, TranslateSizedInterval(offset, size, stride) selects
    // `size` elements starting at `offset` with the given stride and
    // translates the origin back to 0, so the resulting domain matches
    // the array view shape.  Without a bounding box the entire store
    // full domain is read (Req 10.4).
    tensorstore::TensorStore<> source = ts_store_;
    if (bbox.has_value()) {
        const auto &b = *bbox;
        for (int d = 0; d < b.rank; ++d) {
            auto sliced = source | tensorstore::Dims(d).TranslateSizedInterval(b.offsets[d], b.extents[d], b.strides[d]);
            if (!sliced.ok()) {
                std::string err_msg = std::string(sliced.status().message());
                std::string full_msg = "Zarr_Driver: failed to restrict read domain for dimension " + std::to_string(d) + ": " + err_msg;
                throw std::runtime_error(full_msg);
            }
            source = std::move(sliced).value();
        }
    }

    // Read from TensorStore.
    auto read_future = tensorstore::Read(source, array);
    auto read_result = read_future.result();

    if (!read_result.ok()) {
        // Network/auth errors abort, leave target unchanged (R8.9).
        std::string err_msg = std::string(read_result.status().message());
        std::string category = categorize_error(err_msg);
        std::string full_msg = "Zarr_Driver: read failed (" + category + "): " + err_msg;
        throw std::runtime_error(full_msg);
    }

    dst.used_bytes = total_bytes;

#else
    (void)dst;
    (void)meta;
    (void)timestep;
    (void)bbox;
    std::string msg = "Zarr_Driver: TensorStore is not available in this build.";
    throw std::runtime_error(msg);
#endif  // AMIO_HAS_TENSORSTORE
}

// ===================================================================
// flush -- ensure all async TensorStore operations complete.
// ===================================================================

void Zarr_Driver::flush() {
    if (!is_open_) {
        return;  // No-op if not open.
    }

#ifdef AMIO_HAS_TENSORSTORE
    // TensorStore operations are synchronous in our usage (we call
    // .result() on each future).  Nothing additional to flush.
    // If we switch to async writes in the future, we would collect
    // futures and wait on them here.
#endif
}

// ===================================================================
// close -- close TensorStore handles and release resources.
// ===================================================================

void Zarr_Driver::close() {
    if (!is_open_) {
        return;  // No-op if not open.
    }

#ifdef AMIO_HAS_TENSORSTORE
    // Reset TensorStore state.  The TensorStore object's destructor
    // handles cleanup of internal resources.
    ts_store_ = tensorstore::TensorStore<>();
    ts_context_ = tensorstore::Context();
#endif

    is_open_ = false;
    is_write_mode_ = false;
}

#endif  // !AMIO_NCZARR_FALLBACK

// ===================================================================
// describe_variable -- introspect a variable's dtype, shape, and
// timestep count from the open TensorStore store (Req 4.1, 4.2, 4.5).
//
// Defined outside the AMIO_NCZARR_FALLBACK guard above because
// zarr_driver.cpp is compiled in BOTH the TensorStore build and the
// NCZarr fallback build, and the Zarr_Driver vtable references this
// override in either case (zarr_nczarr_fallback.cpp does not define
// it).  The real introspection is compiled only when TensorStore is
// available; otherwise the base-class behavior (found = false) applies
// so the NCZarr fallback reports no describable variable here.
//
// Steps (design §3), mirroring the NetCDF approach:
//   * ts_store_.dtype()   -> element type (inverse of to_ts_dtype); an
//                            unmapped element type => found = false.
//   * ts_store_.domain()  -> rank + per-dimension extent.
//   * Timestep model: Zarr encodes the timestep in the leading array
//     index (design §3).  When the store has rank >= 2 the leading
//     dimension is treated as the time axis: its extent is
//     total_timesteps and the reported per-timestep shape is the
//     remaining dimensions.  A rank-1 store has no separable time axis,
//     so total_timesteps = 1 and the full 1-D shape is reported.
//
// Returns VariableInfo{found = false} when the driver is not open for
// reading, the store's element type has no AMIO dtype mapping, or the
// reported rank is outside [1, AMIO_MAX_RANK].  A false result causes
// the read path to fail the read with AMIO_ERR_BACKEND_FAILURE.
// ===================================================================

VariableInfo Zarr_Driver::describe_variable(const std::string &name) {
    VariableInfo info{};  // found == false by default.

    if (!is_open_ || is_write_mode_) {
        return info;
    }

#ifdef AMIO_HAS_TENSORSTORE
    // The TensorStore store opened by this driver instance represents a
    // single Zarr array; the variable name is not used to look up a
    // sub-array.  Reference it to avoid an unused-parameter warning.
    (void)name;

    // Element type: invert to_ts_dtype.  An element type with no AMIO
    // dtype mapping (bool, complex, string, ...) cannot be described.
    amio_dtype_t dtype{};
    if (!to_amio_dtype(ts_store_.dtype(), dtype)) {
        return info;
    }

    // Shape + rank from the store domain.
    auto domain = ts_store_.domain();
    const tensorstore::DimensionIndex rank = domain.rank();
    if (rank < 1) {
        return info;
    }
    const auto shape_span = domain.shape();  // span<const Index>, Index = int64_t

    // Leading-dimension timestep model (see header comment): treat the
    // leading dimension as the time axis only when there is at least one
    // remaining dimension to report as the per-timestep shape.
    std::int64_t total_timesteps = 1;
    tensorstore::DimensionIndex shape_start = 0;
    if (rank >= 2) {
        total_timesteps = static_cast<std::int64_t>(shape_span[0]);
        shape_start = 1;
    }

    const tensorstore::DimensionIndex reported_rank = rank - shape_start;
    if (reported_rank < 1 || reported_rank > AMIO_MAX_RANK) {
        return info;
    }

    info.shape.rank = static_cast<std::int32_t>(reported_rank);
    for (tensorstore::DimensionIndex d = 0; d < reported_rank; ++d) {
        info.shape.extents[d] = static_cast<std::int64_t>(shape_span[d + shape_start]);
        info.shape.strides[d] = 0;  // contiguous / row-major
    }

    info.dtype = dtype;
    info.total_timesteps = total_timesteps > 0 ? total_timesteps : 1;
    info.found = true;
    return info;

#else
    (void)name;
    return info;
#endif  // AMIO_HAS_TENSORSTORE
}

}  // namespace amio::detail

extern "C" void amio_register_zarr_driver() {}
