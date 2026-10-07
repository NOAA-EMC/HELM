// zarr_driver.hpp -- AMIO Zarr_Driver for Zarr v3.
//
// This header is PRIVATE to the AMIO_Core build (`src/drivers/zarr/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// Zarr_Driver implements the Backend_Driver interface for Zarr v3
// datasets.  Two compile-time variants exist:
//
//   * TensorStore mode (AMIO_HAS_TENSORSTORE defined):
//     Uses Google TensorStore configured for Zarr v3.  Cloud URIs
//     (s3://, gs://, https://) route through TensorStore KvStore
//     HTTP REST transport.  Each array is configured with the
//     zarr3_sharding_indexed codec; chunk and shard shapes must
//     satisfy: every dim is a positive integer and every chunk dim
//     divides the corresponding shard dim.  Payloads are run through
//     a Byte-Shuffle filter + exactly one of {Blosc, Zstandard}.
//
//   * NCZarr fallback mode (AMIO_NCZARR_FALLBACK defined):
//     Serializes through netCDF-c NCZarr using exactly one of
//     {Blosc, Zstandard}.  No sharding, no cloud KvStore.
//     Emits a one-shot diagnostic on initialization that sharding
//     is unavailable.
//
// Registration
// ------------
// The driver registers with BackendFactory under key "zarr3" via
// BackendRegistrar<Zarr_Driver>("zarr3") at static initialization.
//
// Thread safety
// -------------
// A single Zarr_Driver instance is NOT required to be thread-safe;
// the Worker_Pool's per-(dataset, variable) ordering mutex serializes
// calls to the same driver instance (R6.3).
//
// Validates: R8.1, R8.2, R8.3, R8.4, R8.5, R8.6, R8.7, R8.8, R8.9, R8.10

#ifndef AMIO_SRC_DRIVERS_ZARR_ZARR_DRIVER_HPP
#define AMIO_SRC_DRIVERS_ZARR_ZARR_DRIVER_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "drivers/common/var_attributes.hpp"
#include "factory/backend_driver.hpp"

#ifdef AMIO_HAS_TENSORSTORE
#include <tensorstore/context.h>
#include <tensorstore/tensorstore.h>
#endif

namespace amio::detail {

// ---------------------------------------------------------------
// ZarrConfig -- parsed Zarr-specific dataset configuration.
// ---------------------------------------------------------------

struct ZarrConfig {
    // Target URI or path (e.g., "s3://bucket/path", "/local/path").
    std::string uri;

    // Chunk shape: per-dimension chunk size (positive integers).
    std::vector<std::int64_t> chunk_shape;

    // Shard shape: per-dimension shard size (positive integers).
    // Each chunk dim must evenly divide the corresponding shard dim.
    std::vector<std::int64_t> shard_shape;

    // Array shape: per-dimension total array size.
    std::vector<std::int64_t> array_shape;

    // Compression codec: must be one of {"blosc", "zstandard"}.
    std::string codec;

    // Data type string for TensorStore (e.g., "float32", "int64").
    std::string dtype_str;
};

// ---------------------------------------------------------------
// Zarr_Driver -- Backend_Driver implementation for Zarr v3.
// ---------------------------------------------------------------

class Zarr_Driver : public Backend_Driver {
   public:
    Zarr_Driver() = default;
    ~Zarr_Driver() override = default;

    // Non-copyable, non-movable.
    Zarr_Driver(const Zarr_Driver &) = delete;
    Zarr_Driver &operator=(const Zarr_Driver &) = delete;
    Zarr_Driver(Zarr_Driver &&) = delete;
    Zarr_Driver &operator=(Zarr_Driver &&) = delete;

    // open_write -- validate config and configure TensorStore spec.
    //
    // Validates:
    //   - All required fields present (R8.10)
    //   - Chunk dims divide shard dims (R8.3)
    //   - Codec is one of {blosc, zstandard} (R8.4)
    //   - Cloud URIs route through TensorStore KvStore (R8.2)
    //
    // Throws on validation failure or if TensorStore is not available.
    void open_write(const conf::Config &config) override;

    // open_read -- validate config and open TensorStore for reading.
    void open_read(const conf::Config &config) override;

    // write -- serialize StagingBuffer through TensorStore with
    // sharding + compression.
    void write(const StagingBuffer &src, const VarMeta &meta) override;

    // read -- read from TensorStore into StagingBuffer.
    void read(StagingBuffer &dst, const VarMeta &meta, std::int64_t timestep, const std::optional<BoundingBox> &bbox) override;

    // flush -- ensure all async TensorStore operations complete.
    void flush() override;

    // close -- close TensorStore handles.
    void close() override;

    // describe_variable -- report a variable's element type, shape, and
    // timestep count by introspecting the open TensorStore store
    // (ts_store_.domain() / ts_store_.dtype()).  Returns
    // VariableInfo{found = false} when the driver is not open for
    // reading or the store's element type has no AMIO dtype mapping.
    // In the NCZarr fallback build (no TensorStore) the default
    // found = false is returned.  (Req 4.1, 4.2, 4.5)
    VariableInfo describe_variable(const std::string &name) override;

    // ----- Static utility methods (public for testability) -----

    // Determine if a URI is a cloud URI (s3://, gs://, https://).
    static bool is_cloud_uri(const std::string &uri);

    // Categorize a network/auth error for proper error reporting (R8.9).
    static std::string categorize_error(const std::string &message);

    // Convert amio_dtype_t to TensorStore dtype string.
    static std::string dtype_to_string(amio_dtype_t dtype);

    // Convert amio_dtype_t to element size in bytes.
    static std::size_t dtype_size(amio_dtype_t dtype);

   private:
    // Parse and validate Zarr-specific configuration from conf::Config.
    // Returns the parsed ZarrConfig.
    // Throws on missing required fields (R8.10) or invalid values (R8.3).
    ZarrConfig parse_zarr_config(const conf::Config &config);

    // Validate that chunk dims divide shard dims.
    // Throws on violation (R8.3).
    void validate_sharding(const ZarrConfig &cfg);

    // Validate codec selection (must be blosc or zstandard).
    // Throws on invalid codec (R8.4).
    void validate_codec(const ZarrConfig &cfg);

    // State.
    bool is_open_ = false;
    bool is_write_mode_ = false;
    ZarrConfig config_;

    // CF/UGRID convention metadata + per-variable attributes parsed
    // from the dataset manifest.  Written as Zarr attributes (.zattrs)
    // via netCDF-c in NCZarr mode, or into the TensorStore JSON spec
    // metadata in TensorStore mode.
    DatasetAttributes attributes_;
    bool global_attrs_written_ = false;

    // One-shot diagnostic flag: emitted once per driver instance on
    // first open in NCZarr fallback mode (R8.8).
    bool sharding_diagnostic_emitted_ = false;

#ifdef AMIO_HAS_TENSORSTORE
    // TensorStore context and store handle.
    tensorstore::Context ts_context_;
    tensorstore::TensorStore<> ts_store_;
#endif

#ifdef AMIO_NCZARR_FALLBACK
    // NCZarr fallback state: netCDF-c file handle.
    int ncid_ = -1;
#endif
};

}  // namespace amio::detail

#endif  // AMIO_SRC_DRIVERS_ZARR_ZARR_DRIVER_HPP
