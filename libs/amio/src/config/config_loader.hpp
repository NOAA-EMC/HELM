// config_loader.hpp -- AMIO Config_Loader for YAML/JSON manifests.
//
// This header is PRIVATE to the AMIO_Core build (`src/config/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// The Config_Loader parses runtime YAML or JSON configuration
// manifests and populates a Config struct that drives the
// construction of Staging_Pool, Worker_Pool, Backend_Factory,
// and Prefetch_Queue.  It provides:
//
//   * `parse(path) -> Config`: load and validate a manifest file.
//   * `parse_string(content, format) -> Config`: load and validate a
//     manifest from an in-memory string, applying the identical
//     validation, ValidationError reporting, and return codes as
//     `parse(path)`. A supported internal path (not test-only) used by
//     the string-based AMIO detail entry points.
//   * `serialize(Config) -> string`: emit a YAML string from a
//     Config struct with round-trip guarantee on field set, values,
//     and structural nesting (R11.5).
//   * Single-pass schema validation reporting the first failing
//     rule with file/line/field path (R11.4).
//   * Enforcement of all numeric ranges:
//       buffer_count      [1, 4096]
//       buffer_capacity   [1, 1 GiB]
//       threads           [1, 256]
//       prefetch.depth    [1, 1024]
//       read_timeout_s    [1, 3600]
//       staging_timeout_ms[1, 60000]
//   * Enforcement of lossless-codec allow-list (R11.6, R11.7).
//
// CONF integration
// ----------------
// The loader delegates to HELM::CONF's Config::from_file /
// Config::from_string for YAML/JSON parsing, then reads values
// via CONF's typed accessors (get_int, get_string, etc.) with
// dotted-path keys matching the manifest schema.
//
// Thread safety
// -------------
// parse() and serialize() are stateless free-function-like methods;
// they are safe to call concurrently from any thread.
//
// Validates: R1.2, R1.3, R1.5, R11.3, R11.4, R11.5, R11.6, R11.7

#ifndef AMIO_SRC_CONFIG_CONFIG_LOADER_HPP
#define AMIO_SRC_CONFIG_CONFIG_LOADER_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "amio/amio_errors.h"

// Forward-declare conf::Config to avoid pulling the full CONF header
// into every translation unit that includes this private header.
namespace conf {
class Config;
}

namespace amio::detail {

// ===================================================================
// Config -- runtime configuration struct.
//
// Captures all fields needed by Staging_Pool, Worker_Pool,
// Backend_Factory, Prefetch_Queue, and the backend drivers.
// ===================================================================

// StagingPoolConfig -- buffer pool sizing.
//
// buffer_count is the initial slot count (a provisioning hint).  The pool
// auto-grows on demand up to max_buffer_count (hard ceiling,
// [1, 4096], default kMaxBufferCount) so an under-provisioned count can
// never crash a host model with AMIO_ERR_STAGING_BACKPRESSURE; set
// max_buffer_count == buffer_count to keep the pool strictly bounded.
struct StagingPoolConfig {
    std::size_t buffer_count = 16;                // [1, 4096]
    std::size_t buffer_capacity_bytes = 1048576;  // [1, 1 GiB] (default 1 MiB)
    std::size_t max_buffer_count = 4096;          // [buffer_count, 4096]
};

// WorkerPoolConfig (config-level) -- thread pool sizing and pinning.
struct WorkerPoolCfg {
    std::size_t threads = 1;         // [1, 256]
    std::vector<int> cpu_cores;      // optional CPU core list
    std::optional<int> numa_domain;  // optional NUMA domain
};

// PrefetchConfig -- look-ahead prefetch settings.
struct PrefetchConfig {
    std::size_t depth = 4;            // [1, 1024]
    std::size_t read_timeout_s = 60;  // [1, 3600]
};

// BackpressureConfig (config-level) -- queue admission control.
struct BackpressureCfg {
    std::size_t low_watermark = 0;
    std::size_t high_watermark = 0;
    std::size_t queue_capacity = 1024;
};

// CodecConfig -- lossless codec settings.
struct CodecConfig {
    // The allow-list of lossless codecs permitted for this manifest.
    // Valid entries: "blosc", "zstandard", "libaec", "lossless_jpeg2000"
    std::vector<std::string> lossless_allow_list;

    // The active codec selection (must be on the allow-list).
    std::string active_codec;
};

// Config -- top-level configuration struct.
struct Config {
    // Staging pool configuration.
    StagingPoolConfig staging_pool;

    // Worker pool configuration.
    WorkerPoolCfg worker_pool;

    // Prefetch configuration.
    PrefetchConfig prefetch;

    // Staging timeout in milliseconds [1, 60000], default 5000.
    std::size_t staging_timeout_ms = 5000;

    // Backpressure configuration.
    BackpressureCfg backpressure;

    // Backend driver key (e.g., "netcdf4", "zarr3", "grib2").
    std::string backend;

    // Codec configuration.
    CodecConfig codec;

    // MPI I/O rank set (empty = all ranks participate).
    std::vector<int> io_ranks;
};

// ===================================================================
// ValidationError -- describes a single schema validation failure.
// ===================================================================

struct ValidationError {
    std::string field_path;  // e.g., "staging_pool.buffer_count"
    std::string message;     // human-readable description
    int line = 0;            // line number in source file (0 if unknown)
};

// ===================================================================
// ConfigLoader -- parse and serialize AMIO configuration manifests.
// ===================================================================

class ConfigLoader {
   public:
    // Numeric range constants (from design.md / requirements).
    static constexpr std::size_t kMinBufferCount = 1;
    static constexpr std::size_t kMaxBufferCount = 4096;
    static constexpr std::size_t kMinBufferCapacity = 1;
    static constexpr std::size_t kMaxBufferCapacity = 1'073'741'824;  // 1 GiB
    static constexpr std::size_t kMinThreads = 1;
    static constexpr std::size_t kMaxThreads = 256;
    static constexpr std::size_t kMinPrefetchDepth = 1;
    static constexpr std::size_t kMaxPrefetchDepth = 1024;
    static constexpr std::size_t kMinReadTimeoutS = 1;
    static constexpr std::size_t kMaxReadTimeoutS = 3600;
    static constexpr std::size_t kMinStagingTimeoutMs = 1;
    static constexpr std::size_t kMaxStagingTimeoutMs = 60000;

    // Lossless codec allow-list: the set of codec names that are
    // permitted in any AMIO manifest.  Any codec not in this set
    // is rejected with AMIO_ERR_LOSSY_CODEC_FORBIDDEN.
    static const std::vector<std::string> &valid_codecs();

    // parse -- load and validate a manifest file.
    //
    // On success, returns AMIO_OK and populates `config_out`.
    // On failure, returns the appropriate AMIO_ERR_* code and
    // populates `error_out` with the first failing validation rule.
    //
    // The file format is auto-detected: .json → JSON, otherwise YAML.
    static amio_err_t parse(const std::string &path, Config &config_out, ValidationError &error_out);

    // parse_string -- parse a manifest from an in-memory string.
    //
    // This is a first-class, supported internal API that mirrors
    // parse(path, ...): it applies the identical single-pass schema
    // validation, reports failures through the same ValidationError
    // structure, and returns the same AMIO_ERR_* codes. The only
    // difference from parse() is the manifest source -- an in-memory
    // string here versus a file path in parse(). Because there is no
    // file, AMIO_ERR_MANIFEST_NOT_FOUND cannot be returned on this path.
    //
    // Used by the string-based AMIO detail entry points
    // (init_from_string / open_dataset_from_string) so callers can open
    // datasets from an in-memory manifest without writing a file to disk.
    //
    // `format` should be "yaml" or "json".
    static amio_err_t parse_string(const std::string &content, const std::string &format, Config &config_out, ValidationError &error_out);

    // serialize -- emit a YAML string from a Config struct.
    //
    // Round-trip guarantee: parse(serialize(config)) == config
    // for all valid Config values (R11.5).
    static std::string serialize(const Config &config);

    // validate -- validate a Config struct against all schema rules.
    //
    // Returns AMIO_OK if valid, or the appropriate error code with
    // `error_out` populated on the first failing rule.
    static amio_err_t validate(const Config &config, ValidationError &error_out);

   private:
    // Populate Config from a conf::Config object using CONF typed accessors.
    // Reads scalar and list values via dotted-path keys matching the manifest schema.
    static amio_err_t populate_from_conf(const conf::Config &manifest, Config &config_out, ValidationError &error_out);

    // Check if a codec name is in the valid codecs list.
    static bool is_valid_codec(const std::string &name);
};

// ===================================================================
// Equality operator for Config (supports round-trip testing).
// ===================================================================

inline bool operator==(const StagingPoolConfig &a, const StagingPoolConfig &b) {
    return a.buffer_count == b.buffer_count && a.buffer_capacity_bytes == b.buffer_capacity_bytes && a.max_buffer_count == b.max_buffer_count;
}

inline bool operator==(const WorkerPoolCfg &a, const WorkerPoolCfg &b) {
    return a.threads == b.threads && a.cpu_cores == b.cpu_cores && a.numa_domain == b.numa_domain;
}

inline bool operator==(const PrefetchConfig &a, const PrefetchConfig &b) {
    return a.depth == b.depth && a.read_timeout_s == b.read_timeout_s;
}

inline bool operator==(const BackpressureCfg &a, const BackpressureCfg &b) {
    return a.low_watermark == b.low_watermark && a.high_watermark == b.high_watermark && a.queue_capacity == b.queue_capacity;
}

inline bool operator==(const CodecConfig &a, const CodecConfig &b) {
    return a.lossless_allow_list == b.lossless_allow_list && a.active_codec == b.active_codec;
}

inline bool operator==(const Config &a, const Config &b) {
    return a.staging_pool == b.staging_pool && a.worker_pool == b.worker_pool && a.prefetch == b.prefetch &&
           a.staging_timeout_ms == b.staging_timeout_ms && a.backpressure == b.backpressure && a.backend == b.backend && a.codec == b.codec &&
           a.io_ranks == b.io_ranks;
}

inline bool operator!=(const Config &a, const Config &b) {
    return !(a == b);
}

}  // namespace amio::detail

#endif  // AMIO_SRC_CONFIG_CONFIG_LOADER_HPP
