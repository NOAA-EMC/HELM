// config_loader.cpp -- AMIO Config_Loader implementation.
//
// Implements manifest parsing, schema validation, and serialization
// for AMIO runtime configuration.  Delegates to HELM::CONF for
// YAML/JSON parsing via conf::Config::from_file / from_string.
//
// Validates: R1.2, R1.3, R1.5, R11.3, R11.4, R11.5, R11.6, R11.7

#include "config/config_loader.hpp"

#include <algorithm>
#include <cmath>
#include <conf/config.hpp>
#include <conf/error.hpp>
#include <conf/value.hpp>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace amio::detail {
namespace {

bool validate_evolution_keys(const conf::Config &manifest, ValidationError &error) {
    static const std::set<std::string> root_keys{"amio_version", "execution", "layout", "destination", "membership", "budgets", "transfer"};
    static const std::map<std::string, std::set<std::string>> nested_keys{
        {"execution", {"mode", "progress"}},
        {"layout", {"kind"}},
        {"destination", {"backend", "path", "create_policy"}},
        {"membership", {"producers", "writers", "consumers"}},
        {"budgets", {"total_bytes", "staging_bytes", "metadata_bytes", "transport_bytes", "view_bytes", "control_reserve_bytes",
                      "total_descriptors", "control_reserve_descriptors"}},
        {"transfer", {"chunk_bytes", "credits_per_peer"}},
    };

    const conf::Value root = manifest.root();
    if (root.kind() != conf::Node_Kind::Map) {
        error.field_path.clear();
        error.message = "versioned AMIO configuration must be a map";
        return false;
    }
    for (const auto &key : root.keys()) {
        if (root_keys.find(key) == root_keys.end()) {
            error.field_path = key;
            error.message = "unknown key in amio_version: 1 configuration";
            return false;
        }
    }
    for (const auto &[section, allowed] : nested_keys) {
        if (!manifest.has(section)) {
            continue;
        }
        const auto node = manifest.at(section);
        if (node.kind() != conf::Node_Kind::Map) {
            error.field_path = section;
            error.message = "expected a map";
            return false;
        }
        for (const auto &key : node.keys()) {
            if (allowed.find(key) == allowed.end()) {
                error.field_path = section + "." + key;
                error.message = "unknown key in amio_version: 1 configuration";
                return false;
            }
        }
    }
    return true;
}

bool required_positive_size(const conf::Config &manifest, std::string_view key, std::size_t &out, ValidationError &error) {
    if (!manifest.has(key)) {
        error.field_path = std::string(key);
        error.message = "required finite budget value is missing";
        return false;
    }
    double value = 0.0;
    try {
        value = manifest.get_double(key);
    } catch (const conf::Conf_Error &exception) {
        error.field_path = std::string(key);
        error.message = exception.what();
        return false;
    }
    constexpr double kLargestExactInteger = 9007199254740991.0;
    if (!std::isfinite(value) || value <= 0.0 || std::floor(value) != value || value > kLargestExactInteger) {
        error.field_path = std::string(key);
        error.message = "value must be a positive integer exactly representable by the configuration parser";
        return false;
    }
    out = static_cast<std::size_t>(value);
    return true;
}

amio_err_t populate_evolution_config(const conf::Config &manifest, Config &config, ValidationError &error) {
    int version = 0;
    try {
        version = manifest.get_int("amio_version");
    } catch (const conf::Conf_Error &exception) {
        error.field_path = "amio_version";
        error.message = exception.what();
        return AMIO_ERR_MANIFEST_INVALID;
    }
    if (version != 1) {
        error.field_path = "amio_version";
        error.message = "only amio_version: 1 is supported";
        return AMIO_ERR_MANIFEST_INVALID;
    }
    if (!validate_evolution_keys(manifest, error)) {
        return AMIO_ERR_MANIFEST_INVALID;
    }

    auto &evolution = config.evolution;
    evolution.enabled = true;
    evolution.version = version;
    try {
        if (!manifest.has("execution.mode") || !manifest.has("execution.progress")) {
            error.field_path = !manifest.has("execution.mode") ? "execution.mode" : "execution.progress";
            error.message = "versioned configuration requires an explicit execution mode and progress policy";
            return AMIO_ERR_MANIFEST_INVALID;
        }
        evolution.execution_mode = manifest.get_string("execution.mode");
        evolution.progress_mode = manifest.get_string("execution.progress");
        evolution.layout = manifest.get_or<std::string>("layout.kind", "step_collection");
        evolution.destination_backend = manifest.get_or<std::string>("destination.backend", "");
        evolution.destination_path = manifest.get_or<std::string>("destination.path", "");
        evolution.create_policy = manifest.get_or<std::string>("destination.create_policy", "exclusive");
        config.backend = evolution.destination_backend;
        if (manifest.has("membership.producers")) evolution.producers = manifest.get_int_list("membership.producers");
        if (manifest.has("membership.writers")) evolution.writers = manifest.get_int_list("membership.writers");
        if (manifest.has("membership.consumers")) evolution.consumers = manifest.get_int_list("membership.consumers");
    } catch (const conf::Conf_Error &exception) {
        error.message = exception.what();
        return AMIO_ERR_MANIFEST_INVALID;
    }

    auto &budgets = evolution.budgets;
    if (!required_positive_size(manifest, "budgets.total_bytes", budgets.total_bytes, error) ||
        !required_positive_size(manifest, "budgets.staging_bytes", budgets.staging_bytes, error) ||
        !required_positive_size(manifest, "budgets.metadata_bytes", budgets.metadata_bytes, error) ||
        !required_positive_size(manifest, "budgets.transport_bytes", budgets.transport_bytes, error) ||
        !required_positive_size(manifest, "budgets.view_bytes", budgets.view_bytes, error) ||
        !required_positive_size(manifest, "budgets.control_reserve_bytes", budgets.control_reserve_bytes, error) ||
        !required_positive_size(manifest, "budgets.total_descriptors", budgets.total_descriptors, error) ||
        !required_positive_size(manifest, "budgets.control_reserve_descriptors", budgets.control_reserve_descriptors, error) ||
        !required_positive_size(manifest, "transfer.chunk_bytes", budgets.chunk_bytes, error) ||
        !required_positive_size(manifest, "transfer.credits_per_peer", budgets.credits_per_peer, error)) {
        return AMIO_ERR_MANIFEST_INVALID;
    }
    return ConfigLoader::validate(config, error);
}

std::string yaml_quote(std::string_view value) {
    std::string quoted{"\""};
    for (const char character : value) {
        if (character == '\\' || character == '"') quoted.push_back('\\');
        if (character == '\n') {
            quoted += "\\n";
        } else if (character == '\r') {
            quoted += "\\r";
        } else if (character == '\t') {
            quoted += "\\t";
        } else {
            quoted.push_back(character);
        }
    }
    quoted.push_back('"');
    return quoted;
}

void write_rank_list(std::ostringstream &out, std::string_view name, const std::vector<int> &ranks) {
    if (ranks.empty()) return;
    out << "  " << name << ":\n";
    for (const int rank : ranks) out << "    - " << rank << '\n';
}

} // namespace

// ===================================================================
// Static codec allow-list
// ===================================================================

const std::vector<std::string> &ConfigLoader::valid_codecs() {
    static const std::vector<std::string> codecs = {"blosc", "zstandard", "libaec", "lossless_jpeg2000"};
    return codecs;
}

bool ConfigLoader::is_valid_codec(const std::string &name) {
    const auto &vc = valid_codecs();
    return std::find(vc.begin(), vc.end(), name) != vc.end();
}

// ===================================================================
// Validation
// ===================================================================

amio_err_t ConfigLoader::validate(const Config &config, ValidationError &error_out) {
    // Single-pass validation: report first failing rule (R11.4).

    if (config.evolution.enabled) {
        const auto &evolution = config.evolution;
        const auto fail = [&error_out](std::string path, std::string message, amio_err_t code = AMIO_ERR_MANIFEST_INVALID) {
            error_out.field_path = std::move(path);
            error_out.message = std::move(message);
            return code;
        };
        if (evolution.version != 1) return fail("amio_version", "only amio_version: 1 is supported");
        if (evolution.execution_mode != "persistent_direct" && evolution.execution_mode != "persistent_aggregated" &&
            evolution.execution_mode != "mpi_stream") {
            return fail("execution.mode", "unsupported execution mode");
        }
        if (evolution.progress_mode != "background" && evolution.progress_mode != "caller") {
            return fail("execution.progress", "progress must be background or caller");
        }
        if (evolution.layout != "step_collection") return fail("layout.kind", "persistent datasets require layout: step_collection", AMIO_ERR_UNSUPPORTED);
        const bool stream = evolution.execution_mode == "mpi_stream";
        if (stream) {
            if (!evolution.destination_backend.empty() || !evolution.destination_path.empty()) {
                return fail("destination", "stream mode does not open a persistent backend or path");
            }
            if (evolution.producers.empty() || evolution.consumers.empty() || !evolution.writers.empty()) {
                return fail("membership", "stream mode requires producer and consumer ranks and no writer ranks");
            }
        } else {
            if (evolution.destination_backend != "netcdf4" && evolution.destination_backend != "zarr3" &&
                evolution.destination_backend != "grib2") {
                return fail("destination.backend", "persistent mode requires netcdf4, zarr3, or grib2");
            }
            if (evolution.destination_path.empty()) return fail("destination.path", "persistent mode requires an explicit destination path");
            if (evolution.create_policy != "exclusive") {
                return fail("destination.create_policy", "only exclusive creation is currently supported", AMIO_ERR_UNSUPPORTED);
            }
            if (evolution.execution_mode == "persistent_aggregated" && (evolution.producers.empty() || evolution.writers.empty())) {
                return fail("membership", "aggregated output requires nonempty producer and writer rank lists");
            }
            if (evolution.execution_mode == "persistent_direct" && (!evolution.writers.empty() || !evolution.consumers.empty())) {
                return fail("membership", "direct persistent output uses producers as writers and has no consumer role");
            }
        }

        std::set<int> roles;
        const auto add_ranks = [&roles](const std::vector<int> &ranks) {
            for (int rank : ranks) {
                if (rank < 0 || !roles.insert(rank).second) return false;
            }
            return true;
        };
        if (!add_ranks(evolution.producers) || !add_ranks(evolution.writers) || !add_ranks(evolution.consumers)) {
            return fail("membership", "rank lists must be nonnegative, unique, and disjoint");
        }

        const auto &budget = evolution.budgets;
        if (budget.total_bytes == 0 || budget.total_descriptors == 0 || budget.staging_bytes == 0 || budget.metadata_bytes == 0 ||
            budget.transport_bytes == 0 || budget.view_bytes == 0 || budget.control_reserve_bytes == 0 ||
            budget.control_reserve_descriptors == 0 || budget.chunk_bytes == 0 || budget.credits_per_peer == 0) {
            return fail("budgets", "all byte, descriptor, chunk, and credit ceilings must be finite and positive");
        }
        std::size_t remaining = budget.total_bytes;
        for (const auto &[path, bytes] : std::vector<std::pair<const char *, std::size_t>>{
                 {"budgets.staging_bytes", budget.staging_bytes}, {"budgets.metadata_bytes", budget.metadata_bytes},
                 {"budgets.transport_bytes", budget.transport_bytes}, {"budgets.view_bytes", budget.view_bytes},
                 {"budgets.control_reserve_bytes", budget.control_reserve_bytes}}) {
            if (bytes > remaining) return fail("budgets.total_bytes", "sum of reserved byte ceilings exceeds total_bytes");
            remaining -= bytes;
            (void)path;
        }
        if (budget.control_reserve_descriptors > budget.total_descriptors) {
            return fail("budgets.total_descriptors", "control reserve must fit the total descriptor ceiling");
        }
        if (budget.chunk_bytes > budget.transport_bytes) return fail("transfer.chunk_bytes", "chunk_bytes must fit transport_bytes");
        return AMIO_OK;
    }

    // staging_pool.buffer_count [1, 4096]
    if (config.staging_pool.buffer_count < kMinBufferCount || config.staging_pool.buffer_count > kMaxBufferCount) {
        error_out.field_path = "staging_pool.buffer_count";
        error_out.message = "buffer_count must be in [1, 4096], got " + std::to_string(config.staging_pool.buffer_count);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // staging_pool.buffer_capacity_bytes [1, 1 GiB]
    if (config.staging_pool.buffer_capacity_bytes < kMinBufferCapacity || config.staging_pool.buffer_capacity_bytes > kMaxBufferCapacity) {
        error_out.field_path = "staging_pool.buffer_capacity_bytes";
        error_out.message = "buffer_capacity_bytes must be in [1, 1073741824], got " + std::to_string(config.staging_pool.buffer_capacity_bytes);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // staging_pool.max_buffer_count [buffer_count, 4096] -- the auto-grow
    // ceiling must be at least the initial count (else the pool could not
    // hold its own provisioned slots) and within the hard limit.
    if (config.staging_pool.max_buffer_count < config.staging_pool.buffer_count || config.staging_pool.max_buffer_count > kMaxBufferCount) {
        error_out.field_path = "staging_pool.max_buffer_count";
        error_out.message = "max_buffer_count must be in [buffer_count, 4096], got " + std::to_string(config.staging_pool.max_buffer_count) +
                            " with buffer_count " + std::to_string(config.staging_pool.buffer_count);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // worker_pool.threads [0, 256]; zero selects synchronous writes
    if (config.worker_pool.threads > kMaxThreads) {
        error_out.field_path = "worker_pool.threads";
        error_out.message = "threads must be in [0, 256] (0 selects synchronous writes), got " + std::to_string(config.worker_pool.threads);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // prefetch.depth [1, 1024]
    if (config.prefetch.depth < kMinPrefetchDepth || config.prefetch.depth > kMaxPrefetchDepth) {
        error_out.field_path = "prefetch.depth";
        error_out.message = "depth must be in [1, 1024], got " + std::to_string(config.prefetch.depth);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // prefetch.read_timeout_s [1, 3600]
    if (config.prefetch.read_timeout_s < kMinReadTimeoutS || config.prefetch.read_timeout_s > kMaxReadTimeoutS) {
        error_out.field_path = "prefetch.read_timeout_s";
        error_out.message = "read_timeout_s must be in [1, 3600], got " + std::to_string(config.prefetch.read_timeout_s);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // staging_timeout_ms [1, 60000]
    if (config.staging_timeout_ms < kMinStagingTimeoutMs || config.staging_timeout_ms > kMaxStagingTimeoutMs) {
        error_out.field_path = "staging_timeout_ms";
        error_out.message = "staging_timeout_ms must be in [1, 60000], got " + std::to_string(config.staging_timeout_ms);
        return AMIO_ERR_MANIFEST_INVALID;
    }

    // Backpressure invariant: 0 <= low < high <= capacity (when configured)
    if (config.backpressure.high_watermark > 0) {
        if (config.backpressure.low_watermark >= config.backpressure.high_watermark) {
            error_out.field_path = "backpressure.low_watermark";
            error_out.message = "low_watermark must be < high_watermark";
            return AMIO_ERR_MANIFEST_INVALID;
        }
        if (config.backpressure.high_watermark > config.backpressure.queue_capacity) {
            error_out.field_path = "backpressure.high_watermark";
            error_out.message = "high_watermark must be <= queue_capacity";
            return AMIO_ERR_MANIFEST_INVALID;
        }
    }

    // Codec allow-list enforcement (R11.6, R11.7).
    for (const auto &codec : config.codec.lossless_allow_list) {
        if (!is_valid_codec(codec)) {
            error_out.field_path = "codec.lossless_allow_list";
            error_out.message = "codec '" + codec + "' is not a recognized lossless codec";
            return AMIO_ERR_LOSSY_CODEC_FORBIDDEN;
        }
    }

    // Active codec must be on the allow-list (if specified).
    if (!config.codec.active_codec.empty()) {
        if (!is_valid_codec(config.codec.active_codec)) {
            error_out.field_path = "codec.active_codec";
            error_out.message = "active codec '" + config.codec.active_codec + "' is not a recognized lossless codec";
            return AMIO_ERR_LOSSY_CODEC_FORBIDDEN;
        }
        // Also check it's on the manifest's own allow-list.
        const auto &al = config.codec.lossless_allow_list;
        if (!al.empty() && std::find(al.begin(), al.end(), config.codec.active_codec) == al.end()) {
            error_out.field_path = "codec.active_codec";
            error_out.message = "active codec '" + config.codec.active_codec + "' is not on the manifest's lossless_allow_list";
            return AMIO_ERR_LOSSY_CODEC_FORBIDDEN;
        }
    }

    return AMIO_OK;
}

// ===================================================================
// populate_from_conf -- read CONF typed accessors into Config struct.
//
// Reads all Config fields from the parsed CONF document using typed
// accessors with dotted-path keys.  Optional keys are guarded with
// manifest.has() before access.  On Key_Not_Found or Type_Mismatch,
// the error is reported with the dotted path in ValidationError.
// After population, delegates to validate() for schema checks.
//
// Validates: R1.3, R1.4, R1.7, R1.8, R1.9, R1.10
// ===================================================================

amio_err_t ConfigLoader::populate_from_conf(const conf::Config &manifest, Config &config_out, ValidationError &error_out) {
    // Reset config to defaults.
    config_out = Config{};

    if (manifest.has("amio_version")) {
        return populate_evolution_config(manifest, config_out, error_out);
    }

    // Track which key is being read so we can report it on failure.
    std::string_view current_key;

    try {
        // -- Staging pool (integer scalars) --
        current_key = "staging_pool.buffer_count";
        if (manifest.has(current_key)) config_out.staging_pool.buffer_count = static_cast<std::size_t>(manifest.get_int(current_key));

        current_key = "staging_pool.buffer_capacity_bytes";
        if (manifest.has(current_key)) config_out.staging_pool.buffer_capacity_bytes = static_cast<std::size_t>(manifest.get_int(current_key));

        current_key = "staging_pool.max_buffer_count";
        if (manifest.has(current_key)) config_out.staging_pool.max_buffer_count = static_cast<std::size_t>(manifest.get_int(current_key));

        // -- Worker pool --
        current_key = "worker_pool.threads";
        if (manifest.has(current_key)) config_out.worker_pool.threads = static_cast<std::size_t>(manifest.get_int(current_key));

        current_key = "worker_pool.cpu_cores";
        if (manifest.has(current_key)) config_out.worker_pool.cpu_cores = manifest.get_int_list(current_key);

        current_key = "worker_pool.numa_domain";
        if (manifest.has(current_key)) config_out.worker_pool.numa_domain = manifest.get_int(current_key);

        // -- Prefetch --
        current_key = "prefetch.depth";
        if (manifest.has(current_key)) config_out.prefetch.depth = static_cast<std::size_t>(manifest.get_int(current_key));

        current_key = "prefetch.read_timeout_s";
        if (manifest.has(current_key)) config_out.prefetch.read_timeout_s = static_cast<std::size_t>(manifest.get_int(current_key));

        // -- Staging timeout --
        current_key = "staging_timeout_ms";
        if (manifest.has(current_key)) config_out.staging_timeout_ms = static_cast<std::size_t>(manifest.get_int(current_key));

        // -- Backpressure --
        current_key = "backpressure.low_watermark";
        if (manifest.has(current_key)) config_out.backpressure.low_watermark = static_cast<std::size_t>(manifest.get_int(current_key));

        current_key = "backpressure.high_watermark";
        if (manifest.has(current_key)) config_out.backpressure.high_watermark = static_cast<std::size_t>(manifest.get_int(current_key));

        current_key = "backpressure.queue_capacity";
        if (manifest.has(current_key)) config_out.backpressure.queue_capacity = static_cast<std::size_t>(manifest.get_int(current_key));

        // -- Backend (string scalar) --
        current_key = "backend";
        if (manifest.has(current_key)) config_out.backend = manifest.get_string(current_key);

        // -- Codec --
        current_key = "codec.active_codec";
        if (manifest.has(current_key)) config_out.codec.active_codec = manifest.get_string(current_key);

        current_key = "codec.lossless_allow_list";
        if (manifest.has(current_key)) config_out.codec.lossless_allow_list = manifest.get_string_list(current_key);

        // -- I/O ranks (integer list) --
        current_key = "io_ranks";
        if (manifest.has(current_key)) config_out.io_ranks = manifest.get_int_list(current_key);

    } catch (const conf::Conf_Error &e) {
        error_out.field_path = std::string(current_key);
        switch (e.code()) {
            case conf::Error_Code::Key_Not_Found:
                error_out.message = "required key not found: " + std::string(current_key);
                return AMIO_ERR_MANIFEST_INVALID;
            case conf::Error_Code::Type_Mismatch:
                error_out.message = "type mismatch at '" + std::string(current_key) + "': " + e.what();
                return AMIO_ERR_MANIFEST_INVALID;
            default:
                error_out.message = e.what();
                return AMIO_ERR_MANIFEST_INVALID;
        }
    }

    // Validate the populated config against schema rules.
    return validate(config_out, error_out);
}

// ===================================================================
// parse -- load and validate a manifest file.
// ===================================================================

amio_err_t ConfigLoader::parse(const std::string &path, Config &config_out, ValidationError &error_out) {
    try {
        conf::Config manifest = conf::Config::from_file(path);
        return populate_from_conf(manifest, config_out, error_out);
    } catch (const conf::Conf_Error &e) {
        switch (e.code()) {
            case conf::Error_Code::File_Not_Found:
                error_out.field_path = "";
                error_out.message = e.what();
                error_out.line = 0;
                return AMIO_ERR_MANIFEST_NOT_FOUND;
            case conf::Error_Code::Parse_Error:
                error_out.field_path = "";
                error_out.message = e.what();
                error_out.line = 0;
                return AMIO_ERR_MANIFEST_INVALID;
            default:
                error_out.field_path = "";
                error_out.message = e.what();
                error_out.line = 0;
                return AMIO_ERR_MANIFEST_INVALID;
        }
    }
}

// ===================================================================
// parse_string -- parse a manifest from an in-memory string.
//
// A first-class, supported internal API mirroring parse(path, ...):
// it shares the identical validation body by delegating to
// populate_from_conf (which runs the same field population and the
// same validate() schema checks), reports failures through the same
// ValidationError structure, and returns the same AMIO_ERR_* codes.
// The only difference from parse() is the manifest source: this path
// parses via conf::Config::from_string instead of from_file, so the
// no-file case AMIO_ERR_MANIFEST_NOT_FOUND does not occur in practice
// for a non-empty string input.
//
// Used by the string-based AMIO detail entry points
// (init_from_string / open_dataset_from_string).
//
// The `format` parameter ("yaml" or "json") is accepted for API
// completeness; CONF's from_string currently auto-detects format.
// ===================================================================

amio_err_t ConfigLoader::parse_string(const std::string &content, const std::string &format, Config &config_out, ValidationError &error_out) {
    (void)format;  // Reserved for future use; CONF auto-detects.

    try {
        conf::Config manifest = conf::Config::from_string(content);
        return populate_from_conf(manifest, config_out, error_out);
    } catch (const conf::Conf_Error &e) {
        switch (e.code()) {
            case conf::Error_Code::File_Not_Found:
                error_out.field_path = "";
                error_out.message = e.what();
                error_out.line = 0;
                return AMIO_ERR_MANIFEST_NOT_FOUND;
            case conf::Error_Code::Parse_Error:
                error_out.field_path = "";
                error_out.message = e.what();
                error_out.line = 0;
                return AMIO_ERR_MANIFEST_INVALID;
            default:
                error_out.field_path = "";
                error_out.message = e.what();
                error_out.line = 0;
                return AMIO_ERR_MANIFEST_INVALID;
        }
    }
}

// ===================================================================
// serialize -- emit a YAML string from a Config struct.
//
// Round-trip guarantee: parse_string(serialize(config), "yaml") == config
// ===================================================================

std::string ConfigLoader::serialize(const Config &config) {
    std::ostringstream out;

    if (config.evolution.enabled) {
        const auto &evolution = config.evolution;
        const auto &budget = evolution.budgets;
        out << "amio_version: " << evolution.version << '\n';
        out << "execution:\n  mode: " << yaml_quote(evolution.execution_mode) << "\n  progress: " << yaml_quote(evolution.progress_mode) << '\n';
        out << "layout:\n  kind: " << yaml_quote(evolution.layout) << '\n';
        if (!evolution.destination_backend.empty() || !evolution.destination_path.empty()) {
            out << "destination:\n";
            if (!evolution.destination_backend.empty()) out << "  backend: " << yaml_quote(evolution.destination_backend) << '\n';
            if (!evolution.destination_path.empty()) out << "  path: " << yaml_quote(evolution.destination_path) << '\n';
            out << "  create_policy: " << yaml_quote(evolution.create_policy) << '\n';
        }
        if (!evolution.producers.empty() || !evolution.writers.empty() || !evolution.consumers.empty()) {
            out << "membership:\n";
            write_rank_list(out, "producers", evolution.producers);
            write_rank_list(out, "writers", evolution.writers);
            write_rank_list(out, "consumers", evolution.consumers);
        }
        out << "budgets:\n"
            << "  total_bytes: " << budget.total_bytes << '\n'
            << "  staging_bytes: " << budget.staging_bytes << '\n'
            << "  metadata_bytes: " << budget.metadata_bytes << '\n'
            << "  transport_bytes: " << budget.transport_bytes << '\n'
            << "  view_bytes: " << budget.view_bytes << '\n'
            << "  control_reserve_bytes: " << budget.control_reserve_bytes << '\n'
            << "  total_descriptors: " << budget.total_descriptors << '\n'
            << "  control_reserve_descriptors: " << budget.control_reserve_descriptors << '\n';
        out << "transfer:\n  chunk_bytes: " << budget.chunk_bytes << "\n  credits_per_peer: " << budget.credits_per_peer << '\n';
        return out.str();
    }

    out << "staging_pool:\n";
    out << "  buffer_count: " << config.staging_pool.buffer_count << "\n";
    out << "  buffer_capacity_bytes: " << config.staging_pool.buffer_capacity_bytes << "\n";
    out << "  max_buffer_count: " << config.staging_pool.max_buffer_count << "\n";

    out << "worker_pool:\n";
    out << "  threads: " << config.worker_pool.threads << "\n";
    if (!config.worker_pool.cpu_cores.empty()) {
        out << "  cpu_cores:\n";
        for (int core : config.worker_pool.cpu_cores) {
            out << "    - " << core << "\n";
        }
    }
    if (config.worker_pool.numa_domain.has_value()) {
        out << "  numa_domain: " << config.worker_pool.numa_domain.value() << "\n";
    }

    out << "prefetch:\n";
    out << "  depth: " << config.prefetch.depth << "\n";
    out << "  read_timeout_s: " << config.prefetch.read_timeout_s << "\n";

    out << "staging_timeout_ms: " << config.staging_timeout_ms << "\n";

    out << "backpressure:\n";
    out << "  low_watermark: " << config.backpressure.low_watermark << "\n";
    out << "  high_watermark: " << config.backpressure.high_watermark << "\n";
    out << "  queue_capacity: " << config.backpressure.queue_capacity << "\n";

    out << "backend: " << config.backend << "\n";

    out << "codec:\n";
    out << "  active_codec: " << config.codec.active_codec << "\n";
    if (!config.codec.lossless_allow_list.empty()) {
        out << "  lossless_allow_list:\n";
        for (const auto &codec : config.codec.lossless_allow_list) {
            out << "    - " << codec << "\n";
        }
    }

    if (!config.io_ranks.empty()) {
        out << "io_ranks:\n";
        for (int rank : config.io_ranks) {
            out << "  - " << rank << "\n";
        }
    }

    return out.str();
}

}  // namespace amio::detail
