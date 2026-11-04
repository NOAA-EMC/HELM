// test_config_loader.cpp -- Unit tests for AMIO Config_Loader.
//
// Tests schema validation, numeric range enforcement, lossless-codec
// allow-list enforcement, parse/serialize round-trip, and error
// reporting.
//
// Validates: R1.2, R1.3, R1.5, R11.3, R11.4, R11.5, R11.6, R11.7

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "config/config_loader.hpp"

using namespace amio::detail;

// ===================================================================
// Helper: write a temporary file and return its path.
// ===================================================================

static std::string write_temp_file(const std::string &content, const std::string &suffix = ".yaml") {
    std::string path = "/tmp/amio_test_config_XXXXXX" + suffix;
    // Use a simple unique name based on address.
    char buf[256];
    std::snprintf(buf, sizeof(buf), "/tmp/amio_test_config_%p%s", static_cast<const void *>(content.c_str()), suffix.c_str());
    path = buf;

    std::ofstream f(path);
    f << content;
    f.close();
    return path;
}

// ===================================================================
// Test: valid YAML manifest parses successfully.
// ===================================================================

static void test_valid_yaml_parse() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 32
  buffer_capacity_bytes: 4194304
worker_pool:
  threads: 4
prefetch:
  depth: 8
  read_timeout_s: 120
staging_timeout_ms: 3000
backpressure:
  low_watermark: 10
  high_watermark: 50
  queue_capacity: 100
  prefetch_capacity: 128
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
    - zstandard
io_ranks:
  - 0
  - 1
  - 2
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_OK);
    assert(config.staging_pool.buffer_count == 32);
    assert(config.staging_pool.buffer_capacity_bytes == 4194304);
    assert(config.worker_pool.threads == 4);
    assert(config.prefetch.depth == 8);
    assert(config.prefetch.read_timeout_s == 120);
    assert(config.staging_timeout_ms == 3000);
    assert(config.backpressure.low_watermark == 10);
    assert(config.backpressure.high_watermark == 50);
    assert(config.backpressure.queue_capacity == 100);
    assert(config.backpressure.prefetch_capacity == 128);
    assert(config.backend == "netcdf4");
    assert(config.codec.active_codec == "blosc");
    assert(config.codec.lossless_allow_list.size() == 2);
    assert(config.codec.lossless_allow_list[0] == "blosc");
    assert(config.codec.lossless_allow_list[1] == "zstandard");
    assert(config.io_ranks.size() == 3);
    assert(config.io_ranks[0] == 0);
    assert(config.io_ranks[1] == 1);
    assert(config.io_ranks[2] == 2);

    std::cout << "  PASS: test_valid_yaml_parse\n";
}

// ===================================================================
// Test: buffer_count out of range [1, 4096].
// ===================================================================

static void test_buffer_count_too_low() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 0
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "staging_pool.buffer_count");

    std::cout << "  PASS: test_buffer_count_too_low\n";
}

static void test_buffer_count_too_high() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 5000
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "staging_pool.buffer_count");

    std::cout << "  PASS: test_buffer_count_too_high\n";
}

// ===================================================================
// Test: buffer_capacity_bytes out of range [1, 1 GiB].
// ===================================================================

static void test_buffer_capacity_too_low() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 0
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "staging_pool.buffer_capacity_bytes");

    std::cout << "  PASS: test_buffer_capacity_too_low\n";
}

static void test_buffer_capacity_too_high() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 2000000000
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "staging_pool.buffer_capacity_bytes");

    std::cout << "  PASS: test_buffer_capacity_too_high\n";
}

// ===================================================================
// Test: threads range [0, 256].
// ===================================================================

/** @brief Zero worker threads passes public manifest validation for synchronous writes. */
static void test_threads_zero() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 0
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_OK);
    assert(config.worker_pool.threads == 0);

    std::cout << "  PASS: test_threads_zero\n";
}

static void test_threads_too_high() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 300
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "worker_pool.threads");

    std::cout << "  PASS: test_threads_too_high\n";
}

// ===================================================================
// Test: prefetch.depth out of range [1, 1024].
// ===================================================================

static void test_prefetch_depth_too_low() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 0
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "prefetch.depth");

    std::cout << "  PASS: test_prefetch_depth_too_low\n";
}

static void test_prefetch_depth_too_high() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 2000
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "prefetch.depth");

    std::cout << "  PASS: test_prefetch_depth_too_high\n";
}

// ===================================================================
// Test: read_timeout_s out of range [1, 3600].
// ===================================================================

static void test_read_timeout_too_low() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 0
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "prefetch.read_timeout_s");

    std::cout << "  PASS: test_read_timeout_too_low\n";
}

static void test_read_timeout_too_high() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 5000
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "prefetch.read_timeout_s");

    std::cout << "  PASS: test_read_timeout_too_high\n";
}

// ===================================================================
// Test: staging_timeout_ms out of range [1, 60000].
// ===================================================================

static void test_staging_timeout_too_low() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 0
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "staging_timeout_ms");

    std::cout << "  PASS: test_staging_timeout_too_low\n";
}

static void test_staging_timeout_too_high() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 70000
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "staging_timeout_ms");

    std::cout << "  PASS: test_staging_timeout_too_high\n";
}

// ===================================================================
// Test: lossy codec rejected (R11.6, R11.7).
// ===================================================================

static void test_lossy_codec_rejected() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: lossy_jpeg
  lossless_allow_list:
    - lossy_jpeg
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_LOSSY_CODEC_FORBIDDEN);
    assert(err.field_path.find("codec") != std::string::npos);

    std::cout << "  PASS: test_lossy_codec_rejected\n";
}

static void test_active_codec_not_on_allow_list() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: netcdf4
codec:
  active_codec: zstandard
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_LOSSY_CODEC_FORBIDDEN);
    assert(err.field_path == "codec.active_codec");

    std::cout << "  PASS: test_active_codec_not_on_allow_list\n";
}

// ===================================================================
// Test: round-trip guarantee (R11.5).
// ===================================================================

static void test_round_trip() {
    Config original;
    original.staging_pool.buffer_count = 64;
    original.staging_pool.buffer_capacity_bytes = 8388608;
    original.worker_pool.threads = 8;
    original.worker_pool.cpu_cores = {0, 1, 2, 3};
    original.worker_pool.numa_domain = 1;
    original.prefetch.depth = 16;
    original.prefetch.read_timeout_s = 300;
    original.staging_timeout_ms = 10000;
    original.backpressure.low_watermark = 5;
    original.backpressure.high_watermark = 20;
    original.backpressure.queue_capacity = 50;
    original.backpressure.prefetch_capacity = 77;
    original.backend = "zarr3";
    original.codec.active_codec = "zstandard";
    original.codec.lossless_allow_list = {"blosc", "zstandard"};
    original.io_ranks = {4, 5, 6, 7};

    // Serialize.
    std::string yaml = ConfigLoader::serialize(original);

    // Re-parse.
    Config reparsed;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", reparsed, err);

    assert(rc == AMIO_OK);
    assert(original == reparsed);

    std::cout << "  PASS: test_round_trip\n";
}

// ===================================================================
// Test: file-based parse with missing file.
// ===================================================================

static void test_missing_file() {
    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse("/nonexistent/path/manifest.yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_NOT_FOUND);

    std::cout << "  PASS: test_missing_file\n";
}

// ===================================================================
// Test: file-based parse with valid file.
// ===================================================================

static void test_file_parse() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 10
  buffer_capacity_bytes: 2048
worker_pool:
  threads: 2
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backend: grib2
codec:
  active_codec: libaec
  lossless_allow_list:
    - libaec
    - lossless_jpeg2000
)";

    std::string path = write_temp_file(yaml);
    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse(path, config, err);

    assert(rc == AMIO_OK);
    assert(config.staging_pool.buffer_count == 10);
    assert(config.backend == "grib2");
    assert(config.codec.active_codec == "libaec");

    // Clean up.
    std::remove(path.c_str());

    std::cout << "  PASS: test_file_parse\n";
}

// ===================================================================
// Test: defaults are applied for omitted optional fields.
// ===================================================================

static void test_defaults() {
    // Minimal valid manifest -- only required fields with valid values.
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1
worker_pool:
  threads: 1
prefetch:
  depth: 1
  read_timeout_s: 1
staging_timeout_ms: 1
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_OK);
    // Backpressure defaults.
    assert(config.backpressure.low_watermark == 0);
    assert(config.backpressure.high_watermark == 0);
    assert(config.backpressure.queue_capacity == 1024);
    assert(config.backpressure.prefetch_capacity == 1024);
    // io_ranks defaults to empty.
    assert(config.io_ranks.empty());
    // cpu_cores defaults to empty.
    assert(config.worker_pool.cpu_cores.empty());
    // numa_domain defaults to nullopt.
    assert(!config.worker_pool.numa_domain.has_value());

    std::cout << "  PASS: test_defaults\n";
}

// ===================================================================
// Test: boundary values (min and max of each range).
// ===================================================================

static void test_boundary_values() {
    // All fields at minimum valid values.
    std::string yaml_min = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1
worker_pool:
  threads: 1
prefetch:
  depth: 1
  read_timeout_s: 1
staging_timeout_ms: 1
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml_min, "yaml", config, err);
    assert(rc == AMIO_OK);

    // All fields at maximum valid values.
    std::string yaml_max = R"(
staging_pool:
  buffer_count: 4096
  buffer_capacity_bytes: 1073741824
worker_pool:
  threads: 256
prefetch:
  depth: 1024
  read_timeout_s: 3600
staging_timeout_ms: 60000
backend: zarr3
codec:
  active_codec: zstandard
  lossless_allow_list:
    - zstandard
)";

    rc = ConfigLoader::parse_string(yaml_max, "yaml", config, err);
    assert(rc == AMIO_OK);
    assert(config.staging_pool.buffer_count == 4096);
    assert(config.staging_pool.buffer_capacity_bytes == 1073741824);
    assert(config.worker_pool.threads == 256);
    assert(config.prefetch.depth == 1024);
    assert(config.prefetch.read_timeout_s == 3600);
    assert(config.staging_timeout_ms == 60000);

    std::cout << "  PASS: test_boundary_values\n";
}

// ===================================================================
// Test: backpressure invariant violation.
// ===================================================================

static void test_backpressure_invariant() {
    // low_watermark >= high_watermark is invalid.
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backpressure:
  low_watermark: 50
  high_watermark: 50
  queue_capacity: 100
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "backpressure.low_watermark");

    std::cout << "  PASS: test_backpressure_invariant\n";
}

static void test_backpressure_high_exceeds_capacity() {
    std::string yaml = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 4
  read_timeout_s: 60
staging_timeout_ms: 5000
backpressure:
  low_watermark: 10
  high_watermark: 200
  queue_capacity: 100
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";

    Config config;
    ValidationError err;
    amio_err_t rc = ConfigLoader::parse_string(yaml, "yaml", config, err);

    assert(rc == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "backpressure.high_watermark");

    std::cout << "  PASS: test_backpressure_high_exceeds_capacity\n";
}

static void test_backpressure_timeout_config() {
    const std::string valid = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 1
prefetch:
  depth: 1
  read_timeout_s: 1
staging_timeout_ms: 5000
backpressure:
  low_watermark: 0
  high_watermark: 2
  queue_capacity: 4
  timeout_ms: 250
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";
    Config config;
    ValidationError err;
    assert(ConfigLoader::parse_string(valid, "yaml", config, err) == AMIO_OK);
    assert(config.backpressure.timeout_ms == 250);
    const auto serialized = ConfigLoader::serialize(config);
    assert(serialized.find("timeout_ms: 250") != std::string::npos);

    const auto invalid = valid.substr(0, valid.find("timeout_ms: 250")) + "timeout_ms: 0" +
                         valid.substr(valid.find("timeout_ms: 250") + std::string("timeout_ms: 250").size());
    assert(ConfigLoader::parse_string(invalid, "yaml", config, err) == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "backpressure.timeout_ms");
    std::cout << "  PASS: test_backpressure_timeout_config\n";
}

static void test_prefetch_capacity_config_bounds() {
    const std::string valid = R"(
staging_pool:
  buffer_count: 1
  buffer_capacity_bytes: 1024
worker_pool:
  threads: 0
prefetch:
  depth: 1
  read_timeout_s: 1
staging_timeout_ms: 1000
backpressure:
  prefetch_capacity: 9
backend: netcdf4
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
)";
    Config config;
    ValidationError error;
    assert(ConfigLoader::parse_string(valid, "yaml", config, error) == AMIO_OK);
    assert(config.backpressure.prefetch_capacity == 9);
    const auto serialized = ConfigLoader::serialize(config);
    assert(serialized.find("prefetch_capacity: 9") != std::string::npos);
    const auto position = valid.find("prefetch_capacity: 9");
    const auto invalid = valid.substr(0, position) + "prefetch_capacity: 0" + valid.substr(position + std::string("prefetch_capacity: 9").size());
    assert(ConfigLoader::parse_string(invalid, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "backpressure.prefetch_capacity");
}

// ===================================================================
// Test: validate() standalone.
// ===================================================================

static void test_validate_standalone() {
    Config config;
    config.staging_pool.buffer_count = 10;
    config.staging_pool.buffer_capacity_bytes = 4096;
    config.worker_pool.threads = 2;
    config.prefetch.depth = 4;
    config.prefetch.read_timeout_s = 60;
    config.staging_timeout_ms = 5000;
    config.backend = "netcdf4";
    config.codec.active_codec = "blosc";
    config.codec.lossless_allow_list = {"blosc"};

    ValidationError err;
    assert(ConfigLoader::validate(config, err) == AMIO_OK);

    // Zero workers is valid synchronous mode. Values above the maximum are not.
    config.worker_pool.threads = 0;
    assert(ConfigLoader::validate(config, err) == AMIO_OK);
    config.worker_pool.threads = 257;
    assert(ConfigLoader::validate(config, err) == AMIO_ERR_MANIFEST_INVALID);
    assert(err.field_path == "worker_pool.threads");

    std::cout << "  PASS: test_validate_standalone\n";
}

// ===================================================================
// Test: valid_codecs() returns expected set.
// ===================================================================

static void test_valid_codecs() {
    const auto &codecs = ConfigLoader::valid_codecs();
    assert(codecs.size() == 4);
    assert(codecs[0] == "blosc");
    assert(codecs[1] == "zstandard");
    assert(codecs[2] == "libaec");
    assert(codecs[3] == "lossless_jpeg2000");

    std::cout << "  PASS: test_valid_codecs\n";
}

static std::string valid_evolution_yaml();

static std::string valid_evolution_yaml() {
    return R"(
amio_version: 1
execution:
  mode: persistent_direct
  progress: caller
destination:
  backend: netcdf4
  path: /tmp/amio-evolution
  create_policy: exclusive
membership:
  producers: [0]
budgets:
  total_bytes: 1000000
  staging_bytes: 200000
  metadata_bytes: 100000
  transport_bytes: 200000
  view_bytes: 100000
  control_reserve_bytes: 10000
  total_descriptors: 256
  control_reserve_descriptors: 8
transfer:
  chunk_bytes: 4096
  credits_per_peer: 4
)";
}

static void test_evolution_mode_is_opt_in_and_bounded() {
    Config config;
    ValidationError error;
    assert(ConfigLoader::parse_string(valid_evolution_yaml(), "yaml", config, error) == AMIO_OK);
    assert(config.evolution.enabled);
    assert(config.evolution.version == 1);
    assert(config.evolution.execution_mode == "persistent_direct");
    assert(config.evolution.progress_mode == "caller");
    assert(config.backend == "netcdf4");
    assert(config.evolution.budgets.total_bytes == 1000000);
    assert(config.evolution.budgets.control_reserve_bytes == 10000);
    Config reparsed;
    assert(ConfigLoader::parse_string(ConfigLoader::serialize(config), "yaml", reparsed, error) == AMIO_OK);
    assert(reparsed == config);

    auto configured = valid_evolution_yaml();
    configured += "worker_pool:\n  threads: 2\n  cpu_cores: [1, 2]\n  numa_domain: 0\n";
    assert(ConfigLoader::parse_string(configured, "yaml", config, error) == AMIO_OK);
    assert(config.worker_pool.threads == 2 && config.worker_pool.cpu_cores == std::vector<int>({1, 2}));
    assert(config.worker_pool.numa_domain == 0);
    assert(ConfigLoader::parse_string(ConfigLoader::serialize(config), "yaml", reparsed, error) == AMIO_OK);
    assert(reparsed == config);
}

static void test_evolution_rejects_unknown_version_and_keys() {
    Config config;
    ValidationError error;
    auto yaml = valid_evolution_yaml();
    const auto version = yaml.find("amio_version: 1");
    yaml.replace(version, std::string("amio_version: 1").size(), "amio_version: 2");
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "amio_version");

    yaml = valid_evolution_yaml();
    yaml += "unexpected_policy: true\n";
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "unexpected_policy");

    yaml = valid_evolution_yaml();
    yaml.insert(yaml.find("  progress: caller"), "  undocumented: true\n");
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "execution.undocumented");
}

static void test_evolution_requires_finite_budget_and_valid_roles() {
    Config config;
    ValidationError error;
    auto yaml = valid_evolution_yaml();
    const auto budget = yaml.find("  total_bytes: 1000000\n");
    yaml.erase(budget, std::string("  total_bytes: 1000000\n").size());
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "budgets.total_bytes");

    yaml = valid_evolution_yaml();
    const auto old_value = yaml.find("  transport_bytes: 200000");
    yaml.replace(old_value, std::string("  transport_bytes: 200000").size(), "  transport_bytes: 800000");
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "budgets.total_bytes");

    yaml = valid_evolution_yaml();
    const auto producers = yaml.find("producers: [0]");
    yaml.replace(producers, std::string("producers: [0]").size(), "producers: [0]\n  writers: [0]");
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "membership");

    yaml = valid_evolution_yaml();
    yaml.replace(yaml.find("persistent_direct"), std::string("persistent_direct").size(), "persistent_aggregated");
    yaml.replace(yaml.find("producers: [0]"), std::string("producers: [0]").size(), "producers: [0]\n  writers: [1]\n  consumers: [2]");
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "membership");

    yaml = valid_evolution_yaml();
    const auto chunk = yaml.find("chunk_bytes: 4096");
    yaml.replace(chunk, std::string("chunk_bytes: 4096").size(), "chunk_bytes: 300000");
    assert(ConfigLoader::parse_string(yaml, "yaml", config, error) == AMIO_ERR_MANIFEST_INVALID);
    assert(error.field_path == "transfer.chunk_bytes");
}

// ===================================================================
// Main
// ===================================================================

int main() {
    std::cout << "Config_Loader unit tests:\n";

    test_valid_yaml_parse();
    test_buffer_count_too_low();
    test_buffer_count_too_high();
    test_buffer_capacity_too_low();
    test_buffer_capacity_too_high();
    test_threads_zero();
    test_threads_too_high();
    test_prefetch_depth_too_low();
    test_prefetch_depth_too_high();
    test_read_timeout_too_low();
    test_read_timeout_too_high();
    test_staging_timeout_too_low();
    test_staging_timeout_too_high();
    test_lossy_codec_rejected();
    test_active_codec_not_on_allow_list();
    test_round_trip();
    test_missing_file();
    test_file_parse();
    test_defaults();
    test_boundary_values();
    test_backpressure_invariant();
    test_backpressure_high_exceeds_capacity();
    test_backpressure_timeout_config();
    test_prefetch_capacity_config_bounds();
    test_validate_standalone();
    test_valid_codecs();
    test_evolution_mode_is_opt_in_and_bounded();
    test_evolution_rejects_unknown_version_and_keys();
    test_evolution_requires_finite_budget_and_valid_roles();

    std::cout << "\nAll Config_Loader tests PASSED.\n";
    return 0;
}
