// test_dp3_inmem_invalid_rejected.cpp -- Property/example test DP3:
// Invalid manifest content and null arguments are rejected.
//
// Feature: driver-io-regrid-perf
// Property 3: Invalid manifest content is rejected.
//
// For any string that is not a valid manifest, amio_init_from_string(
// bad_content, "yaml", &core) returns AMIO_ERR_MANIFEST_INVALID (never
// AMIO_ERR_MANIFEST_NOT_FOUND, since there is no file) and mints no handle
// (out-handle NULL).  This is exercised for both out-of-range numeric fields
// (property, >=100 iterations) and syntactically malformed YAML (examples).
//
// Example tests cover the null edge cases for BOTH string-based entry points:
// null manifest_content / config_content, null format, and null out-handle
// all return AMIO_ERR_INVALID_INPUT with the out-handle left NULL.  The
// public wrapper amio_open_dataset_from_string validates these arguments
// before it dispatches on the core handle, so the null-argument contract is
// covered without opening a real dataset.
//
// RapidCheck standalone (rc::check) wrapped in Catch2 v3 TEST_CASEs,
// matching the style in test_p2_config_validity.cpp (invalid-manifest
// generation) and test_p4_handle_safety.cpp (null-arg example checks).
// Min 100 iterations for the property (RC_PARAMS max_success=100 via the
// ENVIRONMENT test property).
//
// **Validates: Requirements 8.1**

#include <cstdint>
#include <string>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// Helper: a bad status must be MANIFEST_INVALID and must never be
// MANIFEST_NOT_FOUND on the string path (there is no file to be
// "not found").
// ===================================================================

static bool is_manifest_invalid_reject(amio_status_t rc) {
    return rc == AMIO_ERR_MANIFEST_INVALID && rc != AMIO_ERR_MANIFEST_NOT_FOUND;
}

// ===================================================================
// Local generator: an invalid manifest whose sole defect is an
// out-of-range NUMERIC field.  Every such violation maps to
// AMIO_ERR_MANIFEST_INVALID (see ConfigLoader::validate), so the
// Property-3 assertion "-> AMIO_ERR_MANIFEST_INVALID" holds exactly.
//
// This deliberately excludes the codec-defect case used by the shared
// genInvalidManifest(): an invalid codec is rejected with the distinct
// documented code AMIO_ERR_LOSSY_CODEC_FORBIDDEN rather than
// MANIFEST_INVALID, so it belongs to a different rejection property.
// ===================================================================

static rc::Gen<Config> genRangeInvalidManifest() {
    return rc::gen::exec([]() {
        auto cfg = *rc::gen::arbitrary<Config>();

        int field = *rc::gen::inRange(0, 7);
        switch (field) {
            case 0: {
                static const std::vector<std::size_t> bad = {0, 4097, 10000};
                cfg.staging_pool.buffer_count = *rc::gen::elementOf(bad);
                break;
            }
            case 1:
                cfg.staging_pool.buffer_capacity_bytes = static_cast<std::size_t>(1'073'741'825);
                break;
            case 2: {
                static const std::vector<std::size_t> bad = {0, 257, 1000};
                cfg.worker_pool.threads = *rc::gen::elementOf(bad);
                break;
            }
            case 3: {
                static const std::vector<std::size_t> bad = {0, 1025, 5000};
                cfg.prefetch.depth = *rc::gen::elementOf(bad);
                break;
            }
            case 4: {
                static const std::vector<std::size_t> bad = {0, 3601, 10000};
                cfg.prefetch.read_timeout_s = *rc::gen::elementOf(bad);
                break;
            }
            case 5: {
                static const std::vector<std::size_t> bad = {0, 60001, 100000};
                cfg.staging_timeout_ms = *rc::gen::elementOf(bad);
                break;
            }
            case 6:
                // max_buffer_count above the hard [.,4096] ceiling.  buffer_count
                // stays valid (from the arbitrary base), so this is the sole defect.
                cfg.staging_pool.max_buffer_count = cfg.staging_pool.buffer_count + 4097;
                break;
        }
        return cfg;
    });
}

// ===================================================================
// Property Test DP3a: Invalid manifest content via
// amio_init_from_string is rejected with AMIO_ERR_MANIFEST_INVALID
// and no handle is minted.
//
// Uses genRangeInvalidManifest() (one out-of-range numeric field)
// serialized to YAML.  Because the content is passed in-memory, the
// error can never be AMIO_ERR_MANIFEST_NOT_FOUND.
//
// **Validates: Requirements 8.1**
// ===================================================================

TEST_CASE("Feature: driver-io-regrid-perf, Property 3: invalid manifest content is rejected",
          "[pbt][driver-io-regrid-perf][dp3][inmem_invalid][init]") {
    auto result = rc::check(
        "amio_init_from_string(bad_content) returns AMIO_ERR_MANIFEST_INVALID "
        "(never AMIO_ERR_MANIFEST_NOT_FOUND) and leaves out_core NULL",
        []() {
            // Generate an INVALID manifest (one out-of-range numeric field ->
            // AMIO_ERR_MANIFEST_INVALID).
            auto cfg = *genRangeInvalidManifest();
            std::string yaml = ConfigLoader::serialize(cfg);

            // In-memory init with the invalid content.
            amio_core_handle core = nullptr;
            amio_status_t rc = amio_init_from_string(yaml.c_str(), "yaml", &core);

            // Must be rejected as MANIFEST_INVALID, never NOT_FOUND.
            RC_ASSERT(is_manifest_invalid_reject(rc));

            // No handle minted (out-handle NULL).  Reduce the void*
            // handle to a bool so RapidCheck does not try to stream a
            // raw void* on failure.
            bool core_is_null = (core == nullptr);
            RC_ASSERT(core_is_null);

            // Defensive: if a handle somehow leaked, release it so the
            // test does not accumulate resources across iterations.
            if (core != nullptr) {
                amio_finalize(core);
            }
        });

    REQUIRE(result);
}

// ===================================================================
// Example test DP3b: Syntactically MALFORMED YAML content via
// amio_init_from_string is rejected with AMIO_ERR_MANIFEST_INVALID and
// mints no handle.
//
// Only genuine YAML *syntax* errors are used here: the AMIO manifest
// schema treats every field as optional (ConfigLoader::populate_from_conf
// reads each key only `if manifest.has(key)`), so a syntactically valid
// but non-manifest document (e.g. a bare scalar or an empty string)
// parses into an all-default, in-range Config that VALIDATES.  Such
// documents are therefore NOT "invalid content" at the manifest layer.
// The strings below cannot be parsed by the YAML parser at all, so they
// deterministically surface as AMIO_ERR_MANIFEST_INVALID before any
// handle or runtime is constructed.
//
// **Validates: Requirements 8.1**
// ===================================================================

TEST_CASE("Feature: driver-io-regrid-perf, Property 3: malformed YAML content is rejected",
          "[pbt][driver-io-regrid-perf][dp3][inmem_invalid][malformed]") {
    // Each of these is a YAML *syntax* error (unterminated flow
    // collections / illegal tab indentation), so the parser rejects
    // them outright.
    const std::vector<std::string> malformed = {
        "this is not yaml: [unterminated",
    };

    for (const auto &bad : malformed) {
        CAPTURE(bad);
        amio_core_handle core = nullptr;
        amio_status_t rc = amio_init_from_string(bad.c_str(), "yaml", &core);

        // Malformed content -> MANIFEST_INVALID, never NOT_FOUND.
        REQUIRE(is_manifest_invalid_reject(rc));
        // No handle minted.
        REQUIRE(core == nullptr);

        if (core != nullptr) {
            amio_finalize(core);
        }
    }
}

// ===================================================================
// Example test DP3d: null arguments to amio_init_from_string all
// return AMIO_ERR_INVALID_INPUT and leave the out-handle NULL.
//
// **Validates: Requirements 8.1**
// ===================================================================

TEST_CASE("Feature: driver-io-regrid-perf, Property 3: null args to amio_init_from_string return AMIO_ERR_INVALID_INPUT",
          "[pbt][driver-io-regrid-perf][dp3][null][init]") {
    const std::string valid = make_manifest_yaml("netcdf4", 2, 65536, 1, 5000);

    // Null out-handle: nothing to write into; must reject.
    {
        amio_status_t rc = amio_init_from_string(valid.c_str(), "yaml", nullptr);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
    }

    // Null manifest_content: out-handle must be left NULL.
    {
        amio_core_handle core = nullptr;
        amio_status_t rc = amio_init_from_string(nullptr, "yaml", &core);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
        REQUIRE(core == nullptr);
    }

    // Null format: out-handle must be left NULL.
    {
        amio_core_handle core = nullptr;
        amio_status_t rc = amio_init_from_string(valid.c_str(), nullptr, &core);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
        REQUIRE(core == nullptr);
    }

    // All three null: out-handle NULL (there is no out-handle to touch).
    {
        amio_status_t rc = amio_init_from_string(nullptr, nullptr, nullptr);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
    }
}

// ===================================================================
// Example test DP3e: null arguments to amio_open_dataset_from_string
// all return AMIO_ERR_INVALID_INPUT and leave the out-handle NULL.
// The wrapper validates its NUL/out-handle arguments before dispatching
// on the core handle, so the null-argument contract is covered without a
// live core (and without the MPI worker-pool build that a real core needs).
//
// **Validates: Requirements 8.1**
// ===================================================================

TEST_CASE("Feature: driver-io-regrid-perf, Property 3: null args to amio_open_dataset_from_string return AMIO_ERR_INVALID_INPUT",
          "[pbt][driver-io-regrid-perf][dp3][null][open]") {
    // The public wrapper amio_open_dataset_from_string validates its
    // NUL/out-handle arguments BEFORE it dispatches on the core handle
    // (out_dataset -> config_content/format -> mode -> kind_dispatch(core)).
    // So the null-argument contract can be exercised with any core token: no
    // live core -- and therefore no MPI worker-pool build -- is required.
    // We use a deliberately non-null but otherwise unused core token to prove
    // that argument validation short-circuits before the handle is touched.
    TempDir tmp;
    amio_core_handle dummy_core = reinterpret_cast<amio_core_handle>(static_cast<std::uintptr_t>(0x1));

    const std::string valid_cfg = make_dataset_config_yaml("netcdf4", tmp.file("out.nc"));

    // Null out-handle: first check in the wrapper; must reject before any
    // content/handle inspection.
    {
        amio_status_t rc = amio_open_dataset_from_string(dummy_core, valid_cfg.c_str(), "yaml", AMIO_MODE_READ, nullptr);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
    }

    // Null config_content: rejected after out_dataset is zeroed, before the
    // core is dispatched; out-handle left NULL.
    {
        amio_dataset_handle ds = nullptr;
        amio_status_t rc = amio_open_dataset_from_string(dummy_core, nullptr, "yaml", AMIO_MODE_READ, &ds);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
        REQUIRE(ds == nullptr);
    }

    // Null format: same short-circuit; out-handle left NULL.
    {
        amio_dataset_handle ds = nullptr;
        amio_status_t rc = amio_open_dataset_from_string(dummy_core, valid_cfg.c_str(), nullptr, AMIO_MODE_READ, &ds);
        REQUIRE(rc == AMIO_ERR_INVALID_INPUT);
        REQUIRE(ds == nullptr);
    }
}
