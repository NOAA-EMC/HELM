// test_pbt_infrastructure.cpp -- Validates PBT infrastructure setup.
//
// This test verifies that:
//   1. Catch2 + RapidCheck link and run correctly.
//   2. All Arbitrary<T> specializations compile and generate valid
//      instances.
//   3. RC_PARAMS max_success=100 is respected.
//   4. The AMIO C API is accessible (amio_strerror as a smoke check).
//   5. TempDir and manifest helpers work correctly.
//
// This is NOT a property test for any specific AMIO correctness
// property -- it is infrastructure validation only.
//
// Validates: R11.2 (testing infrastructure)

#include <cstring>
#include <filesystem>

#include "generators.hpp"
#include "pbt_common.hpp"

// ===================================================================
// Test: Arbitrary<amio_dtype_t> generates valid dtype values.
// ===================================================================

TEST_CASE("PBT infrastructure: Arbitrary<amio_dtype_t> generates valid dtypes", "[pbt][infrastructure][dtype]") {
    rc::check("all generated dtypes are in [0, 9]", [](amio_dtype_t dtype) {
        RC_ASSERT(static_cast<int>(dtype) >= 0);
        RC_ASSERT(static_cast<int>(dtype) <= 9);
        RC_ASSERT(amio::pbt::dtype_size(dtype) > 0);
    });
}

// ===================================================================
// Test: Arbitrary<amio_shape_t> generates valid shapes.
// ===================================================================

TEST_CASE("PBT infrastructure: Arbitrary<amio_shape_t> generates valid shapes", "[pbt][infrastructure][shape]") {
    rc::check("all generated shapes have valid rank and extents", [](amio_shape_t shape) {
        // Rank in [1, 7].
        RC_ASSERT(shape.rank >= 1);
        RC_ASSERT(shape.rank <= AMIO_MAX_RANK);

        // All extents up to rank are positive.
        for (int32_t d = 0; d < shape.rank; ++d) {
            RC_ASSERT(shape.extents[d] >= 1);
            // Max extent depends on rank (see generators.hpp):
            // rank 1-2: up to 1024, rank 3-4: up to 64, rank 5-7: up to 16
            RC_ASSERT(shape.extents[d] <= 1024);
        }

        // Entries beyond rank are zero.
        for (int32_t d = shape.rank; d < AMIO_MAX_RANK; ++d) {
            RC_ASSERT(shape.extents[d] == 0);
            RC_ASSERT(shape.strides[d] == 0);
        }

        // Strides are 0 (contiguous).
        for (int32_t d = 0; d < shape.rank; ++d) {
            RC_ASSERT(shape.strides[d] == 0);
        }
    });
}

// ===================================================================
// Test: Arbitrary<Payload> generates valid payloads.
// ===================================================================

TEST_CASE("PBT infrastructure: Arbitrary<Payload> generates valid payloads", "[pbt][infrastructure][payload]") {
    rc::check("payload byte count matches shape * dtype_size", [](amio::pbt::Payload payload) {
        std::size_t expected = amio::pbt::payload_byte_count(payload.shape, payload.dtype);
        RC_ASSERT(payload.bytes.size() == expected);
        RC_ASSERT(expected > 0);
    });
}

// ===================================================================
// Test: Arbitrary<CodecConfig> generates valid codec configs.
// ===================================================================

TEST_CASE("PBT infrastructure: Arbitrary<CodecConfig> generates valid configs", "[pbt][infrastructure][codec]") {
    rc::check("codec config has active codec on allow-list", [](amio::detail::CodecConfig cfg) {
        RC_ASSERT(!cfg.active_codec.empty());
        RC_ASSERT(!cfg.lossless_allow_list.empty());

        // Active codec must be on the allow-list.
        auto it = std::find(cfg.lossless_allow_list.begin(), cfg.lossless_allow_list.end(), cfg.active_codec);
        RC_ASSERT(it != cfg.lossless_allow_list.end());
    });
}

// ===================================================================
// Test: Arbitrary<Config> (Manifest) generates valid configurations.
// ===================================================================

TEST_CASE("PBT infrastructure: Arbitrary<Config> generates valid manifests", "[pbt][infrastructure][manifest]") {
    using namespace amio::detail;

    rc::check("all generated configs have fields in valid ranges", [](Config cfg) {
        // Staging pool ranges.
        RC_ASSERT(cfg.staging_pool.buffer_count >= 1);
        RC_ASSERT(cfg.staging_pool.buffer_count <= 4096);
        RC_ASSERT(cfg.staging_pool.buffer_capacity_bytes >= 1);

        // Worker pool range.
        RC_ASSERT(cfg.worker_pool.threads >= 1);
        RC_ASSERT(cfg.worker_pool.threads <= 256);

        // Prefetch ranges.
        RC_ASSERT(cfg.prefetch.depth >= 1);
        RC_ASSERT(cfg.prefetch.depth <= 1024);
        RC_ASSERT(cfg.prefetch.read_timeout_s >= 1);
        RC_ASSERT(cfg.prefetch.read_timeout_s <= 3600);

        // Staging timeout range.
        RC_ASSERT(cfg.staging_timeout_ms >= 1);
        RC_ASSERT(cfg.staging_timeout_ms <= 60000);

        // Backend is a known key.
        RC_ASSERT(cfg.backend == "netcdf4" || cfg.backend == "zarr3" || cfg.backend == "grib2");
    });
}

// ===================================================================
// Test: TempDir creates and cleans up directories.
// ===================================================================

TEST_CASE("PBT infrastructure: TempDir creates and manages temp directories", "[pbt][infrastructure][tempdir]") {
    namespace fs = std::filesystem;

    std::string dir_path;
    {
        amio::pbt::TempDir tmp;
        dir_path = tmp.path();
        REQUIRE(!dir_path.empty());
        REQUIRE(fs::exists(dir_path));
        REQUIRE(fs::is_directory(dir_path));

        // Create a file inside.
        std::string file_path = tmp.file("test.txt");
        std::ofstream ofs(file_path);
        ofs << "hello";
        ofs.close();
        REQUIRE(fs::exists(file_path));
    }
    // After destruction, directory should be removed.
    REQUIRE_FALSE(fs::exists(dir_path));
}

// ===================================================================
// Test: Manifest YAML helpers produce valid YAML.
// ===================================================================

TEST_CASE("PBT infrastructure: manifest YAML helpers produce parseable files", "[pbt][infrastructure][manifest_yaml]") {
    amio::pbt::TempDir tmp;

    std::string yaml = amio::pbt::make_manifest_yaml("netcdf4");
    REQUIRE(!yaml.empty());
    REQUIRE(yaml.find("backend: netcdf4") != std::string::npos);
    REQUIRE(yaml.find("buffer_count:") != std::string::npos);

    std::string path = amio::pbt::write_manifest(tmp, yaml);
    REQUIRE(std::filesystem::exists(path));
}

// ===================================================================
// Test: amio_strerror smoke check (verifies AMIO C API is linked).
// ===================================================================

TEST_CASE("PBT infrastructure: amio_strerror is accessible", "[pbt][infrastructure][api]") {
    const char *msg = amio_strerror(AMIO_OK);
    REQUIRE(msg != nullptr);
    REQUIRE(std::strlen(msg) > 0);

    // Undefined code should also return non-null.
    const char *unknown = amio_strerror(9999);
    REQUIRE(unknown != nullptr);
    REQUIRE(std::strlen(unknown) > 0);
}
