// test_p20_invalid_write_input.cpp -- Property test P20: Invalid write
// input rejection.
//
// For any amio_write with null pointer, unsupported dtype, rank zero,
// or zero/negative extent: returns AMIO_ERR_INVALID_INPUT, no
// Staging_Pool buffer acquired, no task enqueued.
//
// Min 100 iterations.
//
// **Validates: Requirements R2.10**

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::pbt;

// ===================================================================
// Helper: create a minimal initialized AMIO context for write testing.
// Returns an AmioGuard with a valid core handle.
// ===================================================================

namespace {

// Create a minimal manifest and initialize AMIO for write testing.
struct WriteTestContext {
    TempDir dir;
    std::string manifest_path;
    amio_core_handle core = nullptr;
    amio_dataset_handle dataset = nullptr;
    bool valid = false;

    WriteTestContext() {
        std::string yaml = make_manifest_yaml("netcdf4", 4, 65536, 1, 5000);
        manifest_path = write_manifest(dir, yaml);

        amio_status_t rc = amio_init(manifest_path.c_str(), &core);
        if (rc != AMIO_OK || core == nullptr) {
            return;
        }

        // Create a dataset config for write mode.
        std::string ds_yaml = make_dataset_config_yaml("netcdf4", dir.file("output.nc"));
        std::string ds_path = dir.file("dataset.yaml");
        std::ofstream ofs(ds_path);
        ofs << ds_yaml;
        ofs.close();

        rc = amio_open_dataset(core, ds_path.c_str(), AMIO_MODE_WRITE, &dataset);
        if (rc != AMIO_OK || dataset == nullptr) {
            amio_finalize(core);
            core = nullptr;
            return;
        }

        valid = true;
    }

    ~WriteTestContext() {
        if (dataset) {
            amio_close_dataset(dataset);
        }
        if (core) {
            amio_finalize(core);
        }
    }

    WriteTestContext(const WriteTestContext &) = delete;
    WriteTestContext &operator=(const WriteTestContext &) = delete;
};

}  // anonymous namespace

// ===================================================================
// Property Test P20a: Null host_data pointer returns AMIO_ERR_INVALID_INPUT.
//
// For any valid (dtype, shape): amio_write with host_data == nullptr
// returns AMIO_ERR_INVALID_INPUT.
// ===================================================================

TEST_CASE("P20: Invalid write input - null host_data pointer", "[pbt][p20][invalid_write][null_pointer]") {
    auto result = rc::check("null host_data returns AMIO_ERR_INVALID_INPUT", []() {
        WriteTestContext ctx;
        RC_PRE(ctx.valid);

        // Generate a valid dtype and shape.
        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();
        auto shape = *rc::gen::arbitrary<amio_shape_t>();

        // Call amio_write with null host_data.
        amio_io_handle io = nullptr;
        amio_status_t rc_val = amio_write(ctx.dataset, "test_var", nullptr, dtype, &shape, &io);

        RC_ASSERT(rc_val == AMIO_ERR_INVALID_INPUT);
        // No io handle should be created.
        RC_ASSERT(!io);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P20b: Unsupported dtype returns AMIO_ERR_INVALID_INPUT.
//
// For any dtype value outside the valid enum range: amio_write returns
// AMIO_ERR_INVALID_INPUT.
// ===================================================================

TEST_CASE("P20: Invalid write input - unsupported dtype", "[pbt][p20][invalid_write][unsupported_dtype]") {
    auto result = rc::check("unsupported dtype returns AMIO_ERR_INVALID_INPUT", []() {
        WriteTestContext ctx;
        RC_PRE(ctx.valid);

        // Generate an invalid dtype value (outside the valid enum range).
        auto bad_dtype = static_cast<amio_dtype_t>(*rc::gen::elementOf(std::vector<int>{-1, -100, 10, 11, 50, 100, 255, 1000}));

        // Generate a valid shape and payload.
        amio_shape_t shape = {};
        shape.rank = 1;
        shape.extents[0] = 10;

        // Provide a valid (non-null) host pointer.
        std::vector<uint8_t> dummy(80, 0x42);

        amio_io_handle io = nullptr;
        amio_status_t rc_val = amio_write(ctx.dataset, "test_var", dummy.data(), bad_dtype, &shape, &io);

        RC_ASSERT(rc_val == AMIO_ERR_INVALID_INPUT);
        RC_ASSERT(!io);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P20c: Rank zero returns AMIO_ERR_INVALID_INPUT.
//
// For any valid dtype: amio_write with shape.rank == 0 returns
// AMIO_ERR_INVALID_INPUT.
// ===================================================================

TEST_CASE("P20: Invalid write input - rank zero", "[pbt][p20][invalid_write][rank_zero]") {
    auto result = rc::check("rank zero returns AMIO_ERR_INVALID_INPUT", []() {
        WriteTestContext ctx;
        RC_PRE(ctx.valid);

        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();

        // Shape with rank = 0.
        amio_shape_t shape = {};
        shape.rank = 0;

        std::vector<uint8_t> dummy(64, 0xAB);

        amio_io_handle io = nullptr;
        amio_status_t rc_val = amio_write(ctx.dataset, "test_var", dummy.data(), dtype, &shape, &io);

        RC_ASSERT(rc_val == AMIO_ERR_INVALID_INPUT);
        RC_ASSERT(!io);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P20d: Zero or negative extents return AMIO_ERR_INVALID_INPUT.
//
// For any shape generated by genInvalidShape() with zero/negative
// extents: amio_write returns AMIO_ERR_INVALID_INPUT.
// ===================================================================

TEST_CASE("P20: Invalid write input - zero or negative extents", "[pbt][p20][invalid_write][bad_extents]") {
    auto result = rc::check("zero or negative extents return AMIO_ERR_INVALID_INPUT", []() {
        WriteTestContext ctx;
        RC_PRE(ctx.valid);

        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();

        // Generate an invalid shape (zero/negative extents or bad rank).
        auto shape = *genInvalidShape();

        std::vector<uint8_t> dummy(1024, 0xCD);

        amio_io_handle io = nullptr;
        amio_status_t rc_val = amio_write(ctx.dataset, "test_var", dummy.data(), dtype, &shape, &io);

        RC_ASSERT(rc_val == AMIO_ERR_INVALID_INPUT);
        RC_ASSERT(!io);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P20e: Null var_name returns AMIO_ERR_INVALID_INPUT.
//
// For any valid (dtype, shape, host_data): amio_write with
// var_name == nullptr returns AMIO_ERR_INVALID_INPUT.
// ===================================================================

TEST_CASE("P20: Invalid write input - null var_name", "[pbt][p20][invalid_write][null_var_name]") {
    auto result = rc::check("null var_name returns AMIO_ERR_INVALID_INPUT", []() {
        WriteTestContext ctx;
        RC_PRE(ctx.valid);

        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();
        auto shape = *rc::gen::arbitrary<amio_shape_t>();

        std::size_t byte_count = payload_byte_count(shape, dtype);
        RC_PRE(byte_count > 0 && byte_count <= 65536);

        std::vector<uint8_t> data(byte_count, 0xEF);

        amio_io_handle io = nullptr;
        amio_status_t rc_val = amio_write(ctx.dataset, nullptr, data.data(), dtype, &shape, &io);

        RC_ASSERT(rc_val == AMIO_ERR_INVALID_INPUT);
        RC_ASSERT(!io);
    });

    REQUIRE(result);
}
