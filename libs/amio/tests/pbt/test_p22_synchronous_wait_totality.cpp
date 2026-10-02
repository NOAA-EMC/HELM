// test_p22_synchronous_wait_totality.cpp -- Property test P22:
// Synchronous wait status totality.
//
// For any (pending I/O handle, timeout in [0, 86_400_000]ms):
// amio_wait returns exactly one of {AMIO_OK, AMIO_ERR_BACKEND_FAILURE,
// AMIO_ERR_TIMEOUT}; same trichotomy for amio_flush aggregated over
// dataset.
//
// Min 100 iterations.
//
// **Validates: Requirements R3.9, R6.4**

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "factory/backend_factory.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// The allowed return codes for amio_wait and amio_flush (the
// "trichotomy" set).  Any return value MUST be one of these three.
// ===================================================================

namespace {

const std::set<amio_status_t> kWaitTrichotomy = {AMIO_OK, AMIO_ERR_BACKEND_FAILURE, AMIO_ERR_TIMEOUT};

// Extended set for amio_flush which may also return handle errors
// if the dataset handle is invalid.  But for valid handles, the
// trichotomy applies.
const std::set<amio_status_t> kFlushTrichotomy = {AMIO_OK, AMIO_ERR_BACKEND_FAILURE, AMIO_ERR_TIMEOUT};

// NoOpDriver -- a Backend_Driver that accepts writes without performing
// real I/O.  Used by infrastructure tests (P22) that verify wait/flush
// trichotomy, NOT driver serialization behavior.
class NoOpDriver : public Backend_Driver {
   public:
    void open_write(const conf::Config & /*config*/) override {}
    void open_read(const conf::Config & /*config*/) override {}
    void write(const StagingBuffer & /*src*/, const VarMeta & /*meta*/) override {}
    void read(StagingBuffer & /*dst*/, const VarMeta & /*meta*/, std::int64_t /*timestep*/, const std::optional<BoundingBox> & /*bbox*/) override {}
    void flush() override {}
    void close() override {}
};

struct WaitTestContext {
    TempDir dir;
    std::string manifest_path;
    amio_core_handle core = nullptr;
    amio_dataset_handle dataset = nullptr;
    bool valid = false;

    WaitTestContext() {
        // Register a no-op driver so the write path doesn't invoke
        // real netcdf4 parallel I/O (which would crash this
        // infrastructure test).
        BackendFactory::instance().register_driver("netcdf4", []() -> std::unique_ptr<Backend_Driver> { return std::make_unique<NoOpDriver>(); });

        std::string yaml = make_manifest_yaml("netcdf4", 8, 65536, 1, 5000);
        manifest_path = write_manifest(dir, yaml);

        amio_status_t rc = amio_init(manifest_path.c_str(), &core);
        if (rc != AMIO_OK || core == nullptr) {
            return;
        }

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

    ~WaitTestContext() {
        if (dataset) {
            amio_close_dataset(dataset);
        }
        if (core) {
            amio_finalize(core);
        }
    }

    WaitTestContext(const WaitTestContext &) = delete;
    WaitTestContext &operator=(const WaitTestContext &) = delete;
};

}  // anonymous namespace

// ===================================================================
// Property Test P22a: amio_wait returns exactly one of the trichotomy.
//
// For any pending I/O handle obtained from amio_write and any timeout
// in [0, 86_400_000]ms: amio_wait returns exactly one of
// {AMIO_OK, AMIO_ERR_BACKEND_FAILURE, AMIO_ERR_TIMEOUT}.
// ===================================================================

TEST_CASE("P22: Synchronous wait totality - amio_wait trichotomy", "[pbt][p22][wait_totality][amio_wait]") {
    auto result = rc::check("amio_wait returns exactly one of {OK, BACKEND_FAILURE, TIMEOUT}", []() {
        WaitTestContext ctx;
        RC_PRE(ctx.valid);

        // Generate a valid write to get an io_handle.
        amio_shape_t shape = {};
        shape.rank = 1;
        shape.extents[0] = *rc::gen::inRange<int64_t>(1, 100);

        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();
        std::size_t byte_count = payload_byte_count(shape, dtype);
        RC_PRE(byte_count > 0 && byte_count <= 65536);

        std::vector<uint8_t> data(byte_count, 0x42);

        amio_io_handle io = nullptr;
        amio_status_t write_rc = amio_write(ctx.dataset, "wait_var", data.data(), dtype, &shape, &io);
        RC_PRE(write_rc == AMIO_OK && io != nullptr);

        // Generate a timeout in [0, 86_400_000] ms.
        // Use small timeouts for test speed.
        auto timeout_ms = *rc::gen::elementOf(std::vector<int64_t>{0, 1, 10, 50, 100, 500, 1000, 5000});

        // Call amio_wait.
        amio_status_t wait_rc = amio_wait(io, timeout_ms);

        // Verify the return is exactly one of the trichotomy.
        RC_ASSERT(kWaitTrichotomy.count(wait_rc) == 1);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P22b: amio_flush returns exactly one of the trichotomy.
//
// For any dataset with pending writes and any timeout in
// [0, 86_400_000]ms: amio_flush returns exactly one of
// {AMIO_OK, AMIO_ERR_BACKEND_FAILURE, AMIO_ERR_TIMEOUT}.
// ===================================================================

TEST_CASE("P22: Synchronous wait totality - amio_flush trichotomy", "[pbt][p22][wait_totality][amio_flush]") {
    auto result = rc::check("amio_flush returns exactly one of {OK, BACKEND_FAILURE, TIMEOUT}", []() {
        WaitTestContext ctx;
        RC_PRE(ctx.valid);

        // Optionally submit some writes before flushing.
        auto num_writes = *rc::gen::inRange<int>(0, 4);
        for (int i = 0; i < num_writes; ++i) {
            amio_shape_t shape = {};
            shape.rank = 1;
            shape.extents[0] = *rc::gen::inRange<int64_t>(1, 50);

            auto dtype = AMIO_DTYPE_F32;
            std::size_t byte_count = payload_byte_count(shape, dtype);
            if (byte_count == 0 || byte_count > 65536) continue;

            std::vector<uint8_t> data(byte_count, 0x33);
            amio_io_handle io = nullptr;
            amio_write(ctx.dataset, "flush_var", data.data(), dtype, &shape, &io);
        }

        // Generate a timeout in [0, 86_400_000] ms.
        auto timeout_ms = *rc::gen::elementOf(std::vector<int64_t>{0, 1, 10, 100, 500, 1000, 5000});

        // Call amio_flush.
        amio_status_t flush_rc = amio_flush(ctx.dataset, timeout_ms);

        // Verify the return is exactly one of the trichotomy.
        RC_ASSERT(kFlushTrichotomy.count(flush_rc) == 1);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P22c: amio_flush with no pending writes returns AMIO_OK.
//
// For any dataset with zero pending writes: amio_flush returns
// AMIO_OK regardless of timeout value.
// ===================================================================

TEST_CASE("P22: Synchronous wait totality - flush with no pending writes", "[pbt][p22][wait_totality][flush_empty]") {
    auto result = rc::check("amio_flush with no pending writes returns AMIO_OK", []() {
        WaitTestContext ctx;
        RC_PRE(ctx.valid);

        // Generate any timeout value.
        auto timeout_ms = *rc::gen::inRange<int64_t>(0, 86400001);

        // Flush with no pending writes.
        amio_status_t flush_rc = amio_flush(ctx.dataset, timeout_ms);

        // Should return AMIO_OK (no work to do, no failures).
        RC_ASSERT(flush_rc == AMIO_OK);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P22d: amio_wait return value is always in the defined
// error code set (never an undefined code).
//
// For any io_handle and timeout: the return value is a defined
// AMIO_ERR_* code (not an arbitrary integer).
// ===================================================================

TEST_CASE("P22: Synchronous wait totality - return is defined error code", "[pbt][p22][wait_totality][defined_codes]") {
    auto result = rc::check("amio_wait always returns a defined AMIO_ERR_* code", []() {
        WaitTestContext ctx;
        RC_PRE(ctx.valid);

        // Submit a write.
        amio_shape_t shape = {};
        shape.rank = 1;
        shape.extents[0] = 10;

        std::vector<uint8_t> data(40, 0x55);
        amio_io_handle io = nullptr;
        amio_status_t write_rc = amio_write(ctx.dataset, "code_var", data.data(), AMIO_DTYPE_F32, &shape, &io);
        RC_PRE(write_rc == AMIO_OK && io != nullptr);

        auto timeout_ms = *rc::gen::inRange<int64_t>(1, 1000);
        amio_status_t wait_rc = amio_wait(io, timeout_ms);

        // The return must be a defined AMIO_ERR_* code (0..17).
        RC_ASSERT(wait_rc >= 0);
        RC_ASSERT(wait_rc <= 17);
    });

    REQUIRE(result);
}
