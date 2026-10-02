// test_dp2_inmem_open_equals_file.cpp -- Property test DP2: in-memory
// dataset open equals file-based dataset open.
//
// Feature: driver-io-regrid-perf
// Property 2: In-memory open equals file-based open.
//
// For any valid dataset config content string `m` and any valid mode,
// amio_open_dataset_from_string(core, m, "yaml", mode, &ds_a) produces a
// dataset handle equivalent to amio_open_dataset(core, path, mode, &ds_b)
// when `path` names a file whose contents are exactly `m`: the two entry
// points return the SAME status code, and when that status is AMIO_OK each
// mints a non-null dataset handle referencing the same backend key.
//
// RapidCheck standalone (rc::check) wrapped in a Catch2 v3 TEST_CASE,
// matching the end-to-end amio_init + amio_open_dataset style of
// test_p21_snapshot_copy_correctness.cpp.  Min 100 iterations (RC_PARAMS
// max_success=100 via the ENVIRONMENT test property).
//
// A NoOpDriver is registered under "netcdf4" (a backend key the
// BackendFactory dispatches on) so both open entry points reach the same
// driver open outcome WITHOUT invoking the real parallel-netCDF path (which
// would require a live parallel HDF5 file).  The property is about the two
// entry points agreeing, not about performing real I/O.
//
// NOTE ON MPI: like test_dp1_inmem_init_equals_file.cpp, this test does NOT
// initialize MPI and does NOT link the MPI test listener.  amio_init /
// amio_open_dataset construct the LOGS logger via configure_communicator(),
// which only calls MPI_Comm_rank when MPI is already initialized; with MPI
// uninitialized the logger takes its no-MPI path.  The NoOpDriver's open is a
// no-op, so no backend MPI call occurs either.  This keeps the property
// focused on the open-equivalence contract without a live MPI environment.
//
// **Validates: Requirements 2.1, 2.2**

#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "factory/backend_factory.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

namespace {

// NoOpDriver -- a Backend_Driver that accepts open/read/write without
// performing real I/O.  Registered under "netcdf4" so both open entry
// points dispatch to it instead of the real parallel-netCDF path.
class NoOpDriver : public Backend_Driver {
   public:
    void open_write(const conf::Config & /*config*/) override {}
    void open_read(const conf::Config & /*config*/) override {}
    void write(const StagingBuffer & /*src*/, const VarMeta & /*meta*/) override {}
    void read(StagingBuffer & /*dst*/, const VarMeta & /*meta*/, std::int64_t /*timestep*/, const std::optional<BoundingBox> & /*bbox*/) override {}
    void flush() override {}
    void close() override {}
};

// Register the NoOpDriver under "netcdf4" exactly once for the test process.
void ensure_noop_backend_registered() {
    static bool registered = [] {
        BackendFactory::instance().register_driver("netcdf4", []() -> std::unique_ptr<Backend_Driver> { return std::make_unique<NoOpDriver>(); });
        return true;
    }();
    (void)registered;
}

// Build a core handle from an in-memory manifest string.  Returns nullptr on
// failure so the property can precondition it away.
amio_core_handle make_core() {
    // Small buffer counts / capacities suitable for property testing.
    std::string manifest = make_manifest_yaml("netcdf4", 4, 65536, 1, 5000);
    amio_core_handle core = nullptr;
    amio_status_t rc = amio_init_from_string(manifest.c_str(), "yaml", &core);
    if (rc != AMIO_OK) {
        return nullptr;
    }
    return core;
}

}  // anonymous namespace

// ===================================================================
// Property Test DP2: in-memory open equals file-based open.
//
// For a generated valid dataset config, opening via
// amio_open_dataset_from_string and amio_open_dataset (file whose contents
// are byte-identical) yields the SAME status code; when AMIO_OK, both yield
// a non-null dataset handle.
//
// **Validates: Requirements 2.1, 2.2**
// ===================================================================

TEST_CASE("Feature: driver-io-regrid-perf, Property 2: in-memory open equals file-based open", "[pbt][driver-io-regrid-perf][dp2][inmem_open]") {
    ensure_noop_backend_registered();

    auto result = rc::check("amio_open_dataset_from_string(content) agrees with amio_open_dataset(path) for identical dataset config", []() {
        // Generate a valid mode.
        int32_t mode = *rc::gen::element(AMIO_MODE_READ, AMIO_MODE_WRITE);

        // Per-iteration temp directory (RAII cleanup of files).
        TempDir dir;

        // Generate a valid dataset config for the registered "netcdf4"
        // backend.  make_dataset_config_yaml declares the codec on the
        // lossless allow-list and a valid data_model, so the config
        // parses/validates identically through both entry points.
        std::string content = make_dataset_config_yaml("netcdf4", dir.file("output.nc"));

        // Write the SAME content to a file for the path-based open.
        std::string config_path = dir.file("dataset.yaml");
        {
            std::ofstream ofs(config_path);
            ofs << content;
        }

        // Open via the in-memory entry point.
        amio_core_handle core_a = make_core();
        bool core_a_ok = (core_a != nullptr);
        RC_PRE(core_a_ok);
        amio_dataset_handle ds_a = nullptr;
        amio_status_t rc_a = amio_open_dataset_from_string(core_a, content.c_str(), "yaml", mode, &ds_a);

        // Open via the file-based entry point (a fresh core built from the
        // identical manifest; both open the same NoOpDriver under "netcdf4").
        amio_core_handle core_b = make_core();
        bool core_b_ok = (core_b != nullptr);
        RC_PRE(core_b_ok);
        amio_dataset_handle ds_b = nullptr;
        amio_status_t rc_b = amio_open_dataset(core_b, config_path.c_str(), mode, &ds_b);

        // Property: identical config content + identical mode => identical
        // open outcome (same status code).  Compare as ints so RapidCheck
        // does not try to render the opaque handle types on failure.
        bool status_agrees = (rc_a == rc_b);
        RC_ASSERT(status_agrees);

        // On success both entry points mint a non-null dataset handle
        // (referencing the same "netcdf4" backend); on failure neither does.
        // Reduce the opaque void* handles to bools before RC_ASSERT: this
        // RapidCheck build cannot render a void* in a failure message.
        bool ds_a_nonnull = (ds_a != nullptr);
        bool ds_b_nonnull = (ds_b != nullptr);
        if (rc_a == AMIO_OK) {
            RC_ASSERT(ds_a_nonnull);
            RC_ASSERT(ds_b_nonnull);
        } else {
            RC_ASSERT(!ds_a_nonnull);
            RC_ASSERT(!ds_b_nonnull);
        }

        // Close datasets and finalize cores to avoid leaks.
        if (ds_a != nullptr) {
            amio_close_dataset(ds_a);
        }
        if (ds_b != nullptr) {
            amio_close_dataset(ds_b);
        }
        if (core_a != nullptr) {
            amio_finalize(core_a);
        }
        if (core_b != nullptr) {
            amio_finalize(core_b);
        }
    });

    REQUIRE(result);
}
