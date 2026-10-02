// test_read_e2e_smoke.cpp -- End-to-end public-C-ABI read smoke test
// (task 20, design "Testing Strategy").
//
// This is the final-verification smoke test for the read pipeline.  It
// drives the COMPLETE public C API lifecycle exactly as a host
// application would -- through the installed extern "C" entry points
// (amio_api.cpp) -- and asserts that a read served from the runtime
// Staging_Pool succeeds without AMIO_ERR_STAGING_BACKPRESSURE:
//
//   amio_init                          (builds Staging_Pool + Worker_Pool)
//     -> amio_open_dataset(READ)       (invokes Backend_Driver::open_read)
//        -> amio_read                  (serves from a staging buffer)
//           -> amio_release_view       (returns the buffer to the pool)
//        -> amio_close_dataset
//     -> amio_finalize
//
// The original core bug (requirements.md gap #1, Req 1.3) was that
// amio::detail::init() built only a bare AMIO_Core and never
// constructed the Staging_Pool, so EVERY prefetch failed with
// AMIO_ERR_STAGING_BACKPRESSURE before reaching a driver.  The
// strongest regression for that bug is a read through the public ABI
// that (a) succeeds and (b) is explicitly asserted NOT to return
// AMIO_ERR_STAGING_BACKPRESSURE.
//
// To keep the smoke test hermetic and independent of parallel HDF5 /
// TensorStore / g2c availability, it registers a tiny in-process mock
// Backend_Driver under a private factory key ("e2e_smoke_read") before
// amio_init.  The driver describes a fixed rank-1 F32 variable and
// fills the staging buffer with a deterministic pattern on read, so
// the test exercises the real init -> open_read -> prefetch ->
// staging-pool -> read coordinator -> view path end to end, with the
// real public symbols, the real Staging_Pool, and the real
// PrefetchQueue.  The byte-for-byte fidelity of the REAL drivers
// (NetCDF / Zarr / GRIB2) is covered by the driver integration tests
// (unit.read_netcdf4 et al.) and the rp1..rp8 property tests; this
// test pins the C-boundary lifecycle and the no-backpressure contract.
//
// Validates: Req 1.3, 2.1, 9.1 (init builds the pools so a read serves
//            from staging instead of failing with backpressure).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "amio/amio.h"

// Private headers: register a mock driver in the same factory the
// public open_dataset path consults.
#include "factory/backend_driver.hpp"
#include "factory/backend_factory.hpp"
#include "staging/staging_pool.hpp"

namespace {

struct TestResult {
    int passed = 0;
    int failed = 0;
};

TestResult g_result{};

void report_failure(const char *expr, const char *file, int line, const std::string &context) {
    std::fprintf(stderr, "FAIL %s:%d: %s   (%s)\n", file, line, expr, context.c_str());
    ++g_result.failed;
}

#define EXPECT_TRUE(cond, ctx)                                \
    do {                                                      \
        if (!(cond)) {                                        \
            report_failure(#cond, __FILE__, __LINE__, (ctx)); \
        } else {                                              \
            ++g_result.passed;                                \
        }                                                     \
    } while (0)

#define EXPECT_EQ(a, b, ctx)                                                                                             \
    do {                                                                                                                 \
        if ((a) != (b)) {                                                                                                \
            char buf[256];                                                                                               \
            std::snprintf(buf, sizeof(buf), "%s: expected %d, got %d", (ctx), static_cast<int>(b), static_cast<int>(a)); \
            report_failure(#a " == " #b, __FILE__, __LINE__, buf);                                                       \
        } else {                                                                                                         \
            ++g_result.passed;                                                                                           \
        }                                                                                                                \
    } while (0)

// The mock variable: rank-1 [16] F32, single timestep.  Payload =
// 16 * 4 = 64 bytes, well inside the manifest's staging buffers.
constexpr int kElements = 16;

amio::detail::VariableInfo make_info() {
    amio::detail::VariableInfo info{};
    info.found = true;
    info.dtype = AMIO_DTYPE_F32;
    info.shape.rank = 1;
    info.shape.extents[0] = kElements;
    info.total_timesteps = 1;
    return info;
}

// ---------------------------------------------------------------
// SmokeReadDriver -- a minimal readable backend.  open_read succeeds
// (Req 2.1), describe_variable reports a known F32 shape, and read
// fills the staging buffer with a deterministic float pattern within
// capacity (so the read serves from staging, never backpressure).
// ---------------------------------------------------------------
class SmokeReadDriver : public amio::detail::Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {
        opened_read_ = true;
    }

    void write(const amio::detail::StagingBuffer &, const amio::detail::VarMeta &) override {}

    void read(amio::detail::StagingBuffer &dst, const amio::detail::VarMeta &meta, std::int64_t timestep,
              const std::optional<amio::detail::BoundingBox> &) override {
        std::size_t bytes = amio::detail::element_size(meta.dtype);
        for (int d = 0; d < meta.shape.rank; ++d) {
            bytes *= static_cast<std::size_t>(meta.shape.extents[d]);
        }
        if (bytes > dst.capacity_bytes) {
            throw std::runtime_error("SmokeReadDriver: payload exceeds buffer capacity");
        }
        if (dst.data != nullptr) {
            auto *out = reinterpret_cast<float *>(dst.data);
            for (int i = 0; i < kElements; ++i) {
                out[i] = static_cast<float>(timestep) * 100.0f + static_cast<float>(i);
            }
        }
        dst.used_bytes = bytes;
    }

    void flush() override {}
    void close() override {}

    amio::detail::VariableInfo describe_variable(const std::string &) override {
        return make_info();
    }

   private:
    bool opened_read_ = false;
};

// Write a flat-key manifest/dataset config the ConfigLoader accepts.
// The same file serves as both the init manifest (pool sizing) and the
// open_dataset config (backend key); init reads the pool fields and
// open_dataset reads the backend key.
std::string write_config(const std::string &backend_key) {
    const std::string path = "/tmp/amio_test_read_e2e_smoke.yaml";
    std::ofstream ofs(path);
    ofs << "backend: " << backend_key << "\n";
    ofs << "output_path: /tmp/amio_test_read_e2e_smoke.dat\n";
    ofs << "staging_pool:\n";
    ofs << "  buffer_count: 8\n";
    ofs << "  buffer_capacity_bytes: 4096\n";
    ofs << "worker_pool:\n";
    ofs << "  threads: 1\n";
    ofs << "prefetch:\n";
    ofs << "  depth: 4\n";
    ofs << "  read_timeout_s: 60\n";
    ofs << "staging_timeout_ms: 5000\n";
    ofs.close();
    return path;
}

// ---------------------------------------------------------------
// The end-to-end read lifecycle through the public C ABI.
// ---------------------------------------------------------------
void test_public_api_read_no_backpressure() {
    const std::string backend_key = "e2e_smoke_read";

    // Register the mock readable backend BEFORE amio_init so the public
    // open_dataset path finds it via the factory.
    amio::detail::BackendFactory::instance().register_driver(
        backend_key, []() -> std::unique_ptr<amio::detail::Backend_Driver> { return std::make_unique<SmokeReadDriver>(); });

    const std::string config = write_config(backend_key);

    // ---- amio_init: builds the Staging_Pool + Worker_Pool (Req 1.1, 1.2) ----
    amio_core_handle core = nullptr;
    amio_status_t rc_init = amio_init(config.c_str(), &core);
    EXPECT_EQ(rc_init, AMIO_OK, "amio_init succeeds");
    EXPECT_TRUE(core != nullptr, "amio_init returns a core handle");
    if (core == nullptr) {
        return;
    }

    // ---- amio_open_dataset(READ): invokes Backend_Driver::open_read (Req 2.1) ----
    amio_dataset_handle read_ds = nullptr;
    amio_status_t rc_open = amio_open_dataset(core, config.c_str(), AMIO_MODE_READ, &read_ds);
    EXPECT_EQ(rc_open, AMIO_OK, "amio_open_dataset(READ) succeeds");
    EXPECT_TRUE(read_ds != nullptr, "open_dataset returns a read handle");

    if (read_ds != nullptr) {
        // ---- amio_read: must serve from a staging buffer, NOT backpressure (Req 1.3) ----
        amio_view_handle view = nullptr;
        amio_status_t rc_read = amio_read(read_ds, "temperature", /*timestep=*/0, /*bbox=*/nullptr, &view);

        // The crux of the regression: the original bug made this return
        // AMIO_ERR_STAGING_BACKPRESSURE because init() never built the
        // Staging_Pool.  Assert it explicitly does NOT.
        EXPECT_TRUE(rc_read != AMIO_ERR_STAGING_BACKPRESSURE, "amio_read does NOT return AMIO_ERR_STAGING_BACKPRESSURE (Req 1.3)");
        EXPECT_EQ(rc_read, AMIO_OK, "amio_read serves the buffer with AMIO_OK");
        EXPECT_TRUE(view != nullptr, "amio_read returns a Memory_View handle");

        // ---- amio_release_view: returns the buffer to the pool ----
        if (view != nullptr) {
            amio_status_t rc_release = amio_release_view(view);
            EXPECT_EQ(rc_release, AMIO_OK, "amio_release_view succeeds");
        }

        // ---- amio_close_dataset: no outstanding views -> succeeds ----
        amio_status_t rc_close = amio_close_dataset(read_ds);
        EXPECT_EQ(rc_close, AMIO_OK, "amio_close_dataset succeeds after the view is released");
    }

    // ---- amio_finalize: drains the Worker_Pool and releases the pool ----
    amio_status_t rc_final = amio_finalize(core);
    EXPECT_EQ(rc_final, AMIO_OK, "amio_finalize succeeds");

    std::remove(config.c_str());
}

}  // namespace

int main() {
    test_public_api_read_no_backpressure();

    std::fprintf(stdout, "test_read_e2e_smoke: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
