// test_batch_staging_ordering.cpp -- Regression test for the collective-ordering fix in
// amio::detail::write_region_batch(): the staging buffer must be acquired BEFORE the status
// and schema-hash reductions, and a failed acquisition must be reported through those same
// agreement steps rather than as an early return.  If one rank ever returned
// AMIO_ERR_STAGING_BACKPRESSURE while peers proceeded to enqueue NC_COLLECTIVE batch writes,
// the peers would block forever inside nc_put_vara_*.
//
// The test is white-box (single process): it exhausts the core's staging pool directly, then
// asserts that the batch call
//   - returns AMIO_ERR_STAGING_BACKPRESSURE with a null io handle,
//   - does NOT mark the dataset as having started writes (pre-fix behavior set
//     writes_started before the acquisition attempt),
//   - and succeeds after the pool is replenished.
//
// The multi-rank agreement path itself is covered by integration.mpi_netcdf_batch*, which
// drive the same reductions through the collective MPI protocol (schema-mismatch, invalid).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "amio/amio.h"

// Private headers for test setup and for driving the internal batch entry point.
#include "c_boundary/amio_core.hpp"
#include "c_boundary/handle_table.hpp"
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

// ---------------------------------------------------------------
// MockDriver -- minimal region-write-capable Backend_Driver with schema preparation.
// ---------------------------------------------------------------
class MockDriver : public amio::detail::Backend_Driver {
   public:
    bool supports_region_writes() const noexcept override {
        return true;
    }
    bool supports_write_schema_preparation() const noexcept override {
        return true;
    }
    void prepare_write_schema(const std::vector<amio::detail::VarMeta> &) override {}
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const amio::detail::StagingBuffer &, const amio::detail::VarMeta &) override {}
    void read(amio::detail::StagingBuffer &, const amio::detail::VarMeta &, std::int64_t, const std::optional<amio::detail::BoundingBox> &) override {
    }
    void flush() override {}
    void close() override {}
};

std::string write_config_file(const std::string &suffix = "") {
    std::string path = "/tmp/amio_test_batch_order" + suffix + ".yaml";
    std::ofstream ofs(path);
    ofs << "backend: mock_batch_order\n";
    ofs << "staging_pool:\n";
    ofs << "  buffer_count: 2\n";
    ofs << "  buffer_capacity_bytes: 4096\n";
    ofs << "  max_buffer_count: 2\n";
    ofs << "worker_pool:\n";
    ofs << "  threads: 1\n";
    ofs << "prefetch:\n";
    ofs << "  depth: 4\n";
    ofs << "  read_timeout_s: 60\n";
    ofs << "staging_timeout_ms: 50\n";
    ofs.close();
    return path;
}

void test_batch_backpressure_ordering() {
    amio::detail::BackendFactory::instance().register_driver(
        "mock_batch_order", []() -> std::unique_ptr<amio::detail::Backend_Driver> { return std::make_unique<MockDriver>(); });

    std::string path = write_config_file("_ordering");

    amio_core_handle core = nullptr;
    EXPECT_EQ(amio_init(path.c_str(), &core), AMIO_OK, "init");
    EXPECT_TRUE(core != nullptr, "amio_init should succeed");

    amio_dataset_handle ds = nullptr;
    EXPECT_EQ(amio_open_dataset(core, path.c_str(), AMIO_MODE_WRITE, &ds), AMIO_OK, "open_dataset");

    // Prepare a one-variable schema through the public API (region batches require it).
    char name[] = "field";
    amio_write_variable_t variable{};
    variable.struct_size = static_cast<uint32_t>(sizeof(variable));
    variable.version = AMIO_WRITE_VARIABLE_VERSION;
    variable.name = name;
    variable.dtype = AMIO_DTYPE_F32;
    variable.rank = 1;
    variable.global_extents[0] = 8;
    EXPECT_EQ(amio_prepare_write_schema(ds, &variable, 1), AMIO_OK, "prepare_write_schema");

    // Resolve internal records so the test can exhaust the pool deterministically.
    void *dataset_payload = nullptr;
    EXPECT_EQ(
        amio::detail::process_handle_table().lookup(amio::detail::HandleTable::from_ptr(ds), amio::detail::HandleKind::Dataset, &dataset_payload),
        AMIO_OK, "dataset handle resolves");
    auto *record = static_cast<amio::detail::DatasetRecord *>(dataset_payload);
    EXPECT_TRUE(record != nullptr && record->core != nullptr && record->core->staging_pool != nullptr, "core has a staging pool");

    // Exhaust every slot of the bounded pool (buffer_count == max_buffer_count == 2).  The
    // 50 ms staging timeout makes the final acquire return nullptr promptly instead of growing.
    amio::detail::StagingPool *pool = record->core->staging_pool.get();
    std::vector<amio::detail::StagingBuffer *> exhausted;
    while (amio::detail::StagingBuffer *buf = pool->acquire(64)) {
        exhausted.push_back(buf);
    }
    EXPECT_TRUE(exhausted.size() >= pool->total_buffer_count(), "pool exhausted to its ceiling");
    EXPECT_TRUE(pool->acquire(64) == nullptr, "no buffer available after exhaustion");

    // Submit a small batch while the pool is exhausted: the acquisition failure must be
    // reported through the agreement path instead of a post-reduction early return.
    std::vector<float> payload(8, 1.5f);
    amio_shape_t shape{};
    shape.rank = 1;
    shape.extents[0] = 8;
    amio_write_region_t region{};
    region.rank = 1;
    region.global_extents[0] = 8;
    amio_region_write_t write{};
    write.var_name = name;
    write.host_data = payload.data();
    write.dtype = AMIO_DTYPE_F32;
    write.shape = &shape;
    write.region = &region;

    const bool writes_started_before = record->writes_started;
    amio_io_handle io = nullptr;
    EXPECT_EQ(amio_write_region_batch(ds, &write, 1, &io), AMIO_ERR_STAGING_BACKPRESSURE, "backpressure returned by batch");
    EXPECT_TRUE(io == nullptr, "rejected batch returns no io handle");
    // Regression assertion: a failed acquisition must not start writes for the dataset.
    EXPECT_EQ(record->writes_started, writes_started_before, "failed batch acquisition does not start writes");
    EXPECT_TRUE(!record->write_schema_failed, "schema stays healthy after backpressure rejection");

    // Replenish the pool and submit again: the same batch must now succeed.
    for (auto *buf : exhausted) pool->release(buf);
    exhausted.clear();

    EXPECT_EQ(amio_write_region_batch(ds, &write, 1, &io), AMIO_OK, "batch succeeds once pool is free");
    EXPECT_TRUE(io != nullptr, "batch admitted with io handle");

    EXPECT_EQ(amio_wait(io, 5000), AMIO_OK, "batch completes");
    EXPECT_EQ(amio_flush(ds, 5000), AMIO_OK, "flush");

    EXPECT_EQ(amio_close_dataset(ds), AMIO_OK, "close");
    EXPECT_EQ(amio_finalize(core), AMIO_OK, "finalize");
    amio::detail::BackendFactory::instance().clear();
}

}  // namespace

int main() {
    test_batch_backpressure_ordering();
    std::printf("test_batch_staging_ordering: %d passed, %d failed\n", g_result.passed, g_result.failed);
    return g_result.failed == 0 ? 0 : 1;
}
