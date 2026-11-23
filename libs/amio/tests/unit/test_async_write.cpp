// Regression tests for asynchronous ownership, failure, and queue admission.
#include <atomic>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include "c_boundary/amio_core.hpp"
#include "staging/staging_pool.hpp"
#include "workers/worker_pool.hpp"

using namespace amio::detail;

class Driver final : public Backend_Driver {
   public:
    std::function<void(const StagingBuffer &)> action;
    std::atomic<int> flushes{0};
    bool supports_region_writes() const noexcept override {
        return true;
    }
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void write(const StagingBuffer &buf, const VarMeta &) override {
        action(buf);
    }
    void read(StagingBuffer &, const VarMeta &, std::int64_t, const std::optional<BoundingBox> &) override {}
    void flush() override {
        ++flushes;
    }
    void close() override {}
};

static void check(bool condition) {
    if (!condition) throw std::runtime_error("async write regression");
}
static void release_io(amio_io_handle handle, int expected) {
    void *payload = nullptr;
    auto token = HandleTable::from_ptr(handle);
    check(process_handle_table().lookup(token, HandleKind::Io, &payload) == AMIO_OK);
    check(wait(payload, 3000) == expected);
    process_handle_table().release(token, HandleKind::Io);
    delete static_cast<IoRecord *>(payload);
}

int main() {
    AMIO_Core core;
    core.staging_pool = std::make_unique<StagingPool>(16, 4096, 1000);
    core.worker_pool = std::make_unique<WorkerPool>(4);
    DatasetRecord dataset;
    dataset.core = &core;
    dataset.driver = std::make_unique<Driver>();
    auto *driver = static_cast<Driver *>(dataset.driver.get());
    std::atomic<int> writes{0};
    driver->action = [&](const StagingBuffer &buf) {
        check(buf.used_bytes == 4 * sizeof(float));
        float snapshot[4];
        std::memcpy(snapshot, buf.data, sizeof(snapshot));
        check(snapshot[0] == 1 && snapshot[3] == 4);
        ++writes;
    };
    amio_shape_t shape{};
    shape.rank = 1;
    shape.extents[0] = 4;
    std::vector<amio_io_handle> handles;
    for (int n = 0; n < 400; ++n) {
        float data[4] = {1, 2, 3, 4};
        amio_io_handle io = nullptr;
        check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io) == AMIO_OK);
        data[0] = -1;
        handles.push_back(io);
    }
    check(flush(&dataset, 3000) == AMIO_OK);
    check(writes == 400 && dataset.pending_writes == 0 && driver->flushes == 1);
    for (auto io : handles) release_io(io, AMIO_OK);

    driver->action = [](const StagingBuffer &) { throw std::runtime_error("injected backend failure"); };
    float data[4] = {};
    amio_io_handle io = nullptr;
    check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io) == AMIO_OK);
    release_io(io, AMIO_ERR_BACKEND_FAILURE);  // Failure must complete, rather than time out.
    check(flush(&dataset, 3000) == AMIO_ERR_BACKEND_FAILURE);

    WorkerPoolConfig cfg;
    cfg.thread_count = 0;
    cfg.backpressure.queue_capacity = 1;
    core.worker_pool = std::make_unique<WorkerPool>(cfg);
    std::uint64_t seq = 0;
    check(core.worker_pool->submit_write({1, 1}, [] {}, &seq) == AMIO_OK);
    io = nullptr;
    check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io) == AMIO_ERR_QUEUE_FULL);
    check(io == nullptr && dataset.pending_writes == 0);

    core.worker_pool.reset();
    driver->action = [&](const StagingBuffer &) { ++writes; };
    check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io) == AMIO_OK);
    release_io(io, AMIO_OK);
    check(writes == 401);  // Zero workers must perform actual synchronous I/O.

    amio_write_region_t region{};
    region.rank = 1;
    region.global_extents[0] = 8;
    region.offsets[0] = 5;
    check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io, &region) == AMIO_ERR_INVALID_INPUT);
    region.offsets[0] = 4;
    check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io, &region) == AMIO_OK);
    release_io(io, AMIO_OK);
    shape.extents[0] = INT64_MAX;
    check(write(&dataset, "field", data, AMIO_DTYPE_F32, &shape, &io) == AMIO_ERR_INVALID_INPUT);
}
