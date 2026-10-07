// test_rp8_no_write_past_capacity.cpp -- Read-pipeline Property 8:
// No write past capacity.
//
// Design "Correctness Properties", Property 8:
//   A driver read never writes beyond dst.capacity_bytes; if the
//   required payload exceeds capacity the read fails with
//   AMIO_ERR_BACKEND_FAILURE and leaves no partial view.
//
// The read path (PrefetchQueue::sync_fetch) sizes the staging
// acquisition to element_size(dtype) * product(extents) and applies a
// central backstop: after the driver returns, if used_bytes exceeds
// the buffer capacity the fetch is failed with
// AMIO_ERR_BACKEND_FAILURE and the buffer is returned to the pool.
// Drivers that detect the overflow up front instead throw (also
// surfaced as AMIO_ERR_BACKEND_FAILURE).  This property covers both
// behaviours over generated (capacity, payload) pairs.
//
// **Validates: Requirements 4.4**

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "factory/backend_driver.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

namespace {

// CapacityDriver -- attempts to deliver `required_bytes` of payload.
//
// Two failure modes when required_bytes > dst.capacity_bytes:
//   * Throw mode: detect the overflow and throw before writing (the
//     behaviour of the real NetCDF/Zarr/GRIB2 drivers).
//   * Report mode: memcpy only what fits (never past capacity) but
//     report used_bytes = required_bytes, exercising the prefetch
//     queue's post-read capacity backstop.
// When required_bytes <= capacity it writes exactly required_bytes.
class CapacityDriver : public Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void flush() override {}
    void close() override {}
    void write(const StagingBuffer &, const VarMeta &) override {}

    void configure(std::size_t required_bytes, bool throw_on_overflow) {
        std::lock_guard<std::mutex> lock(mu_);
        required_bytes_ = required_bytes;
        throw_on_overflow_ = throw_on_overflow;
    }

    void read(StagingBuffer &dst, const VarMeta &, std::int64_t, const std::optional<BoundingBox> &) override {
        std::lock_guard<std::mutex> lock(mu_);
        if (required_bytes_ > dst.capacity_bytes) {
            if (throw_on_overflow_) {
                // Real-driver behaviour: refuse before writing.
                throw std::runtime_error("CapacityDriver: payload exceeds capacity");
            }
            // Report-mode: never write past capacity, but over-report
            // used_bytes so the read-path backstop must catch it.
            std::memset(dst.data, 0xAB, dst.capacity_bytes);
            dst.used_bytes = required_bytes_;  // > capacity_bytes
            return;
        }
        std::memset(dst.data, 0xCD, required_bytes_);
        dst.used_bytes = required_bytes_;
    }

   private:
    std::mutex mu_;
    std::size_t required_bytes_ = 0;
    bool throw_on_overflow_ = true;
};

}  // namespace

// ===================================================================
// Property RP8: over-capacity payloads fail, never overflow
//
// For any generated buffer capacity C and required payload P:
//   - If P <= C: read succeeds and used_bytes == P <= capacity.
//   - If P > C: read returns AMIO_ERR_BACKEND_FAILURE and no view
//     (buf == nullptr), regardless of whether the driver throws or
//     over-reports used_bytes.
//   - After the failure, the staging pool is fully reclaimed (the
//     over-capacity buffer is not leaked as a partial view).
// ===================================================================

TEST_CASE("RP8: no write past capacity - over-capacity payload fails with BACKEND_FAILURE and no view", "[pbt][rp8][capacity]") {
    auto result = rc::check("payload > buffer capacity yields AMIO_ERR_BACKEND_FAILURE and no view", []() {
        // Buffer capacity C (the pool's per-buffer size).
        auto capacity = *rc::gen::inRange<std::size_t>(16, 4097);

        // Required payload P, spanning both <= C and > C.
        auto required = *rc::gen::inRange<std::size_t>(1, capacity * 2 + 1);

        // Whether the driver throws or over-reports on overflow.
        bool throw_on_overflow = *rc::gen::arbitrary<bool>();

        StagingPool pool(2, capacity, 50);

        auto driver = std::make_shared<CapacityDriver>();
        driver->configure(required, throw_on_overflow);

        // Size the variable info so sync_fetch acquires a buffer of
        // capacity exactly C (request C bytes; all pool buffers are C).
        // We use U8 so element_size==1 and product==C.
        amio_shape_t shape = {};
        shape.rank = 1;
        shape.extents[0] = static_cast<std::int64_t>(capacity);
        auto info = make_var_info(AMIO_DTYPE_U8, shape, 1);

        PrefetchQueue pq(1, 60, &pool, nullptr, driver.get(), 1, "test_var", info, 1);

        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(0, nullptr, &buf);

        if (required <= capacity) {
            // Fits: success, never past capacity.
            RC_ASSERT(status == AMIO_OK);
            RC_ASSERT(buf != nullptr);
            RC_ASSERT(buf->used_bytes == required);
            RC_ASSERT(buf->used_bytes <= buf->capacity_bytes);
            pool.release(buf);
        } else {
            // Over capacity: failure, no view, no partial buffer leaked.
            RC_ASSERT(status == AMIO_ERR_BACKEND_FAILURE);
            RC_ASSERT(buf == nullptr);
            RC_ASSERT(pool.free_buffer_count() == pool.total_buffer_count());
        }
    });

    REQUIRE(result);
}
