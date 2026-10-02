// test_rp1_round_trip_fidelity.cpp -- Read-pipeline Property 1:
// Read round-trip fidelity.
//
// Design "Correctness Properties", Property 1:
//   For every supported amio_dtype_t and shape, writing a payload
//   then reading it back yields byte-equal bytes (read(write(x)) == x).
//
// This is the driver-agnostic PBT-layer round trip: it writes a
// payload through the recording MockBackendDriver (store_payloads),
// then reads it back through the real PrefetchQueue read path and
// asserts byte-for-byte equality across generated dtypes/shapes.
// Real per-driver (NetCDF / Zarr / GRIB2) round trips are covered by
// the integration tests (tasks 13.1, 15.2, 17.1) and P1 in
// test_p1_round_trip.cpp; this property exercises the read-path
// fidelity contract over the full dtype/shape space.
//
// **Validates: Requirements 9.4, 11.3**

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

namespace {

// Cap payload bytes so 100 iterations stay fast.
constexpr std::size_t kMaxPayloadBytes = 16384;

// Generate a bounded shape whose payload (element_size * product)
// stays within kMaxPayloadBytes for the given dtype.
amio_shape_t gen_bounded_shape(amio_dtype_t dtype) {
    const std::size_t elem = dtype_size(dtype);
    const std::size_t max_elems = kMaxPayloadBytes / (elem == 0 ? 1 : elem);

    amio_shape_t shape = {};
    shape.rank = *rc::gen::inRange(1, 4);  // rank in [1, 3]

    std::int64_t per_dim_max = 1;
    if (shape.rank == 1) {
        per_dim_max = static_cast<std::int64_t>(std::min<std::size_t>(max_elems, 1024));
    } else if (shape.rank == 2) {
        per_dim_max = static_cast<std::int64_t>(std::sqrt(static_cast<double>(max_elems)));
        per_dim_max = std::min<std::int64_t>(per_dim_max, 64);
    } else {
        per_dim_max = static_cast<std::int64_t>(std::cbrt(static_cast<double>(max_elems)));
        per_dim_max = std::min<std::int64_t>(per_dim_max, 16);
    }
    if (per_dim_max < 1) per_dim_max = 1;

    for (int d = 0; d < shape.rank; ++d) {
        shape.extents[d] = *rc::gen::inRange<std::int64_t>(1, per_dim_max + 1);
    }
    return shape;
}

}  // namespace

// ===================================================================
// Property RP1: read(write(x)) == x  (byte-equal round trip)
//
// For any generated (dtype, shape, payload):
//   1. Write the payload through the driver (store_payloads on).
//   2. Read it back through PrefetchQueue::get_buffer(0).
//   3. The read-back bytes equal the original payload exactly, and
//      used_bytes equals the payload byte count.
// ===================================================================

TEST_CASE("RP1: read round-trip fidelity - read(write(x)) == x", "[pbt][rp1][round_trip][fidelity]") {
    auto result = rc::check("write then read yields byte-equal payload across dtypes/shapes", []() {
        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();
        amio_shape_t shape = gen_bounded_shape(dtype);

        std::size_t payload_bytes = payload_byte_count(shape, dtype);
        RC_PRE(payload_bytes > 0);
        RC_PRE(payload_bytes <= kMaxPayloadBytes);

        // Generate the original payload bytes.
        std::vector<std::byte> original(payload_bytes);
        for (auto &b : original) {
            b = static_cast<std::byte>(*rc::gen::inRange(0, 256));
        }

        // Staging pool sized to fit the payload (best-fit returns a
        // buffer with capacity >= the requested payload size).
        std::size_t buffer_capacity = std::max(payload_bytes, std::size_t{64});
        StagingPool pool(4, buffer_capacity, 5000);

        auto driver = std::make_shared<MockBackendDriver>();
        driver->set_store_payloads(true);

        // --- Write the payload through the driver ---
        {
            StagingBuffer src;
            src.data = original.data();
            src.capacity_bytes = original.size();
            src.used_bytes = original.size();
            VarMeta meta;
            meta.dataset_id = 1;
            meta.name = "test_var";
            meta.dtype = dtype;
            meta.shape = shape;
            driver->write(src, meta);
        }

        // --- Read it back through the prefetch read path ---
        auto info = make_var_info(dtype, shape, /*total_timesteps=*/1);
        PrefetchQueue pq(1, 60, &pool, nullptr, driver.get(), 1, "test_var", info, 1);

        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(0, nullptr, &buf);
        RC_ASSERT(status == AMIO_OK);
        RC_ASSERT(buf != nullptr);

        // used_bytes equals the payload byte count.
        RC_ASSERT(buf->used_bytes == payload_bytes);

        // Byte-for-byte equality: read(write(x)) == x.
        RC_ASSERT(std::memcmp(buf->data, original.data(), payload_bytes) == 0);

        pool.release(buf);
    });

    REQUIRE(result);
}
