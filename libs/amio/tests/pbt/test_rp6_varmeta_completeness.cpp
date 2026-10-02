// test_rp6_varmeta_completeness.cpp -- Read-pipeline Property 6:
// VarMeta completeness.
//
// Design "Correctness Properties", Property 6:
//   Every Backend_Driver::read invocation receives a VarMeta whose
//   name equals the caller-supplied non-empty variable name and whose
//   dtype/shape match describe_variable.
//
// The MockBackendDriver was extended (additively) to capture the
// VarMeta name/dtype/shape it received on each read.  This property
// drives the PrefetchQueue read path over generated variable names,
// dtypes, and shapes and asserts that every recorded read carries a
// non-empty name equal to the queue's variable name and dtype/shape
// matching the VariableInfo (the describe_variable result the read
// path threads into the queue).
//
// **Validates: Requirements 3.2, 3.3, 4.1, 4.2**

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "generators.hpp"
#include "mock_backend_driver.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

namespace {

// Generate a non-empty variable name from a small alphabet so the
// shrinker stays well-behaved.
std::string gen_var_name() {
    auto len = *rc::gen::inRange<std::size_t>(1, 17);
    std::string name;
    name.reserve(len);
    for (std::size_t i = 0; i < len; ++i) {
        char c = static_cast<char>(*rc::gen::inRange<int>('a', 'z' + 1));
        name.push_back(c);
    }
    return name;
}

// Generate a bounded shape (rank [1,3], small extents) whose payload
// stays within the staging buffer capacity.
amio_shape_t gen_small_shape() {
    amio_shape_t shape = {};
    shape.rank = *rc::gen::inRange(1, 4);
    for (int d = 0; d < shape.rank; ++d) {
        shape.extents[d] = *rc::gen::inRange<std::int64_t>(1, 9);  // [1, 8]
    }
    return shape;
}

}  // namespace

// ===================================================================
// Property RP6: every read receives complete, correct VarMeta
//
// For any generated (var_name, dtype, shape, depth, total_timesteps):
//   schedule_initial() drives min(depth, total) reads; every recorded
//   read has:
//     - meta.name == var_name and is non-empty
//     - meta.dtype == info.dtype
//     - meta.shape rank/extents == info.shape
// ===================================================================

TEST_CASE("RP6: VarMeta completeness - every read carries the variable name, dtype, and shape", "[pbt][rp6][varmeta][completeness]") {
    auto result = rc::check("every Backend_Driver::read receives non-empty name + matching dtype/shape", []() {
        std::string var_name = gen_var_name();
        RC_PRE(!var_name.empty());

        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();
        amio_shape_t shape = gen_small_shape();

        std::size_t payload_bytes = payload_byte_count(shape, dtype);
        RC_PRE(payload_bytes > 0);

        auto depth = *rc::gen::inRange<std::size_t>(1, 9);
        auto total_timesteps = *rc::gen::inRange<std::int64_t>(1, 17);

        // Buffer capacity sized to fit the payload.
        std::size_t buffer_capacity = std::max(payload_bytes, std::size_t{64});
        std::size_t buffer_count = static_cast<std::size_t>(std::min<std::int64_t>(static_cast<std::int64_t>(depth), total_timesteps)) + 4;
        StagingPool pool(buffer_count, buffer_capacity, 5000);

        auto driver = std::make_shared<MockBackendDriver>();

        auto info = make_var_info(dtype, shape, total_timesteps);
        PrefetchQueue pq(depth, 60, &pool, nullptr, driver.get(), 1, var_name, info, total_timesteps);

        // Drives min(depth, total) synchronous reads through the driver.
        pq.schedule_initial();

        auto reads = driver->get_calls(CallRecord::Method::Read);
        RC_ASSERT(!reads.empty());

        for (const auto &rec : reads) {
            // Non-empty name equal to the caller-supplied variable name.
            RC_ASSERT(!rec.name.empty());
            RC_ASSERT(rec.name == var_name);

            // dtype matches describe_variable (the VariableInfo).
            RC_ASSERT(rec.dtype == dtype);

            // shape rank + extents match describe_variable.
            RC_ASSERT(rec.shape.rank == shape.rank);
            for (int d = 0; d < shape.rank; ++d) {
                RC_ASSERT(rec.shape.extents[d] == shape.extents[d]);
            }
        }

        // Drain any completed buffers back to the pool.
        for (std::int64_t t = 0; t < total_timesteps; ++t) {
            StagingBuffer *buf = nullptr;
            if (pq.get_buffer(t, nullptr, &buf) == AMIO_OK && buf != nullptr) {
                pool.release(buf);
            }
        }
    });

    REQUIRE(result);
}
