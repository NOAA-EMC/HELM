// test_rp2_subset_equals_slice.cpp -- Read-pipeline Property 2:
// Subset equals slice.
//
// Design "Correctness Properties", Property 2:
//   A bounding-box read returns exactly the elements
//   full[offset + k*stride] for k in [0, extent) per dimension, and
//   used_bytes == element_size(dtype) * product(extents).
//
// This strengthens the existing p13 (bounding-box selectivity, which
// only checks byte *counts*) by asserting the actual sliced *values*:
// a SliceDriver holds a full row-major array, performs the
// offset/extent/stride gather a real driver would, and the property
// independently recomputes the expected slice and compares byte for
// byte.
//
// **Validates: Requirements 9.2, 9.3, 10.1, 10.2, 10.3, 12.1**

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <vector>

#include "factory/backend_driver.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"
#include "prefetch/prefetch_queue.hpp"
#include "staging/staging_pool.hpp"

using namespace amio::detail;
using namespace amio::pbt;

namespace {

// Row-major strides (in elements) for a shape's extents.
std::vector<std::size_t> row_major_strides(const amio_shape_t &shape) {
    std::vector<std::size_t> strides(static_cast<std::size_t>(shape.rank), 1);
    for (int d = shape.rank - 2; d >= 0; --d) {
        strides[static_cast<std::size_t>(d)] = strides[static_cast<std::size_t>(d + 1)] * static_cast<std::size_t>(shape.extents[d + 1]);
    }
    return strides;
}

// SliceDriver -- holds a full row-major array of `elem_size`-byte
// elements and, on read with a bbox, gathers exactly the elements
// full[offset + k*stride] (per dim) into dst.  With no bbox it copies
// the full array.  This is the slice semantics a real backend driver
// must honour (Req 9.2/9.3/10.1/10.2/10.3); the test validates that
// the read path delivers exactly this gather.
class SliceDriver : public Backend_Driver {
   public:
    void open_write(const conf::Config &) override {}
    void open_read(const conf::Config &) override {}
    void flush() override {}
    void close() override {}
    void write(const StagingBuffer &, const VarMeta &) override {}

    void set_array(const amio_shape_t &shape, amio_dtype_t dtype, std::vector<std::byte> bytes) {
        std::lock_guard<std::mutex> lock(mu_);
        shape_ = shape;
        dtype_ = dtype;
        full_ = std::move(bytes);
    }

    void read(StagingBuffer &dst, const VarMeta &, std::int64_t, const std::optional<BoundingBox> &bbox) override {
        std::lock_guard<std::mutex> lock(mu_);
        const std::size_t elem = dtype_size(dtype_);
        const auto strides = row_major_strides(shape_);

        if (!bbox.has_value()) {
            std::size_t n = std::min(full_.size(), dst.capacity_bytes);
            std::memcpy(dst.data, full_.data(), n);
            dst.used_bytes = n;
            return;
        }

        const BoundingBox &b = bbox.value();
        // Number of selected elements = product(extents).
        std::size_t sel_elems = 1;
        for (int d = 0; d < b.rank; ++d) {
            sel_elems *= static_cast<std::size_t>(b.extents[d]);
        }

        auto *out = dst.data;
        std::size_t out_off = 0;

        // Iterate the selected index space in row-major order, mapping
        // each multi-index k -> source linear index
        // sum_d (offset[d] + k[d]*stride[d]) * row_major_stride[d].
        std::vector<std::int64_t> idx(static_cast<std::size_t>(b.rank), 0);
        for (std::size_t count = 0; count < sel_elems; ++count) {
            std::size_t src_elem = 0;
            for (int d = 0; d < b.rank; ++d) {
                std::int64_t src_d = b.offsets[d] + idx[static_cast<std::size_t>(d)] * b.strides[d];
                src_elem += static_cast<std::size_t>(src_d) * strides[static_cast<std::size_t>(d)];
            }
            std::memcpy(out + out_off, full_.data() + src_elem * elem, elem);
            out_off += elem;

            // Increment the multi-index (last dimension fastest).
            for (int d = b.rank - 1; d >= 0; --d) {
                if (++idx[static_cast<std::size_t>(d)] < b.extents[d]) break;
                idx[static_cast<std::size_t>(d)] = 0;
            }
        }
        dst.used_bytes = sel_elems * elem;
    }

   private:
    std::mutex mu_;
    amio_shape_t shape_ = {};
    amio_dtype_t dtype_ = AMIO_DTYPE_F32;
    std::vector<std::byte> full_;
};

}  // namespace

// ===================================================================
// Property RP2: bbox read returns exactly the slice
//
// For any generated variable shape, dtype, and bbox (offset/extent/
// stride within bounds):
//   - the bytes returned equal full[offset + k*stride] per dim
//   - used_bytes == element_size(dtype) * product(extents)
// ===================================================================

TEST_CASE("RP2: subset equals slice - bbox read returns exactly full[offset + k*stride]", "[pbt][rp2][subset][slice]") {
    auto result = rc::check("a bbox read returns the exact strided slice of the full array", []() {
        auto rank = *rc::gen::inRange(1, 4);  // rank in [1, 3] to bound size
        amio_shape_t shape = {};
        shape.rank = rank;
        for (int d = 0; d < rank; ++d) {
            shape.extents[d] = *rc::gen::inRange<std::int64_t>(2, 17);  // [2, 16]
        }

        auto dtype = *rc::gen::arbitrary<amio_dtype_t>();
        const std::size_t elem = dtype_size(dtype);
        std::size_t full_elems = shape_element_count(shape);
        std::size_t full_bytes = full_elems * elem;
        RC_PRE(full_bytes > 0 && full_bytes <= 65536);

        // Build a valid bbox within bounds, with possible strides > 1.
        amio_bbox_t bbox = {};
        bbox.rank = rank;
        for (int d = 0; d < rank; ++d) {
            std::int64_t ext = shape.extents[d];
            // stride in [1, ext]
            std::int64_t stride = *rc::gen::inRange<std::int64_t>(1, ext + 1);
            // offset in [0, ext-1]
            std::int64_t offset = *rc::gen::inRange<std::int64_t>(0, ext);
            // max extent so that offset + (extent-1)*stride < ext
            std::int64_t max_extent = (ext - 1 - offset) / stride + 1;
            std::int64_t extent = *rc::gen::inRange<std::int64_t>(1, max_extent + 1);
            bbox.offsets[d] = offset;
            bbox.extents[d] = extent;
            bbox.strides[d] = stride;
        }

        // Selected element count and byte count.
        std::size_t sel_elems = 1;
        for (int d = 0; d < rank; ++d) {
            sel_elems *= static_cast<std::size_t>(bbox.extents[d]);
        }
        std::size_t sel_bytes = sel_elems * elem;

        // Generate the full array with distinct per-element content so a
        // wrong gather is detectable.
        std::vector<std::byte> full(full_bytes);
        for (std::size_t i = 0; i < full_bytes; ++i) {
            full[i] = static_cast<std::byte>(*rc::gen::inRange(0, 256));
        }

        // Independently compute the expected slice (row-major gather).
        auto strides = row_major_strides(shape);
        std::vector<std::byte> expected(sel_bytes);
        {
            std::vector<std::int64_t> idx(static_cast<std::size_t>(rank), 0);
            std::size_t out_off = 0;
            for (std::size_t count = 0; count < sel_elems; ++count) {
                std::size_t src_elem = 0;
                for (int d = 0; d < rank; ++d) {
                    std::int64_t src_d = bbox.offsets[d] + idx[static_cast<std::size_t>(d)] * bbox.strides[d];
                    src_elem += static_cast<std::size_t>(src_d) * strides[static_cast<std::size_t>(d)];
                }
                std::memcpy(expected.data() + out_off, full.data() + src_elem * elem, elem);
                out_off += elem;
                for (int d = rank - 1; d >= 0; --d) {
                    if (++idx[static_cast<std::size_t>(d)] < bbox.extents[d]) break;
                    idx[static_cast<std::size_t>(d)] = 0;
                }
            }
        }

        // Set up the read path with the slicing driver.
        std::size_t buffer_capacity = std::max(full_bytes, std::size_t{64});
        StagingPool pool(4, buffer_capacity, 5000);

        auto driver = std::make_shared<SliceDriver>();
        driver->set_array(shape, dtype, full);

        auto info = make_var_info(dtype, shape, 1);
        PrefetchQueue pq(1, 60, &pool, nullptr, driver.get(), 1, "test_var", info, 1);

        StagingBuffer *buf = nullptr;
        amio_status_t status = pq.get_buffer(0, &bbox, &buf);
        RC_ASSERT(status == AMIO_OK);
        RC_ASSERT(buf != nullptr);

        // used_bytes == element_size * product(extents).
        RC_ASSERT(buf->used_bytes == sel_bytes);

        // Returned bytes equal the independently computed slice.
        RC_ASSERT(std::memcmp(buf->data, expected.data(), sel_bytes) == 0);

        pool.release(buf);
    });

    REQUIRE(result);
}
