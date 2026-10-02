// test_p24_grib2_contiguity_gating.cpp -- Property test P24: GRIB2
// contiguity gating.
//
// For any Memory_View to GRIB2_Driver::write: bytes to encoder ==
// row-major flatten; if contiguous+row-major → encoder receives
// underlying pointer directly (no intermediate buffer); otherwise →
// encoder receives contiguous packed buffer.
//
// Min 100 iterations with mixed contiguous/strided views.
//
// **Validates: Requirements R9.4, R9.5**

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "drivers/grib2/grib2_driver.hpp"
#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::detail;
using namespace amio::pbt;

// ===================================================================
// We test the GRIB2_Driver's static helper methods directly:
//   - is_contiguous_row_major(shape)
//   - pack_row_major(src_data, shape, element_size)
//
// These are the core of the contiguity gating logic (R9.4, R9.5).
// Since they are private static methods, we access them through a
// test-friend pattern or by compiling grib2_driver.cpp directly
// into this test (which gives us access to the amio::detail namespace).
// ===================================================================

// ===================================================================
// Helper: compute expected row-major strides for a given shape.
// ===================================================================

namespace {

std::vector<int64_t> compute_row_major_strides(const amio_shape_t &shape) {
    std::vector<int64_t> strides(shape.rank, 0);
    if (shape.rank <= 0) return strides;

    int64_t stride = 1;
    for (int32_t d = shape.rank - 1; d >= 0; --d) {
        strides[d] = stride;
        stride *= shape.extents[d];
    }
    return strides;
}

// Generate a shape with explicit row-major strides (contiguous).
amio_shape_t make_contiguous_row_major_shape(int32_t rank, const std::vector<int64_t> &extents) {
    amio_shape_t shape = {};
    shape.rank = rank;
    for (int32_t d = 0; d < rank; ++d) {
        shape.extents[d] = extents[d];
    }
    // Strides = 0 means contiguous row-major by convention.
    return shape;
}

// Generate a shape with non-contiguous strides (strided view).
amio_shape_t make_strided_shape(int32_t rank, const std::vector<int64_t> &extents, const std::vector<int64_t> &strides) {
    amio_shape_t shape = {};
    shape.rank = rank;
    for (int32_t d = 0; d < rank; ++d) {
        shape.extents[d] = extents[d];
        shape.strides[d] = strides[d];
    }
    return shape;
}

// Compute total elements from shape.
std::size_t total_elements(const amio_shape_t &shape) {
    if (shape.rank <= 0 || shape.rank > AMIO_MAX_RANK) return 0;
    std::size_t count = 1;
    for (int32_t d = 0; d < shape.rank; ++d) {
        if (shape.extents[d] <= 0) return 0;
        count *= static_cast<std::size_t>(shape.extents[d]);
    }
    return count;
}

}  // anonymous namespace

// ===================================================================
// Property Test P24a: Contiguous row-major shapes are detected correctly.
//
// For any valid shape with strides == 0 (convention for contiguous
// row-major): is_contiguous_row_major returns true.
// ===================================================================

TEST_CASE("P24: GRIB2 contiguity gating - contiguous shapes detected", "[pbt][p24][grib2][contiguity][contiguous]") {
    auto result = rc::check("shapes with strides==0 are detected as contiguous row-major", []() {
        // Generate a valid shape (strides default to 0 = contiguous).
        auto shape = *rc::gen::arbitrary<amio_shape_t>();

        // Ensure strides are all zero (contiguous convention).
        for (int32_t d = 0; d < AMIO_MAX_RANK; ++d) {
            shape.strides[d] = 0;
        }

        // Should be detected as contiguous row-major.
        RC_ASSERT(GRIB2_Driver::is_contiguous_row_major(shape) == true);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P24b: Explicit row-major strides are detected correctly.
//
// For any valid shape with explicit strides matching row-major layout:
// is_contiguous_row_major returns true.
// ===================================================================

TEST_CASE("P24: GRIB2 contiguity gating - explicit row-major strides", "[pbt][p24][grib2][contiguity][explicit_strides]") {
    auto result = rc::check("explicit row-major strides are detected as contiguous", []() {
        auto shape = *rc::gen::arbitrary<amio_shape_t>();

        // Set explicit row-major strides.
        auto strides = compute_row_major_strides(shape);
        for (int32_t d = 0; d < shape.rank; ++d) {
            shape.strides[d] = strides[d];
        }

        RC_ASSERT(GRIB2_Driver::is_contiguous_row_major(shape) == true);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P24c: Non-contiguous strides are detected correctly.
//
// For any valid shape with strides that do NOT match row-major layout:
// is_contiguous_row_major returns false.
// ===================================================================

TEST_CASE("P24: GRIB2 contiguity gating - non-contiguous strides detected", "[pbt][p24][grib2][contiguity][non_contiguous]") {
    auto result = rc::check("non-contiguous strides are detected as non-row-major", []() {
        // Generate a shape with rank >= 2 so we can create
        // non-trivial non-contiguous strides.
        amio_shape_t shape = {};
        shape.rank = *rc::gen::inRange(2, 5);

        for (int32_t d = 0; d < shape.rank; ++d) {
            shape.extents[d] = *rc::gen::inRange<int64_t>(2, 16);
        }

        // Compute row-major strides, then perturb one to make
        // it non-contiguous.
        auto strides = compute_row_major_strides(shape);
        for (int32_t d = 0; d < shape.rank; ++d) {
            shape.strides[d] = strides[d];
        }

        // Multiply one stride by a factor > 1 to create a gap.
        int32_t perturb_dim = *rc::gen::inRange(static_cast<int32_t>(0), shape.rank - 1);
        int64_t factor = *rc::gen::inRange<int64_t>(2, 5);
        shape.strides[perturb_dim] *= factor;

        RC_ASSERT(GRIB2_Driver::is_contiguous_row_major(shape) == false);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P24d: pack_row_major produces correct row-major flatten.
//
// For any (data, shape with non-contiguous strides, element_size):
// pack_row_major produces a buffer whose content equals the row-major
// traversal of the source data.
// ===================================================================

TEST_CASE("P24: GRIB2 contiguity gating - pack_row_major correctness", "[pbt][p24][grib2][contiguity][pack_correctness]") {
    auto result = rc::check("pack_row_major produces correct row-major flatten", []() {
        // Generate a small shape (rank 2-3) for tractable testing.
        amio_shape_t shape = {};
        shape.rank = *rc::gen::inRange(2, 4);

        for (int32_t d = 0; d < shape.rank; ++d) {
            shape.extents[d] = *rc::gen::inRange<int64_t>(2, 8);
        }

        // Use element_size = 4 (float-like).
        const std::size_t elem_size = 4;
        std::size_t num_elements = total_elements(shape);
        RC_PRE(num_elements > 0 && num_elements <= 512);

        // Create a strided source buffer.  We'll use strides that
        // are larger than row-major (simulating a subview of a
        // larger array).
        auto rm_strides = compute_row_major_strides(shape);

        // Create "padded" strides: multiply the first dimension's
        // stride by 2 to simulate a view into a larger array.
        std::vector<int64_t> padded_strides = rm_strides;
        padded_strides[0] *= 2;  // double the leading stride

        for (int32_t d = 0; d < shape.rank; ++d) {
            shape.strides[d] = padded_strides[d];
        }

        // Allocate source buffer large enough for the padded layout.
        std::size_t src_total_elements = static_cast<std::size_t>(padded_strides[0]) * static_cast<std::size_t>(shape.extents[0]);
        std::size_t src_bytes = src_total_elements * elem_size;
        std::vector<std::byte> src_data(src_bytes);

        // Fill with sequential byte patterns for verification.
        for (std::size_t i = 0; i < src_bytes; ++i) {
            src_data[i] = static_cast<std::byte>(i & 0xFF);
        }

        // Pack using GRIB2_Driver::pack_row_major.
        auto packed = GRIB2_Driver::pack_row_major(src_data.data(), shape, elem_size);

        // Verify size.
        RC_ASSERT(packed.size() == num_elements * elem_size);

        // Verify content: iterate in row-major order and check
        // each element matches the source at the strided offset.
        std::vector<int64_t> indices(shape.rank, 0);
        for (std::size_t elem = 0; elem < num_elements; ++elem) {
            // Compute source offset from indices and strides.
            std::size_t src_elem_offset = 0;
            for (int32_t d = 0; d < shape.rank; ++d) {
                src_elem_offset += static_cast<std::size_t>(indices[d]) * static_cast<std::size_t>(padded_strides[d]);
            }
            std::size_t src_byte_offset = src_elem_offset * elem_size;
            std::size_t dst_byte_offset = elem * elem_size;

            // Compare element bytes.
            RC_ASSERT(std::memcmp(packed.data() + dst_byte_offset, src_data.data() + src_byte_offset, elem_size) == 0);

            // Increment multi-dimensional index (row-major).
            for (int32_t d = shape.rank - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape.extents[d]) break;
                indices[d] = 0;
            }
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P24e: Contiguous data passes through without packing.
//
// For any contiguous row-major shape: is_contiguous_row_major returns
// true, meaning the encoder would receive the underlying pointer
// directly (no intermediate buffer needed).  We verify that
// pack_row_major on contiguous data produces an identical copy.
// ===================================================================

TEST_CASE("P24: GRIB2 contiguity gating - contiguous data identity", "[pbt][p24][grib2][contiguity][identity]") {
    auto result = rc::check("pack_row_major on contiguous data produces identical output", []() {
        auto shape = *rc::gen::arbitrary<amio_shape_t>();

        // Ensure contiguous (strides = 0).
        for (int32_t d = 0; d < AMIO_MAX_RANK; ++d) {
            shape.strides[d] = 0;
        }

        std::size_t num_elements = total_elements(shape);
        RC_PRE(num_elements > 0 && num_elements <= 4096);

        const std::size_t elem_size = 4;  // float
        std::size_t total_bytes = num_elements * elem_size;

        // Generate random source data.
        std::vector<std::byte> src_data(total_bytes);
        for (std::size_t i = 0; i < total_bytes; ++i) {
            src_data[i] = static_cast<std::byte>(*rc::gen::inRange(0, 256));
        }

        // Verify is_contiguous_row_major.
        RC_ASSERT(GRIB2_Driver::is_contiguous_row_major(shape) == true);

        // pack_row_major should produce an identical copy for
        // contiguous data (since strides=0 means row-major).
        auto packed = GRIB2_Driver::pack_row_major(src_data.data(), shape, elem_size);

        RC_ASSERT(packed.size() == total_bytes);
        RC_ASSERT(std::memcmp(packed.data(), src_data.data(), total_bytes) == 0);
    });

    REQUIRE(result);
}
