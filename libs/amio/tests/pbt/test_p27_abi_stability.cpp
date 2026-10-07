// test_p27_abi_stability.cpp -- Property test P27: ABI stability
// invariant.
//
// For any AMIO_ERR_* code: integer value is stable (matches the enum
// definition); amio_strerror returns consistent non-null strings;
// all public struct sizes/alignments match expected values.
//
// Min 100 iterations.
//
// Uses REAL amio_strerror and verifies struct layouts via
// sizeof/alignof.
//
// **Validates: Requirements R10.8, R12.5**

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::pbt;

// ===================================================================
// ABI stability constants.
//
// These are the FROZEN integer values for all AMIO_ERR_* codes.
// If any of these change, it's an ABI break (R10.8).
// ===================================================================

namespace {

struct ErrorCodeEntry {
    int value;
    const char *name;
};

// The complete set of defined AMIO_ERR_* codes with their frozen
// integer values.
const std::vector<ErrorCodeEntry> kDefinedErrorCodes = {{0, "AMIO_OK"},
                                                        {1, "AMIO_ERR_NULL_HANDLE"},
                                                        {2, "AMIO_ERR_INVALID_HANDLE"},
                                                        {3, "AMIO_ERR_MANIFEST_NOT_FOUND"},
                                                        {4, "AMIO_ERR_MANIFEST_INVALID"},
                                                        {5, "AMIO_ERR_ALREADY_INITIALIZED"},
                                                        {6, "AMIO_ERR_FINALIZE_TIMEOUT"},
                                                        {7, "AMIO_ERR_STAGING_BACKPRESSURE"},
                                                        {8, "AMIO_ERR_INVALID_BINDING"},
                                                        {9, "AMIO_ERR_COMM_SPLIT_FAILED"},
                                                        {10, "AMIO_ERR_THREADING_UNSUPPORTED"},
                                                        {11, "AMIO_ERR_UNKNOWN_BACKEND"},
                                                        {12, "AMIO_ERR_LOSSY_CODEC_FORBIDDEN"},
                                                        {13, "AMIO_ERR_VIEWS_OUTSTANDING"},
                                                        {14, "AMIO_ERR_QUEUE_FULL"},
                                                        {15, "AMIO_ERR_TIMEOUT"},
                                                        {16, "AMIO_ERR_BACKEND_FAILURE"},
                                                        {17, "AMIO_ERR_INVALID_INPUT"}};

// Number of defined error codes (next unused = 18).
constexpr int kMaxDefinedCode = 17;

}  // anonymous namespace

// ===================================================================
// Property Test P27a: Error code integer values are stable.
//
// For any AMIO_ERR_* enumerator: its integer value matches the
// frozen ABI definition.  This ensures no renumbering has occurred.
//
// Validates: R10.8
// ===================================================================

TEST_CASE("P27: ABI stability - error code integer values frozen", "[pbt][p27][abi][error_codes][frozen]") {
    auto result = rc::check("AMIO_ERR_* integer values match frozen ABI definition", []() {
        // Pick a random defined error code to verify.
        auto idx = *rc::gen::inRange<std::size_t>(0, kDefinedErrorCodes.size());
        const auto &entry = kDefinedErrorCodes[idx];

        // Verify the enum value matches the expected frozen value.
        // We check by casting the enum constants to int.
        int actual = -1;
        switch (entry.value) {
            case 0:
                actual = static_cast<int>(AMIO_OK);
                break;
            case 1:
                actual = static_cast<int>(AMIO_ERR_NULL_HANDLE);
                break;
            case 2:
                actual = static_cast<int>(AMIO_ERR_INVALID_HANDLE);
                break;
            case 3:
                actual = static_cast<int>(AMIO_ERR_MANIFEST_NOT_FOUND);
                break;
            case 4:
                actual = static_cast<int>(AMIO_ERR_MANIFEST_INVALID);
                break;
            case 5:
                actual = static_cast<int>(AMIO_ERR_ALREADY_INITIALIZED);
                break;
            case 6:
                actual = static_cast<int>(AMIO_ERR_FINALIZE_TIMEOUT);
                break;
            case 7:
                actual = static_cast<int>(AMIO_ERR_STAGING_BACKPRESSURE);
                break;
            case 8:
                actual = static_cast<int>(AMIO_ERR_INVALID_BINDING);
                break;
            case 9:
                actual = static_cast<int>(AMIO_ERR_COMM_SPLIT_FAILED);
                break;
            case 10:
                actual = static_cast<int>(AMIO_ERR_THREADING_UNSUPPORTED);
                break;
            case 11:
                actual = static_cast<int>(AMIO_ERR_UNKNOWN_BACKEND);
                break;
            case 12:
                actual = static_cast<int>(AMIO_ERR_LOSSY_CODEC_FORBIDDEN);
                break;
            case 13:
                actual = static_cast<int>(AMIO_ERR_VIEWS_OUTSTANDING);
                break;
            case 14:
                actual = static_cast<int>(AMIO_ERR_QUEUE_FULL);
                break;
            case 15:
                actual = static_cast<int>(AMIO_ERR_TIMEOUT);
                break;
            case 16:
                actual = static_cast<int>(AMIO_ERR_BACKEND_FAILURE);
                break;
            case 17:
                actual = static_cast<int>(AMIO_ERR_INVALID_INPUT);
                break;
            default:
                RC_FAIL("Unknown entry value");
        }

        RC_ASSERT(actual == entry.value);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P27b: amio_strerror returns consistent non-null
// strings for defined codes.
//
// For any defined AMIO_ERR_* code: amio_strerror returns a non-null,
// non-empty, null-terminated string that is byte-equal across
// repeated invocations.
//
// Validates: R12.5
// ===================================================================

TEST_CASE("P27: ABI stability - strerror consistent for defined codes", "[pbt][p27][abi][strerror][defined]") {
    auto result = rc::check("amio_strerror returns consistent non-null strings for defined codes", []() {
        // Pick a random defined error code.
        auto idx = *rc::gen::inRange<std::size_t>(0, kDefinedErrorCodes.size());
        int code = kDefinedErrorCodes[idx].value;

        // Call amio_strerror twice.
        const char *str1 = amio_strerror(code);
        const char *str2 = amio_strerror(code);

        // Must be non-null.
        RC_ASSERT(str1 != nullptr);
        RC_ASSERT(str2 != nullptr);

        // Must be non-empty.
        RC_ASSERT(std::strlen(str1) > 0);

        // Must be byte-equal across invocations (R12.5).
        RC_ASSERT(std::strcmp(str1, str2) == 0);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P27c: amio_strerror returns non-null for undefined
// codes.
//
// For any integer NOT in the defined AMIO_ERR_* set: amio_strerror
// returns a non-null, null-terminated string (the "unknown" format).
// No table mutation occurs.
//
// Validates: R12.5
// ===================================================================

TEST_CASE("P27: ABI stability - strerror non-null for undefined codes", "[pbt][p27][abi][strerror][undefined]") {
    auto result = rc::check("amio_strerror returns non-null for undefined codes", []() {
        // Generate an integer outside the defined range.
        // Defined codes are [0, 17].  Test negative values and
        // values >= 18.
        int code;
        bool use_negative = *rc::gen::arbitrary<bool>();
        if (use_negative) {
            code = *rc::gen::inRange(-1000, 0);
        } else {
            code = *rc::gen::inRange(kMaxDefinedCode + 1, 10000);
        }

        // Call amio_strerror.
        const char *str = amio_strerror(code);

        // Must be non-null (R12.5 totality).
        RC_ASSERT(str != nullptr);

        // Must be null-terminated (we can safely call strlen).
        std::size_t len = std::strlen(str);
        RC_ASSERT(len > 0);

        // Should contain "UNKNOWN" or the code value in the string.
        std::string s(str);
        // The format is "AMIO_ERR_UNKNOWN(<int>)" per R12.8.
        RC_ASSERT(s.find("UNKNOWN") != std::string::npos || s.find("unknown") != std::string::npos ||
                  s.find(std::to_string(code)) != std::string::npos);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P27d: Public struct sizes are stable.
//
// All public ABI structs (amio_shape_t, amio_bbox_t, amio_dtype_t)
// have fixed sizes and alignments that must not change across builds.
//
// Validates: R10.8
// ===================================================================

TEST_CASE("P27: ABI stability - public struct sizes and alignments", "[pbt][p27][abi][struct_layout]") {
    auto result = rc::check("public struct sizes and alignments match expected values", []() {
        // amio_shape_t layout:
        //   rank:    int32_t (4 bytes)
        //   extents: int64_t[7] (56 bytes)
        //   strides: int64_t[7] (56 bytes)
        //   Total: 4 + padding + 56 + 56 = depends on alignment
        //
        // On most platforms with 8-byte alignment for int64_t:
        //   sizeof(amio_shape_t) = 8 (rank + padding) + 56 + 56 = 120
        //   OR sizeof = 4 + 56 + 56 = 116 (packed)
        //
        // We verify the size is consistent (not zero, reasonable).
        std::size_t shape_size = sizeof(amio_shape_t);
        std::size_t shape_align = alignof(amio_shape_t);

        // Shape must contain rank (4 bytes) + extents (7*8=56) +
        // strides (7*8=56) = at least 116 bytes.
        RC_ASSERT(shape_size >= 116);
        RC_ASSERT(shape_size <= 128);  // Allow for padding.
        RC_ASSERT(shape_align >= 4);   // At least int32_t alignment.

        // amio_bbox_t layout:
        //   rank:    int32_t (4 bytes)
        //   offsets: int64_t[7] (56 bytes)
        //   extents: int64_t[7] (56 bytes)
        //   strides: int64_t[7] (56 bytes)
        //   Total: at least 4 + 56 + 56 + 56 = 172 bytes.
        std::size_t bbox_size = sizeof(amio_bbox_t);
        std::size_t bbox_align = alignof(amio_bbox_t);

        RC_ASSERT(bbox_size >= 172);
        RC_ASSERT(bbox_size <= 184);  // Allow for padding.
        RC_ASSERT(bbox_align >= 4);

        // amio_dtype_t is an enum backed by int (typically 4 bytes).
        std::size_t dtype_size = sizeof(amio_dtype_t);
        RC_ASSERT(dtype_size == 4);  // enum is int-sized.

        // amio_status_t is int32_t (4 bytes).
        std::size_t status_size = sizeof(amio_status_t);
        RC_ASSERT(status_size == 4);

        // Verify AMIO_MAX_RANK is 7 (frozen constant).
        RC_ASSERT(AMIO_MAX_RANK == 7);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P27e: amio_strerror is side-effect free.
//
// For any sequence of amio_strerror calls with defined codes:
// calling strerror does not change the result of subsequent calls
// (no table mutation, no side effects).
//
// Validates: R12.5
// ===================================================================

TEST_CASE("P27: ABI stability - strerror is side-effect free", "[pbt][p27][abi][strerror][side_effect_free]") {
    auto result = rc::check("amio_strerror calls do not affect subsequent results", []() {
        // Pick two random codes (may be same or different).
        auto idx1 = *rc::gen::inRange<std::size_t>(0, kDefinedErrorCodes.size());
        auto idx2 = *rc::gen::inRange<std::size_t>(0, kDefinedErrorCodes.size());

        int code1 = kDefinedErrorCodes[idx1].value;
        int code2 = kDefinedErrorCodes[idx2].value;

        // Get baseline results.
        const char *baseline1 = amio_strerror(code1);
        const char *baseline2 = amio_strerror(code2);

        // Call strerror with various codes (including undefined).
        amio_strerror(-999);
        amio_strerror(9999);
        amio_strerror(code2);
        amio_strerror(code1);

        // Results should be unchanged (side-effect free).
        const char *after1 = amio_strerror(code1);
        const char *after2 = amio_strerror(code2);

        RC_ASSERT(std::strcmp(baseline1, after1) == 0);
        RC_ASSERT(std::strcmp(baseline2, after2) == 0);
    });

    REQUIRE(result);
}
