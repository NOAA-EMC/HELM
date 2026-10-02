// test_p17_error_code_totality.cpp -- Property test P17: Error code totality
// and strerror correctness.
//
// For any API call with arbitrary inputs: return is AMIO_OK or defined
// AMIO_ERR_*; for any int i to amio_strerror(i): non-null null-terminated
// string; defined codes → byte-equal across invocations, side-effect free;
// undefined codes → "unknown" description, non-null, no table mutation.
//
// Min 100 iterations.
//
// **Validates: Requirements R12.5, R12.6, R12.8**

#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "generators.hpp"
#include "pbt_common.hpp"

using namespace amio::pbt;

// ===================================================================
// The set of all defined AMIO_ERR_* codes (from amio_errors.h).
// ===================================================================

namespace {

// All defined error codes as a vector for iteration.
const std::vector<int> kDefinedCodes = {
    AMIO_OK,                         //  0
    AMIO_ERR_NULL_HANDLE,            //  1
    AMIO_ERR_INVALID_HANDLE,         //  2
    AMIO_ERR_MANIFEST_NOT_FOUND,     //  3
    AMIO_ERR_MANIFEST_INVALID,       //  4
    AMIO_ERR_ALREADY_INITIALIZED,    //  5
    AMIO_ERR_FINALIZE_TIMEOUT,       //  6
    AMIO_ERR_STAGING_BACKPRESSURE,   //  7
    AMIO_ERR_INVALID_BINDING,        //  8
    AMIO_ERR_COMM_SPLIT_FAILED,      //  9
    AMIO_ERR_THREADING_UNSUPPORTED,  // 10
    AMIO_ERR_UNKNOWN_BACKEND,        // 11
    AMIO_ERR_LOSSY_CODEC_FORBIDDEN,  // 12
    AMIO_ERR_VIEWS_OUTSTANDING,      // 13
    AMIO_ERR_QUEUE_FULL,             // 14
    AMIO_ERR_TIMEOUT,                // 15
    AMIO_ERR_BACKEND_FAILURE,        // 16
    AMIO_ERR_INVALID_INPUT           // 17
};

// The set of defined codes for O(1) lookup.
const std::set<int> kDefinedCodeSet(kDefinedCodes.begin(), kDefinedCodes.end());

// Check if a code is in the defined set.
bool is_defined_code(int code) {
    return kDefinedCodeSet.count(code) > 0;
}

}  // anonymous namespace

// ===================================================================
// Property Test P17a: amio_strerror returns non-null, null-terminated
// string for ANY integer input.
//
// For any int i in a wide range: amio_strerror(i) returns a non-null
// pointer to a null-terminated string.
// ===================================================================

TEST_CASE("P17: Error code totality - strerror always returns non-null", "[pbt][p17][error_code][strerror][totality]") {
    auto result = rc::check("amio_strerror(i) returns non-null null-terminated string for any int", []() {
        // Generate arbitrary int values across a wide range.
        auto code = *rc::gen::inRange(-1000, 1000);

        const char *desc = amio_strerror(code);

        // Must be non-null.
        RC_ASSERT(desc != nullptr);

        // Must be null-terminated (strlen must not crash and
        // must return a finite value).
        std::size_t len = std::strlen(desc);
        RC_ASSERT(len > 0);  // description is never empty
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P17b: Defined codes return byte-equal descriptions
// across repeated invocations (idempotency / side-effect free).
//
// For any defined AMIO_ERR_* code: calling amio_strerror twice
// returns the same pointer (or at minimum byte-equal content).
// ===================================================================

TEST_CASE("P17: Error code totality - defined codes byte-equal across calls", "[pbt][p17][error_code][strerror][idempotent]") {
    auto result = rc::check("defined codes return byte-equal descriptions across invocations", []() {
        // Pick a defined code.
        auto code = *rc::gen::elementOf(kDefinedCodes);

        // Call amio_strerror twice.
        const char *desc1 = amio_strerror(code);
        const char *desc2 = amio_strerror(code);

        // Both must be non-null.
        RC_ASSERT(desc1 != nullptr);
        RC_ASSERT(desc2 != nullptr);

        // For defined codes, the pointer should be the same
        // (pointing to the same static string literal).
        RC_ASSERT(desc1 == desc2);

        // Content must be byte-equal.
        RC_ASSERT(std::strcmp(desc1, desc2) == 0);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P17c: Undefined codes return "AMIO_ERR_UNKNOWN(<int>)"
// description, non-null, no table mutation.
//
// For any int NOT in the defined set: amio_strerror returns a string
// containing "AMIO_ERR_UNKNOWN(" and the decimal representation of
// the code.
// ===================================================================

TEST_CASE("P17: Error code totality - undefined codes return unknown description", "[pbt][p17][error_code][strerror][undefined]") {
    auto result = rc::check("undefined codes return AMIO_ERR_UNKNOWN(<int>) description", []() {
        // Generate an int that is NOT a defined code.
        // Strategy: generate from ranges that don't overlap with [0, 17].
        auto code = *rc::gen::suchThat<int>(rc::gen::inRange(-500, 500), [](int c) { return !is_defined_code(c); });

        const char *desc = amio_strerror(code);

        // Must be non-null.
        RC_ASSERT(desc != nullptr);

        // Must be null-terminated.
        std::size_t len = std::strlen(desc);
        RC_ASSERT(len > 0);

        // Must contain "AMIO_ERR_UNKNOWN(".
        std::string desc_str(desc);
        RC_ASSERT(desc_str.find("AMIO_ERR_UNKNOWN(") != std::string::npos);

        // Must contain the decimal representation of the code.
        std::string code_str = std::to_string(code);
        RC_ASSERT(desc_str.find(code_str) != std::string::npos);
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P17d: Calling amio_strerror with undefined codes does
// NOT mutate the defined-code table.
//
// After calling amio_strerror with many undefined codes, all defined
// codes still return their original descriptions unchanged.
// ===================================================================

TEST_CASE("P17: Error code totality - undefined codes do not mutate table", "[pbt][p17][error_code][strerror][no_mutation]") {
    auto result = rc::check("undefined code lookups do not mutate the defined-code table", []() {
        // Snapshot all defined-code descriptions.
        std::vector<std::string> before;
        for (int code : kDefinedCodes) {
            before.push_back(std::string(amio_strerror(code)));
        }

        // Generate and call amio_strerror with several undefined codes.
        auto num_undefined = *rc::gen::inRange<int>(5, 20);
        for (int i = 0; i < num_undefined; ++i) {
            int bad_code = -100 - i;  // guaranteed undefined
            const char *desc = amio_strerror(bad_code);
            RC_ASSERT(desc != nullptr);
        }

        // Also try large positive undefined codes.
        for (int i = 0; i < num_undefined; ++i) {
            int bad_code = 100 + i;  // guaranteed undefined
            const char *desc = amio_strerror(bad_code);
            RC_ASSERT(desc != nullptr);
        }

        // Verify: all defined-code descriptions are unchanged.
        for (std::size_t i = 0; i < kDefinedCodes.size(); ++i) {
            const char *after = amio_strerror(kDefinedCodes[i]);
            RC_ASSERT(after != nullptr);
            RC_ASSERT(std::string(after) == before[i]);
        }
    });

    REQUIRE(result);
}

// ===================================================================
// Property Test P17e: API calls return only AMIO_OK or defined
// AMIO_ERR_* codes.
//
// For any API call with arbitrary (potentially invalid) inputs:
// the return value is always in the defined set.
// ===================================================================

TEST_CASE("P17: Error code totality - API returns only defined codes", "[pbt][p17][error_code][api_returns]") {
    auto result = rc::check("API calls return only AMIO_OK or defined AMIO_ERR_* codes", []() {
        // Exercise various API calls with invalid inputs and
        // verify the return code is always in the defined set.

        // amio_init with null path.
        {
            amio_core_handle core = nullptr;
            amio_status_t rc_val = amio_init(nullptr, &core);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_init with null output pointer.
        {
            amio_status_t rc_val = amio_init("nonexistent.yaml", nullptr);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_finalize with null handle.
        {
            amio_status_t rc_val = amio_finalize(nullptr);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_write with null dataset.
        {
            amio_io_handle io = nullptr;
            amio_status_t rc_val = amio_write(nullptr, "var", nullptr, AMIO_DTYPE_F32, nullptr, &io);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_read with null dataset.
        {
            amio_view_handle view = nullptr;
            amio_status_t rc_val = amio_read(nullptr, "var", 0, nullptr, &view);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_flush with null dataset.
        {
            amio_status_t rc_val = amio_flush(nullptr, 1000);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_wait with null handle.
        {
            amio_status_t rc_val = amio_wait(nullptr, 1000);
            RC_ASSERT(is_defined_code(rc_val));
        }

        // amio_release_view with null handle.
        {
            amio_status_t rc_val = amio_release_view(nullptr);
            RC_ASSERT(is_defined_code(rc_val));
        }
    });

    REQUIRE(result);
}
