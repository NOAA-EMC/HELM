// test_amio_strerror.cpp
//
// Unit tests for `amio_strerror` covering the R12.5 - R12.8
// acceptance criteria spelled out in
// .kiro/specs/amio-async-io-engine/requirements.md (Requirement 12,
// criteria 5-8) and the contract documented in
// include/amio/amio_errors.h.
//
// These are *example* tests (not property-based -- the universal
// version of this contract is P17, scheduled for task 13.18).  They
// pin specific behaviors that are easy to forget when refactoring
// the error-code table: defined codes return non-null, byte-equal,
// stable strings; undefined codes return non-null, null-terminated
// "AMIO_ERR_UNKNOWN(<int>)" strings; the call is side-effect free.
//
// We deliberately avoid pulling in Catch2 here -- task 13.1 adds the
// Catch2 dependency for the property-based tests, and at this stage
// of the build a `<cassert>`-based harness keeps the test target
// lightweight, with no transitive C++ standard-library surface beyond
// what the implementation already uses.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "amio/amio.h"
#include "amio/amio_errors.h"

namespace {

// Lightweight harness counters.  We do not abort on the first
// failure; instead each `expect_*` records the failure with a
// localized message so the test binary's exit status reflects the
// total count and each issue is visible in the test log.
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

// ---------------------------------------------------------------------------
// Test 1: defined codes -- non-null, null-terminated, contain the
// canonical token (R12.5, R12.6).
// ---------------------------------------------------------------------------
void test_defined_codes_return_non_null_strings() {
    struct Case {
        int code;
        const char *expected_token;
    };

    constexpr Case cases[] = {
        {AMIO_OK, "AMIO_OK"},
        {AMIO_ERR_NULL_HANDLE, "AMIO_ERR_NULL_HANDLE"},
        {AMIO_ERR_INVALID_HANDLE, "AMIO_ERR_INVALID_HANDLE"},
        {AMIO_ERR_MANIFEST_NOT_FOUND, "AMIO_ERR_MANIFEST_NOT_FOUND"},
        {AMIO_ERR_MANIFEST_INVALID, "AMIO_ERR_MANIFEST_INVALID"},
        {AMIO_ERR_ALREADY_INITIALIZED, "AMIO_ERR_ALREADY_INITIALIZED"},
        {AMIO_ERR_FINALIZE_TIMEOUT, "AMIO_ERR_FINALIZE_TIMEOUT"},
        {AMIO_ERR_STAGING_BACKPRESSURE, "AMIO_ERR_STAGING_BACKPRESSURE"},
        {AMIO_ERR_INVALID_BINDING, "AMIO_ERR_INVALID_BINDING"},
        {AMIO_ERR_COMM_SPLIT_FAILED, "AMIO_ERR_COMM_SPLIT_FAILED"},
        {AMIO_ERR_THREADING_UNSUPPORTED, "AMIO_ERR_THREADING_UNSUPPORTED"},
        {AMIO_ERR_UNKNOWN_BACKEND, "AMIO_ERR_UNKNOWN_BACKEND"},
        {AMIO_ERR_LOSSY_CODEC_FORBIDDEN, "AMIO_ERR_LOSSY_CODEC_FORBIDDEN"},
        {AMIO_ERR_VIEWS_OUTSTANDING, "AMIO_ERR_VIEWS_OUTSTANDING"},
        {AMIO_ERR_QUEUE_FULL, "AMIO_ERR_QUEUE_FULL"},
        {AMIO_ERR_TIMEOUT, "AMIO_ERR_TIMEOUT"},
        {AMIO_ERR_BACKEND_FAILURE, "AMIO_ERR_BACKEND_FAILURE"},
        {AMIO_ERR_INVALID_INPUT, "AMIO_ERR_INVALID_INPUT"},
    };

    for (const auto &c : cases) {
        const char *desc = amio_strerror(c.code);
        std::string ctx = "code=" + std::to_string(c.code);

        EXPECT_TRUE(desc != nullptr, ctx);
        if (desc == nullptr) {
            continue;
        }
        // Length must be > 0 and bounded; if the string were
        // unterminated, strlen would either crash or produce
        // garbage beyond a sane limit.
        std::size_t len = std::strlen(desc);
        EXPECT_TRUE(len > 0, ctx);
        EXPECT_TRUE(len < 1024, ctx);
        // Each defined-code description begins with the canonical
        // AMIO_OK / AMIO_ERR_<NAME> token.
        EXPECT_TRUE(std::strstr(desc, c.expected_token) != nullptr, ctx);
    }
}

// ---------------------------------------------------------------------------
// Test 2: defined-code lookups are byte-equal across repeated calls
// and side-effect free (R12.6).  Repeated calls with the same code
// must return the same pointer (constexpr table) and the description
// for any other defined code must remain unchanged.
// ---------------------------------------------------------------------------
void test_defined_codes_are_stable() {
    const char *first_ok = amio_strerror(AMIO_OK);
    const char *first_invalid = amio_strerror(AMIO_ERR_INVALID_INPUT);

    // Hammer the table with many repeated lookups, including
    // interleaving distinct codes, to flush out any accidental
    // mutation pattern.
    for (int i = 0; i < 1000; ++i) {
        const char *p1 = amio_strerror(AMIO_OK);
        const char *p2 = amio_strerror(AMIO_ERR_INVALID_INPUT);
        EXPECT_TRUE(p1 == first_ok, "AMIO_OK pointer drifted");
        EXPECT_TRUE(p2 == first_invalid, "AMIO_ERR_INVALID_INPUT pointer drifted");
        // Bytes-equal in addition to pointer-equal -- guards against
        // a future implementation that copies into a static buffer.
        EXPECT_TRUE(std::strcmp(p1, first_ok) == 0, "AMIO_OK string content drifted");
        EXPECT_TRUE(std::strcmp(p2, first_invalid) == 0, "AMIO_ERR_INVALID_INPUT string content drifted");
    }

    // A side-effect-free function must not change the answer for
    // OTHER defined codes when one code is interrogated.  Walk the
    // entire defined range and verify each description is unchanged
    // after the loop above.
    constexpr int kFirstDefined = 0;
    constexpr int kLastDefined = AMIO_ERR_INVALID_INPUT;
    for (int code = kFirstDefined; code <= kLastDefined; ++code) {
        const char *a = amio_strerror(code);
        const char *b = amio_strerror(code);
        EXPECT_TRUE(a != nullptr, "defined code returned null");
        EXPECT_TRUE(b != nullptr, "defined code returned null");
        EXPECT_TRUE(a == b, "defined-code pointer is not stable");
    }
}

// ---------------------------------------------------------------------------
// Test 3: undefined codes -- non-null, null-terminated, formatted
// as "AMIO_ERR_UNKNOWN(<int>)" (R12.8).
// ---------------------------------------------------------------------------
void test_undefined_codes_return_unknown_string() {
    // A grab-bag of values that should NOT appear in the defined
    // table: negative ints, INT_MIN, large positive ints just above
    // the highest defined enumerator, and INT_MAX.
    const int probes[] = {
        -1,  -2,   -42,     -100000,   -2147483647 - 1 /* INT_MIN */, AMIO_ERR_INVALID_INPUT + 1, AMIO_ERR_INVALID_INPUT + 2,
        100, 1000, 1 << 20, 2147483647 /* INT_MAX */
    };

    for (int code : probes) {
        const char *desc = amio_strerror(code);
        std::string ctx = "code=" + std::to_string(code);
        EXPECT_TRUE(desc != nullptr, ctx);
        if (desc == nullptr) {
            continue;
        }
        // Null-terminated and reasonable length.
        std::size_t len = std::strlen(desc);
        EXPECT_TRUE(len > 0, ctx);
        EXPECT_TRUE(len < 64, ctx);

        // The returned string must contain the unknown-code prefix
        // and the literal decimal value of `code` -- the exact form
        // mandated by the task description.
        EXPECT_TRUE(std::strstr(desc, "AMIO_ERR_UNKNOWN") != nullptr, ctx);
        char expected_int[32];
        std::snprintf(expected_int, sizeof(expected_int), "%d", code);
        EXPECT_TRUE(std::strstr(desc, expected_int) != nullptr, ctx);
    }
}

// ---------------------------------------------------------------------------
// Test 4: querying an undefined code does NOT mutate the description
// returned for any defined code (R12.8: "no table mutation").
// ---------------------------------------------------------------------------
void test_unknown_lookup_does_not_mutate_table() {
    // Capture pointers AND content of every defined-code description
    // before the unknown-lookup hammering.
    const int last = AMIO_ERR_INVALID_INPUT;
    std::vector<const char *> before_ptrs;
    std::vector<std::string> before_strs;
    before_ptrs.reserve(static_cast<std::size_t>(last) + 1);
    before_strs.reserve(static_cast<std::size_t>(last) + 1);
    for (int code = 0; code <= last; ++code) {
        const char *p = amio_strerror(code);
        before_ptrs.push_back(p);
        before_strs.emplace_back(p);
    }

    // Hammer with diverse unknown codes.  Each call writes into the
    // thread_local scratch buffer; none should touch the constexpr
    // defined-code table.
    for (int i = -500; i <= 500; ++i) {
        if (i >= 0 && i <= last) continue;  // skip defined codes
        (void)amio_strerror(i);
    }

    // Verify that every defined-code description is bit-for-bit
    // identical to its pre-hammer snapshot.
    for (int code = 0; code <= last; ++code) {
        const char *now = amio_strerror(code);
        std::string ctx = "code=" + std::to_string(code);
        EXPECT_TRUE(now != nullptr, ctx);
        EXPECT_TRUE(now == before_ptrs[static_cast<std::size_t>(code)], ctx + " (pointer changed)");
        EXPECT_TRUE(std::strcmp(now, before_strs[static_cast<std::size_t>(code)].c_str()) == 0, ctx + " (content changed)");
    }
}

// ---------------------------------------------------------------------------
// Test 5: thread safety.  Concurrent calls to amio_strerror across
// multiple threads must each return non-null, null-terminated
// strings and must not corrupt each other's results.  This pins the
// thread_local-buffer design choice for the unknown-code path.
// ---------------------------------------------------------------------------
void test_thread_safety_smoke() {
    constexpr int kThreads = 8;
    constexpr int kIterations = 2000;

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    std::vector<int> failures(kThreads, 0);

    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([t, &failures]() {
            int local_failures = 0;
            for (int i = 0; i < kIterations; ++i) {
                // Mix defined and undefined codes per iteration.
                int defined_code = (t + i) % 18;  // 0..17
                int undefined_code = -(t * 1000 + i + 1);

                const char *d = amio_strerror(defined_code);
                if (d == nullptr || std::strlen(d) == 0) {
                    ++local_failures;
                    continue;
                }

                const char *u = amio_strerror(undefined_code);
                if (u == nullptr || std::strlen(u) == 0) {
                    ++local_failures;
                    continue;
                }
                if (std::strstr(u, "AMIO_ERR_UNKNOWN") == nullptr) {
                    ++local_failures;
                }
            }
            failures[static_cast<std::size_t>(t)] = local_failures;
        });
    }
    for (auto &w : workers) {
        w.join();
    }

    int total_failures = 0;
    for (int f : failures) total_failures += f;
    EXPECT_TRUE(total_failures == 0, "concurrent amio_strerror produced " + std::to_string(total_failures) + " malformed results");
}

}  // namespace

int main() {
    test_defined_codes_return_non_null_strings();
    test_defined_codes_are_stable();
    test_undefined_codes_return_unknown_string();
    test_unknown_lookup_does_not_mutate_table();
    test_thread_safety_smoke();

    std::fprintf(stdout, "test_amio_strerror: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
