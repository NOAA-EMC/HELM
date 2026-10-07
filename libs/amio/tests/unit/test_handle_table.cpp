// test_handle_table.cpp
//
// Unit tests for `amio::detail::HandleTable` covering the
// reuse-after-free / generation-counter contract that the
// C-Boundary relies on (R10.5, R10.6, R10.7).
//
// Because the HandleTable is private to the AMIO_Core build (its
// header lives under `src/c_boundary/`), this test target compiles
// `handle_table.cpp` directly into the test binary rather than
// linking against `libamio.so`.  That keeps the test self-contained
// and means the unit test can run in any build configuration --
// including the early-task scaffolding configurations where the
// public AMIO_C_API entry points return AMIO_ERR_BACKEND_FAILURE.
//
// Test scope:
//
//   * insert returns a non-zero, decodable token of the right
//     kind/payload pair (R10.5).
//   * lookup of a fresh token returns AMIO_OK and the original
//     payload pointer.
//   * lookup of a NULL handle returns AMIO_ERR_NULL_HANDLE.
//   * lookup with the wrong kind returns AMIO_ERR_INVALID_HANDLE.
//   * lookup of a released token returns AMIO_ERR_INVALID_HANDLE
//     (the reuse-after-free case the design pins).  R10.6, R10.7.
//   * reuse of a released slot mints a new generation, so the old
//     token *still* fails lookup even after another insert reuses
//     the slot.
//   * release of a stale token returns AMIO_ERR_INVALID_HANDLE
//     and does NOT corrupt the live token bound to the same slot.
//   * fabricated tokens with out-of-range slot indices fail with
//     AMIO_ERR_INVALID_HANDLE rather than crashing.
//   * the table is safe to use concurrently from multiple threads
//     (smoke-level coverage; the tighter property is covered by
//     P4 in task 13.5).

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "amio/amio_errors.h"
#include "c_boundary/handle_table.hpp"

namespace {

using amio::detail::HandleKind;
using amio::detail::HandleTable;

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

void test_insert_returns_nonzero_token_with_correct_payload() {
    HandleTable t;
    int payload_a = 42;
    int payload_b = 99;

    auto tok_a = t.insert(HandleKind::Core, &payload_a);
    auto tok_b = t.insert(HandleKind::Dataset, &payload_b);

    EXPECT_TRUE(tok_a != 0, "insert returned zero token for first slot");
    EXPECT_TRUE(tok_b != 0, "insert returned zero token for second slot");
    EXPECT_TRUE(tok_a != tok_b, "two fresh inserts produced equal tokens");

    void *out = nullptr;
    EXPECT_TRUE(t.lookup(tok_a, HandleKind::Core, &out) == AMIO_OK, "lookup of fresh Core token failed");
    EXPECT_TRUE(out == &payload_a, "Core lookup returned wrong payload");

    out = nullptr;
    EXPECT_TRUE(t.lookup(tok_b, HandleKind::Dataset, &out) == AMIO_OK, "lookup of fresh Dataset token failed");
    EXPECT_TRUE(out == &payload_b, "Dataset lookup returned wrong payload");

    EXPECT_TRUE(t.size_for_test() == 2u, "size_for_test mismatch after 2 inserts");
}

void test_lookup_null_handle_returns_null_handle_error() {
    HandleTable t;
    void *out = reinterpret_cast<void *>(0xCAFEFACE);  // poison
    auto rc = t.lookup(/*token=*/0, HandleKind::Core, &out);
    EXPECT_TRUE(rc == AMIO_ERR_NULL_HANDLE, "lookup(0) did not return AMIO_ERR_NULL_HANDLE");
    EXPECT_TRUE(out == nullptr, "lookup(0) failed to clear out-payload to NULL");
}

void test_lookup_wrong_kind_returns_invalid_handle() {
    HandleTable t;
    int payload = 5;
    auto tok = t.insert(HandleKind::Dataset, &payload);

    void *out = nullptr;
    auto rc = t.lookup(tok, HandleKind::Core, &out);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "wrong-kind lookup did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(out == nullptr, "wrong-kind lookup leaked a payload pointer");

    // The handle must still be valid for the correct kind.
    out = nullptr;
    rc = t.lookup(tok, HandleKind::Dataset, &out);
    EXPECT_TRUE(rc == AMIO_OK, "correct-kind lookup failed after wrong-kind miss");
    EXPECT_TRUE(out == &payload, "correct-kind lookup returned wrong payload");
}

void test_release_then_lookup_returns_invalid_handle() {
    HandleTable t;
    int payload = 7;
    auto tok = t.insert(HandleKind::Io, &payload);

    EXPECT_TRUE(t.release(tok, HandleKind::Io) == AMIO_OK, "first release failed");

    void *out = reinterpret_cast<void *>(0xDEADBEEF);
    auto rc = t.lookup(tok, HandleKind::Io, &out);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "lookup after release did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(out == nullptr, "stale-lookup leaked a payload pointer");

    // Double-release must also fail without crashing.
    rc = t.release(tok, HandleKind::Io);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "double release did not return AMIO_ERR_INVALID_HANDLE");
}

void test_reuse_after_free_old_token_stays_invalid() {
    // The core invariant for R10.7: even after the table recycles
    // a slot, the OLD token does not silently start pointing at
    // the new owner's payload.  Generation counter difference
    // surfaces this as AMIO_ERR_INVALID_HANDLE.
    HandleTable t;

    int payload_old = 1;
    auto tok_old = t.insert(HandleKind::View, &payload_old);
    EXPECT_TRUE(t.release(tok_old, HandleKind::View) == AMIO_OK, "release(tok_old) failed");

    int payload_new = 2;
    auto tok_new = t.insert(HandleKind::View, &payload_new);

    EXPECT_TRUE(tok_new != tok_old, "recycled slot reused the same token value (no generation bump)");
    EXPECT_TRUE(HandleTable::unpack_slot(tok_new) == HandleTable::unpack_slot(tok_old), "recycled token did not land on the same slot index");

    // Old token still returns AMIO_ERR_INVALID_HANDLE.
    void *out = nullptr;
    auto rc = t.lookup(tok_old, HandleKind::View, &out);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "stale token after recycle did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(out == nullptr, "stale recycle lookup leaked a payload");

    // New token resolves to the new payload.
    out = nullptr;
    rc = t.lookup(tok_new, HandleKind::View, &out);
    EXPECT_TRUE(rc == AMIO_OK, "fresh token after recycle did not return AMIO_OK");
    EXPECT_TRUE(out == &payload_new, "fresh token returned an unexpected payload");

    // Release of a stale token must not invalidate the live token.
    rc = t.release(tok_old, HandleKind::View);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "release of stale token did not return AMIO_ERR_INVALID_HANDLE");

    out = nullptr;
    rc = t.lookup(tok_new, HandleKind::View, &out);
    EXPECT_TRUE(rc == AMIO_OK, "stale-release corrupted the live token");
    EXPECT_TRUE(out == &payload_new, "stale-release corrupted the payload binding");
}

void test_garbage_token_with_oob_slot_index_is_rejected() {
    HandleTable t;
    int payload = 0;
    auto tok = t.insert(HandleKind::Core, &payload);

    // Fabricate a token whose slot index points outside the table.
    auto bogus = HandleTable::pack(/*slot=*/12345u, HandleTable::unpack_generation(tok));

    void *out = nullptr;
    auto rc = t.lookup(bogus, HandleKind::Core, &out);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "OOB-slot lookup did not return AMIO_ERR_INVALID_HANDLE");
    EXPECT_TRUE(out == nullptr, "OOB-slot lookup leaked a payload pointer");

    rc = t.release(bogus, HandleKind::Core);
    EXPECT_TRUE(rc == AMIO_ERR_INVALID_HANDLE, "OOB-slot release did not return AMIO_ERR_INVALID_HANDLE");

    // The legitimate token is unaffected.
    out = nullptr;
    rc = t.lookup(tok, HandleKind::Core, &out);
    EXPECT_TRUE(rc == AMIO_OK, "legitimate lookup broken by OOB-token traffic");
}

void test_concurrent_insert_and_release_is_safe() {
    // Smoke-level coverage of the table's mutex.  Each thread
    // performs a long sequence of (insert, lookup, release) on
    // independent payloads and verifies that every fresh token
    // resolves to its own payload while it is live.  The much
    // tighter universal version of this property lands in P4
    // (task 13.5); the goal here is to catch obvious data races
    // in the slab allocator path during early development.
    HandleTable t;

    constexpr int kThreads = 4;
    constexpr int kIterations = 1000;

    std::atomic<int> race_failures{0};
    std::vector<std::thread> workers;
    workers.reserve(kThreads);

    for (int tid = 0; tid < kThreads; ++tid) {
        workers.emplace_back([&, tid]() {
            std::vector<int> local_payloads(static_cast<std::size_t>(kIterations));
            for (int i = 0; i < kIterations; ++i) {
                local_payloads[static_cast<std::size_t>(i)] = tid * 1'000'000 + i;
            }
            for (int i = 0; i < kIterations; ++i) {
                int *p = &local_payloads[static_cast<std::size_t>(i)];
                auto tok = t.insert(HandleKind::Io, p);
                if (tok == 0) {
                    race_failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                void *out = nullptr;
                if (t.lookup(tok, HandleKind::Io, &out) != AMIO_OK || out != p) {
                    race_failures.fetch_add(1, std::memory_order_relaxed);
                }

                if (t.release(tok, HandleKind::Io) != AMIO_OK) {
                    race_failures.fetch_add(1, std::memory_order_relaxed);
                }

                // Stale lookup after release must always fail.
                out = reinterpret_cast<void *>(0xBADF00D);
                if (t.lookup(tok, HandleKind::Io, &out) != AMIO_ERR_INVALID_HANDLE) {
                    race_failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto &w : workers) {
        w.join();
    }

    EXPECT_TRUE(race_failures.load(std::memory_order_relaxed) == 0,
                "concurrent insert/lookup/release exposed " + std::to_string(race_failures.load()) + " inconsistencies");

    // After the storm, the live-slot count should be zero (every
    // insert was paired with a release).  The free list re-uses
    // slots, so size_for_test counts only live slots.
    EXPECT_TRUE(t.size_for_test() == 0u, "live slot count drifted after balanced insert/release pairs");
}

}  // namespace

int main() {
    test_insert_returns_nonzero_token_with_correct_payload();
    test_lookup_null_handle_returns_null_handle_error();
    test_lookup_wrong_kind_returns_invalid_handle();
    test_release_then_lookup_returns_invalid_handle();
    test_reuse_after_free_old_token_stays_invalid();
    test_garbage_token_with_oob_slot_index_is_rejected();
    test_concurrent_insert_and_release_is_safe();

    std::fprintf(stdout, "test_handle_table: passed=%d failed=%d\n", g_result.passed, g_result.failed);

    return g_result.failed == 0 ? 0 : 1;
}
