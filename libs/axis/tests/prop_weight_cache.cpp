// ─── Property-Based Tests: Weight Cache Round-Trip ───────────────────────────
// Feature: axis-v2-improvements, Property 9: Weight Cache Round-Trip
//
// For any InterpolationMatrix, serialize then deserialize SHALL produce
// bitwise-identical apply results.
//
// Also tests error paths:
// - Deserialize throws on bad magic
// - Deserialize throws on wrong version
// - Deserialize throws on truncated buffer
//
// **Validates: Requirements 5.1, 5.2, 5.3**
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>

#include "common/test_kokkos_env.hpp"
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>

#include <Kokkos_Core.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/weight_cache.hpp>
#include <axis/types.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// ─── Property 9: Weight cache round-trip ─────────────────────────────────────
// Generate a random InterpolationMatrix, serialize via WeightCache::serialize,
// deserialize back via WeightCache::deserialize, apply both to the same random
// source field, and verify destination fields are bitwise identical.
//
// **Validates: Requirements 5.1, 5.2, 5.3**

RC_GTEST_PROP(PropWeightCache, SerializeDeserializeProducesBitwiseIdenticalApply, ()) {
    // Generate dimensions
    const auto n_src = *rc::gen::inRange<std::size_t>(2, 20);
    const auto n_dst = *rc::gen::inRange<std::size_t>(2, 20);
    const auto max_nnz = std::min(n_src * n_dst, static_cast<std::size_t>(50));
    const auto nnz = *rc::gen::inRange<std::size_t>(1, max_nnz + 1);

    // Generate random COO entries
    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", nnz);

    for (std::size_t k = 0; k < nnz; ++k) {
        fl(k) = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 1000.0; });
        fr(k) = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_dst));
        fc(k) = static_cast<axis::index_t>(*rc::gen::inRange<std::size_t>(0, n_src));
    }

    // Generate random frac/area arrays
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = *rc::gen::map(rc::gen::inRange(1, 10001), [](int v) { return static_cast<double>(v) / 10000.0; });
        aa(i) = *rc::gen::map(rc::gen::inRange(1, 10001), [](int v) { return static_cast<double>(v) / 100.0; });
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = *rc::gen::map(rc::gen::inRange(1, 10001), [](int v) { return static_cast<double>(v) / 10000.0; });
        ab(j) = *rc::gen::map(rc::gen::inRange(1, 10001), [](int v) { return static_cast<double>(v) / 100.0; });
    }

    // Build the original InterpolationMatrix
    axis::solver::InterpolationMatrix<Kokkos::HostSpace> original(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                  std::move(aa), std::move(ab), n_src, n_dst);

    // Serialize: first query size, then serialize
    const std::size_t blob_size = axis::solver::WeightCache::serialize(original, nullptr, 0);
    RC_ASSERT(blob_size > 0);

    std::vector<uint8_t> blob(blob_size);
    const std::size_t written = axis::solver::WeightCache::serialize(original, blob.data(), blob_size);
    RC_ASSERT(written == blob_size);

    // Deserialize
    auto restored = axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(blob.data(), blob_size);

    // Verify dimensions match
    RC_ASSERT(restored.n_src() == original.n_src());
    RC_ASSERT(restored.n_dst() == original.n_dst());
    RC_ASSERT(restored.nnz() == original.nnz());

    // Verify underlying arrays are bitwise identical (the round-trip property)
    const auto orig_fl = original.factor_list_view();
    const auto rest_fl = restored.factor_list_view();
    RC_ASSERT(std::memcmp(orig_fl.data(), rest_fl.data(), nnz * sizeof(double)) == 0);

    const auto orig_fr = original.factor_row_view();
    const auto rest_fr = restored.factor_row_view();
    RC_ASSERT(std::memcmp(orig_fr.data(), rest_fr.data(), nnz * sizeof(axis::index_t)) == 0);

    const auto orig_fc = original.factor_col_view();
    const auto rest_fc = restored.factor_col_view();
    RC_ASSERT(std::memcmp(orig_fc.data(), rest_fc.data(), nnz * sizeof(axis::index_t)) == 0);

    const auto orig_fa = original.frac_a_view();
    const auto rest_fa = restored.frac_a_view();
    RC_ASSERT(std::memcmp(orig_fa.data(), rest_fa.data(), n_src * sizeof(double)) == 0);

    const auto orig_fb = original.frac_b_view();
    const auto rest_fb = restored.frac_b_view();
    RC_ASSERT(std::memcmp(orig_fb.data(), rest_fb.data(), n_dst * sizeof(double)) == 0);

    const auto orig_aa = original.area_a_view();
    const auto rest_aa = restored.area_a_view();
    RC_ASSERT(std::memcmp(orig_aa.data(), rest_aa.data(), n_src * sizeof(double)) == 0);

    const auto orig_ab = original.area_b_view();
    const auto rest_ab = restored.area_b_view();
    RC_ASSERT(std::memcmp(orig_ab.data(), rest_ab.data(), n_dst * sizeof(double)) == 0);

    // Generate a random source field and verify apply produces bitwise-identical
    // results using a deterministic scalar reference loop (avoids atomic
    // non-determinism in the parallel COO apply path)
    std::vector<double> src_data(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
        src_data[i] = *rc::gen::map(rc::gen::inRange(-10000, 10001), [](int v) { return static_cast<double>(v) / 100.0; });
    }

    // Deterministic scalar apply using original matrix
    std::vector<double> dst_original(n_dst, 0.0);
    for (std::size_t k = 0; k < nnz; ++k) {
        const auto r = static_cast<std::size_t>(orig_fr.data()[k]);
        const auto c = static_cast<std::size_t>(orig_fc.data()[k]);
        dst_original[r] += orig_fl.data()[k] * src_data[c];
    }

    // Deterministic scalar apply using restored matrix
    std::vector<double> dst_restored(n_dst, 0.0);
    for (std::size_t k = 0; k < nnz; ++k) {
        const auto r = static_cast<std::size_t>(rest_fr.data()[k]);
        const auto c = static_cast<std::size_t>(rest_fc.data()[k]);
        dst_restored[r] += rest_fl.data()[k] * src_data[c];
    }

    // Verify bitwise identity of apply results
    RC_ASSERT(std::memcmp(dst_original.data(), dst_restored.data(), n_dst * sizeof(double)) == 0);
}

// ─── Error Path Tests ────────────────────────────────────────────────────────

TEST(PropWeightCache, DeserializeThrowsOnBadMagic) {
    // Create a minimal valid blob, then corrupt the magic bytes
    const std::size_t n_src = 3;
    const std::size_t n_dst = 2;
    const std::size_t nnz = 4;

    // Build a small matrix
    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", nnz);
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    for (std::size_t k = 0; k < nnz; ++k) {
        fl(k) = static_cast<double>(k) * 0.25;
        fr(k) = static_cast<axis::index_t>(k % n_dst);
        fc(k) = static_cast<axis::index_t>(k % n_src);
    }
    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    axis::solver::InterpolationMatrix<Kokkos::HostSpace> matrix(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                std::move(aa), std::move(ab), n_src, n_dst);

    // Serialize
    const std::size_t blob_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);
    std::vector<uint8_t> blob(blob_size);
    axis::solver::WeightCache::serialize(matrix, blob.data(), blob_size);

    // Corrupt magic bytes (first 4 bytes)
    blob[0] = 0xDE;
    blob[1] = 0xAD;
    blob[2] = 0xBE;
    blob[3] = 0xEF;

    EXPECT_THROW((axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(blob.data(), blob_size)), std::runtime_error);
}

TEST(PropWeightCache, DeserializeThrowsOnWrongVersion) {
    const std::size_t n_src = 3;
    const std::size_t n_dst = 2;
    const std::size_t nnz = 4;

    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", nnz);
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    for (std::size_t k = 0; k < nnz; ++k) {
        fl(k) = static_cast<double>(k) * 0.25;
        fr(k) = static_cast<axis::index_t>(k % n_dst);
        fc(k) = static_cast<axis::index_t>(k % n_src);
    }
    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    axis::solver::InterpolationMatrix<Kokkos::HostSpace> matrix(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                std::move(aa), std::move(ab), n_src, n_dst);

    const std::size_t blob_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);
    std::vector<uint8_t> blob(blob_size);
    axis::solver::WeightCache::serialize(matrix, blob.data(), blob_size);

    // Corrupt version field (bytes 4-7): set to version 99
    uint32_t bad_version = 99;
    std::memcpy(blob.data() + 4, &bad_version, sizeof(uint32_t));

    EXPECT_THROW((axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(blob.data(), blob_size)), std::runtime_error);
}

TEST(PropWeightCache, DeserializeThrowsOnTruncatedBuffer) {
    const std::size_t n_src = 3;
    const std::size_t n_dst = 2;
    const std::size_t nnz = 4;

    Kokkos::View<double *, Kokkos::HostSpace> fl("fl", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fr("fr", nnz);
    Kokkos::View<axis::index_t *, Kokkos::HostSpace> fc("fc", nnz);
    Kokkos::View<double *, Kokkos::HostSpace> fa("fa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> fb("fb", n_dst);
    Kokkos::View<double *, Kokkos::HostSpace> aa("aa", n_src);
    Kokkos::View<double *, Kokkos::HostSpace> ab("ab", n_dst);

    for (std::size_t k = 0; k < nnz; ++k) {
        fl(k) = static_cast<double>(k) * 0.25;
        fr(k) = static_cast<axis::index_t>(k % n_dst);
        fc(k) = static_cast<axis::index_t>(k % n_src);
    }
    for (std::size_t i = 0; i < n_src; ++i) {
        fa(i) = 1.0;
        aa(i) = 1.0;
    }
    for (std::size_t j = 0; j < n_dst; ++j) {
        fb(j) = 1.0;
        ab(j) = 1.0;
    }

    axis::solver::InterpolationMatrix<Kokkos::HostSpace> matrix(std::move(fl), std::move(fr), std::move(fc), std::move(fa), std::move(fb),
                                                                std::move(aa), std::move(ab), n_src, n_dst);

    const std::size_t blob_size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);
    std::vector<uint8_t> blob(blob_size);
    axis::solver::WeightCache::serialize(matrix, blob.data(), blob_size);

    // Truncate: pass only header + partial body (remove last 100 bytes or half)
    const std::size_t truncated_size = axis::solver::WeightCache::HEADER_SIZE + 1;

    EXPECT_THROW((axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(blob.data(), truncated_size)), std::runtime_error);
}

// ─── Kokkos Initialization ───────────────────────────────────────────────────



}  // namespace
