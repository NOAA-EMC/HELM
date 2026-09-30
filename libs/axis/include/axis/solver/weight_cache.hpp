// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_SOLVER_WEIGHT_CACHE_HPP
#define AXIS_SOLVER_WEIGHT_CACHE_HPP

/// @file axis/solver/weight_cache.hpp
/// @brief Minimal binary serialize/deserialize for InterpolationMatrix.
///
/// WeightCache provides a self-describing, architecture-tagged binary format
/// for caching precomputed interpolation weights. The format consists of a
/// 40-byte header followed by contiguous IEEE-754 arrays. No file I/O is
/// performed — the caller owns persistence (HELM Law: AXIS opens no files).
///
/// Binary layout (40-byte header):
///   [magic:4][version:4][endian:1][pad:7][n_src:8][n_dst:8][nnz:8]
///
/// Body arrays (contiguous after header):
///   factor_list[nnz×8]
///   factor_row[nnz×sizeof(index_t)]
///   factor_col[nnz×sizeof(index_t)]
///   frac_a[n_src×8]
///   frac_b[n_dst×8]
///   area_a[n_src×8]
///   area_b[n_dst×8]
///   unmapped_mask[n_dst×4]   — present iff header pad[0] != 0
///
/// The header pad byte pad[0] encodes the unmapped-mask presence flag
/// (0 = absent — blob is byte-identical to the original layout; 1 = present).
/// Format version stays 1: maskless blobs are unchanged and mask-bearing
/// blobs are self-describing.
///
/// Header-only (template) since it is parameterized on MemorySpace.
/// Uses Kokkos mirror views to transfer device data to/from host buffers.

#include <Kokkos_Core.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/types.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace axis::solver {

/// Stateless utility for serializing/deserializing InterpolationMatrix to/from
/// a self-describing binary blob. Accepts raw void* buffers — no file I/O.
struct WeightCache {
    /// Format version. Incremented on incompatible layout changes.
    static constexpr uint32_t FORMAT_VERSION = 1;

    /// Magic bytes identifying an AXIS weight cache blob: "AXWC".
    static constexpr uint32_t MAGIC = 0x43574841;  // 'A','X','W','C' in little-endian

    /// Endianness tag values.
    static constexpr uint8_t ENDIAN_LITTLE = 0x01;
    static constexpr uint8_t ENDIAN_BIG = 0x02;

    /// Header size in bytes (40 bytes total).
    static constexpr std::size_t HEADER_SIZE = 40;

    // ─────────────────────────────────────────────────────────────────────────
    // serialize
    // ─────────────────────────────────────────────────────────────────────────

    /// Serialize an InterpolationMatrix into a binary blob.
    ///
    /// If buf is nullptr, returns the required buffer size without writing.
    /// Otherwise writes the header and array data into buf and returns bytes
    /// written.
    ///
    /// @tparam MS Kokkos MemorySpace of the source InterpolationMatrix
    /// @param m        The interpolation matrix to serialize
    /// @param buf      Destination buffer (nullptr to query size)
    /// @param buf_size Size of the destination buffer in bytes
    /// @return         Bytes written (or required size if buf == nullptr)
    ///
    /// @throws std::runtime_error if buf is non-null but buf_size is too small
    template <class MS>
    static std::size_t serialize(const InterpolationMatrix<MS> &m, void *buf, std::size_t buf_size) {
        const std::size_t n_src = m.n_src();
        const std::size_t n_dst = m.n_dst();
        const std::size_t nnz = m.nnz();
        const bool has_mask = m.has_unmapped_mask();

        const std::size_t required = compute_blob_size(n_src, n_dst, nnz, has_mask);

        // Query mode: return required size
        if (buf == nullptr) {
            return required;
        }

        if (buf_size < required) {
            throw std::runtime_error("axis::solver::WeightCache::serialize: buf_size (" + std::to_string(buf_size) + ") < required (" +
                                     std::to_string(required) + ")");
        }

        auto *dst = static_cast<uint8_t *>(buf);

        // ── Write header ─────────────────────────────────────────────────────
        write_header(dst, n_src, n_dst, nnz, has_mask);
        dst += HEADER_SIZE;

        // ── Create host mirrors of the matrix views ──────────────────────────
        auto h_factor_list = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.factor_list_view());
        auto h_factor_row = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.factor_row_view());
        auto h_factor_col = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.factor_col_view());
        auto h_frac_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.frac_a_view());
        auto h_frac_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.frac_b_view());
        auto h_area_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.area_a_view());
        auto h_area_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.area_b_view());

        // ── Write body arrays contiguously ───────────────────────────────────
        // factor_list [nnz × 8]
        std::memcpy(dst, h_factor_list.data(), nnz * sizeof(double));
        dst += nnz * sizeof(double);

        // factor_row [nnz × sizeof(index_t)]
        std::memcpy(dst, h_factor_row.data(), nnz * sizeof(index_t));
        dst += nnz * sizeof(index_t);

        // factor_col [nnz × sizeof(index_t)]
        std::memcpy(dst, h_factor_col.data(), nnz * sizeof(index_t));
        dst += nnz * sizeof(index_t);

        // frac_a [n_src × 8]
        std::memcpy(dst, h_frac_a.data(), n_src * sizeof(double));
        dst += n_src * sizeof(double);

        // frac_b [n_dst × 8]
        std::memcpy(dst, h_frac_b.data(), n_dst * sizeof(double));
        dst += n_dst * sizeof(double);

        // area_a [n_src × 8]
        std::memcpy(dst, h_area_a.data(), n_src * sizeof(double));
        dst += n_src * sizeof(double);

        // area_b [n_dst × 8]
        std::memcpy(dst, h_area_b.data(), n_dst * sizeof(double));
        dst += n_dst * sizeof(double);

        // unmapped_mask [n_dst × 4] — only when the flag is set
        if (has_mask) {
            auto h_mask = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, m.unmapped_mask_view());
            std::memcpy(dst, h_mask.data(), n_dst * sizeof(int32_t));
            dst += n_dst * sizeof(int32_t);
        }

        return required;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // deserialize
    // ─────────────────────────────────────────────────────────────────────────

    /// Deserialize a binary blob into an InterpolationMatrix.
    ///
    /// Validates the magic, format version, endianness, and total blob size
    /// before constructing the matrix. Throws std::runtime_error on any
    /// validation failure.
    ///
    /// @tparam MS Kokkos MemorySpace for the resulting InterpolationMatrix
    /// @param buf      Source buffer containing the serialized blob
    /// @param buf_size Size of the source buffer in bytes
    /// @return         Reconstructed InterpolationMatrix in the target MemorySpace
    ///
    /// @throws std::runtime_error if magic, version, endianness, or size mismatch
    template <class MS>
    static InterpolationMatrix<MS> deserialize(const void *buf, std::size_t buf_size) {
        if (buf == nullptr) {
            throw std::runtime_error("axis::solver::WeightCache::deserialize: buf is nullptr");
        }

        if (buf_size < HEADER_SIZE) {
            throw std::runtime_error("axis::solver::WeightCache::deserialize: buf_size (" + std::to_string(buf_size) +
                                     ") < minimum header size (40)");
        }

        const auto *src = static_cast<const uint8_t *>(buf);

        // ── Read and validate header ─────────────────────────────────────────
        uint32_t magic{};
        uint32_t version{};
        uint8_t endian{};
        uint8_t mask_flag{};
        uint64_t n_src{};
        uint64_t n_dst{};
        uint64_t nnz{};

        std::memcpy(&magic, src, sizeof(uint32_t));
        src += sizeof(uint32_t);

        std::memcpy(&version, src, sizeof(uint32_t));
        src += sizeof(uint32_t);

        endian = *src;
        src += 1;
        // pad[7]: byte 0 is the unmapped-mask presence flag (0 = no mask);
        // legacy blobs zero-filled all 7 pad bytes and carry no mask.
        mask_flag = *src;
        src += 7;  // remaining pad bytes

        std::memcpy(&n_src, src, sizeof(uint64_t));
        src += sizeof(uint64_t);

        std::memcpy(&n_dst, src, sizeof(uint64_t));
        src += sizeof(uint64_t);

        std::memcpy(&nnz, src, sizeof(uint64_t));
        src += sizeof(uint64_t);

        // Validate magic
        if (magic != MAGIC) {
            throw std::runtime_error(
                "axis::solver::WeightCache::deserialize: invalid magic bytes "
                "(expected 0x" +
                to_hex(MAGIC) + ", got 0x" + to_hex(magic) + ")");
        }

        // Validate version
        if (version != FORMAT_VERSION) {
            throw std::runtime_error(
                "axis::solver::WeightCache::deserialize: format version mismatch "
                "(expected " +
                std::to_string(FORMAT_VERSION) + ", got " + std::to_string(version) + ")");
        }

        // Validate endianness matches native
        const uint8_t native_endian = detect_native_endian();
        if (endian != native_endian) {
            throw std::runtime_error(
                "axis::solver::WeightCache::deserialize: endianness mismatch "
                "(blob is " +
                endian_name(endian) + ", native is " + endian_name(native_endian) + ")");
        }

        // Validate total blob size
        const bool has_mask = (mask_flag != 0);
        const std::size_t expected_size =
            compute_blob_size(static_cast<std::size_t>(n_src), static_cast<std::size_t>(n_dst), static_cast<std::size_t>(nnz), has_mask);

        if (buf_size < expected_size) {
            throw std::runtime_error(
                "axis::solver::WeightCache::deserialize: blob size mismatch "
                "(buf_size=" +
                std::to_string(buf_size) + ", expected=" + std::to_string(expected_size) + " for n_src=" + std::to_string(n_src) +
                ", n_dst=" + std::to_string(n_dst) + ", nnz=" + std::to_string(nnz) + ")");
        }

        // ── Read body arrays into host Views ─────────────────────────────────
        Kokkos::View<double *, Kokkos::HostSpace> h_factor_list("weight_cache::factor_list", nnz);
        Kokkos::View<index_t *, Kokkos::HostSpace> h_factor_row("weight_cache::factor_row", nnz);
        Kokkos::View<index_t *, Kokkos::HostSpace> h_factor_col("weight_cache::factor_col", nnz);
        Kokkos::View<double *, Kokkos::HostSpace> h_frac_a("weight_cache::frac_a", n_src);
        Kokkos::View<double *, Kokkos::HostSpace> h_frac_b("weight_cache::frac_b", n_dst);
        Kokkos::View<double *, Kokkos::HostSpace> h_area_a("weight_cache::area_a", n_src);
        Kokkos::View<double *, Kokkos::HostSpace> h_area_b("weight_cache::area_b", n_dst);

        // factor_list [nnz × 8]
        std::memcpy(h_factor_list.data(), src, nnz * sizeof(double));
        src += nnz * sizeof(double);

        // factor_row [nnz × sizeof(index_t)]
        std::memcpy(h_factor_row.data(), src, nnz * sizeof(index_t));
        src += nnz * sizeof(index_t);

        // factor_col [nnz × sizeof(index_t)]
        std::memcpy(h_factor_col.data(), src, nnz * sizeof(index_t));
        src += nnz * sizeof(index_t);

        // frac_a [n_src × 8]
        std::memcpy(h_frac_a.data(), src, n_src * sizeof(double));
        src += n_src * sizeof(double);

        // frac_b [n_dst × 8]
        std::memcpy(h_frac_b.data(), src, n_dst * sizeof(double));
        src += n_dst * sizeof(double);

        // area_a [n_src × 8]
        std::memcpy(h_area_a.data(), src, n_src * sizeof(double));
        src += n_src * sizeof(double);

        // area_b [n_dst × 8]
        std::memcpy(h_area_b.data(), src, n_dst * sizeof(double));
        src += n_dst * sizeof(double);

        // unmapped_mask [n_dst × 4] — only when the flag is set
        Kokkos::View<int *, Kokkos::HostSpace> h_unmapped_mask("weight_cache::unmapped_mask", has_mask ? n_dst : 0);
        if (has_mask) {
            std::memcpy(h_unmapped_mask.data(), src, n_dst * sizeof(int32_t));
            src += n_dst * sizeof(int32_t);
        }

        // ── Deep-copy to target MemorySpace and construct matrix ─────────────
        Kokkos::View<double *, MS> factor_list("factor_list", nnz);
        Kokkos::View<index_t *, MS> factor_row("factor_row", nnz);
        Kokkos::View<index_t *, MS> factor_col("factor_col", nnz);
        Kokkos::View<double *, MS> frac_a("frac_a", n_src);
        Kokkos::View<double *, MS> frac_b("frac_b", n_dst);
        Kokkos::View<double *, MS> area_a("area_a", n_src);
        Kokkos::View<double *, MS> area_b("area_b", n_dst);

        Kokkos::deep_copy(factor_list, h_factor_list);
        Kokkos::deep_copy(factor_row, h_factor_row);
        Kokkos::deep_copy(factor_col, h_factor_col);
        Kokkos::deep_copy(frac_a, h_frac_a);
        Kokkos::deep_copy(frac_b, h_frac_b);
        Kokkos::deep_copy(area_a, h_area_a);
        Kokkos::deep_copy(area_b, h_area_b);

        InterpolationMatrix<MS> matrix(std::move(factor_list), std::move(factor_row), std::move(factor_col), std::move(frac_a), std::move(frac_b),
                                       std::move(area_a), std::move(area_b), static_cast<std::size_t>(n_src), static_cast<std::size_t>(n_dst));

        if (has_mask) {
            Kokkos::View<int *, MS> unmapped_mask("unmapped_mask", n_dst);
            Kokkos::deep_copy(unmapped_mask, h_unmapped_mask);
            matrix.set_unmapped_mask(std::move(unmapped_mask));
        }

        return matrix;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Utility: compute required blob size from dimensions
    // ─────────────────────────────────────────────────────────────────────────

    /// Compute the total blob size for given matrix dimensions.
    /// @param has_mask true when an n_dst-sized int32 unmapped mask follows area_b.
    static constexpr std::size_t compute_blob_size(std::size_t n_src, std::size_t n_dst, std::size_t nnz, bool has_mask = false) noexcept {
        const std::size_t base = HEADER_SIZE + nnz * sizeof(double)  // factor_list
                                 + nnz * sizeof(index_t)             // factor_row
                                 + nnz * sizeof(index_t)             // factor_col
                                 + n_src * sizeof(double)            // frac_a
                                 + n_dst * sizeof(double)            // frac_b
                                 + n_src * sizeof(double)            // area_a
                                 + n_dst * sizeof(double);           // area_b
        return base + (has_mask ? n_dst * sizeof(int32_t) : 0);
    }

   private:
    /// Detect native endianness at runtime.
    static uint8_t detect_native_endian() noexcept {
        const uint32_t probe = 1;
        const auto *bytes = reinterpret_cast<const uint8_t *>(&probe);
        return (bytes[0] == 1) ? ENDIAN_LITTLE : ENDIAN_BIG;
    }

    /// Write the 40-byte header to the buffer.
    /// @param has_mask when true, pad[0] records the unmapped-mask presence flag.
    static void write_header(uint8_t *dst, std::size_t n_src, std::size_t n_dst, std::size_t nnz, bool has_mask = false) noexcept {
        // magic [4]
        const uint32_t magic = MAGIC;
        std::memcpy(dst, &magic, sizeof(uint32_t));
        dst += sizeof(uint32_t);

        // version [4]
        const uint32_t version = FORMAT_VERSION;
        std::memcpy(dst, &version, sizeof(uint32_t));
        dst += sizeof(uint32_t);

        // endian [1]
        const uint8_t endian = detect_native_endian();
        *dst = endian;
        dst += 1;

        // pad [7] — byte 0 is the unmapped-mask flag, rest zero-filled
        *dst = has_mask ? 1 : 0;
        dst += 1;
        std::memset(dst, 0, 6);
        dst += 6;

        // n_src [8]
        const uint64_t ns = static_cast<uint64_t>(n_src);
        std::memcpy(dst, &ns, sizeof(uint64_t));
        dst += sizeof(uint64_t);

        // n_dst [8]
        const uint64_t nd = static_cast<uint64_t>(n_dst);
        std::memcpy(dst, &nd, sizeof(uint64_t));
        dst += sizeof(uint64_t);

        // nnz [8]
        const uint64_t nz = static_cast<uint64_t>(nnz);
        std::memcpy(dst, &nz, sizeof(uint64_t));
        dst += sizeof(uint64_t);
    }

    /// Convert a uint32_t to hex string (for error messages).
    static std::string to_hex(uint32_t val) {
        static constexpr char digits[] = "0123456789ABCDEF";
        std::string result(8, '0');
        for (int i = 7; i >= 0; --i) {
            result[static_cast<std::size_t>(i)] = digits[val & 0xF];
            val >>= 4;
        }
        return result;
    }

    /// Return human-readable endianness name.
    static std::string endian_name(uint8_t e) {
        switch (e) {
            case ENDIAN_LITTLE:
                return "little-endian";
            case ENDIAN_BIG:
                return "big-endian";
            default:
                return "unknown(0x" + to_hex(static_cast<uint32_t>(e)) + ")";
        }
    }
};

}  // namespace axis::solver

#endif  // AXIS_SOLVER_WEIGHT_CACHE_HPP
