#include "operators/payload_integrity.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <vector>

namespace amio::detail {
namespace {

class Sha256 {
   public:
    void update(const std::uint8_t *data, std::size_t size) {
        total_ += size;
        while (size != 0) {
            const auto take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, take);
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }
    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bits = static_cast<std::uint64_t>(total_) * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_), block_.end(), 0);
            transform(block_.data());
            used_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_), block_.begin() + 56, 0);
        for (unsigned i = 0; i < 8; ++i) block_[63 - i] = static_cast<std::uint8_t>(bits >> (8 * i));
        transform(block_.data());
        std::array<std::uint8_t, 32> result{};
        for (std::size_t i = 0; i < state_.size(); ++i)
            for (unsigned j = 0; j < 4; ++j) result[i * 4 + j] = static_cast<std::uint8_t>(state_[i] >> (24 - 8 * j));
        return result;
    }

   private:
    static constexpr std::array<std::uint32_t, 64> k_ = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    void transform(const std::uint8_t *p) {
        std::array<std::uint32_t, 64> w{};
        for (unsigned i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[i * 4]) << 24) | (std::uint32_t(p[i * 4 + 1]) << 16) | (std::uint32_t(p[i * 4 + 2]) << 8) | p[i * 4 + 3];
        for (unsigned i = 16; i < 64; ++i) {
            auto a = w[i - 15], b = w[i - 2];
            w[i] = (std::rotr(a, 7) ^ std::rotr(a, 18) ^ (a >> 3)) + w[i - 16] + (std::rotr(b, 17) ^ std::rotr(b, 19) ^ (b >> 10)) + w[i - 7];
        }
        auto [a, b, c, d, e, f, g, h] = state_;
        for (unsigned i = 0; i < 64; ++i) {
            auto t1 = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) + ((e & f) ^ (~e & g)) + k_[i] + w[i];
            auto t2 = (std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }
    std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::array<std::uint8_t, 64> block_{};
    std::size_t used_ = 0, total_ = 0;
};

std::size_t dtype_size(std::string_view dtype) {
    if (dtype == "i8" || dtype == "u8") return 1;
    if (dtype == "i16" || dtype == "u16") return 2;
    if (dtype == "i32" || dtype == "u32" || dtype == "f32") return 4;
    if (dtype == "i64" || dtype == "u64" || dtype == "f64") return 8;
    return 0;
}

}  // namespace

amio_status_t sha256_logical_payload(std::string_view dtype, std::span<const std::uint64_t> offsets, std::span<const std::uint64_t> extents,
                                     const void *data, std::size_t bytes, std::string &out_digest) {
    const auto width = dtype_size(dtype);
    if (width == 0 || offsets.size() != extents.size() || offsets.size() > 7 || (bytes != 0 && data == nullptr)) return AMIO_ERR_INVALID_INPUT;
    std::uint64_t elements = 1;
    for (const auto extent : extents) {
        if (extent != 0 && elements > std::numeric_limits<std::uint64_t>::max() / extent) return AMIO_ERR_INVALID_INPUT;
        elements *= extent;
    }
    if (elements > std::numeric_limits<std::size_t>::max() / width || bytes != elements * width) return AMIO_ERR_INVALID_INPUT;
    Sha256 hash;
    const std::string prefix = "AMIO-LOGICAL-SHA256-v1\0";
    hash.update(reinterpret_cast<const std::uint8_t *>(prefix.data()), prefix.size());
    auto add_u64 = [&](std::uint64_t v) {
        std::array<std::uint8_t, 8> b{};
        for (unsigned i = 0; i < 8; ++i) b[i] = static_cast<std::uint8_t>(v >> (8 * i));
        hash.update(b.data(), b.size());
    };
    add_u64(dtype.size());
    hash.update(reinterpret_cast<const std::uint8_t *>(dtype.data()), dtype.size());
    add_u64(extents.size());
    for (std::size_t i = 0; i < extents.size(); ++i) {
        add_u64(offsets[i]);
        add_u64(extents[i]);
    }
    const auto *raw = static_cast<const std::uint8_t *>(data);
    if constexpr (std::endian::native == std::endian::little)
        hash.update(raw, bytes);
    else {
        std::array<std::uint8_t, 4096> canonical{};
        for (std::size_t pos = 0; pos < bytes;) {
            const auto n = std::min(canonical.size(), bytes - pos);
            const auto aligned = n - (n % width);
            for (std::size_t i = 0; i < aligned; i += width)
                for (std::size_t j = 0; j < width; ++j) canonical[i + j] = raw[pos + i + width - 1 - j];
            hash.update(canonical.data(), aligned);
            pos += aligned;
        }
    }
    const auto digest = hash.finish();
    static constexpr char hex[] = "0123456789abcdef";
    out_digest = "sha256-logical-v1:";
    out_digest.reserve(out_digest.size() + 64);
    for (auto b : digest) {
        out_digest.push_back(hex[b >> 4]);
        out_digest.push_back(hex[b & 15]);
    }
    return AMIO_OK;
}

}  // namespace amio::detail
