#include "operators/payload_integrity.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <string>

using namespace amio::detail;

int main() {
    const std::string abc = "abc";
    std::string digest;
    const std::array<std::uint64_t, 1> abc_offset{0}, abc_extent{3};
    assert(sha256_logical_payload("u8", abc_offset, abc_extent, abc.data(), abc.size(), digest) == AMIO_OK);
    assert(digest == "sha256-logical-v1:f766ba187f3c364260c2514311bda61b82bed83c6c86ee4edd3ad18b9b0e0569");

    const std::array<std::uint32_t, 2> native{0x01020304u, 0xa0b0c0d0u};
    const std::array<std::uint64_t, 1> offsets{2}, extents{2};
    std::string first, second;
    assert(sha256_logical_payload("u32", offsets, extents, native.data(), sizeof(native), first) == AMIO_OK);
    auto swapped = native;
    for (auto &value : swapped) value = __builtin_bswap32(value);
    assert(sha256_logical_payload("u32", offsets, extents, swapped.data(), sizeof(swapped), second) == AMIO_OK);
    assert(first != second);
    assert(sha256_logical_payload("u32", offsets, extents, native.data(), sizeof(native) - 1, second) == AMIO_ERR_INVALID_INPUT);
}
