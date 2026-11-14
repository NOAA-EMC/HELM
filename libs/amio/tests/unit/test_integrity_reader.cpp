#include <array>
#include <cassert>
#include <cstring>

#include "operators/integrity_reader.hpp"
#include "operators/payload_integrity.hpp"

using namespace amio::detail;

int main() {
    const std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
    IntegrityRegion first{"a", "f32", {0}, {2}, {}};
    IntegrityRegion second{"a", "f32", {2}, {2}, {}};
    assert(sha256_logical_payload("f32", first.offsets, first.extents, values.data(), 2 * sizeof(float), first.digest) == AMIO_OK);
    assert(sha256_logical_payload("f32", second.offsets, second.extents, values.data() + 2, 2 * sizeof(float), second.digest) == AMIO_OK);
    const std::array<IntegrityRegion, 2> regions{first, second};
    std::size_t calls = 0, verified = 0;
    auto reader = [&](const IntegrityRegion &region, std::size_t limit, std::vector<std::byte> &out) {
        ++calls;
        assert(limit == 2 * sizeof(float));
        const auto start = region.offsets[0];
        out.resize(2 * sizeof(float));
        std::memcpy(out.data(), values.data() + start, out.size());
        return AMIO_OK;
    };
    assert(verify_intersecting_integrity_regions(regions, std::array<std::uint64_t, 1>{1}, std::array<std::uint64_t, 1>{2}, 1024, reader, verified) ==
           AMIO_OK);
    assert(calls == 2 && verified == 2);
    assert(verify_intersecting_integrity_regions(regions, std::array<std::uint64_t, 1>{0}, std::array<std::uint64_t, 1>{1}, 1, reader, verified) ==
           AMIO_ERR_BUDGET_EXCEEDED);
    IntegrityRegionReader corrupt = [&](const IntegrityRegion &region, std::size_t limit, std::vector<std::byte> &out) {
        (void)reader(region, limit, out);
        out[0] ^= std::byte{1};
        return AMIO_OK;
    };
    assert(verify_intersecting_integrity_regions(regions, std::array<std::uint64_t, 1>{0}, std::array<std::uint64_t, 1>{1}, 1024, corrupt,
                                                 verified) == AMIO_ERR_STEP_FAILED);
    assert(verify_intersecting_integrity_regions(regions, std::array<std::uint64_t, 1>{4}, std::array<std::uint64_t, 1>{1}, 1024, reader, verified) ==
           AMIO_ERR_UNSUPPORTED);
}
