#include "operators/integrity_reader.hpp"

#include "operators/payload_integrity.hpp"

#include <algorithm>
#include <limits>

namespace amio::detail {

amio_status_t verify_intersecting_integrity_regions(
    std::span<const IntegrityRegion> regions,
    std::span<const std::uint64_t> selection_offsets,
    std::span<const std::uint64_t> selection_extents,
    std::size_t max_region_bytes,
    const IntegrityRegionReader &reader,
    std::size_t &out_verified_regions) {
    out_verified_regions = 0;
    if (!reader || selection_offsets.size() != selection_extents.size() || selection_offsets.size() > 7)
        return AMIO_ERR_INVALID_INPUT;
    bool intersected = false;
    for (const auto &region : regions) {
        if (region.offsets.size() != selection_offsets.size() || region.extents.size() != selection_extents.size())
            return AMIO_ERR_INVALID_INPUT;
        bool intersects = true;
        std::uint64_t elements = 1;
        for (std::size_t d=0; d<region.extents.size(); ++d) {
            if (region.offsets[d] > std::numeric_limits<std::uint64_t>::max()-region.extents[d] ||
                selection_offsets[d] > std::numeric_limits<std::uint64_t>::max()-selection_extents[d]) return AMIO_ERR_INVALID_INPUT;
            const auto region_end=region.offsets[d]+region.extents[d];
            const auto selection_end=selection_offsets[d]+selection_extents[d];
            intersects = intersects && region.offsets[d] < selection_end && selection_offsets[d] < region_end;
            if (region.extents[d] != 0 && elements > std::numeric_limits<std::uint64_t>::max()/region.extents[d]) return AMIO_ERR_INVALID_INPUT;
            elements *= region.extents[d];
        }
        if (!intersects) continue;
        intersected = true;
        if (region.digest.empty()) return AMIO_ERR_UNSUPPORTED;
        std::size_t width = 0;
        if (region.dtype == "i8" || region.dtype == "u8") width=1;
        else if (region.dtype == "i16" || region.dtype == "u16") width=2;
        else if (region.dtype == "i32" || region.dtype == "u32" || region.dtype == "f32") width=4;
        else if (region.dtype == "i64" || region.dtype == "u64" || region.dtype == "f64") width=8;
        else return AMIO_ERR_UNSUPPORTED;
        if (elements > std::numeric_limits<std::size_t>::max()/width || elements*width > max_region_bytes)
            return AMIO_ERR_BUDGET_EXCEEDED;
        std::vector<std::byte> bytes;
        const auto read_status = reader(region, elements*width, bytes);
        if (read_status != AMIO_OK) return read_status;
        if (bytes.size() != elements*width) return AMIO_ERR_SCHEMA_MISMATCH;
        std::string actual;
        const auto hash_status = sha256_logical_payload(region.dtype, region.offsets, region.extents,
            bytes.data(), bytes.size(), actual);
        if (hash_status != AMIO_OK) return hash_status;
        if (actual != region.digest) return AMIO_ERR_STEP_FAILED;
        ++out_verified_regions;
    }
    return intersected ? AMIO_OK : AMIO_ERR_UNSUPPORTED;
}

} // namespace amio::detail
