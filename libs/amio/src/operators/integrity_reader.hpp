#ifndef AMIO_SRC_OPERATORS_INTEGRITY_READER_HPP
#define AMIO_SRC_OPERATORS_INTEGRITY_READER_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

struct IntegrityRegion {
    std::string field;
    std::string dtype;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> extents;
    std::string digest;
};

using IntegrityRegionReader = std::function<amio_status_t(const IntegrityRegion &, std::size_t, std::vector<std::byte> &)>;

/** Verify each checksum region intersecting a selection. The reader must
 * return the complete recorded region, never merely the clipped overlap.
 */
[[nodiscard]] amio_status_t verify_intersecting_integrity_regions(std::span<const IntegrityRegion> regions,
                                                                  std::span<const std::uint64_t> selection_offsets,
                                                                  std::span<const std::uint64_t> selection_extents, std::size_t max_region_bytes,
                                                                  const IntegrityRegionReader &reader, std::size_t &out_verified_regions);

}  // namespace amio::detail

#endif
