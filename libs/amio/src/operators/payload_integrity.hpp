#ifndef AMIO_SRC_OPERATORS_PAYLOAD_INTEGRITY_HPP
#define AMIO_SRC_OPERATORS_PAYLOAD_INTEGRITY_HPP

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "amio/amio_errors.h"

namespace amio::detail {

/** Hash a logical field region using a versioned, little-endian representation.
 * Input values use native host byte order; output identity is host independent.
 */
[[nodiscard]] amio_status_t sha256_logical_payload(std::string_view dtype, std::span<const std::uint64_t> offsets,
                                                   std::span<const std::uint64_t> extents, const void *data, std::size_t bytes,
                                                   std::string &out_digest);

}  // namespace amio::detail

#endif
