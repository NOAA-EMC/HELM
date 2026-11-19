#ifndef AMIO_SRC_MEMORY_HOST_SNAPSHOT_HPP
#define AMIO_SRC_MEMORY_HOST_SNAPSHOT_HPP

#include <cstddef>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

struct HostSnapshotResult {
    amio_status_t status = AMIO_ERR_INVALID_INPUT;
    std::vector<std::byte> bytes;
    bool source_released = false;
};

/** Copy a validated host buffer into AMIO-owned storage before admission returns. */
[[nodiscard]] HostSnapshotResult snapshot_host_buffer(const void *source, std::size_t bytes);

}  // namespace amio::detail

#endif  // AMIO_SRC_MEMORY_HOST_SNAPSHOT_HPP
