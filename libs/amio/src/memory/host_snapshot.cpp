#include "memory/host_snapshot.hpp"

#include <cstring>
#include <new>
#include <utility>

namespace amio::detail {

HostSnapshotResult snapshot_host_buffer(const void *source, std::size_t bytes) {
    if (source == nullptr || bytes == 0) return {AMIO_ERR_INVALID_INPUT, {}, false};
    try {
        std::vector<std::byte> snapshot(bytes);
        std::memcpy(snapshot.data(), source, bytes);
        return {AMIO_OK, std::move(snapshot), true};
    } catch (const std::bad_alloc &) {
        return {AMIO_ERR_BACKEND_FAILURE, {}, false};
    } catch (...) {
        return {AMIO_ERR_BACKEND_FAILURE, {}, false};
    }
}

}  // namespace amio::detail
