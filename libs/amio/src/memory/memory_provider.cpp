#include <limits>
#include <new>
#include <utility>

#include "memory/host_snapshot.hpp"
#include "memory/memory_descriptor.hpp"

namespace amio::detail {

amio_status_t validate_memory_descriptor(const MemoryDescriptor &descriptor, std::size_t data_bytes) noexcept {
    if (descriptor.ownership != BufferOwnership::Copy) return AMIO_ERR_UNSUPPORTED;
    if (descriptor.kind != MemoryKind::Host && descriptor.kind != MemoryKind::Device) return AMIO_ERR_UNSUPPORTED;
    if (!descriptor.producer_writes_synchronized) return AMIO_ERR_INVALID_POLICY;
    if (descriptor.rank == 0 || descriptor.rank > AMIO_MAX_RANK || descriptor.extents.size() != descriptor.rank ||
        descriptor.strides.size() != descriptor.rank || descriptor.element_bytes == 0 || data_bytes == 0)
        return AMIO_ERR_INVALID_INPUT;

    std::uint64_t elements = 1;
    std::uint64_t expected_stride = descriptor.element_bytes;
    for (std::size_t axis = descriptor.rank; axis-- > 0;) {
        if (descriptor.extents[axis] == 0 || descriptor.strides[axis] != expected_stride) return AMIO_ERR_UNSUPPORTED;
        if (descriptor.extents[axis] > std::numeric_limits<std::uint64_t>::max() / elements ||
            descriptor.extents[axis] > std::numeric_limits<std::uint64_t>::max() / expected_stride)
            return AMIO_ERR_INVALID_INPUT;
        elements *= descriptor.extents[axis];
        expected_stride *= descriptor.extents[axis];
    }
    if (elements > std::numeric_limits<std::size_t>::max() / descriptor.element_bytes || elements * descriptor.element_bytes != data_bytes)
        return AMIO_ERR_INVALID_INPUT;
    if (descriptor.kind == MemoryKind::Device && descriptor.device_id != 0) return AMIO_ERR_UNSUPPORTED;
    return AMIO_OK;
}

amio_status_t snapshot_memory_buffer(const MemoryDescriptor &descriptor, const void *source, std::size_t data_bytes, std::vector<std::byte> &snapshot,
                                     bool &used_fallback) {
    used_fallback = false;
    const auto valid = validate_memory_descriptor(descriptor, data_bytes);
    if (valid != AMIO_OK) return valid;
    if (source == nullptr) return AMIO_ERR_INVALID_INPUT;
    if (descriptor.kind == MemoryKind::Host) {
        auto result = snapshot_host_buffer(source, data_bytes);
        if (result.status == AMIO_OK) snapshot = std::move(result.bytes);
        return result.status;
    }
#ifdef AMIO_HAS_KOKKOS_RUNTIME
    extern amio_status_t snapshot_kokkos_device_buffer(const void *, std::size_t, std::vector<std::byte> &);
    const auto status = snapshot_kokkos_device_buffer(source, data_bytes, snapshot);
    used_fallback = status == AMIO_OK;
    return status;
#else
    return AMIO_ERR_UNSUPPORTED;
#endif
}

}  // namespace amio::detail
