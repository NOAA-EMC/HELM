#ifndef AMIO_SRC_MEMORY_MEMORY_DESCRIPTOR_HPP
#define AMIO_SRC_MEMORY_MEMORY_DESCRIPTOR_HPP

#include "amio/amio_errors.h"
#include "amio/amio_evolution.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace amio::detail {

enum class MemoryKind : std::uint32_t { Host = 0, Device = 1 };
enum class BufferOwnership : std::uint32_t { Copy = 0, Borrowed = 1 };

struct MemoryDescriptor {
    MemoryKind kind = MemoryKind::Host;
    BufferOwnership ownership = BufferOwnership::Copy;
    std::uint32_t device_id = 0;
    bool producer_writes_synchronized = false;
    std::uint32_t rank = 0;
    std::vector<std::uint64_t> extents;
    std::vector<std::uint64_t> strides;
    std::uint64_t element_bytes = 0;
};

[[nodiscard]] amio_status_t validate_memory_descriptor(const MemoryDescriptor &descriptor,
                                                        std::size_t data_bytes) noexcept;
[[nodiscard]] amio_status_t snapshot_memory_buffer(const MemoryDescriptor &descriptor, const void *source,
                                                   std::size_t data_bytes, std::vector<std::byte> &snapshot,
                                                   bool &used_fallback);

} // namespace amio::detail

#endif
