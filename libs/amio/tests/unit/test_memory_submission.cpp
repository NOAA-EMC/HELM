#include "memory/memory_descriptor.hpp"

#include <cassert>
#include <array>
#include <cstdint>
#ifdef AMIO_TEST_HAS_KOKKOS_RUNTIME
#include <Kokkos_Core.hpp>
#endif

using namespace amio::detail;

int main() {
    MemoryDescriptor host;
    host.kind = MemoryKind::Host;
    host.ownership = BufferOwnership::Copy;
    host.producer_writes_synchronized = true;
    host.rank = 2;
    host.element_bytes = 4;
    host.extents = {3, 2};
    host.strides = {8, 4};
    assert(validate_memory_descriptor(host, 24) == AMIO_OK);

    auto strided = host;
    strided.strides[0] = 12;
    assert(validate_memory_descriptor(strided, 24) == AMIO_ERR_UNSUPPORTED);
    auto unsynchronized = host;
    unsynchronized.producer_writes_synchronized = false;
    assert(validate_memory_descriptor(unsynchronized, 24) == AMIO_ERR_INVALID_POLICY);
    auto borrowed = host;
    borrowed.ownership = BufferOwnership::Borrowed;
    assert(validate_memory_descriptor(borrowed, 24) == AMIO_ERR_UNSUPPORTED);
    auto overflow = host;
    overflow.extents = {UINT64_MAX, 2};
    overflow.strides = {8, 4};
    assert(validate_memory_descriptor(overflow, 24) == AMIO_ERR_INVALID_INPUT);
    auto device = host;
    device.kind = MemoryKind::Device;
    device.device_id = 1;
    assert(validate_memory_descriptor(device, 24) == AMIO_ERR_UNSUPPORTED);
    std::vector<std::byte> output;
    bool fallback = false;
    assert(snapshot_memory_buffer(device, reinterpret_cast<const void *>(1), 24, output, fallback) == AMIO_ERR_UNSUPPORTED);
    auto rejected_borrow = borrowed;
    assert(snapshot_memory_buffer(rejected_borrow, reinterpret_cast<const void *>(1), 24, output, fallback) == AMIO_ERR_UNSUPPORTED);
    device.device_id = 0;
    assert(validate_memory_descriptor(device, 24) == AMIO_OK);
#ifdef AMIO_TEST_HAS_KOKKOS_RUNTIME
    Kokkos::initialize();
    std::array<std::byte, 24> device_space_source{};
    for (std::size_t i = 0; i < device_space_source.size(); ++i) device_space_source[i] = static_cast<std::byte>(i);
    assert(snapshot_memory_buffer(device, device_space_source.data(), device_space_source.size(), output, fallback) == AMIO_OK);
    assert(fallback && output.size() == device_space_source.size());
    device_space_source[0] = std::byte{99};
    assert(output[0] == std::byte{0});
    Kokkos::finalize();
#endif
    return 0;
}
