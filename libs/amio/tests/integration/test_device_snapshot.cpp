#include "memory/memory_descriptor.hpp"

#include <Kokkos_Core.hpp>

#include <cassert>
#include <cstddef>
#include <vector>

int main(int argc, char **argv) {
    amio::detail::MemoryDescriptor descriptor;
    descriptor.kind = amio::detail::MemoryKind::Device;
    descriptor.ownership = amio::detail::BufferOwnership::Copy;
    descriptor.device_id = 0;
    descriptor.producer_writes_synchronized = true;
    descriptor.rank = 1;
    descriptor.element_bytes = sizeof(std::byte);
    descriptor.extents = {8};
    descriptor.strides = {1};

    std::vector<std::byte> snapshot;
    bool fallback = false;
    assert(amio::detail::snapshot_memory_buffer(descriptor, reinterpret_cast<const void *>(1), 8,
                                               snapshot, fallback) == AMIO_ERR_UNSUPPORTED);
    assert(snapshot.empty());

    Kokkos::initialize(argc, argv);
    {
        Kokkos::View<std::byte *> source("amio_device_snapshot_source", 8);
        Kokkos::parallel_for("initialize_amio_source", Kokkos::RangePolicy<>(0, 8), KOKKOS_LAMBDA(const int i) {
            source(i) = static_cast<std::byte>(i + 1);
        });
        Kokkos::fence("producer writes complete");

        assert(amio::detail::snapshot_memory_buffer(descriptor, source.data(), 8, snapshot, fallback) == AMIO_OK);
        assert(fallback);
        for (std::size_t i = 0; i < snapshot.size(); ++i) assert(snapshot[i] == static_cast<std::byte>(i + 1));

        // The successful return is the source-release boundary: source may be
        // immediately overwritten while the owned snapshot remains unchanged.
        Kokkos::deep_copy(source, std::byte{99});
        Kokkos::fence("source reuse after snapshot");
        for (std::size_t i = 0; i < snapshot.size(); ++i) assert(snapshot[i] == static_cast<std::byte>(i + 1));
    }
    Kokkos::finalize();
    return 0;
}
