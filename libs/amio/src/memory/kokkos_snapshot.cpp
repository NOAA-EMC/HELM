#include "memory/memory_descriptor.hpp"

#ifdef AMIO_HAS_KOKKOS_RUNTIME
#include <Kokkos_Core.hpp>

#include <new>
#include <vector>

namespace amio::detail {

amio_status_t snapshot_kokkos_device_buffer(const void *source, std::size_t bytes,
                                           std::vector<std::byte> &snapshot) {
    if (source == nullptr || bytes == 0 || !Kokkos::is_initialized() || Kokkos::is_finalized())
        return AMIO_ERR_UNSUPPORTED;
    try {
        // The step reservation is acquired by StepCoordinator before dispatch.
        // Allocate the final owned buffer first, then copy into it directly so
        // there is no unbudgeted intermediate payload allocation.
        snapshot.resize(bytes);
        using device_memory = typename Kokkos::DefaultExecutionSpace::memory_space;
        using unmanaged_device_view = Kokkos::View<const std::byte *, device_memory,
                                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        using unmanaged_host_view = Kokkos::View<std::byte *, Kokkos::HostSpace,
                                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
        unmanaged_device_view input(static_cast<const std::byte *>(source), bytes);
        unmanaged_host_view output(snapshot.data(), bytes);
        Kokkos::deep_copy(output, input);
        Kokkos::DefaultExecutionSpace().fence("AMIO device snapshot source release");
        return AMIO_OK;
    } catch (const std::bad_alloc &) {
        snapshot.clear();
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        snapshot.clear();
        return AMIO_ERR_BACKEND_FAILURE;
    }
}

} // namespace amio::detail
#endif
