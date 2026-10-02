// thread_pinning.hpp -- AMIO per-thread CPU/NUMA pinning abstraction.
//
// This header is PRIVATE to the AMIO_Core build (`src/workers/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// Provides an abstraction layer for binding worker threads to specific
// CPU cores or NUMA domains.  When eckit is available, the
// implementation delegates to eckit resource binding facilities.  When
// eckit is not linked (standalone build), the implementation uses
// platform-specific APIs (pthread_setaffinity_np on Linux) with a
// fallback that returns AMIO_ERR_INVALID_BINDING for non-default
// configurations.
//
// Thread safety
// -------------
// apply_thread_pinning() is intended to be called once per thread at
// thread start.  It is safe to call concurrently from different
// threads (each call affects only the calling thread's affinity).
//
// Validates: R3.2, R3.3

#ifndef AMIO_SRC_WORKERS_THREAD_PINNING_HPP
#define AMIO_SRC_WORKERS_THREAD_PINNING_HPP

#include <cstdint>
#include <vector>

#include "amio/amio_errors.h"

namespace amio::detail {

// ThreadConfig -- configuration for per-thread CPU/NUMA binding.
//
// Fields:
//   cpu_cores   - list of CPU core IDs to pin the thread to.
//                 An empty list means "no pinning" (use default
//                 affinity inherited from the parent process).
//   numa_domain - NUMA domain identifier to bind to.  A value of
//                 -1 means "no NUMA binding" (use default).
//                 When both cpu_cores and numa_domain are specified,
//                 cpu_cores takes precedence.
struct ThreadConfig {
    std::vector<int> cpu_cores;
    int numa_domain = -1;

    // Returns true if no pinning is requested (empty cores and
    // no NUMA domain).
    bool is_default() const noexcept {
        return cpu_cores.empty() && numa_domain < 0;
    }
};

// apply_thread_pinning -- bind the calling thread to the cores or
// NUMA domain specified in `config`.
//
// Behavior:
//   * If config.is_default() is true, the function is a no-op and
//     returns AMIO_OK (host affinity is left unchanged).
//
//   * If cpu_cores is non-empty, the function attempts to set the
//     calling thread's CPU affinity to exactly those cores.  If any
//     core ID does not exist on the host or the process lacks
//     permission to bind to it, the function returns
//     AMIO_ERR_INVALID_BINDING and leaves the calling thread's
//     affinity unchanged.
//
//   * If cpu_cores is empty but numa_domain >= 0, the function
//     attempts to bind the calling thread to all cores in the
//     specified NUMA domain.  If the domain does not exist or the
//     process lacks permission, returns AMIO_ERR_INVALID_BINDING
//     with affinity unchanged.
//
// Returns:
//   AMIO_OK on success.
//   AMIO_ERR_INVALID_BINDING if the requested binding is invalid
//   or cannot be applied.
amio_err_t apply_thread_pinning(const ThreadConfig &config);

// validate_thread_config -- pre-validate a ThreadConfig without
// actually applying it.
//
// This is intended to be called during amio_init (on the calling
// thread) to detect invalid bindings BEFORE creating the WorkerPool.
// If this returns AMIO_ERR_INVALID_BINDING, the caller should fail
// init with that error code and not create the pool (R3.3).
//
// Behavior:
//   * If config.is_default(), returns AMIO_OK.
//   * If cpu_cores is non-empty, validates that all core IDs are
//     within the set of CPUs available to this process.
//   * If numa_domain >= 0, validates that the domain exists and
//     has at least one CPU.
//
// Returns:
//   AMIO_OK if the config is valid.
//   AMIO_ERR_INVALID_BINDING if any binding is invalid.
amio_err_t validate_thread_config(const ThreadConfig &config);

// query_available_cpus -- return the number of CPUs available to
// the calling process.
//
// Returns 0 if the query is not supported on this platform.
int query_available_cpus();

}  // namespace amio::detail

#endif  // AMIO_SRC_WORKERS_THREAD_PINNING_HPP
