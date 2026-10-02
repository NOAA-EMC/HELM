// thread_pinning.cpp -- AMIO per-thread CPU/NUMA pinning implementation.
//
// Platform-specific implementation of thread affinity binding.
//
// When eckit is available (AMIO_HAS_ECKIT defined), delegates to
// eckit resource binding facilities.  Otherwise uses:
//   - Linux: pthread_setaffinity_np / sched_getaffinity
//   - Other platforms: returns AMIO_ERR_INVALID_BINDING for
//     non-default configs (graceful degradation).
//
// Validates: R3.2, R3.3

#include "workers/thread_pinning.hpp"

#include <algorithm>

// Platform detection for CPU affinity APIs.
#if defined(__linux__)
#define AMIO_HAS_PTHREAD_AFFINITY 1
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#elif defined(__APPLE__)
#include <unistd.h>
#define AMIO_HAS_PTHREAD_AFFINITY 0
#else
#define AMIO_HAS_PTHREAD_AFFINITY 0
#endif

namespace amio::detail {

namespace {

#if AMIO_HAS_PTHREAD_AFFINITY

// Validate that all requested core IDs are within the set of CPUs
// available to this process.
bool validate_cpu_cores(const std::vector<int> &cores) {
    if (cores.empty()) {
        return true;
    }

    // Get the set of CPUs available to this process.
    cpu_set_t available;
    CPU_ZERO(&available);
    if (sched_getaffinity(0, sizeof(available), &available) != 0) {
        // Cannot query affinity -- treat as invalid.
        return false;
    }

    for (int core : cores) {
        if (core < 0 || core >= CPU_SETSIZE) {
            return false;
        }
        if (!CPU_ISSET(core, &available)) {
            return false;
        }
    }
    return true;
}

// Get the list of CPUs belonging to a NUMA domain by reading
// /sys/devices/system/node/nodeN/cpulist.
// Returns an empty vector if the domain doesn't exist or can't be read.
std::vector<int> get_numa_domain_cpus(int domain) {
    if (domain < 0) {
        return {};
    }

    std::string path = "/sys/devices/system/node/node" + std::to_string(domain) + "/cpulist";
    std::ifstream f(path);
    if (!f.is_open()) {
        return {};
    }

    std::string line;
    if (!std::getline(f, line)) {
        return {};
    }

    // Parse the cpulist format: "0-3,8-11" or "0,1,2,3"
    std::vector<int> cpus;
    std::istringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
        auto dash = token.find('-');
        if (dash != std::string::npos) {
            int start = std::stoi(token.substr(0, dash));
            int end = std::stoi(token.substr(dash + 1));
            for (int i = start; i <= end; ++i) {
                cpus.push_back(i);
            }
        } else {
            cpus.push_back(std::stoi(token));
        }
    }

    return cpus;
}

amio_err_t pin_to_cores(const std::vector<int> &cores) {
    if (!validate_cpu_cores(cores)) {
        return AMIO_ERR_INVALID_BINDING;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int core : cores) {
        CPU_SET(core, &cpuset);
    }

    pthread_t self = pthread_self();
    int rc = pthread_setaffinity_np(self, sizeof(cpuset), &cpuset);
    if (rc != 0) {
        return AMIO_ERR_INVALID_BINDING;
    }

    return AMIO_OK;
}

#endif  // AMIO_HAS_PTHREAD_AFFINITY

}  // anonymous namespace

amio_err_t apply_thread_pinning(const ThreadConfig &config) {
    // No pinning requested -- no-op, leave host affinity unchanged.
    if (config.is_default()) {
        return AMIO_OK;
    }

#if AMIO_HAS_PTHREAD_AFFINITY
    // CPU core list takes precedence over NUMA domain.
    if (!config.cpu_cores.empty()) {
        return pin_to_cores(config.cpu_cores);
    }

    // NUMA domain binding: resolve domain to its CPU list, then pin.
    if (config.numa_domain >= 0) {
        std::vector<int> domain_cpus = get_numa_domain_cpus(config.numa_domain);
        if (domain_cpus.empty()) {
            // Domain doesn't exist or can't be read.
            return AMIO_ERR_INVALID_BINDING;
        }
        return pin_to_cores(domain_cpus);
    }

    // Should not reach here given is_default() check above.
    return AMIO_OK;

#else
    // Platform does not support pthread affinity.
    // Non-default configs cannot be applied -- return error.
    // This satisfies R3.3: invalid bindings fail init with
    // AMIO_ERR_INVALID_BINDING, leave host affinity untouched.
    return AMIO_ERR_INVALID_BINDING;
#endif
}

int query_available_cpus() {
#if defined(__linux__) || defined(__APPLE__)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return (n > 0) ? static_cast<int>(n) : 0;
#else
    return 0;
#endif
}

amio_err_t validate_thread_config(const ThreadConfig &config) {
    // No pinning requested -- always valid.
    if (config.is_default()) {
        return AMIO_OK;
    }

#if AMIO_HAS_PTHREAD_AFFINITY
    // CPU core list takes precedence over NUMA domain.
    if (!config.cpu_cores.empty()) {
        if (!validate_cpu_cores(config.cpu_cores)) {
            return AMIO_ERR_INVALID_BINDING;
        }
        return AMIO_OK;
    }

    // NUMA domain binding: check that the domain exists and has CPUs.
    if (config.numa_domain >= 0) {
        std::vector<int> domain_cpus = get_numa_domain_cpus(config.numa_domain);
        if (domain_cpus.empty()) {
            return AMIO_ERR_INVALID_BINDING;
        }
        return AMIO_OK;
    }

    return AMIO_OK;

#else
    // Platform does not support pthread affinity.
    // Non-default configs cannot be validated -- return error (R3.3).
    return AMIO_ERR_INVALID_BINDING;
#endif
}

}  // namespace amio::detail
