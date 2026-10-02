// backend_factory.cpp -- AMIO Backend_Factory implementation.
//
// Implements AMIO's own string-keyed singleton registry for
// Backend_Driver implementations:
//
//   * Case-sensitive exact-match lookup (R4.1).
//   * Static-init registration via BackendRegistrar<T> (R4.2).
//   * Unknown/missing/wrong-case keys → AMIO_ERR_UNKNOWN_BACKEND
//     with zero state mutation (R4.6).
//   * Thread-safe registration and lookup via std::shared_mutex.
//   * Supports concurrent read + write datasets on different drivers
//     within a single AMIO_Core (R4.7) — each build() call returns
//     an independent driver instance.
//
// Validates: R4.1, R4.2, R4.3, R4.4, R4.5, R4.6, R4.7, R4.8

#include "factory/backend_factory.hpp"

#include <algorithm>
#include <mutex>

// Optional force-link references to the backend driver translation units.
//
// Each driver static archive defines an (empty) `amio_register_*_driver`
// symbol and a file-scope `BackendRegistrar<T>` whose constructor registers the
// driver's factory key at static-init time.  Referencing those symbols from
// instance() below makes the linker extract the driver archive members so their
// static initializers fire even under `--gc-sections`/IPO or when the archives
// are linked without `--whole-archive` (the libamio.so build and several PBT
// targets respectively).
//
// The references are declared WEAK so that binaries which compile this
// translation unit directly but never link the driver archives -- the hermetic
// unit tests, which register mock drivers instead -- still link: a weak
// undefined reference is not a link error, it simply resolves to null, and the
// null-check in instance() skips the call.  When the drivers ARE linked the
// weak reference resolves to the real symbol and still forces extraction.
#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define AMIO_FORCE_LINK_WEAK __attribute__((weak))
#else
#define AMIO_FORCE_LINK_WEAK
#endif

extern "C" {
AMIO_FORCE_LINK_WEAK void amio_register_netcdf_driver();
AMIO_FORCE_LINK_WEAK void amio_register_zarr_driver();
AMIO_FORCE_LINK_WEAK void amio_register_grib2_driver();
}

namespace amio::detail {

// Singleton instance -- Meyers' singleton, thread-safe per C++11.
BackendFactory &BackendFactory::instance() {
    // Explicitly reference the driver registration functions to force the linker
    // to include the static initializers of all driver translation units,
    // preventing optimization stripping under aggressive compiler flags (e.g. IPO on Intel).
    //
    // Each reference is guarded by a null-check: the symbols are declared weak
    // above, so a binary that does not link a driver archive resolves that symbol
    // to null and the `!= nullptr` check below skips the call (see the comment
    // on the declarations).
    static bool forced = false;
    if (!forced) {
        if (amio_register_netcdf_driver != nullptr) {
            amio_register_netcdf_driver();
        }
        if (amio_register_zarr_driver != nullptr) {
            amio_register_zarr_driver();
        }
        if (amio_register_grib2_driver != nullptr) {
            amio_register_grib2_driver();
        }
        forced = true;
    }

    static BackendFactory factory;
    return factory;
}

bool BackendFactory::register_driver(const std::string &key, BuilderFn builder) {
    if (key.empty()) {
        return false;
    }
    std::unique_lock lock(mu_);
    registry_[key] = std::move(builder);
    return true;
}

std::unique_ptr<Backend_Driver> BackendFactory::build(const std::string &key, amio_err_t &err_out) const {
    // Empty or missing key → error, zero state mutation.
    if (key.empty()) {
        err_out = AMIO_ERR_UNKNOWN_BACKEND;
        return nullptr;
    }

    std::shared_lock lock(mu_);
    auto it = registry_.find(key);
    if (it == registry_.end()) {
        // Key not found — case-sensitive exact-match failed.
        err_out = AMIO_ERR_UNKNOWN_BACKEND;
        return nullptr;
    }

    // Found — invoke the builder to create a fresh driver instance.
    // Each call produces an independent instance, supporting
    // concurrent read + write datasets on different drivers (R4.7).
    err_out = AMIO_OK;
    return it->second();
}

bool BackendFactory::has(const std::string &key) const {
    if (key.empty()) {
        return false;
    }
    std::shared_lock lock(mu_);
    return registry_.find(key) != registry_.end();
}

std::vector<std::string> BackendFactory::registered_keys() const {
    std::shared_lock lock(mu_);
    std::vector<std::string> keys;
    keys.reserve(registry_.size());
    for (const auto &[k, _] : registry_) {
        keys.push_back(k);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

void BackendFactory::clear() {
    std::unique_lock lock(mu_);
    registry_.clear();
}

}  // namespace amio::detail
