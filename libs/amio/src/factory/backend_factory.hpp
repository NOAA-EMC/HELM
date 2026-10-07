// backend_factory.hpp -- AMIO Backend_Factory dispatcher.
//
// This header is PRIVATE to the AMIO_Core build (`src/factory/`).
// It is never installed and must not be referenced from any public
// header under `include/amio/`.
//
// Purpose
// -------
// Backend_Factory is AMIO's own string-keyed singleton registry of
// Backend_Driver implementations.  Each concrete driver registers
// itself at static initialization time via a static
// BackendRegistrar<ConcreteDriver> instance.
//
// The factory provides:
//
//   * Case-sensitive exact-match lookup of registered driver keys.
//   * Thread-safe registration and lookup (std::shared_mutex).
//   * Returns std::unique_ptr<Backend_Driver> on successful lookup.
//   * Returns AMIO_ERR_UNKNOWN_BACKEND on failure with zero state
//     mutation (no partial dataset handle, no driver instantiation).
//   * Support for concurrent read + write datasets on different
//     drivers within a single AMIO_Core context.
//
// No concrete driver class, header, or symbol is part of the public
// API.  The factory key is the only externally visible coupling (R4.8).
//
// Thread safety
// -------------
// All public methods are safe to call concurrently from any thread.
// Registration uses an exclusive lock; lookup uses a shared lock.
//
// Validates: R4.1, R4.2, R4.3, R4.4, R4.5, R4.6, R4.7, R4.8

#ifndef AMIO_SRC_FACTORY_BACKEND_FACTORY_HPP
#define AMIO_SRC_FACTORY_BACKEND_FACTORY_HPP

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "amio/amio_errors.h"
#include "factory/backend_driver.hpp"

namespace amio::detail {

// BackendFactory -- AMIO's own string-keyed singleton registry for
// Backend_Driver implementations.  Concrete drivers register via
// BackendRegistrar<T> at static init time.
class BackendFactory {
   public:
    // Builder function type: creates a new Backend_Driver instance.
    using BuilderFn = std::function<std::unique_ptr<Backend_Driver>()>;

    // instance -- access the singleton factory.
    static BackendFactory &instance();

    // register_driver -- register a builder under the given key.
    //
    // Thread-safe (exclusive lock).  If the key is already registered,
    // the previous registration is silently replaced (last-writer-wins).
    //
    // Returns true if registration succeeded, false if key was empty.
    bool register_driver(const std::string &key, BuilderFn builder);

    // build -- look up a driver by key and instantiate it.
    //
    // Case-sensitive exact-match (R4.1).  On success, returns a
    // unique_ptr to a freshly constructed Backend_Driver.  On failure
    // (key not found, empty key, wrong case), sets `err_out` to
    // AMIO_ERR_UNKNOWN_BACKEND and returns nullptr.
    //
    // Zero state mutation on failure: no partial driver is created,
    // no internal state is modified (R4.6).
    //
    // Thread-safe (shared lock for lookup).
    std::unique_ptr<Backend_Driver> build(const std::string &key, amio_err_t &err_out) const;

    // has -- check if a key is registered (case-sensitive).
    //
    // Thread-safe (shared lock).
    bool has(const std::string &key) const;

    // registered_keys -- return a copy of all registered keys.
    //
    // Thread-safe (shared lock).
    std::vector<std::string> registered_keys() const;

    // clear -- remove all registrations (for testing only).
    //
    // Thread-safe (exclusive lock).
    void clear();

    // Non-copyable, non-movable singleton.
    BackendFactory(const BackendFactory &) = delete;
    BackendFactory &operator=(const BackendFactory &) = delete;
    BackendFactory(BackendFactory &&) = delete;
    BackendFactory &operator=(BackendFactory &&) = delete;

   private:
    BackendFactory() = default;

    mutable std::shared_mutex mu_;
    std::unordered_map<std::string, BuilderFn> registry_;
};

// BackendRegistrar -- RAII static registrar for concrete drivers.
//
// Usage (in a concrete driver .cpp file):
//
//   namespace {
//   amio::detail::BackendRegistrar<NetCDF_Driver> reg_netcdf("netcdf4");
//   }
//
// The registrar registers the driver at static initialization time
// and does NOT unregister on destruction (drivers remain registered
// for the process lifetime).
template <typename ConcreteDriver>
class BackendRegistrar {
   public:
    explicit BackendRegistrar(const std::string &key) {
        BackendFactory::instance().register_driver(key, []() -> std::unique_ptr<Backend_Driver> { return std::make_unique<ConcreteDriver>(); });
    }
};

}  // namespace amio::detail

#endif  // AMIO_SRC_FACTORY_BACKEND_FACTORY_HPP
