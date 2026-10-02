// pbt_common.hpp -- AMIO Property-Based Testing common infrastructure.
//
// Provides reusable helpers shared across all PBT test executables
// (tasks 13.2 - 13.26):
//   * dtype_size, payload_byte_count utilities
//   * Payload struct
//   * TempDir RAII helper
//   * AmioGuard RAII lifecycle wrapper
//   * Manifest YAML generation helpers
//
// CRITICAL CONSTRAINTS:
//   * All property tests exercise the real NetCDF_Driver,
//     Zarr_Driver (NCZarr fallback), and GRIB2_Driver.
//   * Tests link against AMIO::amio_core (the real libamio.so with
//     all drivers registered).
//   * Tests run inside the amio-dev Docker container with all
//     dependencies available (CONF, HALO, LOGS, netCDF, g2c, etc.).
//   * RC_PARAMS max_success=100 is configured via test environment.
//
// Validates: R11.2 (testing infrastructure)

#ifndef AMIO_TESTS_PBT_PBT_COMMON_HPP
#define AMIO_TESTS_PBT_PBT_COMMON_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// Catch2 v3
#include <catch2/catch_test_macros.hpp>

// RapidCheck
#include <rapidcheck.h>

// AMIO public API
#include "amio/amio.h"

// AMIO private headers (accessible via test include paths)
#include "config/config_loader.hpp"
#include "factory/backend_driver.hpp"  // amio::detail::VariableInfo

namespace amio::pbt {

// ===================================================================
// make_var_info -- build an amio::detail::VariableInfo for the
// PrefetchQueue constructor (task 7).
//
// The dtype + extents size the staging acquisition as
// element_size(dtype) * product(extents).  Callers supply a payload
// that fits the test's staging-pool buffer capacity so acquisition
// succeeds and existing byte-count assertions still hold.
// ===================================================================

inline amio::detail::VariableInfo make_var_info(amio_dtype_t dtype, const amio_shape_t &shape, std::int64_t total_timesteps = 1) {
    amio::detail::VariableInfo info{};
    info.found = true;
    info.dtype = dtype;
    info.shape = shape;
    info.total_timesteps = total_timesteps;
    return info;
}

// Convenience overload: build a 1-D VariableInfo with `elements`
// extents of `dtype`.  Used by prefetch scheduling / look-ahead
// tests where only the buffer-sizing payload (not the variable
// shape) is relevant.
inline amio::detail::VariableInfo make_var_info_1d(amio_dtype_t dtype, std::int64_t elements, std::int64_t total_timesteps = 1) {
    amio_shape_t shape = {};
    shape.rank = 1;
    shape.extents[0] = elements;
    return make_var_info(dtype, shape, total_timesteps);
}

// ===================================================================
// dtype_size -- byte size of an amio_dtype_t element.
// ===================================================================

inline std::size_t dtype_size(amio_dtype_t dtype) {
    switch (dtype) {
        case AMIO_DTYPE_F32:
            return 4;
        case AMIO_DTYPE_F64:
            return 8;
        case AMIO_DTYPE_I8:
            return 1;
        case AMIO_DTYPE_I16:
            return 2;
        case AMIO_DTYPE_I32:
            return 4;
        case AMIO_DTYPE_I64:
            return 8;
        case AMIO_DTYPE_U8:
            return 1;
        case AMIO_DTYPE_U16:
            return 2;
        case AMIO_DTYPE_U32:
            return 4;
        case AMIO_DTYPE_U64:
            return 8;
        default:
            return 0;
    }
}

// ===================================================================
// payload_byte_count -- total bytes for a shape + dtype combination.
// ===================================================================

inline std::size_t payload_byte_count(const amio_shape_t &shape, amio_dtype_t dtype) {
    if (shape.rank < 1 || shape.rank > AMIO_MAX_RANK) return 0;
    std::size_t elem_count = 1;
    for (int32_t d = 0; d < shape.rank; ++d) {
        if (shape.extents[d] <= 0) return 0;
        elem_count *= static_cast<std::size_t>(shape.extents[d]);
    }
    return elem_count * dtype_size(dtype);
}

// Alias used by generators.hpp.
inline std::size_t payload_byte_size(const amio_shape_t &shape, amio_dtype_t dtype) {
    return payload_byte_count(shape, dtype);
}

// Element count for a shape.
inline std::size_t shape_element_count(const amio_shape_t &shape) {
    if (shape.rank < 1 || shape.rank > AMIO_MAX_RANK) return 0;
    std::size_t count = 1;
    for (int32_t d = 0; d < shape.rank; ++d) {
        if (shape.extents[d] <= 0) return 0;
        count *= static_cast<std::size_t>(shape.extents[d]);
    }
    return count;
}

// ===================================================================
// Payload -- a typed byte buffer representing a write/read payload.
// ===================================================================

struct Payload {
    amio_dtype_t dtype;
    amio_shape_t shape;
    std::vector<uint8_t> bytes;
};

// ===================================================================
// TempDir -- RAII temporary directory for test output files.
// ===================================================================

class TempDir {
   public:
    TempDir() {
        namespace fs = std::filesystem;
        auto base = fs::temp_directory_path() / "amio_pbt";
        fs::create_directories(base);
        // Create a unique subdirectory via mkdtemp (POSIX).
        std::string tmpl = (base / "XXXXXX").string();
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');
        char *result = mkdtemp(buf.data());
        if (result) {
            path_ = std::string(result);
        } else {
            // Fallback: use a random name.
            path_ = (base / ("t" + std::to_string(std::rand()))).string();
            fs::create_directories(path_);
        }
    }

    ~TempDir() {
        namespace fs = std::filesystem;
        if (!path_.empty()) {
            std::error_code ec;
            fs::remove_all(path_, ec);
        }
    }

    // Non-copyable, movable.
    TempDir(const TempDir &) = delete;
    TempDir &operator=(const TempDir &) = delete;
    TempDir(TempDir &&other) noexcept : path_(std::move(other.path_)) {
        other.path_.clear();
    }
    TempDir &operator=(TempDir &&other) noexcept {
        if (this != &other) {
            namespace fs = std::filesystem;
            if (!path_.empty()) {
                std::error_code ec;
                fs::remove_all(path_, ec);
            }
            path_ = std::move(other.path_);
            other.path_.clear();
        }
        return *this;
    }

    const std::string &path() const {
        return path_;
    }

    // Create a file path within this temp directory.
    std::string file(const std::string &name) const {
        return path_ + "/" + name;
    }

   private:
    std::string path_;
};

// ===================================================================
// AmioGuard -- RAII wrapper for amio_init / amio_finalize lifecycle.
//
// Initializes AMIO from a manifest path and finalizes on destruction.
// ===================================================================

class AmioGuard {
   public:
    explicit AmioGuard(const std::string &manifest_path) : core_(nullptr) {
        amio_status_t rc = amio_init(manifest_path.c_str(), &core_);
        if (rc != AMIO_OK) {
            core_ = nullptr;
        }
        status_ = rc;
    }

    ~AmioGuard() {
        if (core_) {
            amio_finalize(core_);
            core_ = nullptr;
        }
    }

    // Non-copyable, non-movable.
    AmioGuard(const AmioGuard &) = delete;
    AmioGuard &operator=(const AmioGuard &) = delete;

    amio_core_handle handle() const {
        return core_;
    }
    amio_status_t status() const {
        return status_;
    }
    bool ok() const {
        return status_ == AMIO_OK && core_ != nullptr;
    }

   private:
    amio_core_handle core_;
    amio_status_t status_;
};

// ===================================================================
// Manifest YAML generation helpers.
// ===================================================================

// Backend identifiers matching the factory keys.
inline constexpr const char *kBackendNetCDF4 = "netcdf4";
inline constexpr const char *kBackendZarr3 = "zarr3";
inline constexpr const char *kBackendGRIB2 = "grib2";

// Generate a minimal valid manifest YAML string for a given backend.
// The manifest uses small buffer counts and capacities suitable for
// property testing (not performance testing).
inline std::string make_manifest_yaml(const std::string &backend = "netcdf4", std::size_t buffer_count = 4, std::size_t buffer_capacity_bytes = 65536,
                                      std::size_t threads = 1, std::size_t staging_timeout_ms = 5000, const std::string &codec = "blosc") {
    std::string yaml;
    yaml += "staging_pool:\n";
    yaml += "  buffer_count: " + std::to_string(buffer_count) + "\n";
    yaml += "  buffer_capacity_bytes: " + std::to_string(buffer_capacity_bytes) + "\n";
    yaml += "worker_pool:\n";
    yaml += "  threads: " + std::to_string(threads) + "\n";
    yaml += "prefetch:\n";
    yaml += "  depth: 4\n";
    yaml += "  read_timeout_s: 60\n";
    yaml += "staging_timeout_ms: " + std::to_string(staging_timeout_ms) + "\n";
    yaml += "backend: " + backend + "\n";
    yaml += "codec:\n";
    yaml += "  lossless_allow_list:\n";
    yaml += "    - " + codec + "\n";
    yaml += "  active_codec: " + codec + "\n";
    return yaml;
}

// Write a manifest YAML string to a file and return the path.
inline std::string write_manifest(const TempDir &dir, const std::string &yaml, const std::string &filename = "manifest.yaml") {
    std::string path = dir.file(filename);
    std::ofstream ofs(path);
    ofs << yaml;
    ofs.close();
    return path;
}

// Generate a dataset configuration YAML for a specific backend.
inline std::string make_dataset_config_yaml(const std::string &backend, const std::string &output_path, const std::string &codec = "blosc") {
    std::string yaml;
    yaml += "backend: " + backend + "\n";
    yaml += "output_path: " + output_path + "\n";
    yaml += "codec:\n";
    yaml += "  active_codec: " + codec + "\n";
    // Declare the active codec on the lossless allow-list so the backend
    // driver's lossless-codec guard (e.g. NetCDF_Driver::validate_codec,
    // R7.5) accepts it.  The driver reads codec.lossless_allow_list from the
    // dataset config; omitting it leaves the allow-list empty and the open
    // is rejected.
    yaml += "  lossless_allow_list:\n";
    yaml += "    - " + codec + "\n";

    if (backend == "netcdf4") {
        yaml += "data_model: classic\n";
    } else if (backend == "zarr3") {
        yaml += "mode: nczarr\n";
    } else if (backend == "grib2") {
        yaml += "drt: libaec\n";
    }

    return yaml;
}

}  // namespace amio::pbt

#endif  // AMIO_TESTS_PBT_PBT_COMMON_HPP
