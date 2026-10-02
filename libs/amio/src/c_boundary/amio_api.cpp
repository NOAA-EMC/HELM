// amio_api.cpp -- AMIO public C-Boundary translation unit.
//
// This file is the *only* place in the AMIO codebase where the
// `extern "C"` AMIO_C_API entry points (declared in
// `<amio/amio.h>`) are *defined*.  It is PRIVATE to the AMIO_Core
// build and is never installed.
//
// Every entry point follows the same shape:
//
//   1. NULL-handle check               -> AMIO_ERR_NULL_HANDLE
//   2. NULL out-pointer / argument check -> AMIO_ERR_INVALID_INPUT
//   3. HandleTable::lookup with the expected kind:
//        * stale / wrong-kind / generation mismatch
//                                       -> AMIO_ERR_INVALID_HANDLE
//   4. invoke the private C++ entry in `amio::detail::*`
//        wrapped in a try / catch cordon that translates any
//        thrown exception (conf::Conf_Error, std::exception, ...)
//        into a documented AMIO_ERR_* code (R12.2).
//
// This shape is captured in the `kind_dispatch` template below so
// every API entry point stays a thin sliver of code.  Because the
// translation cordon is *the* mechanism by which AMIO_Core's C++
// exceptions never escape into the host application, it is also
// stress-tested by the Property 16 ("Exception bridge invariant")
// PBT in task 13.17.
//
// Validates: R10.5, R10.6, R10.7, R10.8, R12.2

// Flag the build to apply __declspec(dllexport) on Windows when
// AMIO_API decorates a declaration in this translation unit.  On
// ELF / Mach-O targets this define is a no-op; visibility is
// controlled by the `__attribute__((visibility("default")))` part
// of the AMIO_API macro, not by build-flag handshakes.
#define AMIO_BUILDING_LIBRARY 1

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <new>  // std::bad_alloc
#include <type_traits>

#include "amio/amio.h"
#include "c_boundary/amio_core.hpp"
#include "c_boundary/handle_table.hpp"

// The exception cordon translates any C++ exception thrown from
// the private implementation into a documented AMIO_ERR_* code so
// the host application never observes a C++ throw (R12.2).  The
// Exception_Bridge (task 6) handles fine-grained conf::Conf_Error
// mapping; the C-Boundary's cordon is a coarse safety net.

namespace {

using amio::detail::HandleKind;
using amio::detail::HandleTable;
using amio::detail::process_handle_table;

// translate_unknown -- generic last-resort translation used by the
// `catch (...)` arm of every API entry point.  We treat any
// unrecognized exception as a backend failure rather than letting
// it propagate across the C ABI (which is undefined behavior).
constexpr amio_status_t translate_unknown() noexcept {
    return AMIO_ERR_BACKEND_FAILURE;
}

// kind_dispatch -- shared scaffolding for every dataset/io/view
// entry point that requires a lookup-then-call sequence.
//
// `expected` discriminates the handle kind to expect.
// `op` is a callable taking `void *payload` (non-null on entry)
// and returning amio_status_t.
//
// The caller is responsible for any *additional* argument checks
// (e.g. a null `out_io` pointer for amio_write); those checks
// happen before `kind_dispatch` is called so we don't pay the
// table-lookup cost for inputs we already know are bad.
template <typename Op>
amio_status_t kind_dispatch(void *handle, HandleKind expected, Op &&op) noexcept {
    if (handle == nullptr) {
        return AMIO_ERR_NULL_HANDLE;
    }
    void *payload = nullptr;
    const auto token = HandleTable::from_ptr(handle);
    const amio_status_t lookup_rc = process_handle_table().lookup(token, expected, &payload);
    if (lookup_rc != AMIO_OK) {
        return lookup_rc;
    }

    // Exception cordon.  The catch hierarchy translates C++ exceptions
    // to AMIO_ERR_* codes so the host never observes a C++ throw (R12.2).
    try {
        return std::forward<Op>(op)(payload);
    } catch (const std::bad_alloc &) {
        // Out-of-memory in the C++ core surfaces as a backend
        // failure; AMIO_Core does not currently differentiate
        // OOM from other backend errors at the FFI seam.
        std::cerr << "[AMIO ERROR] C-boundary caught std::bad_alloc while dispatching API call" << std::endl;
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << "[AMIO ERROR] C-boundary caught std::exception while dispatching API call: " << e.what() << std::endl;
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        std::cerr << "[AMIO ERROR] C-boundary caught unknown exception while dispatching API call" << std::endl;
        return translate_unknown();
    }
}

}  // namespace

// ---------------------------------------------------------------------
// AMIO_C_API entry points
// ---------------------------------------------------------------------

extern "C" {

AMIO_API amio_status_t amio_open_dataset(amio_core_handle core, const char *config_path, int32_t mode, amio_dataset_handle *out_dataset) {
    if (out_dataset == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_dataset = nullptr;
    if (config_path == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (mode != AMIO_MODE_WRITE && mode != AMIO_MODE_READ) {
        return AMIO_ERR_INVALID_INPUT;
    }
    return kind_dispatch(core, HandleKind::Core,
                         [&](void *payload) -> amio_status_t { return amio::detail::open_dataset(payload, config_path, mode, out_dataset); });
}

AMIO_API amio_status_t amio_open_dataset_from_string(amio_core_handle core, const char *config_content, const char *format, int32_t mode,
                                                     amio_dataset_handle *out_dataset) {
    if (out_dataset == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_dataset = nullptr;
    if (config_content == nullptr || format == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    if (mode != AMIO_MODE_WRITE && mode != AMIO_MODE_READ) {
        return AMIO_ERR_INVALID_INPUT;
    }
    return kind_dispatch(core, HandleKind::Core, [&](void *payload) -> amio_status_t {
        return amio::detail::open_dataset_from_string(payload, config_content, format, mode, out_dataset);
    });
}

AMIO_API amio_status_t amio_close_dataset(amio_dataset_handle dataset) {
    return kind_dispatch(dataset, HandleKind::Dataset, [](void *payload) -> amio_status_t { return amio::detail::close_dataset(payload); });
}

AMIO_API amio_status_t amio_init(const char *manifest_path, amio_core_handle *out_core) {
    if (out_core == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_core = nullptr;
    if (manifest_path == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    // amio_init does NOT consume an existing handle, so it lives
    // outside the kind_dispatch scaffolding -- there is nothing to
    // look up.  We still apply the same exception cordon so the
    // host never observes a C++ throw.
    try {
        return amio::detail::init(manifest_path, out_core);
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (const std::exception &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        return translate_unknown();
    }
}

AMIO_API amio_status_t amio_init_from_string(const char *manifest_content, const char *format, amio_core_handle *out_core) {
    if (out_core == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_core = nullptr;
    if (manifest_content == nullptr || format == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    // Like amio_init, this entry point does NOT consume an existing
    // handle, so it lives outside the kind_dispatch scaffolding --
    // there is nothing to look up.  We still apply the same exception
    // cordon so the host never observes a C++ throw.
    try {
        return amio::detail::init_from_string(manifest_content, format, out_core);
    } catch (const std::bad_alloc &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (const std::exception &) {
        return AMIO_ERR_BACKEND_FAILURE;
    } catch (...) {
        return translate_unknown();
    }
}

AMIO_API amio_status_t amio_finalize(amio_core_handle core) {
    return kind_dispatch(core, HandleKind::Core, [](void *payload) -> amio_status_t { return amio::detail::finalize(payload); });
}

AMIO_API amio_status_t amio_write(amio_dataset_handle dataset, const char *var_name, const void *host_data, amio_dtype_t dtype,
                                  const amio_shape_t *shape, amio_io_handle *out_io) {
    if (out_io == nullptr) {
        // We refuse to validate the dataset handle when the out
        // pointer is invalid; this preserves the contract that
        // we never mutate caller-supplied output on failure.
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_io = nullptr;
    if (var_name == nullptr || host_data == nullptr || shape == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    return kind_dispatch(dataset, HandleKind::Dataset,
                         [&](void *payload) -> amio_status_t { return amio::detail::write(payload, var_name, host_data, dtype, shape, out_io); });
}

AMIO_API amio_status_t amio_describe(amio_dataset_handle dataset, const char *var_name, amio_shape_t *out_shape, int64_t *out_total_timesteps) {
    if (out_shape == nullptr || out_total_timesteps == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_shape = amio_shape_t{};
    *out_total_timesteps = 0;
    if (var_name == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    return kind_dispatch(dataset, HandleKind::Dataset,
                         [&](void *payload) -> amio_status_t { return amio::detail::describe(payload, var_name, out_shape, out_total_timesteps); });
}

AMIO_API amio_status_t amio_read(amio_dataset_handle dataset, const char *var_name, int64_t timestep, const amio_bbox_t *bbox,
                                 amio_view_handle *out_view) {
    if (out_view == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_view = nullptr;
    if (var_name == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    return kind_dispatch(dataset, HandleKind::Dataset,
                         [&](void *payload) -> amio_status_t { return amio::detail::read(payload, var_name, timestep, bbox, out_view); });
}

AMIO_API amio_status_t amio_get_var_attribute_text(amio_dataset_handle dataset, const char *var_name, const char *attr_name, char *out_buf,
                                                   size_t buf_cap, size_t *out_len) {
    if (attr_name == nullptr || out_len == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_len = 0;
    return kind_dispatch(dataset, HandleKind::Dataset, [&](void *payload) -> amio_status_t {
        return amio::detail::get_var_attribute_text(payload, var_name, attr_name, out_buf, buf_cap, out_len);
    });
}

AMIO_API amio_status_t amio_get_var_attribute_double(amio_dataset_handle dataset, const char *var_name, const char *attr_name, double *out_value) {
    if (attr_name == nullptr || out_value == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_value = 0.0;
    return kind_dispatch(dataset, HandleKind::Dataset, [&](void *payload) -> amio_status_t {
        return amio::detail::get_var_attribute_double(payload, var_name, attr_name, out_value);
    });
}

AMIO_API amio_status_t amio_flush(amio_dataset_handle dataset, int64_t timeout_ms) {
    return kind_dispatch(dataset, HandleKind::Dataset, [&](void *payload) -> amio_status_t { return amio::detail::flush(payload, timeout_ms); });
}

AMIO_API amio_status_t amio_close(amio_dataset_handle dataset) {
    return kind_dispatch(dataset, HandleKind::Dataset, [](void *payload) -> amio_status_t { return amio::detail::close(payload); });
}

AMIO_API amio_status_t amio_wait(amio_io_handle io, int64_t timeout_ms) {
    return kind_dispatch(io, HandleKind::Io, [&](void *payload) -> amio_status_t { return amio::detail::wait(payload, timeout_ms); });
}

AMIO_API amio_status_t amio_release_view(amio_view_handle view) {
    return kind_dispatch(view, HandleKind::View, [](void *payload) -> amio_status_t { return amio::detail::release_view(payload); });
}

AMIO_API amio_status_t amio_view_data(amio_view_handle view, const void **out_data, size_t *out_size) {
    if (out_data == nullptr || out_size == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_data = nullptr;
    *out_size = 0;
    return kind_dispatch(view, HandleKind::View,
                         [&](void *payload) -> amio_status_t { return amio::detail::view_data(payload, out_data, out_size); });
}

AMIO_API amio_status_t amio_view_shape(amio_view_handle view, amio_shape_t *out_shape) {
    if (out_shape == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    std::memset(out_shape, 0, sizeof(amio_shape_t));
    return kind_dispatch(view, HandleKind::View, [&](void *payload) -> amio_status_t { return amio::detail::view_shape(payload, out_shape); });
}

AMIO_API amio_status_t amio_view_dtype(amio_view_handle view, amio_dtype_t *out_dtype) {
    if (out_dtype == nullptr) {
        return AMIO_ERR_INVALID_INPUT;
    }
    *out_dtype = AMIO_DTYPE_F32;
    return kind_dispatch(view, HandleKind::View, [&](void *payload) -> amio_status_t { return amio::detail::view_dtype(payload, out_dtype); });
}

// amio_strerror is intentionally NOT defined in this translation
// unit.  Its implementation lives in `amio_strerror.cpp` (task 3.3)
// alongside the static error-code description table.

}  // extern "C"
