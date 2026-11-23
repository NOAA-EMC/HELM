// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/detail/raii_handles.cpp
/// @brief Implementation of Proj_Handle and File_Handle RAII wrappers.

#include "axis/detail/raii_handles.hpp"

#ifdef AXIS_ENABLE_PROJ
#include <proj.h>
#endif

#include <string>
#include <utility>  // std::exchange

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// Proj_Handle implementation (guarded by AXIS_ENABLE_PROJ)
// ─────────────────────────────────────────────────────────────────────────────

#ifdef AXIS_ENABLE_PROJ

Proj_Handle::Proj_Handle(const char *proj_string, PJ_CONTEXT *ctx) : pj_(proj_create(ctx, proj_string)) {
    if (pj_ == nullptr) {
        // Retrieve PROJ error text for a descriptive exception message.
        int err = proj_context_errno(ctx);
        const char *err_text = proj_errno_string(err);
        throw std::runtime_error(std::string("Proj_Handle: proj_create failed: ") + (err_text ? err_text : "unknown PROJ error"));
    }
}

Proj_Handle::~Proj_Handle() noexcept {
    if (pj_ != nullptr) {
        proj_destroy(pj_);
    }
}

Proj_Handle::Proj_Handle(Proj_Handle &&other) noexcept : pj_(std::exchange(other.pj_, nullptr)) {}

Proj_Handle &Proj_Handle::operator=(Proj_Handle &&other) noexcept {
    if (this != &other) {
        if (pj_ != nullptr) {
            proj_destroy(pj_);
        }
        pj_ = std::exchange(other.pj_, nullptr);
    }
    return *this;
}

#endif  // AXIS_ENABLE_PROJ

// ─────────────────────────────────────────────────────────────────────────────
// File_Handle implementation
// ─────────────────────────────────────────────────────────────────────────────

File_Handle::File_Handle(const char *path, const char *mode) : fp_(std::fopen(path, mode)) {
    if (fp_ == nullptr) {
        throw std::runtime_error(std::string("File_Handle: failed to open file: ") + path);
    }
}

File_Handle::~File_Handle() noexcept {
    if (fp_ != nullptr) {
        // Swallow errors — noexcept destructor (Requirement 17.5).
        static_cast<void>(std::fclose(fp_));
    }
}

File_Handle::File_Handle(File_Handle &&other) noexcept : fp_(std::exchange(other.fp_, nullptr)) {}

File_Handle &File_Handle::operator=(File_Handle &&other) noexcept {
    if (this != &other) {
        if (fp_ != nullptr) {
            static_cast<void>(std::fclose(fp_));
        }
        fp_ = std::exchange(other.fp_, nullptr);
    }
    return *this;
}

}  // namespace axis::detail
