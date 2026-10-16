// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

#ifndef AXIS_DETAIL_RAII_HANDLES_HPP
#define AXIS_DETAIL_RAII_HANDLES_HPP

/// @file axis/detail/raii_handles.hpp
/// @brief Move-only RAII wrappers for the two retained C resource handles.
///
/// AXIS retains exactly two C resource handles:
///   - Proj_Handle — wraps PROJ PJ* (optional, guarded by AXIS_ENABLE_PROJ)
///   - File_Handle — wraps std::FILE* (used by GmshWriter for .msh export)
///
/// Both are move-only with noexcept destructors that guarantee resource
/// release on all exit paths, including exceptions (HELM Law #3).
/// AXIS does NOT contain NetCDF_Handle, Grib_Handle, YAML handle, or any
/// other file-format handle — those responsibilities are delegated to AMIO.

#include <cstdio>
#include <stdexcept>

#ifdef AXIS_ENABLE_PROJ
// Keep the public RAII wrapper independent of PROJ's include directories.
// The implementation includes proj.h where the API functions are used.
struct PJconsts;
typedef struct PJconsts PJ;
struct pj_ctx;
typedef struct pj_ctx PJ_CONTEXT;
#endif

namespace axis::detail {

// ─────────────────────────────────────────────────────────────────────────────
// Proj_Handle — RAII wrapper for PROJ PJ* transformation object
// ─────────────────────────────────────────────────────────────────────────────

#ifdef AXIS_ENABLE_PROJ

/// @brief Move-only RAII wrapper for a PROJ PJ* transformation object.
///
/// Acquires a PJ* via proj_create on construction; releases via proj_destroy
/// in the destructor. If proj_create fails (returns nullptr), the constructor
/// throws std::runtime_error with the PROJ error text.
class Proj_Handle {
   public:
    /// Construct from a PROJ pipeline/CRS definition string.
    /// @param proj_string  PROJ pipeline or CRS definition (e.g., "+proj=longlat").
    /// @param ctx          Optional PJ_CONTEXT* (nullptr for the default thread context).
    /// @throws std::runtime_error if proj_create returns nullptr.
    explicit Proj_Handle(const char *proj_string, PJ_CONTEXT *ctx = nullptr);

    /// Destructor: calls proj_destroy if the handle is valid. noexcept.
    ~Proj_Handle() noexcept;

    // Move construction: transfers ownership; source becomes null.
    Proj_Handle(Proj_Handle &&other) noexcept;

    // Move assignment: releases current handle, transfers ownership from other.
    Proj_Handle &operator=(Proj_Handle &&other) noexcept;

    // Non-copyable.
    Proj_Handle(const Proj_Handle &) = delete;
    Proj_Handle &operator=(const Proj_Handle &) = delete;

    /// @brief Access the raw PJ* pointer.
    [[nodiscard]] PJ *get() const noexcept {
        return pj_;
    }

    /// @brief Check whether the handle holds a valid PJ*.
    [[nodiscard]] explicit operator bool() const noexcept {
        return pj_ != nullptr;
    }

   private:
    PJ *pj_ = nullptr;
};

#endif  // AXIS_ENABLE_PROJ

// ─────────────────────────────────────────────────────────────────────────────
// File_Handle — RAII wrapper for std::FILE*
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Move-only RAII wrapper for std::FILE*.
///
/// Acquires a FILE* via std::fopen on construction; releases via std::fclose
/// in the destructor. If fopen fails (returns nullptr), the constructor throws
/// std::runtime_error naming the file path.
class File_Handle {
   public:
    /// Construct by opening a file.
    /// @param path  File path to open.
    /// @param mode  fopen mode string (e.g., "w", "rb").
    /// @throws std::runtime_error if std::fopen returns nullptr.
    explicit File_Handle(const char *path, const char *mode);

    /// Destructor: calls std::fclose if the handle is valid. noexcept.
    ~File_Handle() noexcept;

    // Move construction: transfers ownership; source becomes null.
    File_Handle(File_Handle &&other) noexcept;

    // Move assignment: releases current handle, transfers ownership from other.
    File_Handle &operator=(File_Handle &&other) noexcept;

    // Non-copyable.
    File_Handle(const File_Handle &) = delete;
    File_Handle &operator=(const File_Handle &) = delete;

    /// @brief Access the raw FILE* pointer.
    [[nodiscard]] std::FILE *get() const noexcept {
        return fp_;
    }

    /// @brief Check whether the handle holds a valid FILE*.
    [[nodiscard]] explicit operator bool() const noexcept {
        return fp_ != nullptr;
    }

   private:
    std::FILE *fp_ = nullptr;
};

}  // namespace axis::detail

#endif  // AXIS_DETAIL_RAII_HANDLES_HPP
