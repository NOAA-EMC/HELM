/**
 * @file amio_export.h
 * @brief AMIO public-symbol export macro (C99 surface).
 *
 * Provides the `AMIO_API` macro that decorates every `extern "C"`
 * AMIO_C_API function declaration in the public headers.  The macro
 * resolves to:
 *
 *   - on ELF platforms (Linux / macOS) under GCC / Clang:
 *         `__attribute__((visibility("default")))`
 *     so the function is exported even though the project compiles
 *     with `-fvisibility=hidden`.
 *
 *   - on Windows under MSVC:
 *         `__declspec(dllexport)` when AMIO_BUILDING_LIBRARY is defined,
 *         `__declspec(dllimport)` for consumers.
 *
 *   - everywhere else (or when AMIO_STATIC_BUILD is set): empty.
 *
 * Header-isolation contract (R10.1, R10.2, R13.6):
 *   - Includes nothing.
 *   - No C++ types, no third-party headers, no std:: symbols.
 *   - Compiles cleanly under
 *         `gcc -std=c99 -pedantic -Werror -c amio_export.h`
 *
 * @note Consumers MUST NOT define AMIO_API themselves; the macro is
 *       fully controlled by the AMIO build system.
 */

#ifndef AMIO_AMIO_EXPORT_H
#define AMIO_AMIO_EXPORT_H

/**
 * @def AMIO_API
 * @brief Public symbol visibility macro for AMIO_C_API entry points.
 *
 * Decorates every `extern "C"` AMIO_C_API entry point to control
 * dynamic symbol visibility. On ELF platforms with GCC/Clang, this
 * expands to `__attribute__((visibility("default")))`. On Windows,
 * it expands to `__declspec(dllexport)` or `__declspec(dllimport)`
 * depending on whether the library is being built or consumed.
 *
 * Usage:
 * @code
 * AMIO_API amio_status_t amio_init(const char *manifest_path,
 *                                   amio_core_handle *out_core);
 * @endcode
 */
#if defined(AMIO_STATIC_BUILD)
#define AMIO_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#if defined(AMIO_BUILDING_LIBRARY) || defined(AMIO_BUILDING_DLL)
#if defined(__GNUC__)
#define AMIO_API __attribute__((dllexport))
#else
#define AMIO_API __declspec(dllexport)
#endif
#else
#if defined(__GNUC__)
#define AMIO_API __attribute__((dllimport))
#else
#define AMIO_API __declspec(dllimport)
#endif
#endif
#elif defined(__GNUC__) && (__GNUC__ >= 4)
#define AMIO_API __attribute__((visibility("default")))
#elif defined(__clang__)
#define AMIO_API __attribute__((visibility("default")))
#else
#define AMIO_API
#endif

#endif /* AMIO_AMIO_EXPORT_H */
