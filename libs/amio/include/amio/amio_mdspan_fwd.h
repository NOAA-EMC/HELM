/**
 * @file amio_mdspan_fwd.h
 * @brief Forward declarations of kokkos/mdspan shape descriptors as opaque C structs.
 *
 * Internally, AMIO_Core uses `kokkos/mdspan` (a C++ class template)
 * to express N-dimensional non-owning views over Staging_Pool buffers
 * and host pointers. The public C99 surface MUST NOT see any of those
 * C++ types (R10.1, R10.2, R13.6), so this header declares opaque
 * struct tags that the C-Boundary translation unit is free to cast
 * to/from concrete C++ mdspan instantiations behind the scenes.
 *
 * Header-isolation contract (R10.1, R10.2, R13.6):
 *   - Includes only `<stdint.h>`.
 *   - All forward declarations are opaque (incomplete struct types
 *     used by pointer only).
 *   - No C++ template, namespace, class, or std:: symbol.
 *   - All declarations wrapped in `extern "C"` for C++ inclusion (R10.3).
 */

#ifndef AMIO_AMIO_MDSPAN_FWD_H
#define AMIO_AMIO_MDSPAN_FWD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @struct amio_mdspan_extents_descriptor
 * @brief Opaque forward declaration of the mdspan extents descriptor.
 *
 * Host code interacts with this tag exclusively through pointer-typed
 * parameters; the underlying struct definition is provided privately
 * inside the AMIO_Core build.
 */
struct amio_mdspan_extents_descriptor;

/**
 * @struct amio_mdspan_layout_descriptor
 * @brief Opaque forward declaration of the mdspan layout descriptor.
 */
struct amio_mdspan_layout_descriptor;

/**
 * @struct amio_mdspan_accessor_descriptor
 * @brief Opaque forward declaration of the mdspan accessor descriptor.
 */
struct amio_mdspan_accessor_descriptor;

/**
 * @struct amio_mdspan_view_descriptor
 * @brief Opaque forward declaration of the mdspan view descriptor.
 */
struct amio_mdspan_view_descriptor;

/**
 * @typedef amio_mdspan_extents_t
 * @brief Pointer to opaque mdspan extents descriptor.
 */
typedef struct amio_mdspan_extents_descriptor *amio_mdspan_extents_t;

/**
 * @typedef amio_mdspan_layout_t
 * @brief Pointer to opaque mdspan layout descriptor.
 */
typedef struct amio_mdspan_layout_descriptor *amio_mdspan_layout_t;

/**
 * @typedef amio_mdspan_accessor_t
 * @brief Pointer to opaque mdspan accessor descriptor.
 */
typedef struct amio_mdspan_accessor_descriptor *amio_mdspan_accessor_t;

/**
 * @typedef amio_mdspan_view_t
 * @brief Pointer to opaque mdspan view descriptor.
 */
typedef struct amio_mdspan_view_descriptor *amio_mdspan_view_t;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AMIO_AMIO_MDSPAN_FWD_H */
