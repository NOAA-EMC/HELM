// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/projection_builder.cpp
/// @brief ProjectionBuilder implementation — PROJ-based coordinate transform.
///
/// Guarded by AXIS_ENABLE_PROJ. When PROJ is available, transforms
/// projection-space coordinates to geographic lon/lat using the PROJ C API
/// via detail::Proj_Handle (RAII). When PROJ is not available, provides a
/// stub that throws std::runtime_error.
///
/// For device MemorySpace: transforms on host first (PROJ is CPU-only),
/// then deep_copies the resulting coordinates to the target device space.

#include <axis/detail/memory_traits.hpp>
#include <axis/topology/projection_builder.hpp>

#ifdef AXIS_ENABLE_PROJ
#include <proj.h>

#include <axis/detail/raii_handles.hpp>
#endif

#include <Kokkos_Core.hpp>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace axis::topology {

// ─────────────────────────────────────────────────────────────────────────────
// ProjectionBuilder::build — PROJ-enabled implementation
// ─────────────────────────────────────────────────────────────────────────────

#ifdef AXIS_ENABLE_PROJ

template <class MemorySpace>
StructuredGrid<MemorySpace> ProjectionBuilder::build(const ingest::ProjectedParams &params, const ingest::BufferViews &buffers) {
    // ── Validate inputs ──────────────────────────────────────────────────────

    if (params.proj_string.empty()) {
        throw std::invalid_argument("ProjectionBuilder::build: proj_string is empty");
    }

    if (buffers.ni == 0 || buffers.nj == 0) {
        throw std::invalid_argument("ProjectionBuilder::build: ni and nj must be positive");
    }

    const std::size_t n_points = buffers.ni * buffers.nj;

    if (buffers.center_x.extent(0) != n_points) {
        throw std::invalid_argument("ProjectionBuilder::build: center_x extent (" + std::to_string(buffers.center_x.extent(0)) +
                                    ") does not match ni*nj (" + std::to_string(n_points) + ")");
    }

    if (buffers.center_y.extent(0) != n_points) {
        throw std::invalid_argument("ProjectionBuilder::build: center_y extent (" + std::to_string(buffers.center_y.extent(0)) +
                                    ") does not match ni*nj (" + std::to_string(n_points) + ")");
    }

    // ── Create the PROJ transformation pipeline ──────────────────────────────
    // Transform from the given projection to EPSG:4326 (geographic lon/lat).
    // We use proj_create_crs_to_crs to build a pipeline from the source CRS
    // (the descriptor's proj_string) to WGS84 geographic.

    // Create a PROJ context for thread safety.
    PJ_CONTEXT *ctx = proj_context_create();
    if (ctx == nullptr) {
        throw std::runtime_error("ProjectionBuilder::build: proj_context_create failed");
    }

    // Create the transformation: source CRS -> EPSG:4326 (lon/lat degrees).
    PJ *transform = proj_create_crs_to_crs(ctx,
                                           params.proj_string.c_str(),  // source CRS
                                           "EPSG:4326",                 // target: geographic WGS84
                                           nullptr);                    // area of use (null = global)

    if (transform == nullptr) {
        int err = proj_context_errno(ctx);
        const char *err_text = proj_errno_string(err);
        std::string msg = "ProjectionBuilder::build: proj_create_crs_to_crs failed: ";
        msg += (err_text ? err_text : "unknown PROJ error");
        proj_context_destroy(ctx);
        throw std::runtime_error(msg);
    }

    // Normalize output axis order to lon, lat (PROJ may return lat, lon for
    // geographic CRSs depending on authority definitions).
    PJ *normalized = proj_normalize_for_visualization(ctx, transform);
    if (normalized == nullptr) {
        // Fallback: use the unnormalized transform (some older PROJ versions).
        normalized = transform;
        transform = nullptr;
    } else {
        proj_destroy(transform);
        transform = nullptr;
    }

    // ── Transform coordinates on the host ────────────────────────────────────
    // PROJ is CPU-only, so we always work on host arrays first.

    // Allocate host-side output arrays.
    Kokkos::View<double *, Kokkos::HostSpace> host_lon("proj_builder_host_lon", n_points);
    Kokkos::View<double *, Kokkos::HostSpace> host_lat("proj_builder_host_lat", n_points);

    // Copy input coordinates (from the descriptor's mdspan) to host arrays
    // for the transformation. The mdspan may already be on the host (since
    // the producer fills it), but we access element-by-element to be safe.
    for (std::size_t i = 0; i < n_points; ++i) {
        PJ_COORD input_coord;
        input_coord.xy.x = buffers.center_x[i];
        input_coord.xy.y = buffers.center_y[i];

        PJ_COORD output_coord = proj_trans(normalized, PJ_FWD, input_coord);

        // Check for transformation errors.
        if (output_coord.xy.x == HUGE_VAL || output_coord.xy.y == HUGE_VAL) {
            proj_destroy(normalized);
            proj_context_destroy(ctx);
            throw std::runtime_error("ProjectionBuilder::build: proj_trans failed at point index " + std::to_string(i) +
                                     " (x=" + std::to_string(buffers.center_x[i]) + ", y=" + std::to_string(buffers.center_y[i]) + ")");
        }

        // Longitude in degrees, normalized to [0, 360) to match the range
        // convention of every other AXIS mesh ingest path (PROJ's EPSG:4326
        // output is [-180, 180]; a disjoint raw range makes the engine's 2-D
        // BVH range check reject otherwise-overlapping grids).
        host_lon(i) = std::fmod(output_coord.xy.x, 360.0);
        if (host_lon(i) < 0.0) {
            host_lon(i) += 360.0;
        }
        host_lat(i) = output_coord.xy.y;  // latitude in degrees
    }

    // ── Clean up PROJ resources ──────────────────────────────────────────────
    proj_destroy(normalized);
    proj_context_destroy(ctx);

    // ── Transfer to target MemorySpace ───────────────────────────────────────
    // If MemorySpace is HostSpace, this is a no-op copy (same space).
    // If MemorySpace is a device space, this performs an explicit deep_copy.

    Kokkos::View<double *, MemorySpace> target_lon("proj_builder_target_lon", n_points);
    Kokkos::View<double *, MemorySpace> target_lat("proj_builder_target_lat", n_points);

    Kokkos::deep_copy(target_lon, host_lon);
    Kokkos::deep_copy(target_lat, host_lat);

    // ── Construct and return the StructuredGrid ──────────────────────────────
    // The output grid has geographic coordinates (lon/lat in degrees).
    return StructuredGrid<MemorySpace>(buffers.ni, buffers.nj, std::move(target_lon), std::move(target_lat), CoordinateSystem::SphericalDeg);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations (PROJ-enabled)
// ─────────────────────────────────────────────────────────────────────────────

template StructuredGrid<Kokkos::HostSpace> ProjectionBuilder::build<Kokkos::HostSpace>(const ingest::ProjectedParams &params,
                                                                                       const ingest::BufferViews &buffers);

#ifdef KOKKOS_ENABLE_CUDA
template StructuredGrid<Kokkos::CudaSpace> ProjectionBuilder::build<Kokkos::CudaSpace>(const ingest::ProjectedParams &params,
                                                                                       const ingest::BufferViews &buffers);
#endif

#ifdef KOKKOS_ENABLE_HIP
template StructuredGrid<Kokkos::HIPSpace> ProjectionBuilder::build<Kokkos::HIPSpace>(const ingest::ProjectedParams &params,
                                                                                     const ingest::BufferViews &buffers);
#endif

#else  // !AXIS_ENABLE_PROJ

// ─────────────────────────────────────────────────────────────────────────────
// ProjectionBuilder::build — stub when PROJ is NOT available
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
StructuredGrid<MemorySpace> ProjectionBuilder::build(const ingest::ProjectedParams & /*params*/, const ingest::BufferViews & /*buffers*/) {
    throw std::runtime_error(
        "AXIS built without PROJ support (AXIS_ENABLE_PROJ=OFF). "
        "Cannot transform projected coordinates.");
}

// Explicit template instantiations (stub)

template StructuredGrid<Kokkos::HostSpace> ProjectionBuilder::build<Kokkos::HostSpace>(const ingest::ProjectedParams &params,
                                                                                       const ingest::BufferViews &buffers);

#ifdef KOKKOS_ENABLE_CUDA
template StructuredGrid<Kokkos::CudaSpace> ProjectionBuilder::build<Kokkos::CudaSpace>(const ingest::ProjectedParams &params,
                                                                                       const ingest::BufferViews &buffers);
#endif

#ifdef KOKKOS_ENABLE_HIP
template StructuredGrid<Kokkos::HIPSpace> ProjectionBuilder::build<Kokkos::HIPSpace>(const ingest::ProjectedParams &params,
                                                                                     const ingest::BufferViews &buffers);
#endif

#endif  // AXIS_ENABLE_PROJ

}  // namespace axis::topology
