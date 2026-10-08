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
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace axis::topology {

#ifdef AXIS_ENABLE_PROJ
namespace {

/// Reconstruct a corner lattice in the source projection's Cartesian space.
/// The tensor-product stencil averages adjacent centers for interior corners
/// and linearly extrapolates the outermost corner rows/columns.
/// This must happen before projection to geographic coordinates: averaging
/// transformed lon/lat centers can fold otherwise regular projected cells.
void reconstruct_projected_corners(std::size_t ni, std::size_t nj, const ingest::BufferViews &buffers, std::vector<double> &corner_x,
                                   std::vector<double> &corner_y) {
    if (ni < 2 || nj < 2) {
        throw std::invalid_argument("ProjectionBuilder::build: at least 2x2 centers are required to reconstruct projected corners");
    }
    if (ni == std::numeric_limits<std::size_t>::max() || nj == std::numeric_limits<std::size_t>::max() ||
        ni + 1 > std::numeric_limits<std::size_t>::max() / (nj + 1)) {
        throw std::overflow_error("ProjectionBuilder::build: projected corner extent overflows size_t");
    }

    const std::size_t nip1 = ni + 1;
    const std::size_t n_corners = nip1 * (nj + 1);
    corner_x.resize(n_corners);
    corner_y.resize(n_corners);
    auto axis_stencil = [](std::size_t node, std::size_t count, std::size_t (&cells)[2], double (&weights)[2]) {
        if (node == 0) {
            cells[0] = 0;
            cells[1] = 1;
            weights[0] = 1.5;
            weights[1] = -0.5;
        } else if (node == count) {
            cells[0] = count - 1;
            cells[1] = count - 2;
            weights[0] = 1.5;
            weights[1] = -0.5;
        } else {
            cells[0] = node - 1;
            cells[1] = node;
            weights[0] = 0.5;
            weights[1] = 0.5;
        }
    };
    for (std::size_t j = 0; j <= nj; ++j) {
        std::size_t rows[2];
        double row_weights[2];
        axis_stencil(j, nj, rows, row_weights);
        for (std::size_t i = 0; i <= ni; ++i) {
            std::size_t cols[2];
            double col_weights[2];
            axis_stencil(i, ni, cols, col_weights);
            double x = 0.0;
            double y = 0.0;
            for (int r = 0; r < 2; ++r) {
                for (int c = 0; c < 2; ++c) {
                    const std::size_t center = cols[c] + rows[r] * ni;
                    const double weight = row_weights[r] * col_weights[c];
                    x += weight * buffers.center_x[center];
                    y += weight * buffers.center_y[center];
                }
            }
            const std::size_t node = i + j * nip1;
            corner_x[node] = x;
            corner_y[node] = y;
        }
    }
}

}  // namespace
#endif

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

    if (buffers.ni > std::numeric_limits<std::size_t>::max() / buffers.nj) {
        throw std::overflow_error("ProjectionBuilder::build: center extent overflows size_t");
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

    // ── Reconstruct corners in the source projection space ───────────────────
    std::vector<double> projected_corner_x;
    std::vector<double> projected_corner_y;
    reconstruct_projected_corners(buffers.ni, buffers.nj, buffers, projected_corner_x, projected_corner_y);
    const std::size_t n_corners = projected_corner_x.size();

    // ── Transform centers and corners on the host ────────────────────────────
    // PROJ is CPU-only, so we always work on host arrays first.

    // Allocate host-side output arrays.
    Kokkos::View<double *, Kokkos::HostSpace> host_lon("proj_builder_host_lon", n_points);
    Kokkos::View<double *, Kokkos::HostSpace> host_lat("proj_builder_host_lat", n_points);
    Kokkos::View<double *, Kokkos::HostSpace> host_corner_lon("proj_builder_host_corner_lon", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> host_corner_lat("proj_builder_host_corner_lat", n_corners);

    // Copy input coordinates (from the descriptor's mdspan) to host arrays
    // for the transformation. The mdspan may already be on the host (since
    // the producer fills it), but we access element-by-element to be safe.
    for (std::size_t i = 0; i < n_points; ++i) {
        PJ_COORD input_coord = proj_coord(buffers.center_x[i], buffers.center_y[i], 0.0, 0.0);
        PJ_COORD output_coord = proj_trans(normalized, PJ_FWD, input_coord);
        if (!std::isfinite(output_coord.xy.x) || !std::isfinite(output_coord.xy.y)) {
            proj_destroy(normalized);
            proj_context_destroy(ctx);
            throw std::runtime_error("ProjectionBuilder::build: proj_trans failed at point index " + std::to_string(i) +
                                     " (x=" + std::to_string(buffers.center_x[i]) + ", y=" + std::to_string(buffers.center_y[i]) + ")");
        }

        host_lon(i) = output_coord.xy.x;  // longitude in degrees
        host_lat(i) = output_coord.xy.y;  // latitude in degrees
    }

    for (std::size_t i = 0; i < n_corners; ++i) {
        PJ_COORD input_coord = proj_coord(projected_corner_x[i], projected_corner_y[i], 0.0, 0.0);
        PJ_COORD output_coord = proj_trans(normalized, PJ_FWD, input_coord);
        if (!std::isfinite(output_coord.xy.x) || !std::isfinite(output_coord.xy.y)) {
            proj_destroy(normalized);
            proj_context_destroy(ctx);
            throw std::runtime_error("ProjectionBuilder::build: proj_trans failed at projected corner index " + std::to_string(i));
        }
        host_corner_lon(i) = output_coord.xy.x;
        host_corner_lat(i) = output_coord.xy.y;
    }

    // ── Clean up PROJ resources ──────────────────────────────────────────────
    proj_destroy(normalized);
    proj_context_destroy(ctx);

    // ── Transfer to target MemorySpace ───────────────────────────────────────
    // If MemorySpace is HostSpace, this is a no-op copy (same space).
    // If MemorySpace is a device space, this performs an explicit deep_copy.

    Kokkos::View<double *, MemorySpace> target_lon("proj_builder_target_lon", n_points);
    Kokkos::View<double *, MemorySpace> target_lat("proj_builder_target_lat", n_points);
    Kokkos::View<double *, MemorySpace> target_corner_lon("proj_builder_target_corner_lon", n_corners);
    Kokkos::View<double *, MemorySpace> target_corner_lat("proj_builder_target_corner_lat", n_corners);

    Kokkos::deep_copy(target_lon, host_lon);
    Kokkos::deep_copy(target_lat, host_lat);
    Kokkos::deep_copy(target_corner_lon, host_corner_lon);
    Kokkos::deep_copy(target_corner_lat, host_corner_lat);

    // ── Construct and return the StructuredGrid ──────────────────────────────
    // The output grid has geographic coordinates (lon/lat in degrees).
    StructuredGrid<MemorySpace> grid(buffers.ni, buffers.nj, std::move(target_lon), std::move(target_lat), CoordinateSystem::SphericalDeg);
    grid.set_corners(std::move(target_corner_lon), std::move(target_corner_lat));
    return grid;
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
