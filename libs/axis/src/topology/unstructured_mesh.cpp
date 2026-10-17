// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/unstructured_mesh.cpp
/// @brief Implementation of UnstructuredMesh::compute_areas() via Kokkos
///        parallel kernel, with explicit template instantiations for common
///        memory spaces.
///
/// Area computation strategies:
/// - SphericalDeg / SphericalRad: spherical excess formula (Girard's theorem
///   generalized for polygons). Each polygon is decomposed into triangles
///   anchored at vertex 0; the spherical excess of each triangle is summed.
///   Result is in steradians on the unit sphere.
/// - Cartesian3D: planar polygon area via cross-product summation (the
///   shoelace formula generalized to 3D-embedded polygons).

#include <Kokkos_Core.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <cmath>

namespace axis::topology {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Math constants
// ─────────────────────────────────────────────────────────────────────────────

constexpr double PI = 3.14159265358979323846;
constexpr double DEG_TO_RAD = PI / 180.0;

// ─────────────────────────────────────────────────────────────────────────────
// Spherical geometry helpers (all operate in radians on the unit sphere)
// ─────────────────────────────────────────────────────────────────────────────

/// Convert (lon, lat) in radians to unit-sphere Cartesian (x, y, z).
KOKKOS_INLINE_FUNCTION
void lonlat_to_xyz(double lon, double lat, double &x, double &y, double &z) {
    const double cos_lat = Kokkos::cos(lat);
    x = cos_lat * Kokkos::cos(lon);
    y = cos_lat * Kokkos::sin(lon);
    z = Kokkos::sin(lat);
}

/// Cross product of two 3-D vectors: c = a × b.
KOKKOS_INLINE_FUNCTION
void cross3(double ax, double ay, double az, double bx, double by, double bz, double &cx, double &cy, double &cz) {
    cx = ay * bz - az * by;
    cy = az * bx - ax * bz;
    cz = ax * by - ay * bx;
}

/// Dot product of two 3-D vectors.
KOKKOS_INLINE_FUNCTION
double dot3(double ax, double ay, double az, double bx, double by, double bz) {
    return ax * bx + ay * by + az * bz;
}

/// Magnitude of a 3-D vector.
KOKKOS_INLINE_FUNCTION
double mag3(double x, double y, double z) {
    return Kokkos::sqrt(x * x + y * y + z * z);
}

/// Compute the spherical excess (area on the unit sphere) of a triangle
/// defined by three unit-sphere points using the numerically stable formula:
///   E = 2 * atan2( |a · (b × c)| , 1 + a·b + a·c + b·c )
///
/// This is derived from the tangent half-angle identity and avoids
/// catastrophic cancellation for small triangles.
KOKKOS_INLINE_FUNCTION
double spherical_excess_triangle(double ax, double ay, double az, double bx, double by, double bz, double cx, double cy, double cz) {
    // b × c
    double bcx, bcy, bcz;
    cross3(bx, by, bz, cx, cy, cz, bcx, bcy, bcz);

    // Numerator: |a · (b × c)| = magnitude of scalar triple product
    const double num = Kokkos::fabs(dot3(ax, ay, az, bcx, bcy, bcz));

    // Denominator: 1 + a·b + a·c + b·c
    const double ab = dot3(ax, ay, az, bx, by, bz);
    const double ac = dot3(ax, ay, az, cx, cy, cz);
    const double bc = dot3(bx, by, bz, cx, cy, cz);
    const double den = 1.0 + ab + ac + bc;

    // atan2 must receive the signed denominator: a negative value is valid and
    // identifies a spherical excess larger than a hemisphere. Replacing it
    // with a fixed fallback loses the geometry (and can turn a valid polygon
    // into exactly pi steradians). Exact antipodal/degenerate configurations
    // remain branch-ambiguous and are rejected by source-grid validation when
    // strict geometry is required.
    return 2.0 * Kokkos::atan2(num, den);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// compute_areas() — Kokkos parallel kernel
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
void UnstructuredMesh<MemorySpace>::compute_areas() {
    const std::size_t nc = n_cells();
    if (nc == 0) return;

    // Allocate (or reallocate) the areas View
    cell_areas_ = Kokkos::View<double *, MemorySpace>("UnstructuredMesh::cell_areas", nc);

    // Capture views by value for the Kokkos lambda
    auto coords = node_coords_;
    auto offsets = conn_offsets_;
    auto indices = conn_indices_;
    auto areas = cell_areas_;
    const auto csys = coord_sys_;

    using exec_space = typename MemorySpace::execution_space;

    Kokkos::parallel_for(
        "UnstructuredMesh::compute_areas", Kokkos::RangePolicy<exec_space>(0, static_cast<int>(nc)), KOKKOS_LAMBDA(const int cell) {
            const index_t start = offsets(cell);
            const index_t end = offsets(cell + 1);
            const int nv = static_cast<int>(end - start);

            if (nv < 3) {
                areas(cell) = 0.0;
                return;
            }

            if (csys == CoordinateSystem::Cartesian3D) {
                // ─────────────────────────────────────────────────────────────
                // Cartesian3D: planar polygon area via cross-product summation.
                //
                // For a 3D-embedded planar polygon with vertices v_0..v_{n-1}:
                //   Area = 0.5 * | Σ_{i=0}^{n-1} (v_i × v_{(i+1) mod n}) |
                //
                // This is the generalization of the shoelace formula.
                // ─────────────────────────────────────────────────────────────
                double nx = 0.0, ny = 0.0, nz = 0.0;

                for (int i = 0; i < nv; ++i) {
                    const int j = (i + 1) % nv;
                    const index_t vi = indices(start + i);
                    const index_t vj = indices(start + j);

                    const double vix = coords(vi, 0);
                    const double viy = coords(vi, 1);
                    const double viz = coords(vi, 2);
                    const double vjx = coords(vj, 0);
                    const double vjy = coords(vj, 1);
                    const double vjz = coords(vj, 2);

                    // Accumulate cross product
                    double cx, cy, cz;
                    cross3(vix, viy, viz, vjx, vjy, vjz, cx, cy, cz);
                    nx += cx;
                    ny += cy;
                    nz += cz;
                }

                areas(cell) = 0.5 * mag3(nx, ny, nz);

            } else {
                // ─────────────────────────────────────────────────────────────
                // SphericalDeg / SphericalRad: spherical excess via triangle fan.
                //
                // Decompose the polygon into (nv - 2) triangles anchored at
                // vertex 0. Sum the spherical excess of each triangle.
                // For convex polygons this gives exact area on the unit sphere;
                // typical Earth-system mesh cells are convex or nearly so.
                //
                // Coordinates are assumed to be (lon, lat) in ndim=2 columns.
                // ─────────────────────────────────────────────────────────────
                const double scale = (csys == CoordinateSystem::SphericalDeg) ? DEG_TO_RAD : 1.0;

                // Convert vertex 0 to Cartesian on unit sphere
                const index_t idx0 = indices(start);
                const double lon0 = coords(idx0, 0) * scale;
                const double lat0 = coords(idx0, 1) * scale;
                double x0, y0, z0;
                lonlat_to_xyz(lon0, lat0, x0, y0, z0);

                double total_excess = 0.0;

                for (int i = 1; i < nv - 1; ++i) {
                    const index_t idx1 = indices(start + i);
                    const index_t idx2 = indices(start + i + 1);

                    const double lon1 = coords(idx1, 0) * scale;
                    const double lat1 = coords(idx1, 1) * scale;
                    double x1, y1, z1;
                    lonlat_to_xyz(lon1, lat1, x1, y1, z1);

                    const double lon2 = coords(idx2, 0) * scale;
                    const double lat2 = coords(idx2, 1) * scale;
                    double x2, y2, z2;
                    lonlat_to_xyz(lon2, lat2, x2, y2, z2);

                    total_excess += spherical_excess_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2);
                }

                areas(cell) = total_excess;  // steradians on the unit sphere
            }
        });

    // Fence to ensure areas are fully computed before any subsequent host access
    Kokkos::fence("UnstructuredMesh::compute_areas fence");
    geometry_.area_model = coord_sys_ == CoordinateSystem::Cartesian3D ? AreaModel::PlanarPolygon : AreaModel::SphericalExcess;
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations for common Kokkos memory spaces
// ─────────────────────────────────────────────────────────────────────────────

template class UnstructuredMesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class UnstructuredMesh<Kokkos::CudaSpace>;
#endif

#ifdef KOKKOS_ENABLE_HIP
template class UnstructuredMesh<Kokkos::HIPSpace>;
#endif

}  // namespace axis::topology
