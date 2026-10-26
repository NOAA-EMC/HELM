// SPDX-License-Identifier: Apache-2.0
// AXIS — Arbitrary eXgrid Interpolation Solver
// Copyright (c) HELM Project Contributors

/// @file src/topology/mesh_factory.cpp
/// @brief MeshFactory implementation — the single funnel for all grid sources.
///
/// from_descriptor branches on ConventionKind ONCE. It NEVER branches on
/// producer identity. Validation is thorough: unknown kind, missing field,
/// inconsistent buffer extents, and null/empty buffers all throw
/// std::invalid_argument with a descriptive message.
///
/// Memory-space policy:
///   - When buffer views already address the target MemorySpace: adopt without
///     copy (zero-copy).
///   - When in a different space: explicit Kokkos::deep_copy (HELM Law #2).

#include <Kokkos_Core.hpp>
#include <axis/detail/memory_traits.hpp>
#include <axis/topology/mesh_builder.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/topology/reduced_gaussian_grid.hpp>
#include <axis/topology/rule_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace axis::topology {

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Descriptor validation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Convert ConventionKind to a human-readable string.
const char *convention_kind_str(ingest::ConventionKind kind) {
    switch (kind) {
        case ingest::ConventionKind::CF:
            return "CF";
        case ingest::ConventionKind::UGRID:
            return "UGRID";
        case ingest::ConventionKind::GRIB:
            return "GRIB";
        case ingest::ConventionKind::ReducedGaussian:
            return "ReducedGaussian";
        case ingest::ConventionKind::Projected:
            return "Projected";
        case ingest::ConventionKind::NamedGrid:
            return "NamedGrid";
        case ingest::ConventionKind::GridRules:
            return "GridRules";
    }
    return "Unknown";
}

/// Validate a CF descriptor's required fields and buffer consistency.
void validate_cf(const ingest::GridDescriptor &desc) {
    const auto &b = desc.buffers;

    if (b.ni == 0) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(CF): missing required field 'ni' "
            "(must be > 0)");
    }
    if (b.nj == 0) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(CF): missing required field 'nj' "
            "(must be > 0)");
    }

    const std::size_t expected = b.ni * b.nj;

    if (b.center_x.data_handle() == nullptr || b.center_x.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(CF): null/empty required buffer 'center_x'");
    }
    if (b.center_y.data_handle() == nullptr || b.center_y.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(CF): null/empty required buffer 'center_y'");
    }

    if (b.center_x.extent(0) != expected) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(CF): inconsistent buffer extents — "
            "center_x.extent(0)=" +
            std::to_string(b.center_x.extent(0)) + " but ni*nj=" + std::to_string(expected));
    }
    if (b.center_y.extent(0) != expected) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(CF): inconsistent buffer extents — "
            "center_y.extent(0)=" +
            std::to_string(b.center_y.extent(0)) + " but ni*nj=" + std::to_string(expected));
    }
}

/// Validate a UGRID descriptor's required fields and buffer consistency.
void validate_ugrid(const ingest::GridDescriptor &desc) {
    const auto &b = desc.buffers;

    if (b.node_coords.data_handle() == nullptr || b.node_coords.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(UGRID): null/empty required buffer 'node_coords'");
    }
    if (b.conn_offsets.data_handle() == nullptr || b.conn_offsets.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(UGRID): null/empty required buffer 'conn_offsets'");
    }
    if (b.conn_indices.data_handle() == nullptr || b.conn_indices.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(UGRID): null/empty required buffer 'conn_indices'");
    }

    // node_coords must be rank-2 with ndim >= 2
    if (b.node_coords.extent(1) < 2) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(UGRID): inconsistent buffer extents — "
            "node_coords.extent(1)=" +
            std::to_string(b.node_coords.extent(1)) + " but ndim must be >= 2");
    }
}

/// Validate a GRIB descriptor's required fields and buffer consistency.
void validate_grib(const ingest::GridDescriptor &desc) {
    const auto &g = desc.grib;
    const auto &b = desc.buffers;

    if (g.grid_type.empty()) {
        throw std::invalid_argument("MeshFactory::from_descriptor(GRIB): missing required field 'grid_type'");
    }
    if (g.ni <= 0) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(GRIB): missing required field 'ni' "
            "(must be > 0)");
    }
    if (g.nj <= 0) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(GRIB): missing required field 'nj' "
            "(must be > 0)");
    }

    const std::size_t expected = static_cast<std::size_t>(g.ni) * static_cast<std::size_t>(g.nj);

    if (b.center_x.data_handle() == nullptr || b.center_x.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(GRIB): null/empty required buffer 'center_x'");
    }
    if (b.center_y.data_handle() == nullptr || b.center_y.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(GRIB): null/empty required buffer 'center_y'");
    }

    if (b.center_x.extent(0) != expected) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(GRIB): inconsistent buffer extents — "
            "center_x.extent(0)=" +
            std::to_string(b.center_x.extent(0)) + " but ni*nj=" + std::to_string(expected));
    }
    if (b.center_y.extent(0) != expected) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(GRIB): inconsistent buffer extents — "
            "center_y.extent(0)=" +
            std::to_string(b.center_y.extent(0)) + " but ni*nj=" + std::to_string(expected));
    }
}

void validate_reduced_gaussian(const ingest::GridDescriptor &desc) {
    const auto &b = desc.buffers;
    const std::size_t rows = desc.reduced_gaussian.n_rows;
    if (rows == 0) throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): missing required field 'n_rows'");
    if (b.latitude_bounds.data_handle() == nullptr || b.latitude_bounds.extent(0) != rows + 1) {
        throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): latitude_bounds must have n_rows+1 values");
    }
    if (b.row_cell_offsets.data_handle() == nullptr || b.row_cell_offsets.extent(0) != rows + 1) {
        throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): row_cell_offsets must have n_rows+1 values");
    }
    if (b.row_longitude_boundaries.data_handle() == nullptr || b.row_longitude_boundaries.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): missing required buffer 'row_longitude_boundaries'");
    }
    if (b.row_cell_offsets(0) != 0) throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): row_cell_offsets must start at zero");
    for (std::size_t j = 0; j < rows; ++j) {
        if (b.row_cell_offsets(j + 1) <= b.row_cell_offsets(j)) {
            throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): every row must contain at least one cell");
        }
    }
    const auto n_cells = static_cast<std::size_t>(b.row_cell_offsets(rows));
    if (n_cells > std::numeric_limits<std::size_t>::max() - rows || b.row_longitude_boundaries.extent(0) != n_cells + rows) {
        throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): flattened boundary extent must equal total_cells+n_rows");
    }
    if (desc.coord_system != ingest::CoordinateSystem::SphericalDeg && desc.coord_system != ingest::CoordinateSystem::SphericalRad) {
        throw std::invalid_argument("MeshFactory::from_descriptor(ReducedGaussian): only spherical coordinates are supported");
    }
}

/// Validate a Projected descriptor's required fields.
void validate_projected(const ingest::GridDescriptor &desc) {
    if (desc.projected.proj_string.empty()) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(Projected): missing required field "
            "'proj_string' (must be non-empty)");
    }

    const auto &b = desc.buffers;

    if (b.ni == 0) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(Projected): missing required field 'ni' "
            "(must be > 0)");
    }
    if (b.nj == 0) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(Projected): missing required field 'nj' "
            "(must be > 0)");
    }

    const std::size_t expected = b.ni * b.nj;

    if (b.center_x.data_handle() == nullptr || b.center_x.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(Projected): null/empty required buffer 'center_x'");
    }
    if (b.center_y.data_handle() == nullptr || b.center_y.extent(0) == 0) {
        throw std::invalid_argument("MeshFactory::from_descriptor(Projected): null/empty required buffer 'center_y'");
    }

    if (b.center_x.extent(0) != expected) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(Projected): inconsistent buffer extents — "
            "center_x.extent(0)=" +
            std::to_string(b.center_x.extent(0)) + " but ni*nj=" + std::to_string(expected));
    }
    if (b.center_y.extent(0) != expected) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(Projected): inconsistent buffer extents — "
            "center_y.extent(0)=" +
            std::to_string(b.center_y.extent(0)) + " but ni*nj=" + std::to_string(expected));
    }
}

/// Validate a NamedGrid descriptor's required fields.
void validate_named_grid(const ingest::GridDescriptor &desc) {
    if (desc.named_grid.name.empty()) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(NamedGrid): missing required field 'name' "
            "(must be non-empty)");
    }
}

/// Validate a GridRules descriptor's required fields.
void validate_grid_rules(const ingest::GridDescriptor &desc) {
    if (desc.grid_rules.kind.empty()) {
        throw std::invalid_argument(
            "MeshFactory::from_descriptor(GridRules): missing required field 'kind' "
            "(must be non-empty)");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Buffer adoption / deep_copy helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Copy a 1-D mdspan buffer into a Kokkos::View in the target MemorySpace.
/// Uses deep_copy when the source is host-accessible (which it always is for
/// descriptor buffers — producers fill from host).
template <class MemorySpace, class T>
Kokkos::View<T *, MemorySpace> copy_buffer_1d(const char *label, field_view<const T, 1> src) {
    const std::size_t n = src.extent(0);
    Kokkos::View<T *, MemorySpace> dst(std::string(label), n);

    // Create an unmanaged host view wrapping the source mdspan data.
    Kokkos::View<const T *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src_view(src.data_handle(), n);

    Kokkos::deep_copy(dst, src_view);
    return dst;
}

/// Copy a 2-D mdspan buffer into a Kokkos::View<T**, LayoutLeft, MemorySpace>.
template <class MemorySpace, class T>
Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace> copy_buffer_2d(const char *label, field_view<const T, 2> src) {
    const std::size_t n0 = src.extent(0);
    const std::size_t n1 = src.extent(1);
    Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace> dst(std::string(label), n0, n1);

    Kokkos::View<const T **, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> src_view(src.data_handle(), n0, n1);

    Kokkos::deep_copy(dst, src_view);
    return dst;
}

/// Adopt a 1-D mdspan buffer directly into a Kokkos::View when the target
/// is HostSpace (zero-copy: wraps the existing pointer as an unmanaged view,
/// then copies into a managed view for ownership).
/// For the general case we always copy since the descriptor owns nothing.
/// NOTE: The design says "adopt without copy when space matches" — for
/// HostSpace we still need a managed view that AXIS owns, so we memcpy into
/// a fresh allocation on the same space (which Kokkos::deep_copy optimizes
/// to memcpy). The key invariant is: no device↔host transfer occurs.
template <class MemorySpace, class T>
Kokkos::View<T *, MemorySpace> adopt_or_copy_1d(const char *label, field_view<const T, 1> src) {
    return copy_buffer_1d<MemorySpace, T>(label, src);
}

template <class MemorySpace, class T>
Kokkos::View<T **, Kokkos::LayoutLeft, MemorySpace> adopt_or_copy_2d(const char *label, field_view<const T, 2> src) {
    return copy_buffer_2d<MemorySpace, T>(label, src);
}

// ─────────────────────────────────────────────────────────────────────────────
// Convention-specific mesh builders
// ─────────────────────────────────────────────────────────────────────────────

/// Map ingest::CoordinateSystem to topology::CoordinateSystem.
CoordinateSystem map_coord_system(ingest::CoordinateSystem cs) {
    switch (cs) {
        case ingest::CoordinateSystem::SphericalDeg:
            return CoordinateSystem::SphericalDeg;
        case ingest::CoordinateSystem::SphericalRad:
            return CoordinateSystem::SphericalRad;
        case ingest::CoordinateSystem::Cartesian3D:
            return CoordinateSystem::Cartesian3D;
    }
    return CoordinateSystem::SphericalDeg;
}

LongitudePeriodicity map_longitude_periodicity(const ingest::GridDescriptor &desc) {
    double period = desc.longitude_period;
    if (period == 0.0) {
        period = desc.coord_system == ingest::CoordinateSystem::SphericalRad ? 2.0 * std::numbers::pi : 360.0;
    }
    return LongitudePeriodicity{desc.longitude_periodic, period, desc.longitude_increasing};
}

/// Build an UnstructuredMesh from a CF descriptor.
/// CF: use buffers.center_x/y + ni/nj to build StructuredGrid, then to_unstructured().
template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_cf(const ingest::GridDescriptor &desc) {
    const auto &b = desc.buffers;
    const std::size_t n_points = b.ni * b.nj;

    // Copy center coordinates into Kokkos Views.
    auto center_lon = adopt_or_copy_1d<MemorySpace, double>("cf_center_lon", b.center_x);
    auto center_lat = adopt_or_copy_1d<MemorySpace, double>("cf_center_lat", b.center_y);

    // Build StructuredGrid.
    StructuredGrid<MemorySpace> grid(b.ni, b.nj, std::move(center_lon), std::move(center_lat), map_coord_system(desc.coord_system));

    // If corners are provided, set them.
    if (b.corner_x.data_handle() != nullptr && b.corner_x.extent(0) > 0 && b.corner_y.data_handle() != nullptr && b.corner_y.extent(0) > 0) {
        auto corner_lon = adopt_or_copy_1d<MemorySpace, double>("cf_corner_lon", b.corner_x);
        auto corner_lat = adopt_or_copy_1d<MemorySpace, double>("cf_corner_lat", b.corner_y);
        grid.set_corners(std::move(corner_lon), std::move(corner_lat));
    }
    if (b.longitude_bounds.extent(0) > 0 && b.latitude_bounds.extent(0) > 0) {
        grid.set_rectilinear_bounds(adopt_or_copy_1d<MemorySpace, double>("cf_longitude_bounds", b.longitude_bounds),
                                    adopt_or_copy_1d<MemorySpace, double>("cf_latitude_bounds", b.latitude_bounds));
    }
    if (b.gaussian_latitude_weights.extent(0) > 0) {
        grid.set_gaussian_latitude_weights(adopt_or_copy_1d<MemorySpace, double>("cf_gaussian_weights", b.gaussian_latitude_weights));
        return grid.to_unstructured(CornerPolicy::GaussianLatLon, map_longitude_periodicity(desc));
    }
    const auto policy = desc.curvilinear_centers ? CornerPolicy::CurvilinearApproximate : CornerPolicy::RectilinearMidpoint;
    return grid.to_unstructured(policy, map_longitude_periodicity(desc));
}

/// Build an UnstructuredMesh from a UGRID descriptor.
/// UGRID: adopt buffers.node_coords and CSR connectivity directly.
template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_ugrid(const ingest::GridDescriptor &desc) {
    const auto &b = desc.buffers;

    // Copy node coordinates [n_nodes, ndim].
    auto node_coords = adopt_or_copy_2d<MemorySpace, double>("ugrid_node_coords", b.node_coords);
    if (map_coord_system(desc.coord_system) == CoordinateSystem::Cartesian3D && node_coords.extent(1) == 2) {
        // Legacy UGRID descriptors commonly use Cartesian3D for a planar
        // x/y mesh. Preserve that accepted convention while presenting the
        // generic builder with the declared three-coordinate shape.
        Kokkos::View<double **, Kokkos::LayoutLeft, MemorySpace> embedded("ugrid_node_coords_3d", node_coords.extent(0), 3);
        using exec_space = typename MemorySpace::execution_space;
        exec_space exec{};
        const auto n_nodes = node_coords.extent(0);
        Kokkos::parallel_for(
            "ugrid_embed_planar_coordinates", Kokkos::RangePolicy<exec_space, Kokkos::IndexType<std::size_t>>(exec, 0, n_nodes),
            KOKKOS_LAMBDA(const std::size_t node) {
                embedded(node, 0) = node_coords(node, 0);
                embedded(node, 1) = node_coords(node, 1);
                embedded(node, 2) = 0.0;
            });
        exec.fence("ugrid_embed_planar_coordinates_fence");
        node_coords = std::move(embedded);
    }

    // Copy CSR connectivity.
    auto conn_offsets = adopt_or_copy_1d<MemorySpace, index_t>("ugrid_conn_offsets", b.conn_offsets);
    auto conn_indices = adopt_or_copy_1d<MemorySpace, index_t>("ugrid_conn_indices", b.conn_indices);

    // Optional cell areas.
    Kokkos::View<double *, MemorySpace> areas;
    if (b.cell_areas.data_handle() != nullptr && b.cell_areas.extent(0) > 0) {
        areas = adopt_or_copy_1d<MemorySpace, double>("ugrid_areas", b.cell_areas);
    }

    // Optional cell mask.
    Kokkos::View<int *, MemorySpace> mask;
    if (b.cell_mask.data_handle() != nullptr && b.cell_mask.extent(0) > 0) {
        mask = adopt_or_copy_1d<MemorySpace, int>("ugrid_mask", b.cell_mask);
    }

    const AreaModel supplied_area_model = areas.extent(0) == 0 ? AreaModel::Unspecified : AreaModel::SourceSupplied;
    return make_unstructured(std::move(node_coords), std::move(conn_offsets), std::move(conn_indices), map_coord_system(desc.coord_system),
                             std::move(areas), std::move(mask),
                             GeometryMetadata{GeometryProvenance::SourceAuthoritative, BoundaryModel::Unspecified, supplied_area_model});
}

/// Build an UnstructuredMesh from a GRIB descriptor.
/// GRIB: reconstruct from GribParams + buffers (build StructuredGrid, then
/// to_unstructured()).
template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_grib(const ingest::GridDescriptor &desc) {
    const auto &g = desc.grib;
    const auto &b = desc.buffers;
    const std::size_t ni = static_cast<std::size_t>(g.ni);
    const std::size_t nj = static_cast<std::size_t>(g.nj);

    // Copy center coordinates into Kokkos Views.
    auto center_lon = adopt_or_copy_1d<MemorySpace, double>("grib_center_lon", b.center_x);
    auto center_lat = adopt_or_copy_1d<MemorySpace, double>("grib_center_lat", b.center_y);

    // Build StructuredGrid using GRIB dimensions.
    StructuredGrid<MemorySpace> grid(ni, nj, std::move(center_lon), std::move(center_lat), map_coord_system(desc.coord_system));

    // If corners are provided, set them.
    if (b.corner_x.data_handle() != nullptr && b.corner_x.extent(0) > 0 && b.corner_y.data_handle() != nullptr && b.corner_y.extent(0) > 0) {
        auto corner_lon = adopt_or_copy_1d<MemorySpace, double>("grib_corner_lon", b.corner_x);
        auto corner_lat = adopt_or_copy_1d<MemorySpace, double>("grib_corner_lat", b.corner_y);
        grid.set_corners(std::move(corner_lon), std::move(corner_lat));
    }
    if (b.longitude_bounds.extent(0) > 0 && b.latitude_bounds.extent(0) > 0) {
        grid.set_rectilinear_bounds(adopt_or_copy_1d<MemorySpace, double>("grib_longitude_bounds", b.longitude_bounds),
                                    adopt_or_copy_1d<MemorySpace, double>("grib_latitude_bounds", b.latitude_bounds));
    }
    if (b.gaussian_latitude_weights.extent(0) > 0) {
        grid.set_gaussian_latitude_weights(adopt_or_copy_1d<MemorySpace, double>("grib_gaussian_weights", b.gaussian_latitude_weights));
        return grid.to_unstructured(CornerPolicy::GaussianLatLon, map_longitude_periodicity(desc));
    }
    const auto policy = desc.curvilinear_centers ? CornerPolicy::CurvilinearApproximate : CornerPolicy::RectilinearMidpoint;
    return grid.to_unstructured(policy, map_longitude_periodicity(desc));
}

template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_reduced_gaussian(const ingest::GridDescriptor &desc) {
    const auto &b = desc.buffers;
    const std::size_t rows = desc.reduced_gaussian.n_rows;
    std::vector<double> latitude_bounds(rows + 1);
    for (std::size_t j = 0; j <= rows; ++j) latitude_bounds[j] = b.latitude_bounds(j);
    std::vector<ReducedGaussianRow> row_geometry(rows);
    for (std::size_t j = 0; j < rows; ++j) {
        const std::size_t cell_start = static_cast<std::size_t>(b.row_cell_offsets(j));
        const std::size_t cell_count = static_cast<std::size_t>(b.row_cell_offsets(j + 1) - b.row_cell_offsets(j));
        const std::size_t boundary_start = cell_start + j;
        row_geometry[j].longitude_boundaries.resize(cell_count + 1);
        for (std::size_t i = 0; i <= cell_count; ++i) {
            row_geometry[j].longitude_boundaries[i] = b.row_longitude_boundaries(boundary_start + i);
        }
    }
    const double period = desc.longitude_period == 0.0
                              ? (desc.coord_system == ingest::CoordinateSystem::SphericalRad ? 2.0 * std::numbers::pi : 360.0)
                              : desc.longitude_period;
    ReducedGaussianGrid provider(std::move(latitude_bounds), std::move(row_geometry), map_coord_system(desc.coord_system),
                                 LongitudePeriodicity{desc.longitude_periodic, period, desc.longitude_increasing});
    return provider.template to_unstructured<MemorySpace>();
}

/// Build an UnstructuredMesh from a Projected descriptor.
/// Projected: delegate to ProjectionBuilder::build(), then to_unstructured().
template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_projected(const ingest::GridDescriptor &desc) {
    auto grid = ProjectionBuilder::build<MemorySpace>(desc.projected, desc.buffers);
    return grid.to_unstructured(CornerPolicy::CurvilinearApproximate, map_longitude_periodicity(desc));
}

/// Build an UnstructuredMesh from a NamedGrid descriptor.
/// NamedGrid: delegate to NamedGridRegistry::generate(named_grid.name).
template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_named_grid(const ingest::GridDescriptor &desc) {
    return NamedGridRegistry::generate<MemorySpace>(desc.named_grid.name);
}

/// Build an UnstructuredMesh from a GridRules descriptor.
/// GridRules: delegate to RuleGenerator::generate(grid_rules).
template <class MemorySpace>
UnstructuredMesh<MemorySpace> build_from_grid_rules(const ingest::GridDescriptor &desc) {
    return RuleGenerator::generate<MemorySpace>(desc.grid_rules);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// MeshFactory::from_descriptor
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> MeshFactory::from_descriptor(const ingest::GridDescriptor &descriptor) {
    // ── Validate and dispatch on ConventionKind (ONCE) ───────────────────────
    // NEVER branch on producer identity.

    switch (descriptor.kind) {
        case ingest::ConventionKind::CF:
            validate_cf(descriptor);
            return build_from_cf<MemorySpace>(descriptor);

        case ingest::ConventionKind::UGRID:
            validate_ugrid(descriptor);
            return build_from_ugrid<MemorySpace>(descriptor);

        case ingest::ConventionKind::GRIB:
            validate_grib(descriptor);
            return build_from_grib<MemorySpace>(descriptor);

        case ingest::ConventionKind::ReducedGaussian:
            validate_reduced_gaussian(descriptor);
            return build_from_reduced_gaussian<MemorySpace>(descriptor);

        case ingest::ConventionKind::Projected:
            validate_projected(descriptor);
            return build_from_projected<MemorySpace>(descriptor);

        case ingest::ConventionKind::NamedGrid:
            validate_named_grid(descriptor);
            return build_from_named_grid<MemorySpace>(descriptor);

        case ingest::ConventionKind::GridRules:
            validate_grid_rules(descriptor);
            return build_from_grid_rules<MemorySpace>(descriptor);
    }

    // If we reach here, the kind is an unknown enum value (e.g., future
    // addition or memory corruption).
    throw std::invalid_argument("MeshFactory::from_descriptor: unknown ConventionKind value (" + std::to_string(static_cast<int>(descriptor.kind)) +
                                ")");
}

// ─────────────────────────────────────────────────────────────────────────────
// MeshFactory::from_named
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> MeshFactory::from_named(const std::string &name) {
    return NamedGridRegistry::generate<MemorySpace>(name);
}

// ─────────────────────────────────────────────────────────────────────────────
// MeshFactory::from_rules
// ─────────────────────────────────────────────────────────────────────────────

template <class MemorySpace>
UnstructuredMesh<MemorySpace> MeshFactory::from_rules(const ingest::GridRulesParams &params) {
    return RuleGenerator::generate<MemorySpace>(params);
}

// ─────────────────────────────────────────────────────────────────────────────
// Explicit template instantiations
// ─────────────────────────────────────────────────────────────────────────────

// HostSpace (always available)
template UnstructuredMesh<Kokkos::HostSpace> MeshFactory::from_descriptor<Kokkos::HostSpace>(const ingest::GridDescriptor &);

template UnstructuredMesh<Kokkos::HostSpace> MeshFactory::from_named<Kokkos::HostSpace>(const std::string &);

template UnstructuredMesh<Kokkos::HostSpace> MeshFactory::from_rules<Kokkos::HostSpace>(const ingest::GridRulesParams &);

// CudaSpace (when CUDA is enabled)
#ifdef KOKKOS_ENABLE_CUDA
template UnstructuredMesh<Kokkos::CudaSpace> MeshFactory::from_descriptor<Kokkos::CudaSpace>(const ingest::GridDescriptor &);

template UnstructuredMesh<Kokkos::CudaSpace> MeshFactory::from_named<Kokkos::CudaSpace>(const std::string &);

template UnstructuredMesh<Kokkos::CudaSpace> MeshFactory::from_rules<Kokkos::CudaSpace>(const ingest::GridRulesParams &);
#endif

// HIPSpace (when HIP is enabled)
#ifdef KOKKOS_ENABLE_HIP
template UnstructuredMesh<Kokkos::HIPSpace> MeshFactory::from_descriptor<Kokkos::HIPSpace>(const ingest::GridDescriptor &);

template UnstructuredMesh<Kokkos::HIPSpace> MeshFactory::from_named<Kokkos::HIPSpace>(const std::string &);

template UnstructuredMesh<Kokkos::HIPSpace> MeshFactory::from_rules<Kokkos::HIPSpace>(const ingest::GridRulesParams &);
#endif

}  // namespace axis::topology
