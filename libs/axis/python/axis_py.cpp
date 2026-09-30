// SPDX-License-Identifier: Apache-2.0
// AXIS Python bindings via nanobind
//
// Exposes the AXIS interpolation engine to Python for benchmarking and
// interactive use. Uses numpy arrays as the zero-copy buffer interface.
//
// Bindings cover:
//   - Mesh construction (regular lat-lon, named grids, from_descriptor)
//   - Weight generation with full RegridConfig dictionary
//   - Single-field apply and multi-field batch_apply
//   - Weight cache serialize/deserialize (to_bytes / from_bytes)
//   - Conservation check
//
// Links only against AXIS + nanobind — no AMIO/HALO dependency (Req 12.8).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <Kokkos_Core.hpp>
#include <axis/detail/gnomonic_projector.hpp>
#include <axis/detail/regular_grid_detector.hpp>
#include <axis/detail/spherical_clipper.hpp>
#include <axis/detail/spherical_geometry.hpp>
#include <axis/ingest/grid_descriptor.hpp>
#include <axis/solver/apply.hpp>
#include <axis/solver/conservation.hpp>
#include <axis/solver/gradient_reconstructor.hpp>
#include <axis/solver/interpolation_matrix.hpp>
#include <axis/solver/regrid_config.hpp>
#include <axis/solver/vector_regridder.hpp>
#include <axis/solver/vertical_regridder.hpp>
#include <axis/solver/weight_cache.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/topology/gmsh_writer.hpp>
#include <axis/topology/mesh_factory.hpp>
#include <axis/topology/named_grid_registry.hpp>
#include <axis/topology/projection_builder.hpp>
#include <axis/topology/rule_generator.hpp>
#include <axis/topology/structured_grid.hpp>
#include <axis/topology/unstructured_mesh.hpp>
#include <axis/types.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef AXIS_HAVE_NETCDF
#include <axis/io/esmf_weight_io.hpp>
#endif

namespace nb = nanobind;
using namespace nb::literals;

using HostMesh = axis::topology::UnstructuredMesh<Kokkos::HostSpace>;
using HostMatrix = axis::solver::InterpolationMatrix<Kokkos::HostSpace>;

// ─── Kokkos lifecycle ────────────────────────────────────────────────────────

static bool kokkos_initialized = false;

void ensure_kokkos() {
    if (!kokkos_initialized && !Kokkos::is_initialized()) {
        Kokkos::initialize();
        kokkos_initialized = true;
    }
}

// ─── Helper: Build a regular lat-lon mesh from numpy arrays ──────────────────

HostMesh make_regular_mesh(std::size_t ni, std::size_t nj, double lon_start, double lat_start, double dlon, double dlat) {
    ensure_kokkos();

    const std::size_t n_centers = ni * nj;
    const std::size_t n_corners = (ni + 1) * (nj + 1);

    Kokkos::View<double *, Kokkos::HostSpace> cx("cx", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> cy("cy", n_centers);
    Kokkos::View<double *, Kokkos::HostSpace> crx("crx", n_corners);
    Kokkos::View<double *, Kokkos::HostSpace> cry("cry", n_corners);

    for (std::size_t j = 0; j < nj; ++j) {
        for (std::size_t i = 0; i < ni; ++i) {
            cx(i + j * ni) = lon_start + (static_cast<double>(i) + 0.5) * dlon;
            cy(i + j * ni) = lat_start + (static_cast<double>(j) + 0.5) * dlat;
        }
    }
    for (std::size_t j = 0; j <= nj; ++j) {
        for (std::size_t i = 0; i <= ni; ++i) {
            crx(i + j * (ni + 1)) = lon_start + static_cast<double>(i) * dlon;
            cry(i + j * (ni + 1)) = lat_start + static_cast<double>(j) * dlat;
        }
    }

    axis::topology::StructuredGrid<Kokkos::HostSpace> grid(ni, nj, std::move(cx), std::move(cy), axis::topology::CoordinateSystem::SphericalDeg);
    grid.set_corners(std::move(crx), std::move(cry));

    return grid.to_unstructured();
}

// ─── Helper: Build a projected mesh (e.g. Lambert Conformal) via PROJ ───────

HostMesh make_projected_mesh(std::size_t ni, std::size_t nj, const std::string &proj_string, nb::ndarray<nb::numpy, double, nb::ndim<1>> center_x,
                             nb::ndarray<nb::numpy, double, nb::ndim<1>> center_y) {
    ensure_kokkos();

    axis::ingest::ProjectedParams params;
    params.proj_string = proj_string;

    axis::ingest::BufferViews buffers;
    buffers.ni = ni;
    buffers.nj = nj;
    buffers.center_x = axis::field_view<const double, 1>(center_x.data(), center_x.shape(0));
    buffers.center_y = axis::field_view<const double, 1>(center_y.data(), center_y.shape(0));

    auto grid = axis::topology::ProjectionBuilder::build<Kokkos::HostSpace>(params, buffers);
    return grid.to_unstructured();
}

// ─── Helper: Build an unstructured UGRID mesh from arrays ───────────────────

HostMesh make_ugrid_mesh(nb::ndarray<nb::numpy, double, nb::ndim<2>> node_coords, nb::ndarray<nb::numpy, axis::index_t, nb::ndim<1>> conn_offsets,
                         nb::ndarray<nb::numpy, axis::index_t, nb::ndim<1>> conn_indices) {
    ensure_kokkos();

    axis::ingest::GridDescriptor desc;
    desc.kind = axis::ingest::ConventionKind::UGRID;
    desc.ugrid.topology_dimension = 2;
    desc.ugrid.start_index = 0;

    desc.buffers.node_coords = axis::field_view<const double, 2>(node_coords.data(), node_coords.shape(0), node_coords.shape(1));
    desc.buffers.conn_offsets = axis::field_view<const axis::index_t, 1>(conn_offsets.data(), conn_offsets.shape(0));
    desc.buffers.conn_indices = axis::field_view<const axis::index_t, 1>(conn_indices.data(), conn_indices.shape(0));

    return axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(desc);
}

// ─── Helper: Parse RegridConfig from Python dict ─────────────────────────────

axis::solver::RegridConfig parse_regrid_config(const nb::dict &config) {
    axis::solver::RegridConfig cfg;

    if (config.contains("method")) {
        auto method_obj = config["method"];
        // Accept enum directly or string
        if (nb::isinstance<axis::solver::InterpolationMethod>(method_obj)) {
            cfg.method = nb::cast<axis::solver::InterpolationMethod>(method_obj);
        } else if (nb::isinstance<nb::str>(method_obj)) {
            std::string m = nb::cast<std::string>(method_obj);
            if (m == "bilinear" || m == "Bilinear")
                cfg.method = axis::solver::InterpolationMethod::Bilinear;
            else if (m == "nearest_neighbor" || m == "NearestNeighbor")
                cfg.method = axis::solver::InterpolationMethod::NearestNeighbor;
            else if (m == "bicubic" || m == "Bicubic")
                cfg.method = axis::solver::InterpolationMethod::Bicubic;
            else if (m == "patch" || m == "Patch")
                cfg.method = axis::solver::InterpolationMethod::Patch;
            else if (m == "conservative" || m == "Conservative1stOrder")
                cfg.method = axis::solver::InterpolationMethod::Conservative1stOrder;
            else if (m == "conservative2nd" || m == "Conservative2ndOrder")
                cfg.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
            else
                throw std::invalid_argument("Unknown method: " + m);
        }
    }

    if (config.contains("norm_type")) {
        auto nt = config["norm_type"];
        if (nb::isinstance<axis::solver::NormType>(nt)) {
            cfg.norm_type = nb::cast<axis::solver::NormType>(nt);
        } else if (nb::isinstance<nb::str>(nt)) {
            std::string s = nb::cast<std::string>(nt);
            if (s == "dstarea" || s == "DstArea")
                cfg.norm_type = axis::solver::NormType::DstArea;
            else if (s == "fracarea" || s == "FracArea")
                cfg.norm_type = axis::solver::NormType::FracArea;
            else
                throw std::invalid_argument("Unknown norm_type: " + s);
        }
    }

    if (config.contains("unmapped")) {
        auto um = config["unmapped"];
        if (nb::isinstance<axis::solver::UnmappedAction>(um)) {
            cfg.unmapped = nb::cast<axis::solver::UnmappedAction>(um);
        } else if (nb::isinstance<nb::str>(um)) {
            std::string s = nb::cast<std::string>(um);
            if (s == "error" || s == "Error")
                cfg.unmapped = axis::solver::UnmappedAction::Error;
            else if (s == "ignore" || s == "Ignore")
                cfg.unmapped = axis::solver::UnmappedAction::Ignore;
            else
                throw std::invalid_argument("Unknown unmapped action: " + s);
        }
    }

    if (config.contains("line_type")) {
        auto lt = config["line_type"];
        if (nb::isinstance<axis::solver::LineType>(lt)) {
            cfg.line_type = nb::cast<axis::solver::LineType>(lt);
        } else if (nb::isinstance<nb::str>(lt)) {
            std::string s = nb::cast<std::string>(lt);
            if (s == "cartesian" || s == "Cartesian")
                cfg.line_type = axis::solver::LineType::Cartesian;
            else if (s == "great_circle" || s == "GreatCircle")
                cfg.line_type = axis::solver::LineType::GreatCircle;
            else
                throw std::invalid_argument("Unknown line_type: " + s);
        }
    }

    if (config.contains("use_limiter")) {
        cfg.use_limiter = nb::cast<bool>(config["use_limiter"]);
    }

    return cfg;
}

// ─── Module definition ───────────────────────────────────────────────────────

NB_MODULE(axis_py, m) {
    m.doc() = "AXIS Python bindings — spatial interpolation for Earth-system fields";

    // ─── Build capability flags ──────────────────────────────────────────────
    // Report which optional AXIS features were compiled in, so Python callers
    // can gate functionality (e.g. projected meshes) without triggering a
    // runtime error deep inside the C++ core.
#ifdef AXIS_ENABLE_PROJ
    m.attr("HAVE_PROJ") = true;
#else
    m.attr("HAVE_PROJ") = false;
#endif
#ifdef AXIS_HAVE_NETCDF
    m.attr("HAVE_NETCDF") = true;
#else
    m.attr("HAVE_NETCDF") = false;
#endif

    // ─── InterpolationMethod enum ────────────────────────────────────────────
    nb::enum_<axis::solver::InterpolationMethod>(m, "Method")
        .value("Bilinear", axis::solver::InterpolationMethod::Bilinear)
        .value("NearestNeighbor", axis::solver::InterpolationMethod::NearestNeighbor)
        .value("Bicubic", axis::solver::InterpolationMethod::Bicubic)
        .value("Patch", axis::solver::InterpolationMethod::Patch)
        .value("Conservative", axis::solver::InterpolationMethod::Conservative1stOrder)
        .value("Conservative1stOrder", axis::solver::InterpolationMethod::Conservative1stOrder)
        .value("Conservative2ndOrder", axis::solver::InterpolationMethod::Conservative2ndOrder);

    // ─── NormType enum ───────────────────────────────────────────────────────
    nb::enum_<axis::solver::NormType>(m, "NormType")
        .value("DstArea", axis::solver::NormType::DstArea)
        .value("FracArea", axis::solver::NormType::FracArea);

    // ─── UnmappedAction enum ─────────────────────────────────────────────────
    nb::enum_<axis::solver::UnmappedAction>(m, "UnmappedAction")
        .value("Error", axis::solver::UnmappedAction::Error)
        .value("Ignore", axis::solver::UnmappedAction::Ignore);

    // ─── LineType enum ───────────────────────────────────────────────────────
    nb::enum_<axis::solver::LineType>(m, "LineType")
        .value("Cartesian", axis::solver::LineType::Cartesian)
        .value("GreatCircle", axis::solver::LineType::GreatCircle);

    // ─── Mesh wrapper class ──────────────────────────────────────────────────
    nb::class_<HostMesh>(m, "Mesh").def_prop_ro("n_nodes", &HostMesh::n_nodes).def_prop_ro("n_cells", &HostMesh::n_cells);

    // ─── StructuredGrid wrapper class ────────────────────────────────────────
    nb::class_<axis::topology::StructuredGrid<Kokkos::HostSpace>>(m, "StructuredGrid")
        .def(
            "__init__",
            [](axis::topology::StructuredGrid<Kokkos::HostSpace> *grid, std::size_t ni, std::size_t nj, nb::ndarray<const double, nb::ndim<1>> cx,
               nb::ndarray<const double, nb::ndim<1>> cy) {
                ensure_kokkos();

                Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> cx_in(cx.data(), cx.shape(0));
                Kokkos::View<const double *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> cy_in(cy.data(), cy.shape(0));

                Kokkos::View<double *, Kokkos::HostSpace> cx_v("structured_grid_cx", cx.shape(0));
                Kokkos::View<double *, Kokkos::HostSpace> cy_v("structured_grid_cy", cy.shape(0));
                Kokkos::deep_copy(cx_v, cx_in);
                Kokkos::deep_copy(cy_v, cy_in);

                new (grid) axis::topology::StructuredGrid<Kokkos::HostSpace>(ni, nj, std::move(cx_v), std::move(cy_v),
                                                                             axis::topology::CoordinateSystem::SphericalDeg);
            },
            "ni"_a, "nj"_a, "cx"_a, "cy"_a)
        .def(
            "set_corners",
            [](axis::topology::StructuredGrid<Kokkos::HostSpace> &grid, nb::ndarray<const double, nb::ndim<1>> crx,
               nb::ndarray<const double, nb::ndim<1>> cry) {
                Kokkos::View<double *, Kokkos::HostSpace> crx_v(const_cast<double *>(crx.data()), crx.shape(0));
                Kokkos::View<double *, Kokkos::HostSpace> cry_v(const_cast<double *>(cry.data()), cry.shape(0));
                grid.set_corners(crx_v, cry_v);
            },
            "crx"_a, "cry"_a)
        .def("to_unstructured", &axis::topology::StructuredGrid<Kokkos::HostSpace>::to_unstructured, "Convert structured grid to unstructured mesh");

    // ─── InterpolationMatrix wrapper class ───────────────────────────────────
    nb::class_<HostMatrix>(m, "Matrix")
        .def_prop_ro("nnz", &HostMatrix::nnz)
        .def_prop_ro("n_src", &HostMatrix::n_src)
        .def_prop_ro("n_dst", &HostMatrix::n_dst)
        .def_prop_ro("is_csr", &HostMatrix::is_csr)

        .def_prop_ro(
            "factor_list",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const double> {
                auto view = matrix.factor_list();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const double>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Interpolation weights list")
        .def_prop_ro(
            "factor_col",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const axis::index_t> {
                auto view = matrix.factor_col();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const axis::index_t>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Source column indices list")
        .def_prop_ro(
            "factor_row",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const axis::index_t> {
                auto view = matrix.factor_row();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const axis::index_t>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Destination row indices list")
        .def_prop_ro(
            "frac_a",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const double> {
                auto view = matrix.frac_a();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const double>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Source fractions")
        .def_prop_ro(
            "frac_b",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const double> {
                auto view = matrix.frac_b();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const double>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Destination fractions")
        .def_prop_ro(
            "area_a",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const double> {
                auto view = matrix.area_a();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const double>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Source cell areas")
        .def_prop_ro(
            "area_b",
            [](const HostMatrix &matrix) -> nb::ndarray<nb::numpy, const double> {
                auto view = matrix.area_b();
                std::size_t shape[1] = {view.extent(0)};
                return nb::ndarray<nb::numpy, const double>(view.data_handle(), 1, shape, nb::handle());
            },
            nb::keep_alive<0, 1>(), "Destination cell areas")

        // to_csr() — convert to CSR format for row-parallel apply
        .def("to_csr", &HostMatrix::to_csr, "Convert internal COO representation to CSR format")

        // to_bytes() — serialize to binary blob via WeightCache (Req 12.5)
        .def(
            "to_bytes",
            [](const HostMatrix &matrix) -> nb::bytes {
                // Query required size
                const std::size_t size = axis::solver::WeightCache::serialize(matrix, nullptr, 0);

                // Allocate buffer and serialize
                std::vector<uint8_t> buf(size);
                axis::solver::WeightCache::serialize(matrix, buf.data(), size);

                // Return as Python bytes object
                return nb::bytes(reinterpret_cast<const char *>(buf.data()), size);
            },
            "Serialize the matrix to a binary blob (WeightCache format)")

        // from_bytes() — static method to deserialize (Req 12.5)
        .def_static(
            "from_bytes",
            [](nb::bytes data) -> HostMatrix {
                ensure_kokkos();
                const auto *ptr = reinterpret_cast<const void *>(data.c_str());
                const std::size_t size = data.size();
                return axis::solver::WeightCache::deserialize<Kokkos::HostSpace>(ptr, size);
            },
            "data"_a, "Deserialize a binary blob (WeightCache format) into a Matrix");

    // ─── Mesh construction ───────────────────────────────────────────────────

    // Make a regular lat-lon mesh
    m.def("make_regular_mesh", &make_regular_mesh, "ni"_a, "nj"_a, "lon_start"_a, "lat_start"_a, "dlon"_a, "dlat"_a,
          "Create a regular lat-lon UnstructuredMesh");

    // Make a projected mesh (e.g. Lambert Conformal)
    m.def("make_projected_mesh", &make_projected_mesh, "ni"_a, "nj"_a, "proj_string"_a, "center_x"_a, "center_y"_a,
          "Create a projected UnstructuredMesh using PROJ");

    // Make an unstructured UGRID mesh
    m.def("make_ugrid_mesh", &make_ugrid_mesh, "node_coords"_a, "conn_offsets"_a, "conn_indices"_a, "Create an unstructured UGRID UnstructuredMesh");

    m.def(
        "triangulate_poly_cells",
        [](nb::ndarray<const double, nb::ndim<2>> node_coords, nb::ndarray<const axis::index_t, nb::ndim<2>> conn_raw,
           nb::ndarray<const axis::index_t, nb::ndim<1>> n_edges) -> nb::dict {
            ensure_kokkos();

            const std::size_t n_cells = conn_raw.shape(0);
            const std::size_t max_edges = conn_raw.shape(1);

            if (n_edges.shape(0) != n_cells) {
                throw std::invalid_argument("triangulate_poly_cells: n_edges length must match conn_raw.shape(0)");
            }

            std::vector<axis::index_t> conn_indices;
            std::vector<axis::index_t> conn_offsets;
            conn_offsets.push_back(0);

            for (std::size_t c = 0; c < n_cells; ++c) {
                const std::size_t n_cell_edges = static_cast<std::size_t>(n_edges(c));
                if (n_cell_edges > max_edges) {
                    throw std::invalid_argument("triangulate_poly_cells: n_edges[" + std::to_string(c) + "] exceeds conn_raw.shape(1)");
                }
                if (n_cell_edges < 3) {
                    conn_offsets.push_back(conn_indices.size());
                    continue;
                }

                // Triangulate via standard triangle fan from vertex 0 of the cell
                axis::index_t v0 = conn_raw(c, 0) - 1;
                for (std::size_t j = 1; j < n_cell_edges - 1; ++j) {
                    conn_indices.push_back(v0);
                    conn_indices.push_back(conn_raw(c, j) - 1);
                    conn_indices.push_back(conn_raw(c, j + 1) - 1);
                }
                conn_offsets.push_back(conn_indices.size());
            }

            nb::dict res;
            // Return offsets and indices as numpy arrays
            auto off_uniq = std::make_unique<axis::index_t[]>(conn_offsets.size());
            std::copy(conn_offsets.begin(), conn_offsets.end(), off_uniq.get());
            auto ind_uniq = std::make_unique<axis::index_t[]>(conn_indices.size());
            std::copy(conn_indices.begin(), conn_indices.end(), ind_uniq.get());

            axis::index_t *raw_off = off_uniq.release();
            nb::capsule owner_off(raw_off, [](void *p) noexcept { delete[] static_cast<axis::index_t *>(p); });
            std::size_t shape_off[1] = {conn_offsets.size()};

            axis::index_t *raw_ind = ind_uniq.release();
            nb::capsule owner_ind(raw_ind, [](void *p) noexcept { delete[] static_cast<axis::index_t *>(p); });
            std::size_t shape_ind[1] = {conn_indices.size()};

            res["conn_offsets"] = nb::ndarray<nb::numpy, axis::index_t>(raw_off, 1, shape_off, std::move(owner_off));
            res["conn_indices"] = nb::ndarray<nb::numpy, axis::index_t>(raw_ind, 1, shape_ind, std::move(owner_ind));
            return res;
        },
        "node_coords"_a, "conn_raw"_a, "n_edges"_a, "Triangulate general poly cells into standard triangles");

    m.def(
        "parse_scrip_bounds",
        [](nb::ndarray<const double, nb::ndim<2>> lat_bnds, nb::ndarray<const double, nb::ndim<2>> lon_bnds) -> nb::dict {
            ensure_kokkos();

            const std::size_t n_cells = lat_bnds.shape(0);
            const std::size_t nv = lat_bnds.shape(1);

            if (lon_bnds.shape(0) != n_cells || lon_bnds.shape(1) != nv) {
                throw std::invalid_argument("parse_scrip_bounds: lon_bnds shape must match lat_bnds shape");
            }
            std::vector<double> node_lons;
            std::vector<double> node_lats;
            std::vector<axis::index_t> conn_offsets;
            std::vector<axis::index_t> conn_indices;

            conn_offsets.push_back(0);
            axis::index_t node_counter = 0;

            for (std::size_t idx = 0; idx < n_cells; ++idx) {
                // Filter out repeated padded corners
                std::vector<std::pair<double, double>> cell_vertices;
                for (std::size_t v = 0; v < nv; ++v) {
                    double lat_val = lat_bnds(idx, v);
                    double lon_val = lon_bnds(idx, v);

                    // Skip repeated padded corners (standard CDO SCRIP padding)
                    if (v > 0 && lat_val == lat_bnds(idx, v - 1) && lon_val == lon_bnds(idx, v - 1)) {
                        continue;
                    }
                    cell_vertices.push_back({lon_val, lat_val});
                }

                std::size_t n_vertices = cell_vertices.size();
                if (n_vertices < 3) {
                    // Fallback: if too many repeated, just use the first 3
                    cell_vertices = {
                        {lon_bnds(idx, 0), lat_bnds(idx, 0)}, {lon_bnds(idx, 1), lat_bnds(idx, 1)}, {lon_bnds(idx, 2), lat_bnds(idx, 2)}};
                    n_vertices = 3;
                }

                for (const auto &p : cell_vertices) {
                    // Wrap longitudes to [0, 360]
                    double wrapped_lon = std::fmod(p.first, 360.0);
                    if (wrapped_lon < 0.0) {
                        wrapped_lon += 360.0;
                    }
                    node_lons.push_back(wrapped_lon);
                    node_lats.push_back(p.second);
                    conn_indices.push_back(node_counter);
                    node_counter++;
                }

                conn_offsets.push_back(conn_indices.size());
            }

            nb::dict res;

            // Return offsets, indices, lons, and lats as numpy arrays
            auto off_uniq = std::make_unique<axis::index_t[]>(conn_offsets.size());
            std::copy(conn_offsets.begin(), conn_offsets.end(), off_uniq.get());
            auto ind_uniq = std::make_unique<axis::index_t[]>(conn_indices.size());
            std::copy(conn_indices.begin(), conn_indices.end(), ind_uniq.get());
            auto lon_uniq = std::make_unique<double[]>(node_lons.size());
            std::copy(node_lons.begin(), node_lons.end(), lon_uniq.get());
            auto lat_uniq = std::make_unique<double[]>(node_lats.size());
            std::copy(node_lats.begin(), node_lats.end(), lat_uniq.get());

            axis::index_t *raw_off = off_uniq.release();
            nb::capsule owner_off(raw_off, [](void *p) noexcept { delete[] static_cast<axis::index_t *>(p); });
            std::size_t shape_off[1] = {conn_offsets.size()};

            axis::index_t *raw_ind = ind_uniq.release();
            nb::capsule owner_ind(raw_ind, [](void *p) noexcept { delete[] static_cast<axis::index_t *>(p); });
            std::size_t shape_ind[1] = {conn_indices.size()};

            double *raw_lon = lon_uniq.release();
            nb::capsule owner_lon(raw_lon, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape_lon[1] = {node_lons.size()};

            double *raw_lat = lat_uniq.release();
            nb::capsule owner_lat(raw_lat, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape_lat[1] = {node_lats.size()};

            res["node_lon"] = nb::ndarray<nb::numpy, double>(raw_lon, 1, shape_lon, std::move(owner_lon));
            res["node_lat"] = nb::ndarray<nb::numpy, double>(raw_lat, 1, shape_lat, std::move(owner_lat));
            res["conn_offsets"] = nb::ndarray<nb::numpy, axis::index_t>(raw_off, 1, shape_off, std::move(owner_off));
            res["conn_indices"] = nb::ndarray<nb::numpy, axis::index_t>(raw_ind, 1, shape_ind, std::move(owner_ind));

            return res;
        },
        "lat_bnds"_a, "lon_bnds"_a, "Parse SCRIP-style cell bounds to unstructured nodes and connectivity in C++");

    // Expose GmshWriter ASCII exporter
    m.def(
        "write_gmsh", [](const std::string &filepath, const HostMesh &mesh) { axis::topology::GmshWriter::write<Kokkos::HostSpace>(filepath, mesh); },
        "filepath"_a, "mesh"_a, "Write an UnstructuredMesh to a Gmsh .msh v2.2 ASCII file");

    m.def(
        "generate_mesh_from_rules",
        [](const nb::dict &config) -> HostMesh {
            ensure_kokkos();
            axis::ingest::GridRulesParams rules;

            std::string kind_str = nb::cast<std::string>(config["kind"]);
            rules.kind = kind_str;

            if (config.contains("bbox_min_x")) rules.min_x = nb::cast<double>(config["bbox_min_x"]);
            if (config.contains("bbox_max_x")) rules.max_x = nb::cast<double>(config["bbox_max_x"]);
            if (config.contains("bbox_min_y")) rules.min_y = nb::cast<double>(config["bbox_min_y"]);
            if (config.contains("bbox_max_y")) rules.max_y = nb::cast<double>(config["bbox_max_y"]);
            if (config.contains("r_x")) rules.r_x = nb::cast<double>(config["r_x"]);
            if (config.contains("r_y")) rules.r_y = nb::cast<double>(config["r_y"]);
            if (config.contains("gaussian_n")) rules.gaussian_n = nb::cast<axis::index_t>(config["gaussian_n"]);
            if (config.contains("proj_string")) rules.proj_string = nb::cast<std::string>(config["proj_string"]);

            return axis::topology::RuleGenerator::generate<Kokkos::HostSpace>(rules);
        },
        "config"_a, "Generate an UnstructuredMesh using abstract mathematical rules");

    m.def(
        "reconstruct_gradient",
        [](nb::ndarray<const double, nb::ndim<1>> cell_values, nb::ndarray<const double, nb::ndim<2>> centroids,
           nb::ndarray<const axis::index_t, nb::ndim<1>> adj_offsets, nb::ndarray<const axis::index_t, nb::ndim<1>> adj_indices,
           bool use_limiter) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            std::size_t n_cells = cell_values.shape(0);
            if (centroids.shape(0) != n_cells || centroids.shape(1) != 3) {
                throw std::invalid_argument("centroids shape must be (n_cells, 3)");
            }
            if (adj_offsets.shape(0) != n_cells + 1) {
                throw std::invalid_argument("adj_offsets shape must be (n_cells + 1)");
            }

            auto out_grad_uniq = std::make_unique<double[]>(n_cells * 3);
            double *out_ptr = out_grad_uniq.get();

            Kokkos::View<const double *, Kokkos::HostSpace> val_view(cell_values.data(), n_cells);
            Kokkos::View<const double *[3], Kokkos::HostSpace> cent_view(centroids.data(), n_cells);
            Kokkos::View<const axis::index_t *, Kokkos::HostSpace> off_view(adj_offsets.data(), n_cells + 1);
            Kokkos::View<const axis::index_t *, Kokkos::HostSpace> ind_view(adj_indices.data(), adj_indices.shape(0));
            Kokkos::View<double *[3], Kokkos::HostSpace> grad_view(out_ptr, n_cells);

            axis::solver::GradientReconstructor<Kokkos::HostSpace>::compute(val_view, cent_view, off_view, ind_view, grad_view, use_limiter);

            double *raw_ptr = out_grad_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_cells, 3};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "cell_values"_a, "centroids"_a, "adj_offsets"_a, "adj_indices"_a, "use_limiter"_a = false,
        "Reconstruct cell-centered linear gradients via least-squares over CSR neighbors");

    // Make a named grid (Req 12.4)
    m.def(
        "make_named_mesh",
        [](const std::string &name) -> HostMesh {
            ensure_kokkos();
            return axis::topology::NamedGridRegistry::generate<Kokkos::HostSpace>(name);
        },
        "name"_a, "Generate a named grid (e.g., 'O32', 'F64')");

    // ─── Weight generation (Req 12.2) ────────────────────────────────────────

    // Simple overload: method enum only
    m.def(
        "generate_weights",
        [](const HostMesh &src, const HostMesh &dst, axis::solver::InterpolationMethod method) -> HostMatrix {
            ensure_kokkos();
            axis::solver::RegridConfig cfg;
            cfg.method = method;
            cfg.unmapped = axis::solver::UnmappedAction::Ignore;
            return axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src, dst, cfg);
        },
        "src_mesh"_a, "dst_mesh"_a, "method"_a, "Generate interpolation weights between src and dst meshes");

    // Dict-based overload: full RegridConfig (Req 12.2)
    m.def(
        "generate_weights",
        [](const HostMesh &src, const HostMesh &dst, const nb::dict &config) -> HostMatrix {
            ensure_kokkos();
            axis::solver::RegridConfig cfg = parse_regrid_config(config);
            return axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src, dst, cfg);
        },
        "src_mesh"_a, "dst_mesh"_a, "config"_a, "Generate interpolation weights with a RegridConfig dictionary");

    // ─── Apply weights to a 1-D numpy array (Req 12.3) ──────────────────────

    m.def(
        "apply_weights",
        [](const HostMatrix &matrix, nb::ndarray<nb::numpy, double, nb::ndim<1>> src_arr) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            const std::size_t n_src = src_arr.shape(0);
            const std::size_t n_dst = matrix.n_dst();

            if (n_src != matrix.n_src()) {
                throw std::invalid_argument("src array size (" + std::to_string(n_src) + ") != matrix.n_src (" + std::to_string(matrix.n_src()) +
                                            ")");
            }

            // Wrap numpy source as field_view (zero-copy)
            const double *src_ptr = src_arr.data();
            axis::field_view<const double, 1> src_view(src_ptr, n_src);

            // Allocate destination in an exception-safe unique_ptr
            auto dst_uniq = std::make_unique<double[]>(n_dst);
            double *dst_ptr = dst_uniq.get();
            axis::field_view<double, 1> dst_view(dst_ptr, n_dst);

            // Apply
            axis::solver::apply<Kokkos::HostSpace>(matrix, src_view, dst_view);

            // Return as numpy array (with ownership transfer)
            double *raw_ptr = dst_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            return nb::ndarray<nb::numpy, double>(raw_ptr, {n_dst}, std::move(owner));
        },
        "matrix"_a, "src"_a, "Apply interpolation matrix to source field, return destination array");

    // ─── Batch apply: multi-field SpMV (Req 12.6) ────────────────────────────

    m.def(
        "batch_apply",
        [](const HostMatrix &matrix, nb::ndarray<nb::numpy, double, nb::ndim<2>> src_arr) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            const std::size_t n_src = src_arr.shape(0);
            const std::size_t n_vars = src_arr.shape(1);
            const std::size_t n_dst = matrix.n_dst();

            if (n_src != matrix.n_src()) {
                throw std::invalid_argument("src array shape[0] (" + std::to_string(n_src) + ") != matrix.n_src (" + std::to_string(matrix.n_src()) +
                                            ")");
            }

            // Check memory layout (Req 12.7):
            // AXIS uses layout_left (Fortran-order / column-major).
            // numpy default is C-order (row-major / layout_right).
            // We accept both: if C-order, we interpret (n_src, n_vars) as row-major
            // and manually handle the stride difference by transposing into a
            // contiguous column-major buffer.
            const double *src_ptr = src_arr.data();

            // Detect if array is Fortran-contiguous (column-major)
            // stride(0) == sizeof(double) means column-major (fastest along axis 0)
            bool is_fortran_order = (src_arr.stride(0) == 1);

            // Allocate a column-major source buffer for AXIS
            std::vector<double> src_colmajor;
            if (!is_fortran_order) {
                // C-order (row-major): src_arr(i, v) is at ptr[i*n_vars + v]
                // We need column-major: buf(i, v) at buf[i + v*n_src]
                src_colmajor.resize(n_src * n_vars);
                for (std::size_t v = 0; v < n_vars; ++v) {
                    for (std::size_t i = 0; i < n_src; ++i) {
                        src_colmajor[i + v * n_src] = src_ptr[i * n_vars + v];
                    }
                }
                src_ptr = src_colmajor.data();
            }

            // Build field_view<const double, 2> over the column-major source
            axis::field_view<const double, 2> src_view(src_ptr, n_src, n_vars);

            // Allocate column-major destination buffer (exception-safe unique_ptr)
            auto dst_buf_uniq = std::make_unique<double[]>(n_dst * n_vars);
            double *dst_buf = dst_buf_uniq.get();
            axis::field_view<double, 2> dst_view(dst_buf, n_dst, n_vars);

            // Execute batch apply
            axis::solver::batch_apply<Kokkos::HostSpace>(matrix, src_view, dst_view);

            // Convert back to C-order (row-major) for numpy return
            // numpy expects shape (n_dst, n_vars) in C-order: dst[j*n_vars + v]
            auto result_uniq = std::make_unique<double[]>(n_dst * n_vars);
            double *result = result_uniq.get();
            for (std::size_t v = 0; v < n_vars; ++v) {
                for (std::size_t j = 0; j < n_dst; ++j) {
                    result[j * n_vars + v] = dst_buf[j + v * n_dst];
                }
            }

            // Return as numpy array (n_dst, n_vars) in C-order (with ownership transfer)
            double *raw_ptr = result_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_dst, n_vars};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "matrix"_a, "src"_a,
        "Apply interpolation matrix to multiple fields (cells × variables).\n"
        "Accepts both C-order and Fortran-order 2-D arrays.");

    // ─── NaN-aware Batch apply: multi-field SpMV with re-normalization ───────

    m.def(
        "nan_batch_apply",
        [](const HostMatrix &matrix, nb::ndarray<nb::numpy, double, nb::ndim<2>> src_arr, double na_thres) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            const std::size_t n_src = src_arr.shape(0);
            const std::size_t n_vars = src_arr.shape(1);
            const std::size_t n_dst = matrix.n_dst();

            if (n_src != matrix.n_src()) {
                throw std::invalid_argument("src array shape[0] (" + std::to_string(n_src) + ") != matrix.n_src (" + std::to_string(matrix.n_src()) +
                                            ")");
            }

            const double *src_ptr = src_arr.data();
            const bool is_fortran_contig = (src_arr.stride(0) == 1) && (src_arr.stride(1) == static_cast<nb::ssize_t>(n_src));

            // Allocate a column-major source buffer for AXIS (safe for any NumPy strides)
            std::vector<double> src_colmajor;
            if (!is_fortran_contig) {
                src_colmajor.resize(n_src * n_vars);
                for (std::size_t v = 0; v < n_vars; ++v) {
                    for (std::size_t i = 0; i < n_src; ++i) {
                        src_colmajor[i + v * n_src] = src_arr(i, v);
                    }
                }
                src_ptr = src_colmajor.data();
            }

            // Build field_view<const double, 2> over the column-major source
            axis::field_view<const double, 2> src_view(src_ptr, n_src, n_vars);

            // Allocate column-major destination buffer (exception-safe unique_ptr)
            auto dst_buf_uniq = std::make_unique<double[]>(n_dst * n_vars);
            double *dst_buf = dst_buf_uniq.get();
            axis::field_view<double, 2> dst_view(dst_buf, n_dst, n_vars);

            // Execute nan batch apply
            axis::solver::nan_batch_apply<Kokkos::HostSpace>(matrix, src_view, dst_view, na_thres);

            // Convert back to C-order (row-major) for numpy return
            // numpy expects shape (n_dst, n_vars) in C-order: dst[j*n_vars + v]
            auto result_uniq = std::make_unique<double[]>(n_dst * n_vars);
            double *result = result_uniq.get();
            for (std::size_t v = 0; v < n_vars; ++v) {
                for (std::size_t j = 0; j < n_dst; ++j) {
                    result[j * n_vars + v] = dst_buf[j + v * n_dst];
                }
            }

            // Return as numpy array (n_dst, n_vars) in C-order (with ownership transfer)
            double *raw_ptr = result_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_dst, n_vars};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "matrix"_a, "src"_a, "na_thres"_a,
        "Apply interpolation matrix with on-the-fly NaN-aware SpMV re-normalization.\n"
        "Accepts both C-order and Fortran-order 2-D arrays.");

    // ─── Conservation check ──────────────────────────────────────────────────

    m.def(
        "check_conservation",
        [](const HostMatrix &matrix, nb::ndarray<nb::numpy, double, nb::ndim<1>> src_arr,
           nb::ndarray<nb::numpy, double, nb::ndim<1>> dst_arr) -> nb::dict {
            ensure_kokkos();

            axis::field_view<const double, 1> src_view(src_arr.data(), src_arr.shape(0));
            axis::field_view<const double, 1> dst_view(dst_arr.data(), dst_arr.shape(0));

            auto report = axis::solver::check_conservation<Kokkos::HostSpace>(src_view, dst_view, matrix, axis::solver::NormType::DstArea);

            nb::dict result;
            result["src_integral"] = report.src_integral;
            result["dst_integral"] = report.dst_integral;
            result["absolute_error"] = report.absolute_error;
            result["relative_error"] = report.relative_error;
            return result;
        },
        "matrix"_a, "src"_a, "dst"_a, "Check conservation between source and destination fields");

    // ─── Tripolar grid detection ─────────────────────────────────────────────

    m.def(
        "detect_tripolar_grid",
        [](const HostMesh &mesh, std::size_t ni, std::size_t nj) -> nb::dict {
            ensure_kokkos();
            auto info = axis::detail::detect_tripolar_grid<Kokkos::HostSpace>(mesh, ni, nj);
            nb::dict res;
            res["is_tripolar"] = info.is_tripolar;
            res["ni"] = info.ni;
            res["nj"] = info.nj;
            res["seam_lat"] = info.seam_lat;
            res["seam_lon_center"] = info.seam_lon_center;
            return res;
        },
        "mesh"_a, "ni"_a, "nj"_a, "Detect whether an unstructured mesh represents a folded tripolar grid");

    // ─── Regular and Rectilinear grid detection ──────────────────────────────

    m.def(
        "detect_regular_grid",
        [](const HostMesh &mesh) -> nb::dict {
            ensure_kokkos();
            auto info = axis::detail::detect_regular_grid<Kokkos::HostSpace>(mesh);
            nb::dict res;
            res["is_regular"] = info.is_regular;
            res["lon_min"] = info.lon_min;
            res["lon_max"] = info.lon_max;
            res["delta_lon"] = info.delta_lon;
            res["lat_min"] = info.lat_min;
            res["lat_max"] = info.lat_max;
            res["delta_lat"] = info.delta_lat;
            res["ni"] = info.ni;
            res["nj"] = info.nj;
            return res;
        },
        "mesh"_a, "Detect whether an unstructured mesh represents a uniform regular lat-lon grid");

    m.def(
        "detect_rectilinear_grid",
        [](const HostMesh &mesh) -> nb::dict {
            ensure_kokkos();
            auto info = axis::detail::detect_rectilinear_grid<Kokkos::HostSpace>(mesh);
            nb::dict res;
            res["is_rectilinear"] = info.is_rectilinear;
            res["ni"] = info.ni;
            res["nj"] = info.nj;

            if (info.is_rectilinear) {
                std::vector<double> unique_lons(info.unique_lons.extent(0));
                for (std::size_t i = 0; i < unique_lons.size(); ++i) {
                    unique_lons[i] = info.unique_lons(i);
                }
                std::vector<double> unique_lats(info.unique_lats.extent(0));
                for (std::size_t j = 0; j < unique_lats.size(); ++j) {
                    unique_lats[j] = info.unique_lats(j);
                }
                res["unique_lons"] = unique_lons;
                res["unique_lats"] = unique_lats;
            } else {
                res["unique_lons"] = std::vector<double>{};
                res["unique_lats"] = std::vector<double>{};
            }
            return res;
        },
        "mesh"_a, "Detect whether an unstructured mesh represents a non-uniform rectilinear grid");

    // ─── Adjust by fraction ──────────────────────────────────────────────────

    m.def(
        "adjust_by_fraction",
        [](nb::ndarray<nb::numpy, double, nb::ndim<1>> dst_arr, nb::ndarray<const double, nb::ndim<1>> frac_b) {
            ensure_kokkos();
            const std::size_t n = dst_arr.shape(0);
            if (frac_b.shape(0) != n) {
                throw std::invalid_argument("dst and frac_b dimensions must match");
            }
            axis::field_view<double, 1> dst_view(dst_arr.data(), n);
            axis::field_view<const double, 1> frac_view(frac_b.data(), n);
            axis::solver::adjust_by_fraction<Kokkos::HostSpace>(dst_view, frac_view);
        },
        "dst"_a, "frac_b"_a, "Adjust destination field by fraction (modified in-place)");

    // ─── Vector weight generation ───────────────────────────────────────────

    m.def(
        "generate_vector_weights",
        [](const HostMesh &src, const HostMesh &dst, nb::ndarray<const double, nb::ndim<1>> src_alpha,
           nb::ndarray<const double, nb::ndim<1>> dst_alpha, const nb::dict &config) -> std::pair<HostMatrix, HostMatrix> {
            ensure_kokkos();

            axis::solver::RegridConfig cfg = parse_regrid_config(config);

            Kokkos::View<const double *, Kokkos::HostSpace> src_rot_view(src_alpha.data(), src.n_cells());
            Kokkos::View<const double *, Kokkos::HostSpace> dst_rot_view(dst_alpha.data(), dst.n_cells());

            axis::solver::GridRotation<Kokkos::HostSpace> src_rot{src_rot_view};
            axis::solver::GridRotation<Kokkos::HostSpace> dst_rot{dst_rot_view};

            auto [W_u, W_v] = axis::solver::VectorWeightGenerator<Kokkos::HostSpace>::generate(src, dst, src_rot, dst_rot, cfg);

            return std::make_pair(W_u, W_v);
        },
        "src_mesh"_a, "dst_mesh"_a, "src_alpha"_a, "dst_alpha"_a, "config"_a,
        "Generate coupled vector interpolation weights for U and V wind components");

    m.def(
        "vector_transform",
        [](const HostMatrix &W_u, const HostMatrix &W_v, nb::ndarray<nb::numpy, double, nb::ndim<2>> u,
           nb::ndarray<nb::numpy, double, nb::ndim<2>> v) -> std::pair<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>> {
            ensure_kokkos();

            const std::size_t n_src = u.shape(0);
            const std::size_t n_vars = u.shape(1);
            if (v.shape(0) != n_src || v.shape(1) != n_vars) {
                throw std::invalid_argument("vector_transform: u and v must have identical shapes (n_src, n_vars)");
            }
            const std::size_t n_dst = W_u.n_dst();
            // Setup combined source buffer: [2 * n_src, n_vars]
            // where first n_src rows are U, and next n_src rows are V.
            std::vector<double> uv_colmajor(2 * n_src * n_vars);
            for (std::size_t var = 0; var < n_vars; ++var) {
                for (std::size_t i = 0; i < n_src; ++i) {
                    uv_colmajor[i + var * 2 * n_src] = u(i, var);
                    uv_colmajor[i + n_src + var * 2 * n_src] = v(i, var);
                }
            }

            axis::field_view<const double, 2> uv_view(uv_colmajor.data(), 2 * n_src, n_vars);

            auto dst_u_uniq = std::make_unique<double[]>(n_dst * n_vars);
            auto dst_v_uniq = std::make_unique<double[]>(n_dst * n_vars);

            axis::field_view<double, 2> dst_u_view(dst_u_uniq.get(), n_dst, n_vars);
            axis::field_view<double, 2> dst_v_view(dst_v_uniq.get(), n_dst, n_vars);

            axis::solver::batch_apply<Kokkos::HostSpace>(W_u, uv_view, dst_u_view);
            axis::solver::batch_apply<Kokkos::HostSpace>(W_v, uv_view, dst_v_view);

            // Reconstruct row-major for Python return
            auto res_u_uniq = std::make_unique<double[]>(n_dst * n_vars);
            auto res_v_uniq = std::make_unique<double[]>(n_dst * n_vars);

            for (std::size_t var = 0; var < n_vars; ++var) {
                for (std::size_t j = 0; j < n_dst; ++j) {
                    res_u_uniq[j * n_vars + var] = dst_u_uniq[j + var * n_dst];
                    res_v_uniq[j * n_vars + var] = dst_v_uniq[j + var * n_dst];
                }
            }

            double *raw_u = res_u_uniq.release();
            nb::capsule owner_u(raw_u, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_dst, n_vars};

            double *raw_v = res_v_uniq.release();
            nb::capsule owner_v(raw_v, [](void *p) noexcept { delete[] static_cast<double *>(p); });

            return std::make_pair(nb::ndarray<nb::numpy, double>(raw_u, 2, shape, std::move(owner_u)),
                                  nb::ndarray<nb::numpy, double>(raw_v, 2, shape, std::move(owner_v)));
        },
        "W_u"_a, "W_v"_a, "u"_a, "v"_a, "Perform coupled SpMV remapping for vector fields");

#ifdef AXIS_HAVE_NETCDF
    m.def(
        "write_esmf",
        [](const std::string &filepath, const HostMatrix &matrix) { axis::io::EsmfWeightIO<Kokkos::HostSpace>::write_esmf(filepath, matrix); },
        "filepath"_a, "matrix"_a, "Write the interpolation weights matrix to an ESMF netCDF file");

    m.def(
        "read_esmf",
        [](const std::string &filepath) -> HostMatrix {
            ensure_kokkos();
            return axis::io::EsmfWeightIO<Kokkos::HostSpace>::read_esmf(filepath);
        },
        "filepath"_a, "Read the interpolation weights matrix from an ESMF netCDF file");
#endif

    // ─── Vertical Tension Spline interpolation ──────────────────────────────

    m.def(
        "interpolate_vertical",
        [](nb::ndarray<const double, nb::ndim<2>> src_field, nb::ndarray<const double, nb::ndim<1>> src_levels_1d,
           nb::ndarray<const double, nb::ndim<1>> dst_levels_1d, double tension) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            std::size_t n_col = src_field.shape(0);
            std::size_t n_src_lev = src_field.shape(1);
            std::size_t n_dst_lev = dst_levels_1d.shape(0);

            if (src_levels_1d.shape(0) != n_src_lev) {
                throw std::invalid_argument("src_levels size must match src_field levels dimension");
            }

            // Exception-safe unique_ptr allocation
            auto dst_uniq = std::make_unique<double[]>(n_col * n_dst_lev);
            double *dst_ptr = dst_uniq.get();

            Kokkos::View<const double **, Kokkos::HostSpace> src_view(src_field.data(), n_col, n_src_lev);
            Kokkos::View<double **, Kokkos::HostSpace> dst_view(dst_ptr, n_col, n_dst_lev);
            Kokkos::View<const double *, Kokkos::HostSpace> src_lev_view(src_levels_1d.data(), n_src_lev);
            Kokkos::View<const double *, Kokkos::HostSpace> dst_lev_view(dst_levels_1d.data(), n_dst_lev);

            axis::solver::VerticalRegridder<Kokkos::HostSpace>::interpolate(src_view, dst_view, src_lev_view, dst_lev_view, tension);

            // Relinquish ownership to Python capsule
            double *raw_ptr = dst_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_col, n_dst_lev};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "src_field"_a, "src_levels"_a, "dst_levels"_a, "tension"_a = 0.0, "Interpolate vertical 2D profiles using 1D uniform coordinates");

    m.def(
        "interpolate_vertical_varying",
        [](nb::ndarray<const double, nb::ndim<2>> src_field, nb::ndarray<const double, nb::ndim<2>> src_levels_2d,
           nb::ndarray<const double, nb::ndim<2>> dst_levels_2d, double tension) -> nb::ndarray<nb::numpy, double> {
            ensure_kokkos();

            std::size_t n_col = src_field.shape(0);
            std::size_t n_src_lev = src_field.shape(1);
            std::size_t n_dst_lev = dst_levels_2d.shape(1);

            if (src_levels_2d.shape(0) != n_col || src_levels_2d.shape(1) != n_src_lev) {
                throw std::invalid_argument("src_levels shape must match src_field shape");
            }
            if (dst_levels_2d.shape(0) != n_col) {
                throw std::invalid_argument("dst_levels column dimension must match src_field");
            }

            // Exception-safe unique_ptr allocation
            auto dst_uniq = std::make_unique<double[]>(n_col * n_dst_lev);
            double *dst_ptr = dst_uniq.get();

            Kokkos::View<const double **, Kokkos::HostSpace> src_view(src_field.data(), n_col, n_src_lev);
            Kokkos::View<double **, Kokkos::HostSpace> dst_view(dst_ptr, n_col, n_dst_lev);
            Kokkos::View<const double **, Kokkos::HostSpace> src_lev_view(src_levels_2d.data(), n_col, n_src_lev);
            Kokkos::View<const double **, Kokkos::HostSpace> dst_lev_view(dst_levels_2d.data(), n_col, n_dst_lev);

            axis::solver::VerticalRegridder<Kokkos::HostSpace>::interpolate(src_view, dst_view, src_lev_view, dst_lev_view, tension);

            // Relinquish ownership to Python capsule
            double *raw_ptr = dst_uniq.release();
            nb::capsule owner(raw_ptr, [](void *p) noexcept { delete[] static_cast<double *>(p); });
            std::size_t shape[2] = {n_col, n_dst_lev};
            return nb::ndarray<nb::numpy, double>(raw_ptr, 2, shape, std::move(owner));
        },
        "src_field"_a, "src_levels"_a, "dst_levels"_a, "tension"_a = 0.0, "Interpolate vertical 2D profiles using 2D spatially-varying coordinates");

    // ─── Unified Vec3 Class ──────────────────────────────────────────────────
    nb::class_<axis::detail::Vec3>(m, "Vec3")
        .def(nb::init<double, double, double>())
        .def_rw("x", &axis::detail::Vec3::x)
        .def_rw("y", &axis::detail::Vec3::y)
        .def_rw("z", &axis::detail::Vec3::z);

    // ─── Spherical Geometry math ─────────────────────────────────────────────
    m.def(
        "lonlat_to_xyz",
        [](double lon, double lat) -> axis::detail::Vec3 {
            auto v = axis::detail::spherical::lonlat_to_xyz(lon, lat);
            return axis::detail::Vec3{v.x, v.y, v.z};
        },
        "lon"_a, "lat"_a, "Convert lon/lat in radians to 3D Cartesian position vector");

    m.def(
        "xyz_to_lonlat",
        [](const axis::detail::Vec3 &p) -> std::pair<double, double> {
            axis::detail::spherical::Vec3 v{p.x, p.y, p.z};
            double lon, lat;
            axis::detail::spherical::xyz_to_lonlat(v, lon, lat);
            return {lon, lat};
        },
        "p"_a, "Convert 3D Cartesian position vector back to lon/lat in radians");

    m.def(
        "robust_orient_sphere",
        [](const axis::detail::Vec3 &a, const axis::detail::Vec3 &b, const axis::detail::Vec3 &c) -> double {
            axis::detail::spherical::Vec3 va{a.x, a.y, a.z};
            axis::detail::spherical::Vec3 vb{b.x, b.y, b.z};
            axis::detail::spherical::Vec3 vc{c.x, c.y, c.z};
            return axis::detail::spherical::robust_orient_sphere(va, vb, vc);
        },
        "a"_a, "b"_a, "c"_a, "Compute orientation sign of C relative to arc A->B using adaptive predicates");

    m.def(
        "great_circle_arc_intersection",
        [](const axis::detail::Vec3 &a1, const axis::detail::Vec3 &a2, const axis::detail::Vec3 &b1, const axis::detail::Vec3 &b2) -> nb::object {
            axis::detail::spherical::Vec3 va1{a1.x, a1.y, a1.z};
            axis::detail::spherical::Vec3 va2{a2.x, a2.y, a2.z};
            axis::detail::spherical::Vec3 vb1{b1.x, b1.y, b1.z};
            axis::detail::spherical::Vec3 vb2{b2.x, b2.y, b2.z};
            axis::detail::spherical::Vec3 vp;
            if (axis::detail::spherical::great_circle_arc_intersection(va1, va2, vb1, vb2, vp)) {
                return nb::cast(axis::detail::Vec3{vp.x, vp.y, vp.z});
            }
            return nb::none();
        },
        "a1"_a, "a2"_a, "b1"_a, "b2"_a, "Compute the exact intersection Vec3 of great-circle arcs A1->A2 and B1->B2, or None");

    // ─── Gnomonic Tangent Projection Math ────────────────────────────────────
    m.def(
        "gnomonic_forward",
        [](const axis::detail::Vec3 &center, const axis::detail::Vec3 &point) -> std::pair<double, double> {
            double u, v;
            axis::detail::GnomonicProjector::forward(center, point, u, v);
            return {u, v};
        },
        "center"_a, "point"_a, "Project point on the sphere onto the tangent plane at center");

    m.def(
        "gnomonic_inverse",
        [](const axis::detail::Vec3 &center, double u, double v) -> axis::detail::Vec3 {
            return axis::detail::GnomonicProjector::inverse(center, u, v);
        },
        "center"_a, "u"_a, "v"_a, "Reconstruct a unit-sphere point from tangent-plane coordinates");

    m.def(
        "bilinear_weights",
        [](nb::ndarray<const double, nb::ndim<1>> quad_u, nb::ndarray<const double, nb::ndim<1>> quad_v, double pu, double pv) -> nb::object {
            if (quad_u.shape(0) != 4 || quad_v.shape(0) != 4) {
                throw std::invalid_argument("quad_u and quad_v must have exactly 4 vertices");
            }
            double weights[4];
            bool ok = axis::detail::GnomonicProjector::bilinear_weights(quad_u.data(), quad_v.data(), pu, pv, weights);
            if (ok) {
                std::vector<double> out_weights(weights, weights + 4);
                return nb::cast(out_weights);
            }
            return nb::none();
        },
        "quad_u"_a, "quad_v"_a, "pu"_a, "pv"_a, "Solve the inverse bilinear problem for point pu, pv in quadrilateral");

    // ─── SphericalPolygon 32 Capacity Struct ─────────────────────────────────
    nb::class_<axis::detail::SphericalPolygon<32>>(m, "SphericalPolygon")
        .def(nb::init<>())
        .def_rw("n", &axis::detail::SphericalPolygon<32>::n)
        .def("area", &axis::detail::SphericalPolygon<32>::area)
        .def(
            "add_vertex",
            [](axis::detail::SphericalPolygon<32> &poly, const axis::detail::Vec3 &v) {
                if (poly.n >= 32) {
                    throw std::runtime_error("Polygon vertex capacity (32) exceeded");
                }
                poly.verts[poly.n++] = v;
            },
            "v"_a, "Add a unit-sphere Cartesian position vector vertex to the polygon");
}
