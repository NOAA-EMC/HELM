# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple

import numpy as np
import xarray as xr

from . import axis_py


class ScripMeshInfo(NamedTuple):
    """Parsed SCRIP mesh nodes and connectivity arrays."""

    node_lon: np.ndarray
    node_lat: np.ndarray
    conn_offsets: np.ndarray
    conn_indices: np.ndarray


# Unstructured spatial dimension tags commonly used in climate datasets
UNSTRUCTURED_DIMS = {
    "ncol",
    "grid_size",
    "nCells",
    "nVertices",
    "nNodes",
    "nFaces",
    "nEdges",
    "n_node",
    "n_face",
    "n_edge",
    "n_cells",
    "n_vertices",
    "node",
    "face",
    "vertex",
    "cell",
    "n_pts",
}


def _get_non_spatial_dims(ds: xr.Dataset) -> set[str]:
    """Identify and filter out non-spatial dimensions (Time, Z, Member)."""
    spatial_keywords = {
        "lat",
        "lon",
        "x",
        "y",
        "node",
        "face",
        "element",
        "cell",
        "n_pts",
        "ncol",
        "ncells",
        "grid_size",
        "vert",
        "vertex",
        "vertices",
        "tile",
    }
    non_spatial = set()
    for d in ds.dims:
        d_lower = str(d).lower()
        if not any(kw in d_lower for kw in spatial_keywords):
            non_spatial.add(str(d))
    return non_spatial


def _find_coord(ds: xr.Dataset, name: str) -> xr.DataArray | None:
    """Find a coordinate array based on standard_name, axis, or name heuristics."""
    for c in ds.coords:
        da = ds[c]
        if da.attrs.get("standard_name") == name:
            return da
        if name == "latitude" and da.attrs.get("axis") == "Y":
            return da
        if name == "longitude" and da.attrs.get("axis") == "X":
            return da
    return None


def _get_mesh_info(
    ds: xr.Dataset,
    method: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray, tuple[int, ...], tuple[str, ...], bool]:
    """Detect grid type and extract coordinate DataArrays and dimensions."""
    non_spatial_dims = _get_non_spatial_dims(ds)

    lat = None
    lon = None

    # CF search priority
    lat = _find_coord(ds, "latitude")
    if lat is None:
        for v in ["lat", "latCell", "lat_face", "lat_node", "latitude"]:
            if v in ds:
                lat = ds[v]
                break

    if lat is None:
        raise KeyError("Could not find latitude coordinates in dataset.")

    lon_name = str(lat.name).replace("lat", "lon").replace("LAT", "LON").replace("latitude", "longitude")
    if lon_name in ds:
        lon = ds[lon_name]
    else:
        lon = _find_coord(ds, "longitude")

    if lon is None:
        for v in ["lon", "lonCell", "lon_face", "lon_node", "longitude"]:
            if v in ds:
                lon = ds[v]
                break

    if lon is None:
        raise KeyError("Could not find matching longitude coordinates.")

    # Drop non-spatial dimensions if present
    lat_isel = {d: 0 for d in non_spatial_dims if d in lat.dims}
    lon_isel = {d: 0 for d in non_spatial_dims if d in lon.dims}
    if lat_isel:
        lat = lat.isel(lat_isel, drop=True)
    if lon_isel:
        lon = lon.isel(lon_isel, drop=True)

    # Detect if unstructured
    is_unstructured = False
    if "mesh" in lat.attrs and "location" in lat.attrs:
        is_unstructured = True
    elif any(d in lat.dims for d in UNSTRUCTURED_DIMS):
        is_unstructured = True
    else:
        for var in ds.variables:
            if ds[var].attrs.get("cf_role") == "mesh_topology":
                is_unstructured = True
                break

    if not is_unstructured:
        if lat.ndim == 3:
            is_unstructured = True
        elif lat.ndim == 2:
            # If 2D curvilinear with NO lambert conformal conic mapping, treat as unstructured quad mesh
            has_lcc = False
            for var in ds.variables:
                if ds[var].attrs.get("grid_mapping_name") == "lambert_conformal_conic":
                    has_lcc = True
                    break
            if not has_lcc:
                is_unstructured = True

    if is_unstructured:
        return lon, lat, lat.shape, tuple(str(d) for d in lat.dims), True
    else:
        # Structured grid: both latitude and longitude are spatial dimensions!
        if lat.ndim == 1:
            # Regular grid with 1D lat and 1D lon.
            # Return full 2D spatial dims and shape
            dims = (str(lat.name), str(lon.name))
            shape = (len(lat), len(lon))
            return lon, lat, shape, dims, False
        else:
            # Curvilinear grid with 2D lat and 2D lon
            return lon, lat, lat.shape, tuple(str(d) for d in lat.dims), False


def _try_bounds_curvilinear_mesh(ds: xr.Dataset, lat: xr.DataArray, lon: xr.DataArray) -> "axis_py.Mesh | None":
    """Build a quad/polygon mesh from explicit CF ``bounds`` variables on 2-D coords.

    Returns None when the dataset has no resolvable 2-D bounds arrays, so the
    caller can fall back to synthesizing corners from centers.
    """
    lat_b_name = lat.attrs.get("bounds")
    lon_b_name = lon.attrs.get("bounds")
    if not lat_b_name or not lon_b_name:
        return None
    if lat_b_name not in ds or lon_b_name not in ds:
        return None

    lb = np.asarray(ds[lat_b_name].values, dtype=np.float64)
    ob = np.asarray(ds[lon_b_name].values, dtype=np.float64)
    if lb.ndim != 3 or ob.ndim != 3 or lb.shape != ob.shape:
        return None
    if lb.shape[0] != lat.shape[0] or lb.shape[1] != lon.shape[1]:
        return None
    nv = lb.shape[2]
    if nv < 3:
        return None

    # Drop a repeated closing vertex (ring stored with first == last).
    wrap_dup = np.all(lb[:, :, 0] == lb[:, :, -1]) and np.all(ob[:, :, 0] == ob[:, :, -1])
    if wrap_dup:
        lb, ob = lb[:, :, :-1], ob[:, :, :-1]
        nv -= 1
        if nv < 3:
            return None

    interior_dup = np.any((lb[:, :, 1:] == lb[:, :, :-1]) & (ob[:, :, 1:] == ob[:, :, :-1]))
    if not interior_dup:
        node_coords = np.asfortranarray(np.column_stack([ob.ravel(), lb.ravel()]))
        n_cells = lat.size
        conn_offsets = np.arange(0, n_cells * nv + 1, nv, dtype=np.int64)
        conn_indices = np.arange(n_cells * nv, dtype=np.int64)
        return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)

    # Per-cell filter of repeated padded corners (SCRIP-style padding).
    node_lons: list[float] = []
    node_lats: list[float] = []
    conn_offsets_list = [0]
    for c in range(lb.shape[0] * lb.shape[1]):
        lats_c = lb.reshape(-1, nv)[c]
        lons_c = ob.reshape(-1, nv)[c]
        prev = None
        n_here = 0
        for vi in range(nv):
            if prev is not None and lats_c[vi] == prev[0] and lons_c[vi] == prev[1]:
                continue
            prev = (lats_c[vi], lons_c[vi])
            node_lats.append(lats_c[vi])
            node_lons.append(lons_c[vi])
            n_here += 1
        if n_here < 3:
            return None  # malformed bounds: fall back to synthesis
        conn_offsets_list.append(len(node_lons))

    node_coords = np.asfortranarray(np.column_stack([np.array(node_lons), np.array(node_lats)]))
    return axis_py.make_ugrid_mesh(
        node_coords, np.array(conn_offsets_list, dtype=np.int64), np.arange(len(node_lons), dtype=np.int64)
    )


def _synthesize_curvilinear_corners(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize (ny+1, nx+1) corner coordinates from (ny, nx) cell centers via fast 2D slicing."""
    ny, nx = lon.shape
    pad_lon = np.pad(lon, 1, mode="edge")
    pad_lat = np.pad(lat, 1, mode="edge")

    # Sum the 4 surrounding padded elements
    sum_lon = pad_lon[:-1, :-1] + pad_lon[:-1, 1:] + pad_lon[1:, :-1] + pad_lon[1:, 1:]
    sum_lat = pad_lat[:-1, :-1] + pad_lat[:-1, 1:] + pad_lat[1:, :-1] + pad_lat[1:, 1:]

    clon = sum_lon / 4.0
    clat = sum_lat / 4.0
    return clon, clat


def _rectilinear_cell_edges(centers: np.ndarray, clamp: tuple[float, float] | None = None) -> np.ndarray:
    """CF cell edges from a 1-D array of cell centers.

    Interior edges are midpoints of adjacent centers; the outer edges extend
    half a cell beyond the first/last center. When ``clamp`` is given (e.g.
    latitude to [-90, 90]) edges are clipped into it, so a pole-inclusive
    center vector (linspace(-90, 90, n)) yields pole-anchored cells instead
    of overflowing past the pole (where sin() turns around and the spherical
    cell area would go negative).
    """
    c = np.asarray(centers, dtype=np.float64)
    n = c.size
    if n < 2:
        lo, hi = c[0] - 0.5, c[0] + 0.5
        if clamp is not None:
            lo, hi = max(clamp[0], lo), min(clamp[1], hi)
        return np.array([lo, hi], dtype=np.float64)
    edges = np.empty(n + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (c[:-1] + c[1:])
    edges[0] = c[0] - 0.5 * (c[1] - c[0])
    edges[-1] = c[-1] + 0.5 * (c[-1] - c[-2])
    if clamp is not None:
        edges = np.clip(edges, clamp[0], clamp[1])
    return edges


def _make_regular_mesh_from_centers(lons: np.ndarray, lats: np.ndarray) -> "axis_py.Mesh":
    """Build a rectilinear mesh from centers using native extrapolated bounds."""
    lon_edges = _rectilinear_cell_edges(lons)
    lat_edges = _rectilinear_cell_edges(lats, clamp=(-90.0, 90.0))
    return _structured_mesh_from_centers(
        lons,
        lats,
        policy=axis_py.CornerPolicy.RectilinearMidpoint,
        rectilinear_bounds=(lon_edges, lat_edges),
    )


def _structured_mesh_from_centers(
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    policy: axis_py.CornerPolicy,
    longitude_periodic: bool = False,
    longitude_period: float = 360.0,
    rectilinear_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    gaussian_weights: np.ndarray | None = None,
) -> "axis_py.Mesh":
    """Use the C++ structured-grid corner and geometry policies for center data."""
    lon_values = np.asarray(lons, dtype=np.float64)
    lat_values = np.asarray(lats, dtype=np.float64)
    if lon_values.ndim == 1 and lat_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values, indexing="xy")
    elif lon_values.ndim == 2 and lat_values.shape == lon_values.shape:
        lon_grid, lat_grid = lon_values, lat_values
    else:
        raise ValueError("center coordinates must be matching 2-D arrays or 1-D rectilinear axes")
    ny, nx = lon_grid.shape
    grid = axis_py.StructuredGrid(nx, ny, lon_grid.ravel(), lat_grid.ravel(), axis_py.CoordinateSystem.SphericalDeg)
    seam = axis_py.LongitudePeriodicity(longitude_periodic, longitude_period, True)
    if rectilinear_bounds is not None:
        grid.set_rectilinear_bounds(rectilinear_bounds[0], rectilinear_bounds[1])
    if gaussian_weights is not None:
        grid.set_gaussian_latitude_weights(gaussian_weights)
    return grid.to_unstructured(policy, seam)


def _triangulate_mpas_mesh(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate arbitrary polygon cells (like MPAS Voronoi cells) into triangles."""
    non_spatial_dims = _get_non_spatial_dims(ds)

    v_lat = ds["latVertex"]
    v_lon = ds["lonVertex"]
    v_conn = ds["verticesOnCell"]

    # Filter non-spatial dimensions individually to prevent mismatched dimension indexing
    isel_lat = {d: 0 for d in non_spatial_dims if d in v_lat.dims}
    if isel_lat:
        v_lat = v_lat.isel(isel_lat, drop=True)

    isel_lon = {d: 0 for d in non_spatial_dims if d in v_lon.dims}
    if isel_lon:
        v_lon = v_lon.isel(isel_lon, drop=True)

    isel_conn = {d: 0 for d in non_spatial_dims if d in v_conn.dims}
    if isel_conn:
        v_conn = v_conn.isel(isel_conn, drop=True)

    # Normalize longitudes and latitudes to degrees
    node_lat = v_lat.values
    node_lon = v_lon.values
    if np.any(np.abs(node_lat) > 2.0 * np.pi):
        pass  # Already degrees
    else:
        node_lat = np.degrees(node_lat)
        node_lon = np.degrees(node_lon)

    # Wrap longitudes to [0, 360]
    node_lon = np.mod(node_lon, 360.0)

    conn_raw = v_conn.values
    n_edges = ds["nEdgesOnCell"].values if "nEdgesOnCell" in ds else np.full(conn_raw.shape[0], conn_raw.shape[1])

    n_cells, max_edges = conn_raw.shape
    max_tris = max_edges - 2
    j = np.arange(1, max_tris + 1)
    mask = j[None, :] < (n_edges[:, None] - 1)

    v0 = np.repeat(conn_raw[:, 0:1], max_tris, axis=1) - 1
    v1 = conn_raw[:, 1:-1] - 1
    v2 = conn_raw[:, 2:] - 1

    element_conn = np.stack([v0[mask], v1[mask], v2[mask]], axis=1).flatten()
    np.repeat(np.arange(n_cells), max_tris)[mask.flatten()]

    return node_lon, node_lat, element_conn.astype(np.int64)


def _parse_scrip_bounds(ds: xr.Dataset) -> ScripMeshInfo:
    """Parse unstructured SCRIP-style cell centers and 2D bounds into general polygon nodes/connectivity offsets/indices."""
    # Find longitude/latitude coordinates
    lat = None
    for v in ["lat", "latCell", "latitude"]:
        if v in ds:
            lat = ds[v]
            break
    if lat is None:
        raise KeyError("Could not find latitude coordinates in dataset.")

    lon_name = str(lat.name).replace("lat", "lon").replace("LAT", "LON").replace("latitude", "longitude")
    if lon_name in ds:
        lon = ds[lon_name]
    else:
        for v in ["lon", "lonCell", "longitude"]:
            if v in ds:
                lon = ds[v]
                break
    if lon is None:
        raise KeyError("Could not find longitude coordinates.")

    lat_bnds_name = lat.attrs.get("bounds", "lat_bnds")
    lon_bnds_name = lon.attrs.get("bounds", "lon_bnds")

    lat_bnds = np.asarray(ds[lat_bnds_name].values, dtype=np.float64)
    lon_bnds = np.asarray(ds[lon_bnds_name].values, dtype=np.float64)

    res = axis_py.parse_scrip_bounds(lat_bnds, lon_bnds)
    return ScripMeshInfo(res["node_lon"], res["node_lat"], res["conn_offsets"], res["conn_indices"])


def _get_ugrid_info(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract standard UGRID mesh connectivity and node coordinates."""
    mesh_var = None
    for var in ds.variables:
        if ds[var].attrs.get("cf_role") == "mesh_topology":
            mesh_var = var
            break

    if mesh_var is None:
        raise KeyError("Could not find CF UGRID mesh_topology variable.")

    attrs = ds[mesh_var].attrs
    node_coords_names = attrs.get("node_coordinates", "").split()
    face_conn_name = attrs.get("face_node_connectivity", "")

    node_lon = ds[node_coords_names[0]].values
    node_lat = ds[node_coords_names[1]].values
    face_conn = ds[face_conn_name].values

    # Adjust for 1-based indexing in some UGRID files
    start_index = ds[face_conn_name].attrs.get("start_index", 0)
    if start_index == 1:
        face_conn = face_conn - 1

    return node_lon, node_lat, face_conn.astype(np.int64).flatten()


def create_axis_mesh(ds: xr.Dataset, method: str | None = None) -> axis_py.Mesh:
    """Build a native C++ Kokkos-parallel AXIS Mesh from an xarray dataset."""
    lon, lat, shape, dims, is_unstructured = _get_mesh_info(ds, method)

    if is_unstructured:
        # 1. MPAS (Arbitrary polygonal cells)
        if "verticesOnCell" in ds and "latVertex" in ds:
            # Check if method is cell-centered (nearest/conservative/conservative2nd)
            cell_centered = (method is not None) and (method.lower() in ["nearest", "conservative", "conservative2nd"])

            # Normalize and wrap coordinates
            v_lat = ds["latVertex"]
            v_lon = ds["lonVertex"]
            non_spatial_dims = _get_non_spatial_dims(ds)
            isel_lat = {d: 0 for d in non_spatial_dims if d in v_lat.dims}
            if isel_lat:
                v_lat = v_lat.isel(isel_lat, drop=True)
            isel_lon = {d: 0 for d in non_spatial_dims if d in v_lon.dims}
            if isel_lon:
                v_lon = v_lon.isel(isel_lon, drop=True)

            node_lat = v_lat.values
            node_lon = v_lon.values
            if not np.any(np.abs(node_lat) > 2.0 * np.pi):
                node_lat = np.degrees(node_lat)
                node_lon = np.degrees(node_lon)
            node_lon = np.mod(node_lon, 360.0)
            node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))

            if cell_centered:
                # Raw polygonal MPAS mesh directly (no triangulation!)
                v_conn = ds["verticesOnCell"]
                isel_conn = {d: 0 for d in non_spatial_dims if d in v_conn.dims}
                if isel_conn:
                    v_conn = v_conn.isel(isel_conn, drop=True)

                conn_raw = v_conn.values
                n_edges = ds["nEdgesOnCell"].values if "nEdgesOnCell" in ds else np.full(conn_raw.shape[0], conn_raw.shape[1])

                n_cells, max_edges = conn_raw.shape
                conn_offsets = np.zeros(n_cells + 1, dtype=np.int64)
                conn_indices: list[int] = []
                cell_mask = np.ones(n_cells, dtype=np.int32)
                for cell, raw_arity in enumerate(n_edges):
                    arity = int(raw_arity)
                    if arity < 0 or arity > max_edges:
                        raise ValueError(f"MPAS cell {cell} has invalid nEdgesOnCell={arity}")
                    if arity < 3:
                        # Keep source-field indexing stable while marking
                        # inactive/empty cells explicitly. The repeated-node
                        # placeholder is never used by remapping because its
                        # mask is zero.
                        if node_coords.shape[0] == 0:
                            raise ValueError("cannot represent masked MPAS cells without any mesh vertices")
                        conn_indices.extend([0, 0, 0])
                        cell_mask[cell] = 0
                    else:
                        conn_indices.extend((conn_raw[cell, :arity] - 1).astype(np.int64).tolist())
                    conn_offsets[cell + 1] = len(conn_indices)

                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, np.asarray(conn_indices, dtype=np.int64), cell_mask)
            else:
                # Triangulated MPAS (for bilinear/bicubic/patch) - C++ accelerated
                v_conn = ds["verticesOnCell"]
                isel_conn = {d: 0 for d in non_spatial_dims if d in v_conn.dims}
                if isel_conn:
                    v_conn = v_conn.isel(isel_conn, drop=True)

                conn_raw = v_conn.values.astype(np.int64)
                n_edges = (
                    ds["nEdgesOnCell"].values.astype(np.int64)
                    if "nEdgesOnCell" in ds
                    else np.full(conn_raw.shape[0], conn_raw.shape[1], dtype=np.int64)
                )

                tri_res = axis_py.triangulate_poly_cells(node_coords, conn_raw, n_edges)
                conn_offsets = tri_res["conn_offsets"]
                conn_indices = tri_res["conn_indices"]
                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
        # 2. SCRIP 2D Bounds format
        elif "lat_bnds" in ds or any("bounds" in ds[v].attrs for v in ds.variables if v in ["lat", "lon"]):
            node_lon, node_lat, conn_offsets, conn_indices = _parse_scrip_bounds(ds)
            node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))
            return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
        # 3. Curvilinear (2D) or Cubed-Sphere (3D) coordinate arrays fallback
        elif lat.ndim in [2, 3]:
            if lat.ndim == 2:
                # Prefer explicit CF cell bounds when the 2-D coordinate
                # variables carry a `bounds` attribute: the true corners are
                # exact, while synthesizing corners from centers shrinks
                # boundary cells by half a cell (same bug family as the
                # 1-D center-as-corner issue fixed in
                # _make_regular_mesh_from_centers).
                mesh = _try_bounds_curvilinear_mesh(ds, lat, lon)
                if mesh is not None:
                    return mesh
                # 2D Curvilinear grid
                ni, nj = lon.shape[1], lon.shape[0]
                cx = lon.values.ravel()
                cy = lat.values.ravel()
                grid = axis_py.StructuredGrid(ni, nj, cx, cy, axis_py.CoordinateSystem.SphericalDeg)
                return grid.to_unstructured(axis_py.CornerPolicy.CurvilinearApproximate, axis_py.LongitudePeriodicity())
            else:
                # 3D Cubed-Sphere grid (ntiles, ny, nx)
                ntiles, ny, nx = lon.shape
                node_coords_list = []
                conn_indices_list = []
                node_offset = 0

                # Check if dataset contains supergrid corner variables (e.g. x and y of shape (2*ny+1, 2*nx+1))
                has_supergrid_xy = "x" in ds and "y" in ds and ds["x"].ndim == 2 and ds["x"].shape == (2 * ny + 1, 2 * nx + 1)

                for t in range(ntiles):
                    if has_supergrid_xy:
                        clon_t = ds["x"].values[0::2, 0::2]
                        clat_t = ds["y"].values[0::2, 0::2]
                    else:
                        clon_t, clat_t = _synthesize_curvilinear_corners(lon[t].values, lat[t].values)

                    coords_t = np.column_stack([clon_t.ravel(), clat_t.ravel()])
                    node_coords_list.append(coords_t)

                    ni, nj = nx, ny
                    nip1 = ni + 1
                    i_grid, j_grid = np.meshgrid(np.arange(ni), np.arange(nj))
                    bl = (i_grid + j_grid * nip1 + node_offset).ravel()
                    br = ((i_grid + 1) + j_grid * nip1 + node_offset).ravel()
                    tr = ((i_grid + 1) + (j_grid + 1) * nip1 + node_offset).ravel()
                    tl = (i_grid + (j_grid + 1) * nip1 + node_offset).ravel()

                    indices_t = np.column_stack([bl, br, tr, tl]).astype(np.int64).ravel()
                    conn_indices_list.append(indices_t)

                    node_offset += len(coords_t)

                node_coords = np.asfortranarray(np.concatenate(node_coords_list))
                conn_indices = np.concatenate(conn_indices_list)
                conn_offsets = np.arange(0, len(conn_indices) + 1, 4, dtype=np.int64)

                return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
        # 4. CF-UGRID standard
        else:
            node_lon, node_lat, element_conn = _get_ugrid_info(ds)
            node_coords = np.asfortranarray(np.column_stack([node_lon, node_lat]))
            conn_offsets = np.arange(0, len(element_conn) + 1, 3, dtype=np.int64)
            conn_indices = element_conn.astype(np.int64)
            return axis_py.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)
    else:
        # Structured: regular or rectilinear/projected
        if lon.ndim == 1 and lat.ndim == 1:
            # Regular grid: derive true CF cell edges from the 1-D centers
            # (midpoints, clamped to ±90 at the poles) instead of passing the
            # centers themselves as corners.
            return _make_regular_mesh_from_centers(np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))
        else:
            # 2D Curvilinear or Projected grid
            # If coordinates have a grid_mapping or PROJ metadata, build a projected mesh
            grid_mapping = None
            for v in ds.variables:
                if "grid_mapping_name" in ds[v].attrs:
                    grid_mapping = ds[v].attrs["grid_mapping_name"]
                    break

            if grid_mapping == "lambert_conformal_conic":
                # Generate local projection string and coordinates
                # Host models can configure custom LCC params here, otherwise fallback to CONUS standard
                if not getattr(axis_py, "HAVE_PROJ", False):
                    raise RuntimeError(
                        f"This dataset uses a '{grid_mapping}' grid_mapping, which "
                        "requires projected-coordinate support, but axis_py was built without "
                        "PROJ (AXIS_ENABLE_PROJ=OFF). Rebuild AXIS with -DAXIS_ENABLE_PROJ=ON "
                        "to regrid projected grids."
                    ) from None
                proj_string = "+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
                ni, nj = shape[1], shape[0]
                center_x = lon.values.ravel()
                center_y = lat.values.ravel()
                return axis_py.make_projected_mesh(ni, nj, proj_string, center_x, center_y)
            else:
                # Flat regular 2D fallback: rows share lon, columns share lat
                ni, nj = shape[1], shape[0]
                lon_1d = np.asarray(lon.values[:, 0] if ni == 1 else lon.values[0, :], dtype=np.float64)
                lat_1d = np.asarray(lat.values[:, 0], dtype=np.float64)
                lon_1d = np.asarray(lon.values[0, :], dtype=np.float64)
                lat_1d = np.asarray(lat.values[:, 0], dtype=np.float64)

                return _make_regular_mesh_from_centers(lon_1d, lat_1d)


# ==============================================================================
# Low-Level Explicit Geometry API
# ==============================================================================


class Geometry:
    """Base abstract class representing any physical coordinate layout in AXIS."""

    def to_mesh(self) -> axis_py.Mesh:
        """Convert this geometry to a unified C++ UnstructuredMesh."""
        raise NotImplementedError()


class XarrayGeometry(Geometry):
    """Wraps an xarray Dataset or DataArray to defer mesh construction."""

    def __init__(self, ds: xr.Dataset | xr.DataArray, method: str | None = None):
        self.ds = ds
        self.method = method

    def to_mesh(self) -> axis_py.Mesh:
        # Keep 100% of existing, verified auto-detection and triangulation/centering logic!
        ds_normalized = self.ds.to_dataset(name="_tmp_data") if isinstance(self.ds, xr.DataArray) else self.ds
        return create_axis_mesh(ds_normalized, self.method)


class RectilinearGrid(Geometry):
    """
    Represent a standard 2D lat-lon grid with 1-D coordinate vectors.
    """

    def __init__(self, lons: np.ndarray, lats: np.ndarray, *, longitude_periodic: bool = False, longitude_period: float = 360.0):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.longitude_periodic = bool(longitude_periodic)
        self.longitude_period = float(longitude_period)

    def to_mesh(self) -> axis_py.Mesh:
        lon_bounds = _rectilinear_cell_edges(self.lons)
        lat_bounds = _rectilinear_cell_edges(self.lats, clamp=(-90.0, 90.0))
        return _structured_mesh_from_centers(
            self.lons,
            self.lats,
            policy=axis_py.CornerPolicy.RectilinearMidpoint,
            longitude_periodic=self.longitude_periodic,
            longitude_period=self.longitude_period,
            rectilinear_bounds=(lon_bounds, lat_bounds),
        )


class GaussianGrid(Geometry):
    """Regular longitude grid on Gaussian latitudes, with exact quadrature bands.

    ``lats`` must be the Gaussian--Legendre nodes expressed as latitude angles,
    in either ascending or descending order. If weights are omitted, AXIS
    derives them from those nodes and rejects non-Gaussian latitude centers.
    """

    def __init__(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        latitude_weights: np.ndarray | None = None,
        *,
        longitude_periodic: bool = True,
        longitude_period: float = 360.0,
    ):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.latitude_weights = None if latitude_weights is None else np.asarray(latitude_weights, dtype=np.float64)
        self.longitude_periodic = bool(longitude_periodic)
        self.longitude_period = float(longitude_period)

    def to_mesh(self) -> axis_py.Mesh:
        if self.lons.ndim != 1 or self.lats.ndim != 1 or self.lats.size == 0:
            raise ValueError("GaussianGrid requires non-empty one-dimensional longitude and latitude centers")
        if self.latitude_weights is None:
            mu, weights_ascending = np.polynomial.legendre.leggauss(self.lats.size)
            gaussian_lats_ascending = np.degrees(np.arcsin(mu))
            if np.allclose(self.lats, gaussian_lats_ascending, rtol=0.0, atol=1e-8):
                weights = weights_ascending
            elif np.allclose(self.lats, gaussian_lats_ascending[::-1], rtol=0.0, atol=1e-8):
                weights = weights_ascending[::-1]
            else:
                raise ValueError("latitude centers are not Gaussian-Legendre nodes; provide authoritative latitude_weights")
        else:
            weights = self.latitude_weights
            if weights.shape != self.lats.shape:
                raise ValueError("latitude_weights must have one entry per latitude row")
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 2.0, rtol=0.0, atol=1e-10):
            raise ValueError("latitude_weights must be finite, positive, and sum to 2")

        # The C++ Gaussian policy accumulates authoritative weights into exact
        # latitude boundaries. Longitude edges are midpoint/extrapolated from
        # the center vector and may span an explicitly periodic seam.
        lat_bounds = np.empty(self.lats.size + 1, dtype=np.float64)
        if self.lats.size == 1 or self.lats[0] > self.lats[-1]:
            lat_bounds[0] = 90.0
            lat_bounds[1:] = np.degrees(np.arcsin(np.clip(1.0 - np.cumsum(weights), -1.0, 1.0)))
        else:
            lat_bounds[0] = -90.0
            lat_bounds[1:] = np.degrees(np.arcsin(np.clip(-1.0 + np.cumsum(weights), -1.0, 1.0)))
        lon_bounds = _rectilinear_cell_edges(self.lons)
        return _structured_mesh_from_centers(
            self.lons,
            self.lats,
            policy=axis_py.CornerPolicy.GaussianLatLon,
            longitude_periodic=self.longitude_periodic,
            longitude_period=self.longitude_period,
            rectilinear_bounds=(lon_bounds, lat_bounds),
        )


class CurvilinearGrid(Geometry):
    """
    Represent a 2D curvilinear grid with 2-D coordinate matrices.
    """

    def __init__(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        proj_string: str | None = None,
        *,
        corner_policy: str = "curvilinear_approximate",
        longitude_periodic: bool = False,
        longitude_period: float = 360.0,
        corner_lons: np.ndarray | None = None,
        corner_lats: np.ndarray | None = None,
    ):
        self.lons = np.asarray(lons, dtype=np.float64)
        self.lats = np.asarray(lats, dtype=np.float64)
        self.proj_string = proj_string
        self.corner_policy = corner_policy
        self.longitude_periodic = bool(longitude_periodic)
        self.longitude_period = float(longitude_period)
        self.corner_lons = None if corner_lons is None else np.asarray(corner_lons, dtype=np.float64)
        self.corner_lats = None if corner_lats is None else np.asarray(corner_lats, dtype=np.float64)

    def to_mesh(self) -> axis_py.Mesh:
        if self.proj_string:
            return axis_py.make_projected_mesh(
                self.lons.shape[1],
                self.lons.shape[0],
                self.proj_string,
                self.lons.ravel(),
                self.lats.ravel(),
            )
        else:
            ny, nx = self.lons.shape
            if self.lats.shape != self.lons.shape:
                raise ValueError("curvilinear longitude and latitude arrays must have matching shapes")
            policies = {
                "require_explicit": axis_py.CornerPolicy.RequireExplicit,
                "gaussian_lat_lon": axis_py.CornerPolicy.GaussianLatLon,
                "rectilinear_midpoint": axis_py.CornerPolicy.RectilinearMidpoint,
                "curvilinear_approximate": axis_py.CornerPolicy.CurvilinearApproximate,
            }
            try:
                policy = policies[self.corner_policy]
            except KeyError as exc:
                raise ValueError(f"unknown corner_policy {self.corner_policy!r}; expected one of {tuple(policies)}") from exc
            if (self.corner_lons is None) != (self.corner_lats is None):
                raise ValueError("corner_lons and corner_lats must be provided together")
            if self.corner_lons is not None:
                expected = (ny + 1) * (nx + 1)
                if self.corner_lons.size != expected or self.corner_lats.size != expected:
                    raise ValueError(f"explicit corner arrays must contain {(ny + 1)} by {(nx + 1)} values")
                grid = axis_py.StructuredGrid(nx, ny, self.lons.ravel(), self.lats.ravel(), axis_py.CoordinateSystem.SphericalDeg)
                grid.set_corners(self.corner_lons.ravel(), self.corner_lats.ravel())
                return grid.to_unstructured(
                    policy,
                    axis_py.LongitudePeriodicity(self.longitude_periodic, self.longitude_period, True),
                )
            return _structured_mesh_from_centers(
                self.lons,
                self.lats,
                policy=policy,
                longitude_periodic=self.longitude_periodic,
                longitude_period=self.longitude_period,
            )


class ReducedGaussianGrid(Geometry):
    """Latitude-dependent longitude rows converted to mixed-arity CSR cells."""

    def __init__(
        self,
        latitude_bounds: np.ndarray,
        longitude_bounds_by_row: list[np.ndarray],
        *,
        longitude_periodic: bool = True,
        longitude_period: float = 360.0,
        increasing: bool = True,
        conforming: bool = True,
    ):
        self.latitude_bounds = np.asarray(latitude_bounds, dtype=np.float64)
        self.longitude_bounds_by_row = [np.asarray(row, dtype=np.float64) for row in longitude_bounds_by_row]
        self.longitude_periodic = bool(longitude_periodic)
        self.longitude_period = float(longitude_period)
        self.increasing = bool(increasing)
        self.conforming = bool(conforming)

    def to_mesh(self) -> axis_py.Mesh:
        return axis_py.make_reduced_gaussian_mesh(
            self.latitude_bounds,
            self.longitude_bounds_by_row,
            self.longitude_periodic,
            self.longitude_period,
            self.increasing,
            self.conforming,
        )


class MultiFaceGrid(Geometry):
    """Explicitly connected structured faces, suitable for cubed-sphere layouts.

    Each face dictionary contains ``ni``, ``nj``, and flattened ``node_coords``
    with shape ``((ni+1)*(nj+1), 2 or 3)``. Connections name face/edge pairs
    and whether the second edge's vertex order is reversed. Optional vertex
    equivalence groups contain ``(face, i, j)`` triples for multi-face corners.
    """

    def __init__(
        self,
        faces: list[dict],
        connections: list[dict],
        *,
        coordinate_system: axis_py.CoordinateSystem = axis_py.CoordinateSystem.SphericalDeg,
        vertex_equivalences: list[list[tuple[int, int, int]]] | None = None,
    ):
        self.faces = faces
        self.connections = connections
        self.coordinate_system = coordinate_system
        self.vertex_equivalences = vertex_equivalences or []

    def to_mesh(self) -> axis_py.Mesh:
        return axis_py.make_multiface_mesh(self.faces, self.connections, self.coordinate_system, self.vertex_equivalences)


class UnstructuredMesh(Geometry):
    """
    Represent an arbitrary unstructured polygon grid (e.g. MPAS, FVCOM, SCRIP).
    """

    def __init__(
        self,
        node_coords: np.ndarray,
        connectivity_offsets: np.ndarray,
        connectivity_indices: np.ndarray,
    ):
        self.coords = np.asfortranarray(node_coords, dtype=np.float64)
        self.offsets = np.asarray(connectivity_offsets, dtype=np.int64)
        self.indices = np.asarray(connectivity_indices, dtype=np.int64)

    def to_mesh(self) -> axis_py.Mesh:
        return axis_py.make_ugrid_mesh(self.coords, self.offsets, self.indices)


class RuleGeometry(Geometry):
    """
    Geometry generated purely from abstract mathematical rules.
    """

    def __init__(self, config: dict):
        self.config = config

    def to_mesh(self) -> axis_py.Mesh:
        return axis_py.generate_mesh_from_rules(self.config)


class GridFactory:
    """
    Convenience factory to convert high-level containers to Geometry classes.
    """

    @staticmethod
    def from_xarray(
        ds: xr.Dataset | xr.DataArray,
        lon_var: str | None = None,
        lat_var: str | None = None,
        method: str | None = None,
    ) -> Geometry:
        """
        Build an explicit AXIS geometry from an xarray container.
        """
        if lon_var and lat_var:
            lons = ds[lon_var].values
            lats = ds[lat_var].values
            if lons.ndim == 1:
                return RectilinearGrid(lons, lats)
            return CurvilinearGrid(lons, lats)
        return XarrayGeometry(ds, method=method)

    @staticmethod
    def from_dict(mapping: dict) -> Geometry:
        """
        Build an explicit AXIS geometry from a dict of coordinate arrays.

        Accepts either ``lon``/``lat`` or ``lons``/``lats`` keys; 1-D arrays yield a
        RectilinearGrid, higher-dimensional arrays a CurvilinearGrid.
        """
        lons = np.asarray(mapping.get("lon") if mapping.get("lon") is not None else mapping.get("lons"))
        lats = np.asarray(mapping.get("lat") if mapping.get("lat") is not None else mapping.get("lats"))
        if lons.ndim == 1:
            return RectilinearGrid(lons, lats)
        return CurvilinearGrid(lons, lats)
