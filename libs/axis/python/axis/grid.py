# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from . import _core
from .errors import AxisCapabilityError, GridError
from .types import GridFamily

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
    """Identify and filter out non-spatial dimensions (Time, Z, Member).

    Multi-character keywords match as substrings; single-character index dims
    (``i``/``j``/``k``/``n``/``m``, common on cubed-sphere and structured grids)
    match by exact name so that e.g. ``time`` is not mistaken for ``i``.
    """
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
        "nv",
        "nbnds",
        "bnds",
        "edge",
    }
    spatial_exact = {"i", "j", "k", "n", "m", "p", "q"}
    non_spatial = set()
    for d in ds.dims:
        d_lower = str(d).lower()
        if d_lower in spatial_exact:
            continue
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


def _try_bounds_curvilinear_mesh(ds: xr.Dataset, lat: xr.DataArray, lon: xr.DataArray) -> _core.Mesh | None:
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
        return _core.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)

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
    return _core.make_ugrid_mesh(
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


def _make_regular_mesh_from_centers(lons: np.ndarray, lats: np.ndarray, cell_mask: np.ndarray | None = None) -> _core.Mesh:
    """Build a quad-cell mesh from 1-D cell-center vectors using true CF edges.

    Replaces the old center-as-corner convention (make_regular_mesh treats
    lat_start/lon_start as lower cell CORNERS, so passing centers shifted the
    whole grid by half a cell and overflowed the north pole). The (nj+1)x(ni+1)
    node lattice with CCW quads is fed through make_ugrid_mesh, which keeps
    the regular/nonuniform rectangle fast-paths available (all cells are quads).
    """
    lon_edges = _rectilinear_cell_edges(lons)
    lat_edges = _rectilinear_cell_edges(lats, clamp=(-90.0, 90.0))
    ni = lon_edges.size - 1
    nj = lat_edges.size - 1

    clat, clon = np.meshgrid(lat_edges, lon_edges, indexing="ij")
    node_coords = np.asfortranarray(np.column_stack([clon.ravel(), clat.ravel()]))

    # Vectorized CCW quad connectivity: cell (j, i) -> [bl, br, tr, tl]
    ncol = ni + 1
    ii, jj = np.meshgrid(np.arange(ni, dtype=np.int64), np.arange(nj, dtype=np.int64), indexing="xy")
    bl = ii + jj * ncol
    conn = np.stack([bl, bl + 1, bl + ncol + 1, bl + ncol], axis=-1).ravel().astype(np.int64)
    offsets = np.arange(0, conn.size + 1, 4, dtype=np.int64)
    return _core.make_ugrid_mesh(node_coords, offsets, conn, _as_int_mask(cell_mask))


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


def _parse_scrip_bounds(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    lat_bnds = ds[lat_bnds_name].values
    lon_bnds = ds[lon_bnds_name].values

    n_cells, nv = lat_bnds.shape

    node_lons = []
    node_lats = []
    conn_offsets = [0]
    conn_indices = []

    node_counter = 0
    for idx in range(n_cells):
        lats_c = lat_bnds[idx]
        lons_c = lon_bnds[idx]

        # Filter out repeated padded corners
        cell_vertices = []
        for vi in range(nv):
            # Skip repeated padded corners (standard CDO SCRIP padding)
            if vi > 0 and lats_c[vi] == lats_c[vi - 1] and lons_c[vi] == lons_c[vi - 1]:
                continue
            cell_vertices.append((lons_c[vi], lats_c[vi]))

        n_vertices = len(cell_vertices)
        if n_vertices < 3:
            # Fallback: if too many repeated, just use the first 3
            cell_vertices = [(lons_c[0], lats_c[0]), (lons_c[1], lats_c[1]), (lons_c[2], lats_c[2])]
            n_vertices = 3

        for lon_val, lat_val in cell_vertices:
            node_lons.append(lon_val)
            node_lats.append(lat_val)
            conn_indices.append(node_counter)
            node_counter += 1

        conn_offsets.append(len(conn_indices))

    return (
        np.mod(np.array(node_lons), 360.0),
        np.array(node_lats),
        np.array(conn_offsets, dtype=np.int64),
        np.array(conn_indices, dtype=np.int64),
    )


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


_NAMED_FAMILIES = {
    "C": GridFamily.CUBED_SPHERE,
    "O": GridFamily.UGRID,
    "N": GridFamily.UGRID,
    "F": GridFamily.UGRID,
    "R": GridFamily.UGRID,
    "G": GridFamily.UGRID,
}


def _to_degrees(lon: np.ndarray, lat: np.ndarray, *, wrap_lon: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Normalize coordinates: radians (max|lat| <= pi) -> degrees, lon -> [0, 360).

    Detection keys off latitude only (matching the verified legacy rule): a
    degree-valued grid has max|lat| up to 90 (> pi), while radians keep
    |lat| <= pi/2. Longitudes are then wrapped into [0, 360) unless
    ``wrap_lon`` is False — used for 1-D rectilinear center vectors, where a
    per-value mod would destroy monotonicity (e.g. ``[-170..170]`` →
    ``[190,303,57,170]``) and corrupt both the mesh and the dateline-spanning
    rotation angles. A globally-shifted rectilinear grid is instead brought
    into [0, 360) by a single constant offset that preserves monotonicity.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lat.size and np.abs(lat).max(initial=0.0) <= np.pi + 1e-12:
        lon = np.degrees(lon)
        lat = np.degrees(lat)
    if wrap_lon:
        lon = np.mod(lon, 360.0)
    elif lon.size:
        # Monotonicity-preserving shift: bring the minimum into [0, 360) with a
        # single constant offset (e.g. [-170..170] -> [190..530]), so a
        # dateline-spanning 1-D grid keeps increasing cell edges.
        lon = lon - 360.0 * np.floor(lon.min(initial=0.0) / 360.0)
    return lon, lat


def _signed_area_lonlat(ring_lon: np.ndarray, ring_lat: np.ndarray) -> float:
    """Shoelace signed area in (lon*cos(mid_lat), lat) space; positive = CCW."""
    if ring_lon.size < 3:
        return 0.0
    xs = ring_lon * np.cos(np.radians(ring_lat.mean()))
    ys = ring_lat
    return float(0.5 * np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1]) + 0.5 * (xs[-1] * ys[0] - xs[0] * ys[-1]))


def _enforce_ccw(node_coords: np.ndarray, conn_offsets: np.ndarray, conn_indices: np.ndarray) -> np.ndarray:
    """Reverse clockwise rings in place; returns the (possibly reordered) indices."""
    out = conn_indices.copy()
    for c in range(conn_offsets.size - 1):
        lo, hi = int(conn_offsets[c]), int(conn_offsets[c + 1])
        if hi - lo < 3:
            continue
        ring = out[lo:hi]
        pts = node_coords[ring]
        if _signed_area_lonlat(pts[:, 0], pts[:, 1]) < 0.0:
            out[lo:hi] = ring[::-1]
    return out


def _fan_triangulate(conn_offsets: np.ndarray, conn_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fan-triangulate polygon rings: cell (v0..vn-1) -> triangles (v0, vi, vi+1).

    Returns ``(tri_offsets, tri_indices, cell_of_tri)`` where ``cell_of_tri``
    maps each produced triangle back to its source polygon index (so a
    per-polygon ``cell_mask`` can be broadcast to per-triangle rows).
    """
    new_idx: list[int] = []
    new_off = [0]
    cell_of_tri: list[int] = []
    for c in range(conn_offsets.size - 1):
        ring = conn_indices[int(conn_offsets[c]) : int(conn_offsets[c + 1])]
        for i in range(1, len(ring) - 1):
            new_idx.extend((int(ring[0]), int(ring[i]), int(ring[i + 1])))
            new_off.append(len(new_idx))
            cell_of_tri.append(c)
    return (
        np.asarray(new_off, dtype=np.int64),
        np.asarray(new_idx, dtype=np.int64),
        np.asarray(cell_of_tri, dtype=np.int64),
    )


def _proj_string_from_attrs(attrs: dict[str, Any]) -> str | None:
    """Build a PROJ string from CF grid_mapping attributes (T031).

    Returns None for mapping names the engine route does not handle, letting
    callers fall through to plain coordinate detection.
    """
    name = str(attrs.get("grid_mapping_name", "")).lower()
    if name == "lambert_conformal_conic":
        sp = np.atleast_1d(attrs.get("standard_parallel", attrs.get("latitude_of_projection_origin", 0.0)))
        lat_0 = attrs.get("latitude_of_projection_origin", float(sp[0]))
        lon_0 = attrs.get("longitude_of_central_meridian", attrs.get("standard_longitude", 0.0))
        x_0 = attrs.get("false_easting", 0.0)
        y_0 = attrs.get("false_northing", 0.0)
        return f"+proj=lcc +lat_1={float(sp[0])} +lat_2={float(sp[-1])} +lat_0={float(lat_0)} +lon_0={float(lon_0)} +x_0={float(x_0)} +y_0={float(y_0)} +datum=WGS84 +units=m +no_defs"
    if name == "polar_stereographic":
        lat_0 = attrs.get("latitude_of_projection_origin", 90.0)
        lon_0 = attrs.get("longitude_of_central_meridian", attrs.get("straight_vertical_longitude_from_pole", 0.0))
        lat_ts = attrs.get("standard_parallel", attrs.get("latitude_of_true_scale", lat_0))
        k = attrs.get("scale_factor_at_projection_origin", 1.0)
        x_0 = attrs.get("false_easting", 0.0)
        y_0 = attrs.get("false_northing", 0.0)
        return f"+proj=stere +lat_0={float(lat_0)} +lon_0={float(lon_0)} +lat_ts={float(np.atleast_1d(lat_ts)[0])} +k={float(k)} +x_0={float(x_0)} +y_0={float(y_0)} +datum=WGS84 +units=m +no_defs"
    return None


def _cf_find_projection_axis(obj: xr.Dataset, axis: str) -> xr.DataArray | None:
    """Locate the projection x (or y) coordinate: standard_name, axis attr, then name."""
    std = "projection_x_coordinate" if axis == "x" else "projection_y_coordinate"
    for v in obj.variables:
        da = obj[v]
        if da.attrs.get("standard_name") == std or str(da.attrs.get("axis", "")).upper() == axis.upper():
            return da
    try:
        import cf_xarray  # noqa: F401

        if axis.upper() in obj.cf:
            return cast(xr.DataArray, obj.cf[axis.upper()])
    except Exception:
        pass
    return obj[axis] if axis in obj.variables else None


def _spherical_midpoints(
    lon_a: np.ndarray, lat_a: np.ndarray, lon_b: np.ndarray, lat_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Great-circle midpoints of (lon_a,lat_a)-(lon_b,lat_b) pairs, degrees."""
    xyz_a = _lonlat_to_xyz(lon_a, lat_a)
    xyz_b = _lonlat_to_xyz(lon_b, lat_b)
    mid = xyz_a + xyz_b
    norm = np.linalg.norm(mid, axis=1)
    valid = norm > 1e-12  # antipodal pairs: degenerate, keep first point
    mid[valid] /= norm[valid, None]
    lon = np.degrees(np.arctan2(mid[:, 1], mid[:, 0])) % 360.0
    lat = np.degrees(np.arcsin(np.clip(mid[:, 2] / np.where(valid, norm, 1.0), -1.0, 1.0)))
    lat = np.where(valid, lat, lat_a)
    return lon, lat


def _lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    r = np.radians(lon_deg)
    p = np.radians(lat_deg)
    return np.column_stack([np.cos(p) * np.cos(r), np.cos(p) * np.sin(r), np.sin(p)])


def _interval_bounds_to_vertices(lon_b: np.ndarray, lat_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert CF interval bounds (…, nv=2) to polygon corner vertices (…, nv=4) (AC2).

    A 2-value bound is the [min, max] interval of one coordinate across the
    cell's four corners. The canonical CCW quad is assembled from the lon/lat
    interval corners (SW, SE, NE, NW); exact corner arrays (nv >= 3) pass
    through untouched.
    """
    lb = np.asarray(lat_b, dtype=np.float64)
    ob = np.asarray(lon_b, dtype=np.float64)
    if lb.shape[-1] == 2 and ob.shape[-1] == 2:
        lo0, lo1 = np.minimum(ob[..., 0], ob[..., 1]), np.maximum(ob[..., 0], ob[..., 1])
        la0, la1 = np.minimum(lb[..., 0], lb[..., 1]), np.maximum(lb[..., 0], lb[..., 1])
        return np.stack([lo0, lo1, lo1, lo0], axis=-1), np.stack([la0, la0, la1, la1], axis=-1)
    return ob, lb


def _curvilinear_bounds_mesh(lon_b: np.ndarray, lat_b: np.ndarray, cell_mask: np.ndarray | None = None) -> _core.Mesh:
    """Build a polygon mesh from (ny, nx, nv) explicit CF bounds arrays.

    Drops repeated closing/padded vertices per cell (port of the verified
    ``_try_bounds_curvilinear_mesh`` array logic, decoupled from xarray).
    """
    lb = np.asarray(lat_b, dtype=np.float64)
    ob = np.asarray(lon_b, dtype=np.float64)
    if lb.ndim != 3 or ob.ndim != 3 or lb.shape != ob.shape:
        raise GridError(f"bounds must be 3-D (ny, nx, nv) matching coord shape {lb.shape[:2]}, got lon {ob.shape} / lat {lb.shape}")
    nv = lb.shape[2]
    node_lons: list[float] = []
    node_lats: list[float] = []
    offsets = [0]
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
            raise GridError(f"cell {c} has fewer than 3 unique vertices in its bounds — malformed bounds array")
        offsets.append(len(node_lons))
    node_coords = np.asfortranarray(np.column_stack([np.asarray(node_lons), np.asarray(node_lats)]))
    conn_offsets = np.asarray(offsets, dtype=np.int64)
    conn_indices = np.arange(len(node_lons), dtype=np.int64)
    conn_indices = _enforce_ccw(node_coords, conn_offsets, conn_indices)
    return _core.make_ugrid_mesh(node_coords, conn_offsets, conn_indices, _as_int_mask(cell_mask))


def _quad_mesh_from_corners(clon: np.ndarray, clat: np.ndarray, cell_mask: np.ndarray | None = None) -> _core.Mesh:
    """Build a CCW quad mesh from (ny+1, nx+1) corner arrays."""
    ny1, nx1 = clon.shape
    node_coords = np.asfortranarray(np.column_stack([clon.ravel(), clat.ravel()]))
    ii, jj = np.meshgrid(np.arange(nx1 - 1, dtype=np.int64), np.arange(ny1 - 1, dtype=np.int64), indexing="xy")
    bl = ii + jj * nx1
    conn = np.stack([bl, bl + 1, bl + nx1 + 1, bl + nx1], axis=-1).ravel().astype(np.int64)
    offsets = np.arange(0, conn.size + 1, 4, dtype=np.int64)
    return _core.make_ugrid_mesh(node_coords, offsets, conn, _as_int_mask(cell_mask))


def _cubed_sphere_tile_mesh(
    lon: np.ndarray, lat: np.ndarray, supergrid: tuple[np.ndarray, np.ndarray] | None, cell_mask: np.ndarray | None = None
) -> _core.Mesh:
    """Assemble 6-tile (ntiles, ny, nx) center arrays (or supergrid corners) into one global mesh."""
    ntiles, ny, nx = lon.shape
    node_coords_list: list[np.ndarray] = []
    conn_indices_list: list[np.ndarray] = []
    node_offset = 0
    for t in range(ntiles):
        if supergrid is not None:
            sx, sy = supergrid
            clon_t = sx[0::2, 0::2]
            clat_t = sy[0::2, 0::2]
        else:
            clon_t, clat_t = _synthesize_curvilinear_corners(lon[t], lat[t])
        coords_t = np.column_stack([clon_t.ravel(), clat_t.ravel()])
        node_coords_list.append(coords_t)
        nip1 = nx + 1
        i_grid, j_grid = np.meshgrid(np.arange(nx), np.arange(ny))
        bl = (i_grid + j_grid * nip1 + node_offset).ravel()
        br = ((i_grid + 1) + j_grid * nip1 + node_offset).ravel()
        tr = ((i_grid + 1) + (j_grid + 1) * nip1 + node_offset).ravel()
        tl = (i_grid + (j_grid + 1) * nip1 + node_offset).ravel()
        conn_indices_list.append(np.column_stack([bl, br, tr, tl]).astype(np.int64).ravel())
        node_offset += len(coords_t)
    node_coords = np.asfortranarray(np.concatenate(node_coords_list))
    conn_indices = np.concatenate(conn_indices_list)
    conn_offsets = np.arange(0, conn_indices.size + 1, 4, dtype=np.int64)
    return _core.make_ugrid_mesh(node_coords, conn_offsets, conn_indices, _as_int_mask(cell_mask))


def _as_int_mask(mask: np.ndarray | None) -> np.ndarray | None:
    """Flatten a per-cell wet/dry mask to a C-contiguous int32 1-D array (or None)."""
    if mask is None:
        return None
    return np.ascontiguousarray(np.asarray(mask).ravel(), dtype=np.int32)


def _cf_find(obj: xr.Dataset, name: str) -> xr.DataArray | None:
    """CF-standard coord lookup: cf-xarray first, attrs, then name heuristics.

    Returns None when absent. Records nothing — callers track searched patterns.
    """
    try:
        import cf_xarray  # noqa: F401  (runtime dependency)

        if name in obj.cf:
            return cast(xr.DataArray, obj.cf[name])
    except Exception:
        pass
    found = _find_coord(obj, name)  # standard_name/axis attrs (legacy helper)
    if found is not None:
        return found
    aliases = {
        "latitude": ("lat", "latCell", "lat_face", "lat_node", "latitude"),
        "longitude": ("lon", "lonCell", "lon_face", "lon_node", "longitude"),
    }
    for v in aliases.get(name, ()):
        if v in obj.coords or v in obj.variables:
            return obj[v]
    return None


class Grid:
    """A frozen, reusable, immutable description of one spatial layout.

    Constructed once, shared across any number of regridders and datasets
    (FR-001, FR-007). Engine meshes are built lazily per method-variant and
    cached; detection/normalization runs exactly once at construction.

    Example:
        >>> g = Grid(ds)                          # auto-detect family
        >>> g = Grid("C96")                       # named cubed-sphere, no file
        >>> g = Grid(lon=lon_vec, lat=lat_vec)    # explicit rectilinear

    Note:
        ICON grids are constructed from their vertex coordinates and triangle
        face-connectivity via :meth:`Grid.from_ugrid` (the ``location="cell"``
        polygon route). *Named* ICON tokens are out of scope for v1 (R11):
        the engine's ``N<num>`` family is the ECMWF **reduced Gaussian** grid,
        not the ICON diamond grid, so ``Grid("N320")`` produces a Gaussian
        mesh — diamond-grid construction from a name alone is not provided.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        # Set-once: each slot may be written during construction only; any
        # later mutation raises (FR-007 — construct once, reuse everywhere).
        try:
            object.__getattribute__(self, name)
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f"Grid is immutable (construct once, reuse everywhere) — cannot set {name!r}")

    __slots__ = (
        "_family",
        "_dims",
        "_shape",
        "_n_cells",
        "_periodic",
        "_line_type",
        "_tripolar",
        "_name",
        "_payload",
        "_meshes",
        "_fingerprint",
    )

    # Slot declarations for static typing (assigned once during construction).
    _family: GridFamily
    _dims: tuple[str, ...]
    _shape: tuple[int, ...]
    _n_cells: int
    _periodic: bool
    _line_type: str
    _tripolar: bool | None
    _name: str | None
    _payload: dict[str, Any]
    _meshes: dict[str, _core.Mesh]
    _fingerprint: str

    def __init__(
        self,
        obj: xr.Dataset | xr.DataArray | dict[str, Any] | str | None = None,
        *,
        lon: np.ndarray | xr.DataArray | None = None,
        lat: np.ndarray | xr.DataArray | None = None,
        bounds: tuple[Any, ...] | None = None,
        mesh: xr.DataArray | str | None = None,
        location: Literal["auto", "cell", "node", "edge"] = "auto",
        periodic: bool | None = None,
        line_type: Literal["great_circle", "cartesian"] | None = None,
        tripolar: bool | None = None,
    ) -> None:
        if line_type not in (None, "great_circle", "cartesian"):
            raise GridError(f"line_type must be 'great_circle' or 'cartesian', got {line_type!r}")
        if isinstance(obj, str):
            if lon is not None or lat is not None or mesh is not None:
                raise GridError("named-grid construction (obj=<str>) takes no lon/lat/mesh overrides")
            self._init_named(obj)
        elif lon is not None or lat is not None:
            if lon is None or lat is None:
                raise GridError("both lon= and lat= are required for the explicit-arrays route")
            self._init_arrays(np.asarray(lon), np.asarray(lat), bounds=bounds)
        elif mesh is not None:
            self._init_mesh(obj if isinstance(obj, (xr.Dataset, xr.DataArray)) else None, mesh, location)
        elif isinstance(obj, dict):
            self._init_dict(obj)
        elif isinstance(obj, (xr.Dataset, xr.DataArray)):
            self._init_xarray(obj, location=location)
        else:
            raise GridError(
                "Grid needs a Dataset/DataArray, a dict of coord arrays, a named-grid string "
                "('C96', 'O96', 'F64', 'N320', 'R80', 'grid128'), or explicit lon=/lat= arrays; "
                f"got {type(obj).__name__}"
            )
        if line_type is not None:
            object.__setattr__(self, "_line_type", str(line_type))
        elif not hasattr(self, "_line_type"):
            object.__setattr__(self, "_line_type", "great_circle")
        object.__setattr__(self, "_tripolar", tripolar)
        if periodic is not None:
            object.__setattr__(self, "_periodic", bool(periodic))
        if not hasattr(self, "_meshes"):
            object.__setattr__(self, "_meshes", {})

    # ─── alternate constructors (thin wrappers) ─────────────────────────────

    @classmethod
    def from_xarray(
        cls,
        obj: xr.Dataset | xr.DataArray,
        *,
        lon: np.ndarray | xr.DataArray | None = None,
        lat: np.ndarray | xr.DataArray | None = None,
        bounds: tuple[Any, ...] | None = None,
        mesh: xr.DataArray | str | None = None,
        location: Literal["auto", "cell", "node", "edge"] = "auto",
        **geometry: Any,
    ) -> Grid:
        """Detect (or override) a grid from an xarray Dataset/DataArray."""
        return cls(obj, lon=lon, lat=lat, bounds=bounds, mesh=mesh, location=location, **geometry)

    @classmethod
    def from_arrays(
        cls,
        lon: np.ndarray,
        lat: np.ndarray,
        *,
        bounds: tuple[Any, ...] | None = None,
        **geometry: Any,
    ) -> Grid:
        """1-D+1-D → rectilinear; 2-D → curvilinear; 3-D (6,ny,nx) → cubed-sphere."""
        return cls(lon=np.asarray(lon), lat=np.asarray(lat), bounds=bounds, **geometry)

    @classmethod
    def from_named(cls, name: str, **geometry: Any) -> Grid:
        """A registry grid ('C96', 'O96', 'F64', 'N320', 'R80', 'grid128') — no file I/O."""
        return cls(name, **geometry)

    @classmethod
    def from_ugrid(
        cls,
        node_coords: np.ndarray,
        conn_offsets: np.ndarray,
        conn_indices: np.ndarray,
        *,
        location: str = "cell",
        cell_mask: np.ndarray | None = None,
        dims: tuple[str, ...] | str | None = None,
        **geometry: Any,
    ) -> Grid:
        """Raw CSR unstructured mesh arrays (UGRID/ICON polygons, 0-based indices).

        ICON is the triangle special case: all rings of three vertices are
        detected as ``GridFamily.ICON``; named ICON tokens are out of scope
        for v1 (R11) — see :class:`Grid`.
        """
        coords = np.asfortranarray(np.asarray(node_coords, dtype=np.float64))
        offs = np.asarray(conn_offsets, dtype=np.int64)
        inds = np.asarray(conn_indices, dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise GridError(f"node_coords must be (n_nodes, 2) lon/lat, got shape {coords.shape}")
        if offs.size == 0 or inds.size == 0 or offs[0] != 0 or offs[-1] != inds.size:
            raise GridError("conn_offsets/conn_indices are not a valid CSR pair (offsets[0]==0, offsets[-1]==len(indices))")
        if inds.size and (inds.min() < 0 or inds.max() >= coords.shape[0]):
            raise GridError(f"connectivity indices outside [0, {coords.shape[0]}) — check start_index normalization")
        lon, lat = _to_degrees(coords[:, 0].copy(), coords[:, 1].copy())
        coords = np.asfortranarray(np.column_stack([lon, lat]))
        inds = _enforce_ccw(coords, offs, inds)
        n_cells = int(offs.size - 1)
        nv = np.diff(offs)
        loc = "node" if location == "node" else "cell"
        if loc == "node":
            n_cells = int(coords.shape[0])
        elif not np.all((nv >= 3) | (nv == 0) | (nv == 1)):
            raise GridError(
                f"every cell needs >= 3 vertices, or 0 (dry) / 1 (point cloud); got ring sizes {sorted(set(nv.tolist()))}"
            )
        family = GridFamily.POINTS if np.all(nv == 1) else (GridFamily.ICON if np.all((nv == 3) | (nv == 0)) else GridFamily.UGRID)
        g = object.__new__(Grid)
        g._family = family
        if dims is not None:
            g._dims = tuple(str(d) for d in (dims if isinstance(dims, (tuple, list)) else (dims,)))
        else:
            g._dims = ("nnode",) if loc == "node" else ("ncol",)
        g._shape = (n_cells,)
        g._n_cells = n_cells
        g._periodic = False
        g._line_type = "great_circle"
        g._tripolar = None
        g._name = None
        g._payload = {
            "kind": "csr",
            "node_coords": coords,
            "conn_offsets": offs,
            "conn_indices": inds,
            "location": loc,
            "cell_mask": cell_mask,
        }
        g._meshes = {}
        g._fingerprint = _fingerprint(family.value, g._shape, n_cells, coords)
        return g

    @classmethod
    def from_points(cls, lon: np.ndarray, lat: np.ndarray, **geometry: Any) -> Grid:
        """A sparse point cloud (flight tracks, obs sites): degenerate zero-area cells."""
        return cls.from_ugrid(
            np.column_stack([np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)]),
            np.arange(0, int(np.asarray(lon).size) + 1, 1, dtype=np.int64),
            np.arange(int(np.asarray(lon).size), dtype=np.int64),
            **geometry,
        )

    @classmethod
    def from_dict(cls, mapping: dict[str, Any], **geometry: Any) -> Grid:
        """A mapping with 'lon'/'lat' (and optional 'bounds') arrays."""
        lon = mapping.get("lon", mapping.get("lons"))
        lat = mapping.get("lat", mapping.get("lats"))
        if lon is None or lat is None:
            raise GridError(f"dict needs 'lon' and 'lat' keys, got {sorted(mapping)}")
        return cls.from_arrays(lon, lat, bounds=mapping.get("bounds"), **geometry)

    # ─── route implementations ──────────────────────────────────────────────

    def _init_named(self, name: str) -> None:
        token = str(name).strip()
        fam_char = token[0].upper() if token[:1].isalpha() else "G"
        if fam_char not in _NAMED_FAMILIES:
            raise GridError(
                f"unknown named-grid {name!r}: expected a family letter "
                "(C=cubed-sphere, O/N/F/R=Gaussian, G=NOAA GRIB) followed by a positive number"
            )
        try:
            probe = _core.make_named_mesh(token)
            layout: dict[str, Any] = dict(_core.named_grid_layout(token))
            centers = np.asarray(_core.named_grid_cell_centers(token), dtype=np.float64).reshape(-1, 2)
        except Exception as exc:
            raise GridError(f"named grid {name!r} is not registered in the engine: {exc}") from None
        n = int(probe.n_cells)
        self._family = _NAMED_FAMILIES[fam_char]
        self._n_cells = n
        self._periodic = True  # global named grids wrap at the dateline
        self._name = token
        # Structured named grids (F/R/regular-GRIB/cubed-sphere) are delivered in
        # their natural row/column layout rather than a flat ncol list: the engine
        # emits cells in a deterministic rectangular order, so the flat axis can be
        # reshaped and labelled with (lat, lon) or (tile, j, i) coordinates.
        # Genuinely unstructured ones (O/N reduced Gaussian, projected GRIB) keep
        # ncol but still get lon/lat (or x/y) coordinates attached.
        ni = int(layout["ni"])
        nj = int(layout["nj"])
        n_tiles = int(layout["n_tiles"])
        projected = bool(layout["projected"])
        payload: dict[str, Any] = {"kind": "named", "mesh": probe, "centers": centers, "projected": projected}
        if projected:
            self._dims = ("ncol",)
            self._shape = (n,)
            self._line_type = "cartesian"
            payload["x"] = centers[:, 0]
            payload["y"] = centers[:, 1]
        elif n_tiles == 6:
            self._dims = ("tile", "j", "i")
            self._shape = (n_tiles, nj, ni)
            payload["lon"] = centers[:, 0].reshape(self._shape)
            payload["lat"] = centers[:, 1].reshape(self._shape)
        elif ni * nj == n:
            self._dims = ("lat", "lon")
            self._shape = (nj, ni)
            if bool(layout["row_uniform_lon"]):
                payload["lat"] = np.ascontiguousarray(centers[:, 1].reshape(nj, ni)[:, 0])
                payload["lon"] = np.ascontiguousarray(centers[:, 0].reshape(nj, ni)[0, :])
            else:
                payload["lat"] = centers[:, 1].reshape(nj, ni)
                payload["lon"] = centers[:, 0].reshape(nj, ni)
        else:
            self._dims = ("ncol",)
            self._shape = (n,)
            payload["lon"] = centers[:, 0]
            payload["lat"] = centers[:, 1]
        self._payload = payload
        token_digest = hashlib.sha256(token.encode()).digest()
        self._fingerprint = _fingerprint(
            self._family.value, self._shape, n, np.frombuffer(token_digest[:8], dtype=np.uint64).astype(np.float64)
        )

    def _init_arrays(self, lon: np.ndarray, lat: np.ndarray, bounds: tuple[Any, ...] | None = None) -> None:
        if lon.ndim != lat.ndim:
            raise GridError(f"lon and lat must have matching dimensions, got lon.ndim={lon.ndim}, lat.ndim={lat.ndim}")
        self._name = None
        # 1-D rectilinear center vectors must stay monotonic: a per-value
        # [0,360) wrap would scramble e.g. [-170..170] into [190,303,57,170],
        # corrupting the mesh and dateline-spanning rotation angles.
        lon, lat = _to_degrees(lon, lat, wrap_lon=lon.ndim != 1)
        if lon.ndim == 1:
            self._family = GridFamily.RECTILINEAR
            self._dims = ("lat", "lon")
            self._shape = (lat.size, lon.size)
            self._n_cells = lat.size * lon.size
            span = float(lon.max(initial=0.0) - lon.min(initial=0.0))
            self._periodic = span >= 350.0
            self._payload = {"kind": "rect", "lon": lon, "lat": lat}
            self._fingerprint = _fingerprint("rectilinear", self._shape, self._n_cells, lon, lat)
        elif lon.ndim == 2:
            self._family = GridFamily.CURVILINEAR
            self._dims = ("y", "x")
            self._shape = lon.shape
            self._n_cells = int(lon.size)
            self._periodic = bool(lon[:, -1:].max(initial=0) - lon[:, :1].min(initial=0) >= 350.0)
            payload: dict[str, Any] = {"kind": "curv", "lon": lon, "lat": lat}
            if bounds is not None:
                lon_b, lat_b = (np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64))
                lon_b, lat_b = _interval_bounds_to_vertices(lon_b, lat_b)
                payload["bounds"] = (np.mod(lon_b, 360.0), lat_b)
            self._payload = payload
            fp_arrays: list[np.ndarray] = [lon, lat] + ([payload["bounds"][0], payload["bounds"][1]] if "bounds" in payload else [])
            self._fingerprint = _fingerprint("curvilinear", self._shape, self._n_cells, *fp_arrays)
        elif lon.ndim == 3:
            if lon.shape[0] != 6:
                raise GridError(f"3-D lon/lat must be (6, ny, nx) cubed-sphere tiles, got shape {lon.shape}")
            self._family = GridFamily.CUBED_SPHERE
            self._dims = ("tile", "j", "i")
            self._shape = lon.shape
            self._n_cells = int(lon.size)
            self._periodic = True
            payload = {"kind": "cs", "lon": lon, "lat": lat}
            if bounds is not None:
                payload["supergrid"] = (np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64))
            self._payload = payload
            self._fingerprint = _fingerprint("cubed_sphere", self._shape, self._n_cells, lon, lat)
        else:
            raise GridError(
                f"lon/lat must be 1-D, 2-D or 3-D (6,ny,nx); got ndim={lon.ndim} — time-varying coordinates are not supported"
            )

    def _init_mesh(self, obj: xr.Dataset | xr.DataArray | None, mesh: xr.DataArray | str, location: str) -> None:
        if location not in ("auto", "cell", "node", "edge"):
            raise GridError(f"location must be one of 'auto'/'cell'/'node'/'edge', got {location!r}")
        ds = None
        if obj is not None:
            ds = obj if isinstance(obj, xr.Dataset) else obj.to_dataset(name="_axis_tmp")
        mesh_name = str(mesh.name) if isinstance(mesh, xr.DataArray) else str(mesh)
        if ds is None or mesh_name not in ds:
            raise GridError(f"mesh={mesh_name!r} not found in the provided dataset")
        mvar = ds[mesh_name]
        attrs = mvar.attrs
        if attrs.get("cf_role") != "mesh_topology":
            raise GridError(f"variable {mesh_name!r} lacks cf_role='mesh_topology' — not a UGRID mesh")
        if location == "edge":
            grid = self._from_ugrid_edge(ds, mesh_name)
            for attr in Grid.__slots__:
                object.__setattr__(self, attr, getattr(grid, attr))
            return
        node_names = str(attrs.get("node_coordinates", "")).split()
        if len(node_names) != 2:
            raise GridError(
                f"mesh {mesh_name!r} has no parseable node_coordinates attribute (got {attrs.get('node_coordinates')!r})"
            )
        conn_name = str(attrs.get("face_node_connectivity", ""))
        if not conn_name or conn_name not in ds:
            raise GridError(f"mesh {mesh_name!r} declares face_node_connectivity={conn_name!r}, which is not in the dataset")
        conn = ds[conn_name].values
        start_index = int(ds[conn_name].attrs.get("start_index", 0))
        if start_index not in (0, 1):
            raise GridError(f"unsupported start_index={start_index} on {conn_name!r} (0 or 1 only)")
        if conn.ndim == 2:
            # padded rectangle: build variable-length CSR from fill values
            fill = ds[conn_name].encoding.get("_FillValue", ds[conn_name].attrs.get("_FillValue", -1))
            counts = np.sum(conn != fill, axis=1) if fill is not None else np.full(conn.shape[0], conn.shape[1])
            offs = np.zeros(conn.shape[0] + 1, dtype=np.int64)
            offs[1:] = np.cumsum(counts)
            mask = np.arange(conn.shape[1])[None, :] < counts[:, None]
            inds = conn[mask]
        else:
            inds = conn
            nv_per = 3
            offs = np.arange(0, inds.size + nv_per, nv_per, dtype=np.int64)
        inds = np.asarray(inds, dtype=np.int64) - start_index
        node_lon = ds[node_names[0]].values.astype(np.float64)
        node_lat = ds[node_names[1]].values.astype(np.float64)
        # Keep the dataset's own element-dim name (nCells, ...) so DataArrays
        # built on those dims validate against this grid.
        elem_dim = str(ds[conn_name].dims[0]) if location != "node" else str(ds[node_names[0]].dims[0])
        grid = Grid.from_ugrid(np.column_stack([node_lon, node_lat]), offs, inds, location=location, dims=(elem_dim,))
        for attr in Grid.__slots__:
            object.__setattr__(self, attr, getattr(grid, attr))

    def _init_dict(self, mapping: dict[str, Any]) -> None:
        lon = mapping.get("lon", mapping.get("lons"))
        lat = mapping.get("lat", mapping.get("lats"))
        if lon is None or lat is None:
            raise GridError(f"dict needs 'lon' and 'lat' keys, got {sorted(mapping)}")
        self._init_arrays(np.asarray(lon), np.asarray(lat), bounds=mapping.get("bounds"))

    def _init_xarray(self, obj: xr.Dataset | xr.DataArray, *, location: str = "auto") -> None:
        ds = obj.to_dataset(name="_axis_tmp") if isinstance(obj, xr.DataArray) else obj
        if location == "auto":
            location = "cell"
        searched = [
            "cf-xarray: ds.cf['longitude'] / ds.cf['latitude'] (standard_name/axis/units)",
            "attrs: standard_name='longitude'/'latitude', axis='X'/'Y'",
            "names: lon/lat/longitude/latitude/lonCell/latCell/lon_face/lat_face/lon_node/lat_node",
            "UGRID: cf_role='mesh_topology' + node_coordinates + face_node_connectivity",
            "MPAS: latVertex/lonVertex + verticesOnCell (+nEdgesOnCell)",
            "SCRIP: lat_bnds/lon_bnds bounds variables",
        ]
        # UGRID mesh_topology first — it is unambiguous when present
        mesh_var: str | None = None
        for var in ds.variables:
            if ds[var].attrs.get("cf_role") == "mesh_topology":
                mesh_var = str(var)
                break
        if mesh_var is not None:
            try:
                if location == "edge":
                    g = self._from_ugrid_edge(ds, mesh_var)
                else:
                    g = Grid(mesh=mesh_var, obj=ds, location=cast(Literal["auto", "cell", "node", "edge"], location))
            except GridError as exc:
                raise GridError(f"UGRID mesh {mesh_var!r} present but unusable: {exc}", searched=searched) from None
            for attr in Grid.__slots__:
                object.__setattr__(self, attr, getattr(g, attr))
            return
        # MPAS-style vertex connectivity
        if "verticesOnCell" in ds and ("latVertex" in ds or "latitude" in {str(v).lower() for v in ds.variables}):
            try:
                g = self._from_mpas(ds, location=location)
            except GridError as exc:
                raise GridError(str(exc), searched=searched) from None
            for attr in Grid.__slots__:
                object.__setattr__(self, attr, getattr(g, attr))
            return
        # Projected grids (Lambert conformal, polar stereographic): a
        # grid_mapping variable with recognized parameters AND real projection
        # x/y coordinate variables routes through the engine's PROJ inverse-
        # projection builder (T031), with cartesian line semantics. A bare
        # grid_mapping attr over geographic lon/lat degrees is not projected.
        mapping_var: str | None = next((str(v) for v in ds.variables if "grid_mapping_name" in ds[v].attrs), None)
        if (
            mapping_var is not None
            and _proj_string_from_attrs(ds[mapping_var].attrs) is not None
            and _cf_find_projection_axis(ds, "x") is not None
            and _cf_find_projection_axis(ds, "y") is not None
        ):
            self._init_projected(ds, mapping_var)
            return
        lat = _cf_find(ds, "latitude")
        lon = _cf_find(ds, "longitude")
        if lat is None or lon is None:
            missing = [] if lat is not None else ["latitude"]
            if lon is None:
                missing.append("longitude")
            raise GridError(f"could not detect {' and '.join(missing)} coordinate(s) in the dataset", searched=searched)
        # Drop non-spatial (time/level/member) slices from the coords
        non_spatial = _get_non_spatial_dims(ds)
        lat = lat.isel(dict.fromkeys(non_spatial_dims_intersection(lat, non_spatial), 0), drop=True) if non_spatial else lat
        lon = lon.isel(dict.fromkeys(non_spatial_dims_intersection(lon, non_spatial), 0), drop=True) if non_spatial else lon
        if lat.ndim > 3 or lon.ndim > 3:
            raise GridError(f"spatial coordinates have {max(lat.ndim, lon.ndim)} dims — time-varying coordinates are not supported")
        bounds = None
        lb_name = lat.attrs.get("bounds")
        ob_name = lon.attrs.get("bounds")
        if lb_name and ob_name and lb_name in ds and ob_name in ds:
            lb = np.asarray(ds[lb_name].values, dtype=np.float64)
            ob = np.asarray(ds[ob_name].values, dtype=np.float64)
            if lb.ndim == 3 and ob.ndim == 3 and lb.shape == ob.shape:
                bounds = (ob, lb)
        try:
            self._init_arrays(np.asarray(lon.values), np.asarray(lat.values), bounds=bounds)
        except GridError as exc:
            raise GridError(str(exc), searched=searched) from None
        # Prefer the dataset's own dimension names over the generic defaults.
        if lat.ndim == 1 and lon.ndim == 1:
            object.__setattr__(self, "_dims", (str(lat.dims[0]), str(lon.dims[0])))
        else:
            object.__setattr__(self, "_dims", tuple(str(d) for d in lat.dims))

    def _init_projected(self, ds: xr.Dataset, mapping_var: str) -> None:
        """Projected (x/y-in-meters) grid via the engine's PROJ builder (T031)."""
        attrs = ds[mapping_var].attrs
        proj_string = _proj_string_from_attrs(attrs)
        if not getattr(_core, "HAVE_PROJ", False):
            raise AxisCapabilityError(
                f"this dataset uses a {attrs.get('grid_mapping_name')!r} grid_mapping, which requires "
                "projected-coordinate support, but the engine was built without PROJ. "
                "Rebuild AXIS with -DAXIS_ENABLE_PROJ=ON to regrid projected grids."
            )
        x = _cf_find_projection_axis(ds, "x")
        y = _cf_find_projection_axis(ds, "y")
        if x is None or y is None:
            raise GridError(
                f"grid_mapping {mapping_var!r} found but projection x/y coordinate variables are missing "
                "(looked for standard_name='projection_x_coordinate'/'projection_y_coordinate', axis='X'/'Y', "
                "and names x/y)"
            )
        x = _drop_non_spatial(x, _get_non_spatial_dims(ds))
        y = _drop_non_spatial(y, _get_non_spatial_dims(ds))
        if x.ndim == 1 and y.ndim == 1:
            gx, gy = np.meshgrid(np.asarray(x.values, dtype=np.float64), np.asarray(y.values, dtype=np.float64), indexing="xy")
        elif x.ndim == 2 and y.ndim == 2:
            gx, gy = np.asarray(x.values, dtype=np.float64), np.asarray(y.values, dtype=np.float64)
        else:
            raise GridError(f"projected x/y coordinates must both be 1-D or both 2-D, got x.ndim={x.ndim}, y.ndim={y.ndim}")
        nj, ni = gy.shape
        self._family = GridFamily.RECTILINEAR
        self._dims = (str(y.dims[-2 if y.ndim == 2 else 0]), str(x.dims[-1]))
        self._shape = (nj, ni)
        self._n_cells = ni * nj
        self._periodic = False
        self._name = None
        self._line_type = "cartesian"  # projected cells live in the plane
        self._payload = {
            "kind": "proj",
            "proj_string": proj_string,
            "ni": ni,
            "nj": nj,
            "center_x": np.ascontiguousarray(gx.ravel(), dtype=np.float64),
            "center_y": np.ascontiguousarray(gy.ravel(), dtype=np.float64),
        }
        if x.ndim == 1 and y.ndim == 1:
            self._payload["x"], self._payload["y"] = np.asarray(x.values, float), np.asarray(y.values, float)
        self._fingerprint = _fingerprint("projected", self._shape, self._n_cells, gx, gy)

    def _from_ugrid_edge(self, ds: xr.Dataset, mesh_var: str) -> Grid:
        """UGRID location='edge': edge midpoints as degenerate point-cells (AC4).

        Values on mesh edges are regrid as point samples at each edge's
        spherical midpoint; the edge dimension name is preserved.
        """
        attrs = ds[mesh_var].attrs
        edge_conn_name = str(attrs.get("edge_node_connectivity", ""))
        node_names = str(attrs.get("node_coordinates", "")).split()
        if not edge_conn_name or edge_conn_name not in ds or len(node_names) != 2:
            raise GridError(
                f"location='edge' needs the mesh's edge_node_connectivity role; mesh {mesh_var!r} "
                f"declares edge_node_connectivity={edge_conn_name!r}"
            )
        econn = np.asarray(ds[edge_conn_name].values, dtype=np.int64)
        start = int(ds[edge_conn_name].attrs.get("start_index", 0))
        econn = econn - start
        node_lon = np.asarray(ds[node_names[0]].values, dtype=np.float64)
        node_lat = np.asarray(ds[node_names[1]].values, dtype=np.float64)
        node_lon, node_lat = _to_degrees(node_lon, node_lat)
        a, b = econn[:, 0], econn[:, 1]
        mid_lon, mid_lat = _spherical_midpoints(node_lon[a], node_lat[a], node_lon[b], node_lat[b])
        return Grid.from_points(mid_lon, mid_lat, dims=(str(ds[edge_conn_name].dims[0]),))

    def _from_mpas(self, ds: xr.Dataset, *, location: str = "cell") -> Grid:
        non_spatial = _get_non_spatial_dims(ds)
        if location == "edge":
            if "lonEdge" not in ds or "latEdge" not in ds:
                raise GridError("MPAS location='edge' needs lonEdge/latEdge coordinate variables")
            elon = np.asarray(_drop_non_spatial(ds["lonEdge"], non_spatial).values, dtype=np.float64)
            elat = np.asarray(_drop_non_spatial(ds["latEdge"], non_spatial).values, dtype=np.float64)
            elon, elat = _to_degrees(elon, elat)
            return Grid.from_points(elon, elat, dims=(str(ds["lonEdge"].dims[0]),))
        v_lat = _drop_non_spatial(ds["latVertex"], non_spatial)
        v_lon = _drop_non_spatial(ds["lonVertex"], non_spatial)
        v_conn = _drop_non_spatial(ds["verticesOnCell"], non_spatial)
        node_lat = np.asarray(v_lat.values, dtype=np.float64)
        node_lon = np.asarray(v_lon.values, dtype=np.float64)
        conn_raw = np.asarray(v_conn.values, dtype=np.int64)
        if not np.any(np.abs(node_lat) > 2.0 * np.pi):
            node_lat = np.degrees(node_lat)
            node_lon = np.degrees(node_lon)
        node_lon = np.mod(node_lon, 360.0)
        n_edges = (
            np.asarray(ds["nEdgesOnCell"].values, dtype=np.int64)
            if "nEdgesOnCell" in ds
            else np.full(conn_raw.shape[0], conn_raw.shape[1])
        )
        offs = np.zeros(n_edges.size + 1, dtype=np.int64)
        offs[1:] = np.cumsum(n_edges)
        mask = np.arange(conn_raw.shape[1])[None, :] < n_edges[:, None]
        inds = conn_raw[mask] - 1  # MPAS is 1-based
        if inds.size and (inds.min() < 0 or inds.max() >= node_lat.size):
            raise GridError("MPAS verticesOnCell indices fall outside the vertex array")
        return Grid.from_ugrid(np.column_stack([node_lon, node_lat]), offs, inds, dims=(str(v_conn.dims[0]),))

    # ─── properties & identity ──────────────────────────────────────────────

    @property
    def family(self) -> GridFamily:
        """Detected grid family."""
        return self._family

    @property
    def dims(self) -> tuple[str, ...]:
        """Spatial dimension name(s) as they appear in source data."""
        return self._dims

    @property
    def shape(self) -> tuple[int, ...]:
        """Spatial shape (1-D grids: ``(n_cells,)``; rectilinear: ``(ny, nx)``)."""
        return self._shape

    @property
    def n_cells(self) -> int:
        """Total number of grid cells."""
        return self._n_cells

    @property
    def periodic(self) -> bool:
        """Whether longitude is treated as periodic."""
        return self._periodic

    @property
    def line_type(self) -> str:
        """Inter-cell geometry: ``great_circle`` or ``cartesian``."""
        return self._line_type

    @property
    def tripolar(self) -> bool | None:
        """Tripolar-grid detection (``None`` = not applicable / undetected)."""
        return self._tripolar

    @property
    def name(self) -> str | None:
        """Named-grid token (e.g. ``'C96'``) when built from the registry."""
        return self._name

    @property
    def fingerprint(self) -> str:
        """Stable 16-hex digest over family, shape, and coordinate bytes."""
        return self._fingerprint

    def __repr__(self) -> str:
        bits = [f"family={self._family.value}", f"shape={self._shape}", f"n_cells={self._n_cells}", f"periodic={self._periodic}"]
        if self._name:
            bits.insert(0, f"name={self._name!r}")
        return f"Grid({', '.join(bits)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Grid) and self._fingerprint == other._fingerprint

    def __hash__(self) -> int:
        return hash(self._fingerprint)

    def to_mesh(self, variant: str = "poly", cell_mask: np.ndarray | None = None) -> _core.Mesh:
        """Engine mesh for a method variant: 'poly' (raw polygons) or 'tri' (fan-triangulated).

        ``cell_mask`` (per-cell int/bool, nonzero = active) is fused into the mesh
        as the engine's wet/dry source mask (FR-012). A mesh built with a mask is
        kept in a separate cache entry (``"<variant>+mask"``) so the unmasked mesh
        stays reusable; the mask must be in source-flattened cell order (row-major
        over the grid's cell indexing).
        """
        if cell_mask is not None:
            key = f"{variant}+mask"
            cached = self._meshes.get(key)
            if cached is not None:
                return cached
        cached = self._meshes.get(variant)
        if cached is not None and cell_mask is None:
            return cached
        mesh: _core.Mesh
        kind = self._payload["kind"]
        if kind == "named":
            if cell_mask is not None:
                raise GridError("src_mask is not supported on named grids (the engine generates their mesh)")
            mesh = self._payload["mesh"]  # engine-generated: single canonical mesh
        elif kind == "rect":
            mesh = _make_regular_mesh_from_centers(self._payload["lon"], self._payload["lat"], cell_mask)
        elif kind == "proj":
            pay = self._payload
            mesh = _core.make_projected_mesh(pay["ni"], pay["nj"], pay["proj_string"], pay["center_x"], pay["center_y"])
        elif kind == "curv":
            if "bounds" in self._payload:
                lb, la = self._payload["bounds"]
                mesh = _curvilinear_bounds_mesh(lb, la, cell_mask)
            else:
                clon, clat = _synthesize_curvilinear_corners(self._payload["lon"], self._payload["lat"])
                mesh = _quad_mesh_from_corners(clon, clat, cell_mask)
        elif kind == "cs":
            mesh = _cubed_sphere_tile_mesh(self._payload["lon"], self._payload["lat"], self._payload.get("supergrid"), cell_mask)
        elif kind == "csr":
            coords = self._payload["node_coords"]
            offs = self._payload["conn_offsets"]
            inds = self._payload["conn_indices"]
            mask = cell_mask if cell_mask is not None else self._payload.get("cell_mask")
            if self._payload.get("location") == "node":
                n = coords.shape[0]
                mesh = _core.make_ugrid_mesh(
                    coords, np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.int64), _as_int_mask(mask)
                )
            elif variant == "tri":
                t_off, t_ind, cell_of_tri = _fan_triangulate(offs, inds)
                im = _as_int_mask(mask)
                tri_mask = im[cell_of_tri] if im is not None else None
                mesh = _core.make_ugrid_mesh(coords, t_off, t_ind, tri_mask)
            else:
                mesh = _core.make_ugrid_mesh(coords, offs, inds, _as_int_mask(mask))
        else:  # pragma: no cover
            raise GridError(f"internal error: unknown payload kind {kind!r}")
        self._meshes[key if cell_mask is not None else variant] = mesh
        return mesh


def non_spatial_dims_intersection(da: xr.DataArray, non_spatial: set[str]) -> set[str]:
    return {d for d in da.dims if d in non_spatial}


def _drop_non_spatial(da: xr.DataArray, non_spatial: set[str]) -> xr.DataArray:
    isel = {d: 0 for d in da.dims if str(d) in non_spatial}
    return da.isel(isel, drop=True) if isel else da


def _fingerprint(family: str, shape: tuple[int, ...], n_cells: int, *arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{family}|{shape}|{n_cells}|".encode())
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]
