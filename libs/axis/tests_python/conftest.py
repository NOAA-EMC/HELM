# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for AXIS Python API integration tests.

Ensures Kokkos is initialized before any test that imports _core.
Kokkos initialization is handled internally by the _core module
(via ensure_kokkos()), so this conftest primarily provides shared
fixtures for mesh construction used across multiple tests.

Grid-family fixtures (US1/US2) build the six supported families from
deterministic synthetic arrays where possible, and from on-demand cached
fetchers (C96 tiles, MPAS mesh) for real-world data. Fetchers live in
``libs/axis/benchmarks/`` and are imported lazily inside each fixture so
network failures skip (not error) dependent tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))


def _sine_bell(lon2d, lat2d):
    """Smooth analytic scalar field on the sphere (cos(lat) bell in lon/lat)."""
    lam = np.radians(lon2d)
    phi = np.radians(lat2d)
    return (np.cos(phi) ** 2) * (1.0 + 0.5 * np.cos(lam))


@pytest.fixture
def analytic_scalar_field():
    """Factory: build a DataArray of the smooth sine-bell field on a grid."""

    def _make(lons, lats, dims=None, extra_dims=None):
        lons = np.asarray(lons, dtype=np.float64)
        lats = np.asarray(lats, dtype=np.float64)
        if lons.ndim == 1:
            lon2d, lat2d = np.meshgrid(lons, lats)
        else:
            lon2d, lat2d = lons, lats
        vals = _sine_bell(lon2d, lat2d)
        if dims is None:
            dims = ("lat", "lon") if lons.ndim == 1 else ("y", "x")
        da = xr.DataArray(vals, dims=dims)
        if extra_dims:
            da = da.expand_dims(extra_dims)
        return da

    return _make


@pytest.fixture
def small_src_mesh():
    """Create a small 8x4 regular lat-lon source mesh."""
    from axis import _core

    ni, nj = 8, 4
    lon_start, lat_start = 0.0, -90.0
    dlon = 360.0 / ni
    dlat = 180.0 / nj
    return _core.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


@pytest.fixture
def small_dst_mesh():
    """Create a small 4x2 regular lat-lon destination mesh."""
    from axis import _core

    ni, nj = 4, 2
    lon_start, lat_start = 0.0, -90.0
    dlon = 360.0 / ni
    dlat = 180.0 / nj
    return _core.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


@pytest.fixture
def bilinear_matrix(small_src_mesh, small_dst_mesh):
    """Generate bilinear interpolation weights between src and dst meshes."""
    from axis import _core

    return _core.generate_weights(small_src_mesh, small_dst_mesh, _core.Method.Bilinear)


# ─── Grid-family fixtures (US1/US2 — six families) ──────────────────────


@pytest.fixture
def rectilinear_ds():
    """1-degree global lat-lon Dataset with CF-standard coords + bounds."""
    lons = np.arange(0.5, 360.0, 1.0)
    lats = np.linspace(-89.5, 89.5, 180)
    lon2d, lat2d = np.meshgrid(lons, lats)
    lon_bnds = np.stack([lons - 0.5, lons + 0.5], axis=-1)
    lat_bnds = np.stack([lats - 0.5, lats + 0.5], axis=-1)
    da_lon = xr.DataArray(
        lons,
        dims=("lon",),
        attrs={"standard_name": "longitude", "units": "degrees_east", "axis": "X", "bounds": "lon_bnds"},
    )
    da_lat = xr.DataArray(
        lats,
        dims=("lat",),
        attrs={"standard_name": "latitude", "units": "degrees_north", "axis": "Y", "bounds": "lat_bnds"},
    )
    air = xr.DataArray(_sine_bell(lon2d, lat2d), dims=("lat", "lon"), attrs={"units": "K"})
    return xr.Dataset(
        {
            "air": air,
            "lon_bnds": xr.DataArray(lon_bnds, dims=("lon", "bnds")),
            "lat_bnds": xr.DataArray(lat_bnds, dims=("lat", "bnds")),
        },
        coords={"lon": da_lon, "lat": da_lat},
    )


@pytest.fixture
def curvilinear_ds():
    """CMIP6-style curvilinear grid: 2-D lat/lon with explicit CF bounds."""
    ny, nx = 64, 64
    y, x = np.meshgrid(np.linspace(0, 1, ny), np.linspace(0, 1, nx), indexing="ij")
    lon = 180 + 60 * (x - 0.5) + 5 * np.sin(2 * np.pi * y)
    lat = 40 + 20 * (y - 0.5) + 3 * np.cos(2 * np.pi * x)
    d_lon = np.gradient(lon, axis=1) / 2
    d_lat = np.gradient(lat, axis=0) / 2
    lon_b = np.stack([lon - d_lon, lon + d_lon], axis=-1)
    lat_b = np.stack([lat - d_lat, lat + d_lat], axis=-1)
    da_lon = xr.DataArray(lon, dims=("y", "x"), attrs={"standard_name": "longitude", "units": "degrees_east", "bounds": "lon_bnds"})
    da_lat = xr.DataArray(lat, dims=("y", "x"), attrs={"standard_name": "latitude", "units": "degrees_north", "bounds": "lat_bnds"})
    air = xr.DataArray(_sine_bell(lon, lat), dims=("y", "x"))
    return xr.Dataset(
        {"air": air, "lon_bnds": (("y", "x", "bnds"), lon_b), "lat_bnds": (("y", "x", "bnds"), lat_b)},
        coords={"lon": da_lon, "lat": da_lat},
    )


@pytest.fixture
def c96_tiles():
    """Six FV3 C96 cubed-sphere tile files (on-demand cached fetch)."""
    try:
        from fetch_c96 import fetch_c96_tiles

        return fetch_c96_tiles()
    except Exception as exc:  # network / cache failure -> skip dependents
        pytest.skip(f"C96 tiles unavailable: {exc}")


@pytest.fixture
def c96_ds(c96_tiles):
    """Assembled 6-tile FV3 cubed-sphere Dataset from the real tile files.

    The NOAA fix files carry a supergrid per tile: ``x``/``y`` of shape
    (2*ny+1, 2*nx+1) in degrees (lon, lat). Cell centers are the odd-odd
    supergrid points; the (2*ny+1, 2*nx+1) corner arrays are passed as the
    supergrid payload so the mesh uses exact corners (AC3).
    """
    tiles = [xr.open_dataset(f) for f in c96_tiles]
    # cell centers: interior supergrid points at odd indices
    lon = np.stack([t["x"].values[1::2, 1::2] for t in tiles])
    lat = np.stack([t["y"].values[1::2, 1::2] for t in tiles])
    da_lon = xr.DataArray(lon, dims=("tile", "j", "i"), attrs={"standard_name": "longitude", "units": "degrees_east"})
    da_lat = xr.DataArray(lat, dims=("tile", "j", "i"), attrs={"standard_name": "latitude", "units": "degrees_north"})
    air = xr.DataArray(_sine_bell(lon, lat), dims=("tile", "j", "i"))
    return xr.Dataset({"air": air}, coords={"lon": da_lon, "lat": da_lat})


@pytest.fixture
def mpas_ds():
    """Real MPAS x1.2562 mesh Dataset (on-demand cached fetch)."""
    try:
        from fetch_mpas import fetch_mpas_grid

        grid_nc = fetch_mpas_grid("x1.2562")
    except Exception as exc:
        pytest.skip(f"MPAS mesh unavailable: {exc}")
    return xr.open_dataset(grid_nc)


@pytest.fixture
def icon_mesh():
    """Synthetic ICON-style triangular mesh: unique nodes + (n,3) connectivity.

    A staggered triangular lattice (equilateral-style, CCW triangles) over a
    regional lon/lat window. Exercises the triangle-polygon ingest path.
    """
    nx, ny = 16, 12
    lon0, lat0, dx, dy = 100.0, 20.0, 2.0, 1.7
    # node (i, j): lon staggered by dx/2 on odd rows
    node_lons, node_lats = [], []
    for j in range(ny):
        for i in range(nx):
            node_lons.append(lon0 + i * dx + (j % 2) * dx / 2)
            node_lats.append(lat0 + j * dy)

    def nid(i, j):
        return j * nx + i

    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            if j % 2 == 0:
                # up triangle: (i,j), (i+1,j), (i,j+1); down: (i+1,j), (i+1,j+1), (i,j+1)
                tris.extend([nid(i, j), nid(i + 1, j), nid(i, j + 1)])
                tris.extend([nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)])
            else:
                tris.extend([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1)])
                tris.extend([nid(i, j), nid(i + 1, j + 1), nid(i, j + 1)])
    node_coords = np.asfortranarray(np.column_stack([node_lons, node_lats]))
    conn = np.asarray(tris, dtype=np.int64)
    offsets = np.arange(0, conn.size + 1, 3, dtype=np.int64)
    return {"node_coords": node_coords, "conn_offsets": offsets, "conn_indices": conn}


@pytest.fixture
def flight_points():
    """Sparse point cloud (flight track): 1-D lon/lat lists."""
    t = np.linspace(0, 1, 200)
    lon = 260 + 40 * t
    lat = 30 + 25 * np.sin(np.pi * t)
    return {"lon": lon, "lat": lat}


@pytest.fixture
def local_cluster():
    """A small single-machine distributed Dask cluster for US6 tests."""
    pytest.importorskip("dask.distributed")
    from dask.distributed import Client, LocalCluster

    with LocalCluster(n_workers=2, threads_per_worker=1, processes=True, memory_limit="512MB", dashboard_address=None) as cluster:
        with Client(cluster) as client:
            yield client
