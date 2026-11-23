# SPDX-License-Identifier: Apache-2.0
"""
Pytest configuration for AXIS Python API integration tests.

Ensures Kokkos is initialized before any test that imports axis_py.
Kokkos initialization is handled internally by the axis_py module
(via ensure_kokkos()), so this conftest primarily provides shared
fixtures for mesh construction used across multiple tests.
"""

import pytest


@pytest.fixture
def small_src_mesh():
    """Create a small 8x4 regular lat-lon source mesh."""
    from axis import axis_py

    ni, nj = 8, 4
    lon_start, lat_start = 0.0, -90.0
    dlon = 360.0 / ni
    dlat = 180.0 / nj
    return axis_py.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


@pytest.fixture
def small_dst_mesh():
    """Create a small 4x2 regular lat-lon destination mesh."""
    from axis import axis_py

    ni, nj = 4, 2
    lon_start, lat_start = 0.0, -90.0
    dlon = 360.0 / ni
    dlat = 180.0 / nj
    return axis_py.make_regular_mesh(ni, nj, lon_start, lat_start, dlon, dlat)


@pytest.fixture
def bilinear_matrix(small_src_mesh, small_dst_mesh):
    """Generate bilinear interpolation weights between src and dst meshes."""
    from axis import axis_py

    return axis_py.generate_weights(small_src_mesh, small_dst_mesh, axis_py.Method.Bilinear)


@pytest.fixture
def dask_regrid_case():
    """Return a small source/target grid and a lazy field for optional Dask tests."""
    import numpy as np
    import xarray as xr

    da = pytest.importorskip("dask.array")

    src_lons = np.linspace(0.0, 360.0, 8, endpoint=False)
    src_lats = np.linspace(-60.0, 60.0, 5)
    dst_lons = np.linspace(0.0, 360.0, 12, endpoint=False)
    dst_lats = np.linspace(-60.0, 60.0, 7)

    source_grid = xr.Dataset({"lon": ("lon", src_lons), "lat": ("lat", src_lats)})
    target_grid = xr.Dataset({"lon": ("lon", dst_lons), "lat": ("lat", dst_lats)})
    values = da.ones((2, src_lats.size, src_lons.size), chunks=(1, src_lats.size, src_lons.size))
    field = xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": [0, 1], "lat": src_lats, "lon": src_lons},
        name="temperature",
        attrs={"units": "K", "long_name": "test temperature"},
    )
    return source_grid, target_grid, field
