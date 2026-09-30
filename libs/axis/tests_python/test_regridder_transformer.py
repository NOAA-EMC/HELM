# SPDX-License-Identifier: Apache-2.0
"""Scikit-learn-style ``transform`` entry point over the new fit-once API.

The legacy instantiate-then-``fit`` pattern is retired (FR-040); a
``Regridder`` is fitted at construction and ``transform`` is the alias for
``__call__``. This test keeps the sklearn-pipeline guarantee on the new surface.
"""

import numpy as np
import xarray as xr
from axis import Grid, Regridder


def _sample():
    lons = np.linspace(-180, 180, 10)
    lats = np.linspace(-90, 90, 10)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    da = xr.DataArray(
        np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid)),
        coords={"lat": lats, "lon": lons},
        dims=["lat", "lon"],
    )
    return da.to_dataset(name="sst"), da


def test_transform_matches_call():
    ds_in, da_in = _sample()
    src = Grid(lon=ds_in.lon.values, lat=ds_in.lat.values)
    dst = Grid(lon=np.linspace(-180, 180, 5), lat=np.linspace(-90, 90, 5))

    rg = Regridder(src, dst, method="bilinear")
    xr.testing.assert_allclose(rg.transform(da_in), rg(da_in))


def test_transform_with_grid_objects():
    ds_in, da_in = _sample()
    src = Grid(lon=ds_in.lon.values, lat=ds_in.lat.values)
    dst = Grid(lon=np.linspace(-180, 180, 5), lat=np.linspace(-90, 90, 5))

    res = Regridder(src, dst, method="bilinear").transform(da_in)
    assert res.shape == (5, 5)
    assert "lat" in res.coords
    assert "lon" in res.coords
