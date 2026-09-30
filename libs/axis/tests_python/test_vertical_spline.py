# SPDX-License-Identifier: Apache-2.0
# test_vertical_spline.py
#
# Tension-spline vertical column interpolation and 3-D pipelines (ported to the
# redesigned VerticalRegridder/regrid_3d API — T038).

import numpy as np
import xarray as xr
from axis import Grid, VerticalRegridder, regrid_3d


def test_vertical_spline_uniform_1d():
    # 2 columns, each with 5 source levels
    n_cols = 2
    src_levels = np.array([1000.0, 850.0, 700.0, 500.0, 300.0], dtype=np.float64)
    dst_levels = np.array([900.0, 600.0], dtype=np.float64)

    # Quadratic source profile: f(lev) = (lev/1000)^2
    src_field = np.zeros((n_cols, 5), dtype=np.float64)
    for c in range(n_cols):
        src_field[c] = (src_levels / 1000.0) ** 2

    regridder = VerticalRegridder(tension=0.0)  # Cubic spline
    dst_field = regridder(src_field, src_levels, dst_levels)

    assert dst_field.shape == (2, 2)
    expected_900 = (900.0 / 1000.0) ** 2  # 0.81
    expected_600 = (600.0 / 1000.0) ** 2  # 0.36

    # Mathematical approximation error of spline on 5 nodes is ~1.9%-2.8%
    np.testing.assert_allclose(dst_field[:, 0], expected_900, rtol=3e-2)
    np.testing.assert_allclose(dst_field[:, 1], expected_600, rtol=3e-2)


def test_vertical_spline_spatially_varying_2d():
    # 2 columns, spatially varying levels
    src_levels = np.array([[1000.0, 800.0, 600.0], [950.0, 750.0, 550.0]], dtype=np.float64)
    dst_levels = np.array([[900.0, 700.0], [850.0, 650.0]], dtype=np.float64)

    src_field = np.ones((2, 3), dtype=np.float64) * 42.0

    regridder = VerticalRegridder(tension=2.0)
    dst_field = regridder(src_field, src_levels, dst_levels)

    assert dst_field.shape == (2, 2)
    # Linear flat profile must preserve value exactly
    np.testing.assert_allclose(dst_field, 42.0, rtol=1e-12)


def test_chained_regrid_3d_pipeline():
    # 4x4 horizontal grid, 3 vertical levels
    lons_in = np.linspace(-10, 10, 4)
    lats_in = np.linspace(-10, 10, 4)
    levs_in = np.array([1000.0, 700.0, 300.0])

    temp = np.ones((3, 4, 4), dtype=np.float64) * 15.0  # (lev, lat, lon)
    da_in = xr.DataArray(temp, dims=("lev", "lat", "lon"), coords={"lev": levs_in, "lat": lats_in, "lon": lons_in})

    src = Grid(lon=lons_in, lat=lats_in)
    dst = Grid(lon=np.linspace(-8, 8, 2), lat=np.linspace(-8, 8, 2))
    levs_out = np.array([900.0, 500.0])

    da_out = regrid_3d(da_in, src, dst, levs_in, levs_out, method="bilinear", tension=0.0, vertical_dim="lev")

    assert da_out.shape == (2, 2, 2)  # (lev, lat, lon)
    assert da_out.dims == ("lev", "lat", "lon")
    np.testing.assert_allclose(da_out.values, 15.0, rtol=1e-12)
