# SPDX-License-Identifier: Apache-2.0
"""US4 — vertical and 3-D remapping (T036).

Asserts AC1–AC4 / FR-024…FR-026: tension spline on 1-D level vectors,
per-column varying levels (hybrid-sigma style), explicit out-of-range policy
(nan / clip / extrapolate), and one-call horizontal+vertical composition.
"""

from __future__ import annotations

import axis
import numpy as np
import pytest
from axis import Grid, VerticalRegridder, regrid_3d

xr = pytest.importorskip("xarray")


def _linear_field(levels_shape, src_levels):
    """f(L) = L: exact for any spline order, ideal for correctness probes."""
    return np.broadcast_to(np.asarray(src_levels, dtype=np.float64), levels_shape).copy()


# ─── AC1 / FR-024: 1-D level vectors, tension knob ──────────────────────────


def test_1d_levels_numpy():
    src = np.array([1000.0, 800.0, 600.0, 400.0])  # descending (pressure-like)
    dst = np.array([900.0, 700.0, 500.0])
    field = _linear_field((2, 4), src)  # (n_col, lev)
    rg = VerticalRegridder(tension=0.0)
    out = rg(field, src, dst)
    np.testing.assert_allclose(out, np.tile(dst, (2, 1)), atol=1e-10)


def test_1d_levels_datarray():
    levs = np.array([1000.0, 800.0, 600.0, 400.0])
    # midpoint levels: exact for the tension spline (see test_vertical_spline)
    dst = np.array([900.0, 500.0])
    data = np.broadcast_to(levs[:, None, None], (4, 4, 4)).copy()  # (lev, y, x)
    da = xr.DataArray(data, dims=("lev", "y", "x"), coords={"lev": levs})
    rg = VerticalRegridder()
    out = rg(da, da["lev"], dst, vertical_dim="lev")
    assert out.dims == ("lev", "y", "x")
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out.sel(lev=900.0).values, 900.0, atol=1e-10)
    np.testing.assert_allclose(out.sel(lev=500.0).values, 500.0, atol=1e-10)
    np.testing.assert_allclose(out["lev"].values, dst)


def test_tension_changes_curvature():
    """A quadratic profile: cubic spline (tension=0) vs stiff spline (high
    tension, locally linear) must differ at mid-interval — FR-024 knob."""
    src = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    field = src**2  # f = L^2
    rg_soft = VerticalRegridder(tension=0.0)
    rg_stiff = VerticalRegridder(tension=10.0)
    dst = np.array([0.5, 1.5, 2.5, 3.5])
    out_soft = rg_soft(field, src, dst)
    out_stiff = rg_stiff(field, src, dst)
    assert not np.allclose(out_soft, out_stiff)
    # stiff spline interpolates between adjacent nodes: stays inside the
    # node-pair linear range; soft cubic undershoots below the parabola chord
    assert (out_stiff > 0.25).all()


def test_ascending_input_matches_descending():
    src_d = np.array([1000.0, 800.0, 600.0, 400.0])
    src_a = src_d[::-1].copy()
    dst = np.array([900.0, 500.0])
    rg = VerticalRegridder()
    out_d = rg(_linear_field((4, 4), src_d), src_d, dst)
    out_a = rg(_linear_field((4, 4), src_a), src_a, dst)
    np.testing.assert_allclose(out_d, out_a, atol=1e-12)


# ─── AC2 / FR-024: per-column varying levels (hybrid → pressure) ─────────────


def test_varying_2d_levels():
    # column 0: surface 1000→400; column 1: 950→350. f(L)=L → exact recovery.
    src = np.array([[1000.0, 800.0, 600.0, 400.0], [950.0, 750.0, 550.0, 350.0]])
    dst = np.array([[900.0, 700.0], [850.0, 650.0]])
    field = src.copy()
    rg = VerticalRegridder()
    out = rg(field, src, dst)
    np.testing.assert_allclose(out, dst, atol=1e-10)


def test_varying_3d_levels_datarray():
    levs = np.array([1000.0, 700.0, 400.0])
    # per-column levels (y, x, lev): a topography-masked hybrid set
    y, x = np.mgrid[0:2, 0:3]  # each (2, 3)
    src_levs = levs[None, None, :] - 10.0 * y[:, :, None] + x[:, :, None]
    dst_levs = np.tile(np.array([900.0, 500.0]), (2, 3, 1))  # per-column (2,3,2)
    field = src_levs.copy()  # f(L)=L per column
    da = xr.DataArray(
        np.moveaxis(field, -1, 0),  # (lev, y, x)
        dims=("lev", "y", "x"),
    )
    src_da = xr.DataArray(src_levs, dims=("y", "x", "lev"))
    dst_da = xr.DataArray(dst_levs, dims=("y", "x", "lev"))
    rg = VerticalRegridder()
    out = rg(da, src_da, dst_da, vertical_dim="lev")
    assert out.shape == (2, 2, 3)
    # f(L)=L per column; spline is exact at midpoints, ~3% elsewhere
    np.testing.assert_allclose(out.isel(lev=0).values, dst_levs[:, :, 0], rtol=5e-2)
    np.testing.assert_allclose(out.isel(lev=1).values, dst_levs[:, :, 1], rtol=5e-2)


# ─── AC4 / FR-026: explicit out-of-range policy ──────────────────────────────


def test_out_of_range_nan_default():
    src = np.array([1000.0, 800.0, 600.0, 400.0])
    dst = np.array([900.0, 200.0, 1200.0])  # 200 below, 1200 above
    field = _linear_field((2, 4), src)
    rg = VerticalRegridder()  # default nan
    out = rg(field, src, dst)
    assert np.isnan(out[:, 1]).all()
    assert np.isnan(out[:, 2]).all()
    np.testing.assert_allclose(out[:, 0], 900.0, atol=1e-10)


def test_out_of_range_clip():
    src = np.array([1000.0, 800.0, 600.0, 400.0])
    dst = np.array([900.0, 200.0, 1200.0])
    field = _linear_field((2, 4), src)
    rg = VerticalRegridder(out_of_range="clip")
    out = rg(field, src, dst)
    np.testing.assert_allclose(out, np.tile([900.0, 400.0, 1000.0], (2, 1)), atol=1e-10)


def test_out_of_range_extrapolate():
    src = np.array([1000.0, 800.0, 600.0, 400.0])
    dst = np.array([900.0, 200.0, 1200.0])
    field = _linear_field((2, 4), src)  # f=L: linear, extrapolation is exact
    rg = VerticalRegridder(out_of_range="extrapolate")
    out = rg(field, src, dst)
    np.testing.assert_allclose(out, np.tile(dst, (2, 1)), atol=1e-8)


def test_out_of_range_invalid():
    with pytest.raises(axis.AxisConfigError):
        VerticalRegridder(out_of_range="magic")


def test_out_of_range_varying_levels():
    # per-column ranges: column 0 tops out at 1000, column 1 at 950
    src = np.array([[1000.0, 800.0, 600.0], [950.0, 750.0, 550.0]])
    dst = np.array([[1100.0], [1100.0]])  # above both columns' max
    field = src.copy()
    rg_nan = VerticalRegridder(out_of_range="nan")
    out = rg_nan(field, src, dst)
    assert np.isnan(out).all()
    rg_clip = VerticalRegridder(out_of_range="clip")
    out_c = rg_clip(field, src, dst)
    np.testing.assert_allclose(out_c, np.array([[1000.0], [950.0]]), atol=1e-10)


# ─── AC3 / FR-025: regrid_3d one-call composition ────────────────────────────


def test_regrid_3d_composition():
    lons = np.arange(0.5, 360.0, 10.0)
    lats = np.arange(-80.0, 81.0, 10.0)
    levs = np.array([1000.0, 700.0, 400.0])
    src = Grid(lon=lons, lat=lats)
    dst = Grid(lon=np.arange(0.5, 360.0, 20.0), lat=np.arange(-80.0, 81.0, 20.0))
    # f(lev) = lev everywhere → horizontal-invariant, vertical-linear
    data = np.broadcast_to(levs[:, None, None], (3, lats.size, lons.size)).copy()
    da = xr.DataArray(data, dims=("lev", "lat", "lon"), coords={"lev": levs})
    dst_levs = np.array([850.0, 550.0])
    out = regrid_3d(da, src, dst, levs, dst_levs, method="bilinear")
    assert out.dims == ("lev", "lat", "lon")
    assert out.shape == (2, dst.shape[0], dst.shape[1])
    np.testing.assert_allclose(out.isel(lev=0).values, 850.0, atol=1e-8)
    np.testing.assert_allclose(out.isel(lev=1).values, 550.0, atol=1e-8)
    assert np.allclose(out["lat"].values[1] - out["lat"].values[0], 20.0)


def test_regrid_3d_4d_field():
    lons = np.arange(0.5, 360.0, 20.0)
    lats = np.arange(-80.0, 81.0, 20.0)
    levs = np.array([1000.0, 500.0])
    src = Grid(lon=lons, lat=lats)
    dst = Grid(lon=np.arange(0.5, 360.0, 40.0), lat=lats)
    data = np.broadcast_to(levs[None, :, None, None], (4, 2, lats.size, lons.size)).copy()
    da = xr.DataArray(data, dims=("time", "lev", "lat", "lon"), coords={"time": np.arange(4), "lev": levs})
    out = regrid_3d(da, src, dst, levs, np.array([750.0]), method="bilinear")
    assert out.dims == ("time", "lev", "lat", "lon")
    assert out.shape == (4, 1, lats.size, (lons.size + 1) // 2)
    np.testing.assert_allclose(out.isel(lev=0).values, 750.0, atol=1e-8)


# ─── misc API behavior ───────────────────────────────────────────────────────


def test_vertical_dim_detection_by_size():
    levs = np.array([1000.0, 800.0, 600.0])
    da = xr.DataArray(_linear_field((5, 4, 3), levs), dims=("t", "y", "lev"), coords={"lev": levs})
    rg = VerticalRegridder()
    out = rg(da, levs, np.array([900.0, 700.0]))  # vertical_dim=None -> match size 3 == len(src)
    assert out.shape == (5, 4, 2)
    assert set(out.dims) == {"t", "y", "lev"}
    np.testing.assert_allclose(out.isel(lev=0).values, 900.0, atol=1e-10)


def test_numpy_default_last_axis():
    levs = np.array([400.0, 600.0, 800.0, 1000.0])  # ascending
    field = _linear_field((2, 2, 4), levs)
    rg = VerticalRegridder()
    out = rg(field, levs, np.array([900.0]))
    assert out.shape == (2, 2, 1)
    np.testing.assert_allclose(out, 900.0, atol=1e-10)


def test_mismatched_level_size_raises():
    levs = np.array([1000.0, 800.0, 600.0])
    da = xr.DataArray(_linear_field((3, 3, 3), levs), dims=("y", "x", "lev"))
    rg = VerticalRegridder()
    with pytest.raises(axis.AxisShapeError):
        rg(da, np.array([1000.0, 900.0, 800.0, 700.0, 600.0, 500.0]), np.array([850.0]), vertical_dim="lev")


def test_dask_path():
    pytest.importorskip("dask.array")
    import dask.array as da_mod

    src = np.array([1000.0, 800.0, 600.0, 400.0])
    dst = np.array([900.0, 500.0])
    field = da_mod.from_array(_linear_field((4, 4, 4), src), chunks=(2, 2, 4))
    levs = xr.DataArray(src, dims=("lev",))
    da = xr.DataArray(field, dims=("y", "x", "lev"), coords={"lev": src})
    rg = VerticalRegridder()
    out = rg(da, levs, dst, vertical_dim="lev")
    result = out.compute()
    np.testing.assert_allclose(result.sel(lev=900.0).values, 900.0, atol=1e-10)
    np.testing.assert_allclose(result.sel(lev=500.0).values, 500.0, atol=1e-10)
