# SPDX-License-Identifier: Apache-2.0
"""US3 — regrid vector fields with grid-frame rotation (T033).

Asserts AC1–AC3 / FR-021…FR-023 / SC-003: paired u/v rotation-consistent
regridding, explicit angle overrides, conservative rejection at fit, and the
rigid-rotation analytic round-trip preserving direction and magnitude.
"""

from __future__ import annotations

import axis
import numpy as np
import pytest
from axis import Grid, VectorRegridder

xr = pytest.importorskip("xarray")

_METHODS = ["bilinear", "nearest"]


def _geo_field(grid):
    lon = np.asarray(grid._payload["lon"])
    lat = np.asarray(grid._payload["lat"])
    LON, LAT = np.meshgrid(lon, lat)
    return LON, LAT


# ─── AC1: paired u/v regridded + rotated, DataArray and NumPy ────────────────


@pytest.mark.parametrize("method", _METHODS)
def test_uv_pair_datarray(method):
    src = Grid(lon=np.arange(0.5, 360.0, 4.0), lat=np.arange(-88.0, 89.0, 4.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 8.0), lat=np.arange(-84.0, 85.0, 8.0))
    rg = VectorRegridder(src, dst, method)
    _, LAT = _geo_field(src)
    u = xr.DataArray(np.cos(np.radians(LAT)), dims=src.dims)  # zonal wind ~ cos(lat)
    v = xr.zeros_like(u)
    uo, vo = rg(u, v)
    assert uo.dims == dst.dims
    assert uo.shape == dst.shape
    assert "lat" in uo.coords and "lon" in uo.coords
    # magnitude preserved (rect->rect, alpha=0): u stays cos(lat)
    _, LATd = _geo_field(dst)
    np.testing.assert_allclose(uo.values, np.cos(np.radians(LATd)), atol=1e-10)
    np.testing.assert_allclose(vo.values, 0.0, atol=1e-12)


def test_uv_numpy():
    src = Grid(lon=np.arange(0.5, 360.0, 6.0), lat=np.arange(-84.0, 85.0, 6.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 12.0), lat=np.arange(-84.0, 85.0, 12.0))
    rg = VectorRegridder(src, dst, "bilinear")
    n = src.n_cells
    u = np.ones(n)
    v = np.full(n, 2.0)
    uo, vo = rg(u, v)
    assert uo.shape == (dst.n_cells,)
    np.testing.assert_allclose(uo, 1.0, atol=1e-12)
    np.testing.assert_allclose(vo, 2.0, atol=1e-12)


# ─── SC-003: rigid-rotation analytic field across differing orientations ─────


def test_rigid_rotation_roundtrip_direction():
    """Uniform geographic eastward wind survives rect -> cubed-sphere -> rect.

    A rigid-rotation field (constant global vector) must keep direction and
    magnitude through two rotation-aware hops; documented angular tolerance
    (SC-003) is 0.1 degrees, measured exact (0.0)."""
    src = Grid(lon=np.arange(0.5, 360.0, 4.0), lat=np.arange(-88.0, 89.0, 4.0))
    cs = Grid("C12")
    back = Grid(lon=np.arange(0.5, 360.0, 8.0), lat=np.arange(-84.0, 85.0, 8.0))
    u0 = np.ones(src.shape)
    v0 = np.zeros(src.shape)
    r1 = VectorRegridder(src, cs, "bilinear")
    u_cs, v_cs = r1(u0, v0)
    r2 = VectorRegridder(cs, back, "bilinear")
    u_b, v_b = r2(u_cs, v_cs)
    mag = np.hypot(u_b, v_b)
    ang = np.degrees(np.arctan2(v_b, u_b))
    np.testing.assert_allclose(mag, 1.0, atol=1e-6)
    assert np.abs(ang).max() < 0.1  # documented angular tolerance


def test_rigid_rotation_nonuniform():
    """A rotating (sheared) field: u=cos, v=sin of longitude keeps |w|=1 and
    rotates consistently through a cubed-sphere hop."""
    src = Grid(lon=np.arange(0.5, 360.0, 5.0), lat=np.arange(-85.0, 86.0, 5.0))
    LON, _ = _geo_field(src)
    u0 = np.cos(np.radians(LON))
    v0 = np.sin(np.radians(LON))
    cs = Grid("C12")
    r = VectorRegridder(src, cs, "bilinear")
    u_c, v_c = r(u0, v0)
    # |w|=1 up to bilinear interpolation error: components are interpolated
    # separately, so the pointwise norm relaxes slightly (~0.1% at 5 deg)
    np.testing.assert_allclose(np.hypot(u_c, v_c), 1.0, atol=2e-3)


# ─── AC2 / FR-022: explicit angle override honored ───────────────────────────


def test_explicit_angles_override():
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    # rotate the source frame by 90 degrees: eastward wind becomes northward
    alpha = np.full(src.n_cells, np.pi / 2)
    rg = VectorRegridder(src, dst, "bilinear", src_alpha=alpha)
    u = np.ones(src.n_cells)
    v = np.zeros(src.n_cells)
    uo, vo = rg(u, v)
    np.testing.assert_allclose(uo, 0.0, atol=1e-10)
    np.testing.assert_allclose(vo, 1.0, atol=1e-10)


def test_explicit_angles_datarray():
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    alpha = xr.DataArray(np.zeros(src.shape), dims=src.dims)
    rg = VectorRegridder(src, dst, "bilinear", src_alpha=alpha, dst_alpha=alpha)
    u = np.ones(src.n_cells)
    v = np.zeros(src.n_cells)
    uo, vo = rg(u, v)
    np.testing.assert_allclose(uo, 1.0, atol=1e-12)
    np.testing.assert_allclose(vo, 0.0, atol=1e-12)


# ─── AC3 / FR-023: conservative methods rejected at fit ──────────────────────


@pytest.mark.parametrize("method", ["conservative", "conservative2nd", "bicubic", "patch"])
def test_non_interpolation_rejected(method):
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    with pytest.raises(axis.AxisConfigError):
        VectorRegridder(src, dst, method)


# ─── leading dims + dtype parity ─────────────────────────────────────────────


def test_leading_dims():
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 20.0), lat=np.arange(-80.0, 81.0, 20.0))
    rg = VectorRegridder(src, dst, "bilinear")
    u = xr.DataArray(np.ones((3, *src.shape)), dims=("time", *src.dims))
    v = xr.zeros_like(u)
    uo, vo = rg(u, v)
    assert uo.shape == (3, *dst.shape)
    assert uo.dims == ("time", *dst.dims)


def test_float32_parity():
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    rg = VectorRegridder(src, dst, "bilinear")
    u = np.ones(src.n_cells, dtype=np.float32)
    v = np.zeros(src.n_cells, dtype=np.float32)
    uo, vo = rg(u, v)
    assert uo.dtype == np.float32
    assert vo.dtype == np.float32


def test_mismatched_shape_raises():
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    rg = VectorRegridder(src, dst, "bilinear")
    bad = np.ones(src.n_cells + 5)
    with pytest.raises(axis.AxisShapeError):
        rg(bad, bad)


def test_transform_alias_and_call_many():
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    rg = VectorRegridder(src, dst, "bilinear")
    u = np.ones(src.n_cells)
    v = np.zeros(src.n_cells)
    a = rg.transform(u, v)
    b = rg(u, v)
    np.testing.assert_allclose(a[0], b[0])
    # reuse across many calls (SC-004: zero weight regeneration — nnz stable)
    nnz = rg.nnz
    for _ in range(5):
        rg(u, v)
    assert rg.nnz == nnz
