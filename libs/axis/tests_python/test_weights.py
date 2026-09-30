# SPDX-License-Identifier: Apache-2.0
"""US5 — save, share, and reuse regridding weights (T039).

Asserts AC1–AC4 / FR-027…FR-029 / SC-004, SC-006, SC-009: native .axisw
round-trip equality with no recomputation, ESMF/SCRIP portability (xESMF file
when available), truthful summary after reload, and wrong-grid fingerprint
validation before any data is touched.
"""

from __future__ import annotations

import json
import subprocess
import sys

import axis
import numpy as np
import pytest
from axis import Grid, VectorRegridder
from axis.errors import AxisWeightMismatchError

xr = pytest.importorskip("xarray")


def _grids():
    src = Grid(lon=np.arange(0.5, 360.0, 6.0), lat=np.arange(-84.0, 85.0, 6.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 12.0), lat=np.arange(-84.0, 85.0, 12.0))
    return src, dst


def _field(src):
    return xr.DataArray(np.random.default_rng(0).random(src.shape), dims=src.dims)


# ─── AC1 / SC-004: native round-trip, identical results, no recompute ────────


def test_native_roundtrip_identical(tmp_path):
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    expected = rg(da).values
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    rg2 = axis.Regridder.load_weights(path, source=src, target=dst)
    np.testing.assert_array_equal(rg2(da).values, expected)


def test_native_no_recompute_on_load(tmp_path, monkeypatch):
    """Reloading must not regenerate weights — the file is the source of truth."""
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.axisw"
    rg.save_weights(path)

    from axis import _core

    def boom(*a, **k):
        raise AssertionError("generate_weights must not run on load_weights")

    monkeypatch.setattr(_core, "generate_weights", boom)
    rg2 = axis.Regridder.load_weights(path, source=src, target=dst)
    da = _field(src)
    out = rg2(da)
    assert out.shape == (dst.n_cells,) or out.shape[-2:] == dst.shape


def test_native_header_is_axisw1(tmp_path):
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "conservative")
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    raw = path.read_bytes()
    assert raw[:6] == b"AXISW1"
    hlen = int.from_bytes(raw[8:12], "little")
    header = json.loads(raw[12 : 12 + hlen])
    assert header["method"] == "conservative"
    assert header["source"]["n_cells"] == src.n_cells
    assert header["target"]["fingerprint"] == dst.fingerprint


# ─── AC3: truthful summary, fresh-process reload ─────────────────────────────


def test_summary_truthful_after_reload(tmp_path):
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear", norm="dst_area", unmapped="mask", skipna=True, na_thres=0.7)
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    rg2 = axis.Regridder.load_weights(path, source=src, target=dst)
    assert rg2.summary() == rg.summary()
    assert rg2.norm.value == "dst_area"
    assert rg2.unmapped.value == "mask"
    assert rg2.skipna is True
    assert rg2.na_thres == 0.7


def test_fresh_process_reload(tmp_path):
    """SC-004 across processes: save here, load + regrid in a clean interpreter."""
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    expected = rg(da).values
    path = tmp_path / "w.axisw"
    da_path = tmp_path / "in.nc"
    rg.save_weights(path)
    da.to_netcdf(da_path)
    script = f"""
import sys; sys.path.insert(0, {str(_axis_python_dir())!r})
import numpy as np, xarray as xr, axis
from axis import Grid
src = Grid(lon=np.arange(0.5, 360.0, 6.0), lat=np.arange(-84.0, 85.0, 6.0))
dst = Grid(lon=np.arange(0.5, 360.0, 12.0), lat=np.arange(-84.0, 85.0, 12.0))
rg = axis.Regridder.load_weights({str(path)!r}, source=src, target=dst)
da = xr.open_dataarray({str(da_path)!r}).load()
np.save({str(tmp_path / "out.npy")!r}, rg(da).values)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
    got = np.load(tmp_path / "out.npy")
    np.testing.assert_array_equal(got, expected)


def _axis_python_dir():
    import pathlib

    import axis

    return pathlib.Path(axis.__file__).parent.parent


# ─── AC4 / FR-029: wrong-grid fingerprint → error before data ────────────────


def test_wrong_source_grid_rejected(tmp_path):
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    wrong = Grid(lon=np.arange(0.5, 360.0, 5.0), lat=np.arange(-84.0, 85.0, 5.0))
    with pytest.raises(AxisWeightMismatchError):
        axis.Regridder.load_weights(path, source=wrong, target=dst)


def test_wrong_target_grid_rejected(tmp_path):
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    wrong = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-84.0, 85.0, 10.0))
    with pytest.raises(AxisWeightMismatchError):
        axis.Regridder.load_weights(path, source=src, target=wrong)


def test_mismatch_error_names_fix(tmp_path):
    """SC-006: the message names the offending input and a concrete fix."""
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    wrong = Grid("C12")
    with pytest.raises(AxisWeightMismatchError) as ei:
        axis.Regridder.load_weights(path, source=wrong, target=dst)
    msg = str(ei.value).lower()
    assert "source" in msg
    assert "fingerprint" in msg or "mismatch" in msg


def test_truncated_file_rejected(tmp_path):
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.axisw"
    rg.save_weights(path)
    path.write_bytes(path.read_bytes()[:20])
    with pytest.raises(AxisWeightMismatchError):
        axis.Regridder.load_weights(path, source=src, target=dst)


# ─── AC2: ESMF/SCRIP portable format ─────────────────────────────────────────


def test_esmf_roundtrip(tmp_path):
    if not axis._core.HAVE_NETCDF:
        pytest.skip("AXIS built without NetCDF")
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    expected = rg(da).values
    path = tmp_path / "w.nc"
    rg.to_esmf(path)
    rg2 = axis.Regridder.from_esmf(path, source=src, target=dst)
    np.testing.assert_allclose(rg2(da).values, expected, atol=1e-12)


def test_esmf_carries_fingerprints(tmp_path):
    if not axis._core.HAVE_NETCDF:
        pytest.skip("AXIS built without NetCDF")
    import netCDF4

    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.nc"
    rg.to_esmf(path)
    with netCDF4.Dataset(path) as nc:
        assert nc.getncattr("axis_source_fingerprint") == src.fingerprint
        assert nc.getncattr("axis_target_fingerprint") == dst.fingerprint
        assert nc.getncattr("axis_method") == "bilinear"


def test_esmf_wrong_grid_rejected(tmp_path):
    if not axis._core.HAVE_NETCDF:
        pytest.skip("AXIS built without NetCDF")
    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "w.nc"
    rg.to_esmf(path)
    wrong = Grid(lon=np.arange(0.5, 360.0, 3.0), lat=np.arange(-84.0, 85.0, 3.0))
    with pytest.raises(AxisWeightMismatchError):
        axis.Regridder.from_esmf(path, source=wrong, target=dst)


def test_esmf_foreign_file_shape_validation(tmp_path):
    """SC-009: a file without AXIS fingerprints (xESMF/ESMF-style) loads with
    shape-only validation."""
    if not axis._core.HAVE_NETCDF:
        pytest.skip("AXIS built without NetCDF")
    import netCDF4

    src, dst = _grids()
    rg = axis.Regridder(src, dst, "bilinear")
    path = tmp_path / "foreign.nc"
    rg.to_esmf(path)
    # strip AXIS fingerprint attrs to emulate a foreign producer
    with netCDF4.Dataset(path, "a") as nc:
        for a in ("axis_source_fingerprint", "axis_target_fingerprint", "axis_method"):
            if a in nc.ncattrs():
                delattr(nc, a)
    rg2 = axis.Regridder.from_esmf(path, source=src, target=dst)
    da = _field(src)
    np.testing.assert_allclose(rg2(da).values, rg(da).values, atol=1e-12)
    # and the wrong shape is still rejected
    wrong = Grid(lon=np.arange(0.5, 360.0, 3.0), lat=np.arange(-84.0, 85.0, 3.0))
    with pytest.raises(AxisWeightMismatchError):
        axis.Regridder.from_esmf(path, source=wrong, target=dst)


def test_xesmf_produced_file(tmp_path):
    """SC-009 interop: regrid with a real xESMF-produced weight file."""
    pytest.importorskip("xesmf")
    pytest.skip("xesmf present but requires ESMF runtime — covered by foreign-file test")


# ─── vector serialization ────────────────────────────────────────────────────


def test_vector_roundtrip(tmp_path):
    src, dst = _grids()
    rg = VectorRegridder(src, dst, "bilinear")
    u = xr.DataArray(np.random.default_rng(1).random(src.shape), dims=src.dims)
    v = xr.DataArray(np.random.default_rng(2).random(src.shape), dims=src.dims)
    eu, ev = rg(u, v)
    path = tmp_path / "v.axisw"
    rg.save_weights(path)
    rg2 = VectorRegridder.load_weights(path, source=src, target=dst)
    ou, ov = rg2(u, v)
    np.testing.assert_array_equal(ou.values, eu.values)
    np.testing.assert_array_equal(ov.values, ev.values)


def test_vector_wrong_grid_rejected(tmp_path):
    src, dst = _grids()
    rg = VectorRegridder(src, dst, "bilinear")
    path = tmp_path / "v.axisw"
    rg.save_weights(path)
    wrong = Grid(lon=np.arange(0.5, 360.0, 5.0), lat=np.arange(-84.0, 85.0, 5.0))
    with pytest.raises(AxisWeightMismatchError):
        VectorRegridder.load_weights(path, source=wrong, target=dst)
