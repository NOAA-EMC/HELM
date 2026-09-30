# SPDX-License-Identifier: Apache-2.0
"""US7 — regrid from the command line (T047).

Asserts AC1–AC3 / FR-035/FR-036 / SC-008 per contracts/cli.md: regrid every
method, weights round-trip in both directions (CLI↔Python), exit codes
{0,1,2}, actionable stderr diagnostics, and `list methods|grids`.
"""

from __future__ import annotations

import subprocess
import sys

import axis
import numpy as np
import pytest
from axis import Grid

xr = pytest.importorskip("xarray")

CLI = [sys.executable, "-m", "axis.cli"]


# ─── fixtures: source/target NetCDF files ────────────────────────────────────


@pytest.fixture()
def nc_files(tmp_path):
    src_lons = np.arange(0.5, 360.0, 10.0)
    src_lats = np.arange(-80.0, 81.0, 10.0)
    dst_lons = np.arange(0.5, 360.0, 20.0)
    dst_lats = np.arange(-80.0, 81.0, 20.0)
    rng = np.random.default_rng(42)
    tas = rng.random((src_lats.size, src_lons.size))
    pr = rng.random((src_lats.size, src_lons.size))
    ds_src = xr.Dataset(
        {"tas": (("lat", "lon"), tas), "pr": (("lat", "lon"), pr)},
        coords={"lat": src_lats, "lon": src_lons},
    )
    ds_dst = xr.Dataset(coords={"lat": dst_lats, "lon": dst_lons})
    s = tmp_path / "src.nc"
    t = tmp_path / "dst.nc"
    ds_src.to_netcdf(s)
    ds_dst.to_netcdf(t)
    return s, t


def run(*args, expect=0):
    proc = subprocess.run(CLI + list(args), capture_output=True, text=True)
    assert proc.returncode == expect, f"rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    return proc


# ─── AC1: regrid every method, target coords attached ────────────────────────


@pytest.mark.parametrize("method", ["bilinear", "nearest", "conservative"])
def test_regrid_all_methods(nc_files, tmp_path, method):
    s, t = nc_files
    out = tmp_path / "out.nc"
    run("regrid", "-s", str(s), "-t", str(t), "-o", str(out), "-m", method)
    ds = xr.open_dataset(out)
    assert ds["tas"].shape[-2:] == (9, 18)  # dst: 9 lats x 18 lons
    assert np.allclose(ds["lat"].values, np.arange(-80.0, 81.0, 20.0))
    assert np.allclose(ds["lon"].values, np.arange(0.5, 360.0, 20.0))
    assert not np.isnan(ds["tas"].values).all()


def test_regrid_variable_selection(nc_files, tmp_path):
    s, t = nc_files
    out = tmp_path / "out.nc"
    run("regrid", "-s", str(s), "-t", str(t), "-o", str(out), "-v", "tas")
    ds = xr.open_dataset(out)
    assert "tas" in ds
    assert "pr" not in ds


def test_regrid_matches_library(nc_files, tmp_path):
    s, t = nc_files
    out = tmp_path / "out.nc"
    run("regrid", "-s", str(s), "-t", str(t), "-o", str(out), "-m", "bilinear")
    ds = xr.open_dataset(out)
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 20.0), lat=np.arange(-80.0, 81.0, 20.0))
    da = xr.open_dataset(s)["tas"]
    expected = axis.Regridder(src, dst, "bilinear")(da)
    np.testing.assert_allclose(ds["tas"].values, expected.values, atol=1e-12)


def test_regrid_flags(nc_files, tmp_path):
    s, t = nc_files
    out = tmp_path / "out.nc"
    run(
        "regrid",
        "-s",
        str(s),
        "-t",
        str(t),
        "-o",
        str(out),
        "-m",
        "conservative",
        "--norm",
        "dst-area",
        "--periodic",
        "--skipna",
        "--na-thres",
        "0.5",
        "--line-type",
        "great-circle",
    )


def test_regrid_keep_attrs(nc_files, tmp_path):
    s, t = nc_files
    # tag the source var with an attribute (load fully, then close, before rewriting)
    ds = xr.open_dataset(s)
    ds["tas"].attrs = {"units": "K", "long_name": "air temperature"}
    ds = ds.load()
    ds.close()
    ds.to_netcdf(s)
    out = tmp_path / "out.nc"
    run("regrid", "-s", str(s), "-t", str(t), "-o", str(out))
    assert xr.open_dataset(out)["tas"].attrs.get("units") == "K"
    out2 = tmp_path / "out2.nc"
    run("regrid", "-s", str(s), "-t", str(t), "-o", str(out2), "--no-keep-attrs")
    assert "units" not in xr.open_dataset(out2)["tas"].attrs


# ─── AC2: weights round-trip in both directions ──────────────────────────────


def test_weights_native_cli_to_python(nc_files, tmp_path):
    s, t = nc_files
    w = tmp_path / "w.axisw"
    run("weights", "-s", str(s), "-t", str(t), "-o", str(w), "-m", "bilinear", "--format", "native")
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 20.0), lat=np.arange(-80.0, 81.0, 20.0))
    rg = axis.Regridder.load_weights(w, source=src, target=dst)
    da = xr.open_dataset(s)["tas"]
    out = rg(da)
    assert out.shape[-2:] == dst.shape


def test_weights_esmf_cli_to_python(nc_files, tmp_path):
    s, t = nc_files
    w = tmp_path / "w.nc"
    run("weights", "-s", str(s), "-t", str(t), "-o", str(w), "-m", "bilinear", "--format", "esmf")
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 20.0), lat=np.arange(-80.0, 81.0, 20.0))
    rg = axis.Regridder.from_esmf(w, source=src, target=dst)
    da = xr.open_dataset(s)["tas"]
    assert rg(da).shape[-2:] == dst.shape


def test_weights_python_to_cli(nc_files, tmp_path):
    """A weight file produced by the Python API is consumed by the CLI."""
    s, t = nc_files
    w = tmp_path / "from_py.axisw"
    src = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    dst = Grid(lon=np.arange(0.5, 360.0, 20.0), lat=np.arange(-80.0, 81.0, 20.0))
    axis.Regridder(src, dst, "bilinear").save_weights(w)
    # CLI `weights` can re-emit / validate it: use regrid with precomputed weights
    out = tmp_path / "out.nc"
    run("regrid", "-s", str(s), "-t", str(t), "-o", str(out), "--weights", str(w))
    assert xr.open_dataset(out)["tas"].shape[-2:] == (9, 18)


# ─── AC3 / FR-036: exit codes and actionable stderr ──────────────────────────


def test_unknown_method_exit_2(nc_files):
    s, t = nc_files
    proc = run("regrid", "-s", str(s), "-t", str(t), "-o", "/tmp/x.nc", "-m", "notamethod", expect=2)
    assert "invalid choice" in proc.stderr.lower() or "notamethod" in proc.stderr.lower()


def test_missing_file_exit_1(nc_files, tmp_path):
    s, t = nc_files
    proc = run("regrid", "-s", str(tmp_path / "nope.nc"), "-t", str(t), "-o", str(tmp_path / "x.nc"), expect=1)
    assert "nope.nc" in proc.stderr
    assert "does not exist" in proc.stderr.lower() or "not found" in proc.stderr.lower()


def test_no_such_variable_exit_1(nc_files, tmp_path):
    s, t = nc_files
    proc = run("regrid", "-s", str(s), "-t", str(t), "-o", str(tmp_path / "o.nc"), "-v", "nonexistent_var", expect=1)
    assert "nonexistent_var" in proc.stderr


def test_incompatible_grids_exit_1(tmp_path):
    # points cloud target vs conservative source -> fit-time AxisConfigError
    s_lon = np.arange(0.5, 360.0, 20.0)
    s_lat = np.arange(-80.0, 81.0, 20.0)
    ds_src = xr.Dataset({"tas": (("lat", "lon"), np.ones((9, 18)))}, coords={"lat": s_lat, "lon": s_lon})
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    ds_src.to_netcdf(a)
    # target: a single point (1x1 grid) conservative is fine; force a real error:
    # unknown grid structure -> GridError at detection
    ds_bad = xr.Dataset({"not_coords": (("q",), np.arange(3))})
    ds_bad.to_netcdf(b)
    proc = run("regrid", "-s", str(a), "-t", str(b), "-o", str(tmp_path / "o.nc"), expect=1)
    assert proc.stderr.strip()  # actionable message present


def test_no_args_exit_2():
    proc = subprocess.run(CLI, capture_output=True, text=True)
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


# ─── list subcommand ─────────────────────────────────────────────────────────


def test_list_methods():
    proc = run("list", "methods")
    lines = proc.stdout.split()
    for m in ["bilinear", "bicubic", "patch", "nearest", "conservative", "conservative2nd"]:
        assert m in lines


def test_list_grids():
    proc = run("list", "grids")
    out = proc.stdout
    for fam in "CFGNOR":
        assert fam in out
