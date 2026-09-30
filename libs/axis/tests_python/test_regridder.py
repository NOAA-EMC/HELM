# SPDX-License-Identifier: Apache-2.0
"""US1 — regrid a field from one grid to another (T020 + T021).

Red until the ``Regridder`` rewrite (T022/T023) lands. Asserts the contract in
``contracts/python-api.md`` §Regridder and the method×family matrix in
``data-model.md`` §2: fit-once/call-many, xarray/NumPy/Dataset paths, leading-
dim preservation, target-coord attachment, attrs handling, fail-fast on
impossible pairings, dtype preservation (FR-018), and the unmapped policy triad
(FR-017).
"""

from __future__ import annotations

import axis
import numpy as np
import pytest
from axis import Grid

xr = pytest.importorskip("xarray")


# ─── helpers ─────────────────────────────────────────────────────────────────


def _rect_src():
    return Grid(lon=np.arange(0.5, 360.0, 2.0), lat=np.arange(-89.5, 90.0, 2.0))


def _rect_dst():
    return Grid(lon=np.arange(0.5, 360.0, 5.0), lat=np.arange(-89.5, 90.0, 5.0))


def _field(grid, *, dtype=np.float64, leading=()):
    lon, lat = grid._payload["lon"], grid._payload["lat"]
    lon2d, lat2d = np.meshgrid(lon, lat)
    vals = (np.cos(np.radians(lat2d)) ** 2) * (1.0 + 0.5 * np.cos(np.radians(lon2d)))
    vals = np.broadcast_to(vals.astype(dtype), tuple(leading) + grid.shape).copy()
    dims = [f"extra{i}" for i in range(len(leading))] + list(grid.dims)
    return xr.DataArray(vals, dims=dims, attrs={"units": "K", "long_name": "sine-bell"})


# ─── AC1: construct grids → fit → call, leading dims preserved ────────────────


def test_fit_call_datarray_basic():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    out = rg(da)
    assert isinstance(out, xr.DataArray)
    assert out.shape == dst.shape
    assert set(out.dims) == set(dst.dims)


def test_leading_dims_preserved():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src, leading=(3,))  # (extra0, lat, lon)
    out = rg(da)
    assert out.dims[0] == "extra0"
    assert out.shape[0] == 3
    assert out.shape[1:] == dst.shape


def test_multiple_leading_dims():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "conservative")
    da = _field(src, leading=(4, 2))  # (a, b, lat, lon)
    out = rg(da)
    assert out.shape[:-2] == (4, 2)
    assert out.shape[-2:] == dst.shape


# ─── AC2: target coords attached, attrs kept (FR-013/FR-014) ──────────────────


def test_target_coords_attached():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    # give dst grids real coord values so attachment is observable
    out = rg(da)
    for d in dst.dims:
        assert d in out.coords, f"target dim {d!r} coord missing from output"


def test_attrs_preserved_by_default():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    out = rg(da)
    assert out.attrs.get("units") == "K"
    assert out.attrs.get("long_name") == "sine-bell"


def test_keep_attrs_false():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    out = rg(da, keep_attrs=False)
    assert out.attrs == {} or "units" not in out.attrs


# ─── AC3: NumPy in → NumPy out (FR-016) ───────────────────────────────────────


def test_numpy_path():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    arr = np.asarray(_field(src).values)
    out = rg(arr)
    assert isinstance(out, np.ndarray)
    assert out.shape == dst.shape


def test_numpy_leading_dims():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    arr = np.broadcast_to(np.asarray(_field(src).values), (5,) + src.shape).copy()
    out = rg(arr)
    assert out.shape == (5,) + dst.shape


# ─── AC4: Dataset passthrough (FR-015) ────────────────────────────────────────


def test_dataset_passthrough():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    ds = xr.Dataset(
        {
            "air": _field(src),
            "const": ("time", np.arange(4.0)),  # not on source dims → passthrough
        },
    )
    out = rg(ds)
    assert isinstance(out, xr.Dataset)
    assert out["air"].shape[-2:] == dst.shape
    assert "const" in out.variables
    assert out["const"].shape == (4,)


# ─── AC5: fail-fast impossible pairing (FR-010) ───────────────────────────────


def test_impossible_pairing_raises_at_construction():
    # bicubic onto a point cloud is structurally impossible (data-model §2).
    pts = Grid.from_points(np.linspace(0, 360, 50), np.linspace(-80, 80, 50))
    src = _rect_src()
    with pytest.raises(axis.AxisConfigError):
        axis.Regridder(src, pts, "bicubic")


# ─── AC6: transform/regrid aliases behave identically ─────────────────────────


def test_aliases_equivalent():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src)
    a = rg(da)
    b = rg.transform(da)
    c = rg.regrid(da)
    np.testing.assert_allclose(a.values, b.values)
    np.testing.assert_allclose(a.values, c.values)


# ─── SC-001: ≤3-call ergonomics ───────────────────────────────────────────────


def test_three_call_ergonomics():
    # construct grid(s) + fit + apply — at most 3 calls to go from data to output.
    rg = axis.Regridder(_rect_src(), _rect_dst(), "bilinear")
    out = rg(_field(_rect_src()))
    assert out.shape == _rect_dst().shape


# ─── SC-004: zero weight regeneration on reuse ────────────────────────────────


def test_fit_once_reuse_no_regeneration(monkeypatch):
    from axis import _core

    calls = {"n": 0}
    real = _core.generate_weights

    def _counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_core, "generate_weights", _counting)
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    assert calls["n"] == 1, "fit should generate weights exactly once"
    da = _field(src)
    for _ in range(5):
        rg(da)
    assert calls["n"] == 1, "repeated apply must not regenerate weights (SC-004)"


# ─── FR-018: dtype preservation ───────────────────────────────────────────────


def test_float32_preserved():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    da = _field(src, dtype=np.float32)
    out = rg(da)
    assert out.dtype == np.float32


def test_float64_default():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    out = rg(_field(src, dtype=np.float64))
    assert out.dtype == np.float64


# ─── FR-017: unmapped policy triad ────────────────────────────────────────────


def _unmapped_pair():
    """Regional source, global target: polar rows have no source coverage."""
    src = Grid(lon=np.arange(0.0, 360.0, 10.0), lat=np.arange(-40.0, 41.0, 10.0))
    dst = Grid(lon=np.arange(0.0, 360.0, 5.0), lat=np.arange(-85.0, 86.0, 5.0))
    return src, dst


def test_unmapped_nan_default():
    src, dst = _unmapped_pair()
    rg = axis.Regridder(src, dst, "bilinear", unmapped="nan")
    out = rg(_field(src))
    vals = np.asarray(out.values)
    assert np.isnan(vals).any(), "polar rows beyond the regional source must be NaN"
    assert not np.isnan(vals).all()


def test_unmapped_error_raises_when_unmapped():
    src, dst = _unmapped_pair()
    with pytest.raises(axis.AxisUnmappedError):
        axis.Regridder(src, dst, "bilinear", unmapped="error")


def test_unmapped_mask_produces_masked_output():
    src, dst = _unmapped_pair()
    rg = axis.Regridder(src, dst, "bilinear", unmapped="mask")
    out = rg(_field(src))
    vals = np.asarray(out.values)
    assert np.isnan(vals).any(), "mask policy must mark unmapped cells invalid"
    assert "unmapped" in out.coords, "mask policy attaches a boolean 'unmapped' coord"
    mark = np.asarray(out.coords["unmapped"].values)
    assert mark.any() and not mark.all()
    assert np.isnan(vals[mark]).all()


# ─── T021: method × family support matrix (data-model §2) ─────────────────────

ALL_METHODS = ["bilinear", "bicubic", "patch", "nearest", "conservative", "conservative2nd"]


@pytest.fixture
def family_grids():
    """One representative grid per family, for the matrix tests."""
    rect = Grid(lon=np.arange(0.5, 360.0, 4.0), lat=np.arange(-88.5, 89.0, 4.0))
    LON, LAT = np.meshgrid(np.arange(0.5, 360.0, 6.0), np.arange(-84.0, 85.0, 6.0))
    curv = Grid(lon=LON + 2 * np.sin(np.radians(LAT)), lat=LAT)
    cs = Grid("C6")
    # UGRID quads (2x2 cells)
    node_lon = np.array([0, 10, 20, 0, 10, 20, 0, 10, 20], float)
    node_lat = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20], float)
    offs = np.array([0, 4, 8, 12, 16], np.int64)
    inds = np.array([0, 1, 4, 3, 1, 2, 5, 4, 3, 4, 7, 6, 4, 5, 8, 7], np.int64)
    ugrid = Grid.from_ugrid(np.column_stack([node_lon, node_lat]), offs, inds)
    tri_offs = np.array([0, 3, 6, 9, 12, 15, 18, 21, 24], np.int64)
    tri_inds = np.array(
        [0, 1, 3, 1, 4, 3, 1, 2, 4, 2, 5, 4, 3, 4, 6, 4, 7, 6, 3, 6, 7, 4, 5, 7],
        np.int64,
    )
    icon = Grid.from_ugrid(np.column_stack([node_lon, node_lat]), tri_offs, tri_inds)
    points = Grid.from_points(np.linspace(0, 360, 40), np.linspace(-80, 80, 40))
    return {"RECT": rect, "CURV": curv, "CS": cs, "UGRID": ugrid, "ICON": icon, "POINTS": points}


# The matrix from data-model.md §2 (per-family support; a pairing is valid iff
# the method is supported on BOTH families).
SUPPORTED = {
    "bilinear": {"RECT", "CURV", "CS", "UGRID", "ICON", "POINTS"},
    "nearest": {"RECT", "CURV", "CS", "UGRID", "ICON", "POINTS"},
    "bicubic": {"RECT", "CURV", "CS"},
    "patch": {"RECT", "CURV", "CS"},
    "conservative": {"RECT", "CURV", "CS", "UGRID", "ICON"},
    "conservative2nd": {"RECT", "CURV", "CS", "UGRID", "ICON"},
}


@pytest.mark.parametrize("method", ALL_METHODS)
def test_all_methods_rect_to_curv(method, family_grids):
    # rect↔curv are universally supported (both in every method's set).
    rg = axis.Regridder(family_grids["RECT"], family_grids["CURV"], method)
    assert rg.fitted is True


@pytest.mark.parametrize("method", ALL_METHODS)
@pytest.mark.parametrize("family", ["RECT", "CURV", "CS", "UGRID", "ICON", "POINTS"])
def test_matrix_as_target(method, family, family_grids):
    """method onto a family: supported iff family ∈ SUPPORTED[method]."""
    src = family_grids["RECT"]
    dst = family_grids[family]
    ok = family in SUPPORTED[method]
    if ok:
        rg = axis.Regridder(src, dst, method)
        assert rg.fitted is True
    else:
        with pytest.raises(axis.AxisConfigError):
            axis.Regridder(src, dst, method)


@pytest.mark.parametrize("method", ["bicubic", "patch"])
@pytest.mark.parametrize("family", ["UGRID", "ICON", "POINTS"])
def test_matrix_as_source_rejected(method, family, family_grids):
    """Unsupported source families raise regardless of target."""
    src = family_grids[family]
    dst = family_grids["RECT"]
    with pytest.raises(axis.AxisConfigError):
        axis.Regridder(src, dst, method)


@pytest.mark.parametrize("method", ["conservative", "conservative2nd"])
def test_matrix_points_source_rejected(method, family_grids):
    src = family_grids["POINTS"]
    dst = family_grids["RECT"]
    with pytest.raises(axis.AxisConfigError):
        axis.Regridder(src, dst, method)


# ─── US1 Independent Test: sine-bell analytic accuracy ───────────────────────


def _sine_bell(lon2d, lat2d):
    return (np.cos(np.radians(lat2d)) ** 2) * (1.0 + 0.5 * np.cos(np.radians(lon2d)))


def test_bilinear_identity_exact():
    """Self-regrid must reproduce the source field bit-for-bit (exact weights)."""
    src = Grid(lon=np.arange(0.5, 360.0, 1.0), lat=np.linspace(-89.5, 89.5, 180))
    LON, LAT = np.meshgrid(np.asarray(src._payload["lon"]), np.asarray(src._payload["lat"]))
    da = xr.DataArray(_sine_bell(LON, LAT), dims=src.dims)
    rg = axis.Regridder(src, src, "bilinear")
    out = rg(da)
    np.testing.assert_allclose(out.values, da.values, atol=1e-15)


def test_bilinear_matches_analytic_within_truncation():
    """1° → 0.25° bilinear on a smooth sine bell: error bounded by the method's
    O(h²) truncation on the analysis grid (≈1e-4), not by the engine."""
    src = Grid(lon=np.arange(0.5, 360.0, 1.0), lat=np.linspace(-89.5, 89.5, 180))
    dst = Grid(lon=np.arange(0.125, 360.0, 0.25), lat=np.linspace(-89.875, 89.875, 720))
    LON, LAT = np.meshgrid(np.asarray(src._payload["lon"]), np.asarray(src._payload["lat"]))
    da = xr.DataArray(_sine_bell(LON, LAT), dims=src.dims)
    rg = axis.Regridder(src, dst, "bilinear")
    out = rg(da)
    LON2, LAT2 = np.meshgrid(np.asarray(dst._payload["lon"]), np.asarray(dst._payload["lat"]))
    err = np.abs(out.values - _sine_bell(LON2, LAT2))
    assert err.max() < 5e-4, f"bilinear truncation too large: {err.max()}"


def test_conservative_preserves_global_mean():
    """Conservative 1st-order on a global-to-global pair preserves the areally
    weighted mean within the engine's conservation tolerance (SC-003)."""
    src = Grid(lon=np.arange(0.5, 360.0, 2.0), lat=np.linspace(-89.0, 89.0, 90))
    dst = Grid(lon=np.arange(0.25, 360.0, 0.5), lat=np.linspace(-89.75, 89.75, 360))
    LON, LAT = np.meshgrid(np.asarray(src._payload["lon"]), np.asarray(src._payload["lat"]))
    da = xr.DataArray(_sine_bell(LON, LAT), dims=src.dims)
    rg = axis.Regridder(src, dst, "conservative")
    out = rg(da)
    LON2, LAT2 = np.meshgrid(np.asarray(dst._payload["lon"]), np.asarray(dst._payload["lat"]))
    w_src = np.cos(np.radians(LAT))
    w_dst = np.cos(np.radians(LAT2))
    mean_in = float((da.values * w_src).sum() / w_src.sum())
    mean_out = float((np.nan_to_num(out.values) * w_dst).sum() / w_dst.sum())
    assert abs(mean_out - mean_in) < 1e-10 * max(1.0, abs(mean_in)) or abs(mean_out - mean_in) / abs(mean_in) < 1e-6


@pytest.mark.parametrize("name", ["F64", "F128", "C96", "R6", "O64"])
def test_conservative_into_named_grid_is_nonempty(name: str) -> None:
    """Conservative regridding must produce weights when a *named* grid is the
    destination (regression).

    The reduced-Gaussian (F/O) generators once emitted cell rings wound
    clockwise in lon/lat while the rest of the engine (R/C, and the Sutherland-
    Hodgman clipper's winding-derived inside-tests) assumes counter-clockwise.
    That silently yielded an all-zero weight matrix (``nnz == 0``) for any
    named destination in the F/O families, which the default ``unmapped`` policy
    then rendered as all-NaN. A constant field must regrid to that same constant
    wherever the destination cell is covered by the source (partition of unity).
    """
    src = Grid(lon=np.arange(0.5, 360.0, 2.0), lat=np.arange(-89.5, 90.0, 2.0))
    da = xr.DataArray(np.ones(src.shape, dtype=np.float64), dims=src.dims)
    rg = axis.Regridder(src, Grid(name), "conservative")
    assert rg.nnz > 0, f"conservative into {name} produced an empty weight matrix"
    out = rg(da)
    mapped = out.values[np.isfinite(out.values)]
    assert mapped.size > 0.9 * out.values.size, f"{name}: too few destination cells covered"
    np.testing.assert_allclose(mapped, 1.0, rtol=1e-10, atol=1e-10)


# ─── introspection (FR-019) ───────────────────────────────────────────────────


def test_summary_and_repr():
    src, dst = _rect_src(), _rect_dst()
    rg = axis.Regridder(src, dst, "bilinear")
    s = rg.summary()
    assert isinstance(s, str) and s
    assert "bilinear" in repr(rg)
    assert rg.nnz > 0
    assert rg.fitted is True


def test_method_string_coercion():
    src, dst = _rect_src(), _rect_dst()
    a = axis.Regridder(src, dst, "bilinear")
    b = axis.Regridder(src, dst, axis.Method.BILINEAR)
    da = _field(src)
    np.testing.assert_allclose(a(da).values, b(da).values)
