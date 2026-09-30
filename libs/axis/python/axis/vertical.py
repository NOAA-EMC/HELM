# SPDX-License-Identifier: Apache-2.0
"""Vertical (per-column) remapping and 3-D composition (FR-024…FR-026).

``VerticalRegridder`` interpolates a field along its vertical axis with a
tension spline evaluated in the C++ engine (``interpolate_vertical`` for
shared 1-D level vectors, ``interpolate_vertical_varying`` for per-column
2-D/3-D level sets — hybrid-sigma to pressure style). Levels may be given in
either direction; ascending order is normalised internally and the requested
output order restored.

Out-of-range target levels (FR-026) are handled explicitly in this layer —
the engine has no policy and its end-segment evaluation is unreliable outside
the source bracket — so ``out_of_range`` ∈ {nan, clip, extrapolate} is never a
silent surprise. ``regrid_3d`` composes a horizontal ``Regridder`` with a
vertical pass in one call (FR-025).
"""

from __future__ import annotations

from typing import Any, Literal, cast

import numpy as np
import xarray as xr

from . import _core
from .errors import AxisConfigError, AxisShapeError
from .types import OutOfRange

_OUT_OF_RANGE = {o.value for o in OutOfRange}


def _as_out_of_range(v: str) -> str:
    try:
        s = str(v).lower()
    except Exception:
        raise AxisConfigError(f"out_of_range must be one of {sorted(_OUT_OF_RANGE)}, got {v!r}") from None
    if s not in _OUT_OF_RANGE:
        raise AxisConfigError(f"out_of_range must be one of {sorted(_OUT_OF_RANGE)}, got {v!r} (FR-026)")
    return s


def _levels_array(levels: Any, name: str) -> np.ndarray:
    arr = np.asarray(levels.data if isinstance(levels, xr.DataArray) else levels, dtype=np.float64)
    if arr.ndim == 0 or 0 in arr.shape:
        raise AxisShapeError(f"{name} must be a non-empty level array")
    return arr


class VerticalRegridder:
    """Per-column vertical tension-spline interpolator.

    Example:
        >>> rg = VerticalRegridder(tension=0.0, out_of_range="nan")
        >>> temp_plev = rg(temp_hlev, src_levs, dst_levs, vertical_dim="lev")
    """

    def __init__(self, *, tension: float = 0.0, out_of_range: Literal["nan", "clip", "extrapolate"] = "nan") -> None:
        tension = float(tension)
        if tension < 0.0:
            raise AxisConfigError(f"tension must be non-negative, got {tension}")
        self.tension = tension
        self.out_of_range = _as_out_of_range(out_of_range)

    def __repr__(self) -> str:
        return f"axis.VerticalRegridder(tension={self.tension}, out_of_range={self.out_of_range!r})"

    # ─── apply ────────────────────────────────────────────────────────────────

    def __call__(
        self,
        field: xr.DataArray | np.ndarray,
        src_levels: Any,
        dst_levels: Any,
        *,
        vertical_dim: str | None = None,
        keep_attrs: bool = True,
    ) -> Any:
        src = _levels_array(src_levels, "src_levels")
        dst = _levels_array(dst_levels, "dst_levels")
        if isinstance(field, xr.DataArray):
            return self._call_xarray(field, src_levels, dst_levels, src, dst, vertical_dim, keep_attrs)
        return self._call_numpy(np.asarray(field), src, dst)

    transform = __call__

    # ─── numpy path ───────────────────────────────────────────────────────────

    def _call_numpy(self, field: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        n_src = src.shape[-1]
        if field.shape[-1] != n_src:
            raise AxisShapeError(f"field trailing dim {field.shape[-1]} != src_levels length {n_src}")
        out = self._interpolate(field, src, dst)
        return out

    def _interpolate(self, field: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Core: normalise direction, engine spline, restore order, apply policy."""
        f = np.ascontiguousarray(field, dtype=np.float64)
        s = np.ascontiguousarray(src, dtype=np.float64)
        d = np.ascontiguousarray(dst, dtype=np.float64)
        n_src = s.shape[-1]
        n_dst = d.shape[-1]
        flat_f = f.reshape(-1, n_src)
        # per-column (or shared) ascending normalisation
        if s.ndim == 1:
            descending = bool(n_src > 1 and s[0] > s[-1])
            if descending:
                s = s[::-1].copy()
                flat_f = flat_f[:, ::-1].copy()
            flat_s = s
        else:
            fs = s.reshape(-1, n_src)
            desc_mask = (fs[:, 0] > fs[:, -1]) if n_src > 1 else np.zeros(fs.shape[0], dtype=bool)
            if desc_mask.any():
                fs = fs.copy()
                flat_f = flat_f.copy()
                fs[desc_mask] = fs[desc_mask, ::-1]
                flat_f[desc_mask] = flat_f[desc_mask, ::-1]
            flat_s = fs
        # destination order: remember original to restore
        if d.ndim == 1:
            dst_desc = bool(n_dst > 1 and d[0] > d[-1])
            d_eval = d[::-1].copy() if dst_desc else d
            flat_d = np.broadcast_to(d_eval, (flat_f.shape[0], n_dst)).copy()
        else:
            fd = d.reshape(-1, n_dst)
            dst_desc = False  # per-column dst: caller chooses order, no global flip
            d_eval = fd
            flat_d = fd
        in_range = self._in_range_mask(flat_s, flat_d)
        if s.ndim == 1 and d.ndim == 1:
            res = np.array(_core.interpolate_vertical(np.ascontiguousarray(flat_f), flat_s, d_eval, self.tension))
            res = res.reshape(flat_f.shape[0], n_dst)
        else:
            res = np.array(
                _core.interpolate_vertical_varying(
                    np.ascontiguousarray(flat_f), np.ascontiguousarray(flat_s), np.ascontiguousarray(flat_d), self.tension
                )
            )
        res = self._apply_policy(res, flat_f, flat_s, flat_d, in_range)
        if dst_desc:
            res = res[:, ::-1].copy()
        return cast(np.ndarray, res.reshape(f.shape[:-1] + (n_dst,)))

    def _in_range_mask(self, flat_s: np.ndarray, flat_d: np.ndarray) -> np.ndarray:
        """Bool (n_col, n_dst): target level lies within the column's bracket."""
        if flat_s.ndim == 1:
            lo, hi = flat_s[0], flat_s[-1]
            return cast(np.ndarray, (flat_d >= lo - 1e-12) & (flat_d <= hi + 1e-12))
        lo = flat_s[:, 0][:, None]
        hi = flat_s[:, -1][:, None]
        return cast(np.ndarray, (flat_d >= lo - 1e-12) & (flat_d <= hi + 1e-12))

    def _apply_policy(
        self, res: np.ndarray, flat_f: np.ndarray, flat_s: np.ndarray, flat_d: np.ndarray, in_range: np.ndarray
    ) -> np.ndarray:
        if self.out_of_range == "nan":
            res = res.copy()
            res[~in_range] = np.nan
            return res
        if self.out_of_range == "clip":
            return self._clip(res, flat_f, flat_s, flat_d, in_range)
        # extrapolate: linear from the nearest interior segment
        return self._extrapolate(res, flat_f, flat_s, flat_d, in_range)

    def _clip(
        self,
        res: np.ndarray,
        flat_f: np.ndarray,
        flat_s: np.ndarray,
        flat_d: np.ndarray,
        in_range: np.ndarray,
    ) -> np.ndarray:
        out = res.copy()
        n_src = flat_s.shape[-1]
        below = flat_d < (flat_s[0] if flat_s.ndim == 1 else flat_s[:, 0][:, None]) - 1e-12
        above = flat_d > (flat_s[-1] if flat_s.ndim == 1 else flat_s[:, -1][:, None]) + 1e-12
        if flat_s.ndim == 1:
            out[below] = flat_f[0, 0]
            out[above] = flat_f[0, n_src - 1]
        else:
            for c in range(flat_f.shape[0]):
                out[c, below[c]] = flat_f[c, 0]
                out[c, above[c]] = flat_f[c, n_src - 1]
        return out

    def _extrapolate(
        self,
        res: np.ndarray,
        flat_f: np.ndarray,
        flat_s: np.ndarray,
        flat_d: np.ndarray,
        in_range: np.ndarray,
    ) -> np.ndarray:
        out = res.copy()
        n_src = flat_s.shape[-1]
        if n_src < 2:
            out[~in_range] = np.nan
            return out
        below = flat_d < (flat_s[0] if flat_s.ndim == 1 else flat_s[:, 0][:, None]) - 1e-12
        above = flat_d > (flat_s[-1] if flat_s.ndim == 1 else flat_s[:, -1][:, None]) + 1e-12
        for c in range(flat_f.shape[0]):
            s = flat_s if flat_s.ndim == 1 else flat_s[c]
            y = flat_f[c]
            # below: linear using first two nodes
            cols = np.where(below if below.ndim == 1 else below[c])[0]
            if cols.size:
                slope = (y[1] - y[0]) / (s[1] - s[0])
                d = flat_d if flat_d.ndim == 1 else flat_d[c]
                out[c, cols] = y[0] + slope * (d[cols] - s[0])
            cols = np.where(above if above.ndim == 1 else above[c])[0]
            if cols.size:
                slope = (y[-1] - y[-2]) / (s[-1] - s[-2])
                d = flat_d if flat_d.ndim == 1 else flat_d[c]
                out[c, cols] = y[-1] + slope * (d[cols] - s[-1])
        return out

    # ─── xarray path ──────────────────────────────────────────────────────────

    def _call_xarray(
        self,
        field: xr.DataArray,
        src_levels: Any,
        dst_levels: Any,
        src: np.ndarray,
        dst: np.ndarray,
        vertical_dim: str | None,
        keep_attrs: bool,
    ) -> xr.DataArray:
        vd = self._resolve_vertical_dim(field, src, vertical_dim)
        if field.sizes[vd] != src.shape[-1]:
            raise AxisShapeError(f"field dim {vd!r} size {field.sizes[vd]} != src_levels length {src.shape[-1]}")
        # internal temp name for the output vertical dim, so a per-column
        # dst level array (whose last dim may also be named ``lev``) does not
        # collide with the source core dim during apply_ufunc alignment.
        dst_name = f"{vd}__axis"
        # transpose vertical to last
        order = [d for d in field.dims if d != vd] + [vd]
        field_t = field.transpose(*order)
        # broadcast/prepare level arrays as core-dim-only DataArrays
        src_da = self._as_level_da(src_levels, src, vd, field_t)
        dst_da = self._as_level_da(dst_levels, dst, dst_name, field_t)
        if isinstance(dst_levels, xr.DataArray):
            dst_da = dst_levels.rename({str(dst_levels.dims[-1]): dst_name})
        out = xr.apply_ufunc(
            self._kernel,
            field_t,
            src_da,
            dst_da,
            input_core_dims=[[vd], [str(src_da.dims[-1])], [dst_name]],
            output_core_dims=[[dst_name]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[field.dtype],
            dask_gufunc_kwargs={"output_sizes": {dst_name: int(dst.shape[-1])}, "allow_rechunk": True},
        )
        # restore the original dimension order (vertical slot renamed if needed)
        target_order = [dst_name if d == vd else d for d in field.dims]
        out = out.transpose(*target_order)
        out = out.rename({dst_name: vd})
        # attach target levels as coordinate
        lev_coord = dst if dst.ndim == 1 else None
        if lev_coord is not None:
            out = out.assign_coords({vd: (vd, lev_coord)})
        out.name = field.name
        out.attrs = dict(field.attrs) if keep_attrs else {}
        return cast(xr.DataArray, out)

    def _resolve_vertical_dim(self, field: xr.DataArray, src: np.ndarray, vertical_dim: str | None) -> str:
        dims = field.dims
        if vertical_dim is not None:
            if vertical_dim not in dims:
                raise AxisShapeError(f"vertical_dim {vertical_dim!r} not in field dims {dims}")
            return vertical_dim
        n_src = src.shape[-1]
        # a dim whose size matches the level count, preferring a coordinate named like it
        candidates = [str(d) for d in dims if field.sizes[str(d)] == n_src]
        if not candidates:
            raise AxisShapeError(f"no field dim matches src_levels length {n_src}; pass vertical_dim explicitly")
        for d in candidates:
            if d in field.coords and field.coords[d].dims == (d,):
                return d
        if len(candidates) == 1:
            return candidates[0]
        raise AxisShapeError(f"ambiguous vertical dim among {candidates}; pass vertical_dim explicitly")

    def _as_level_da(self, levels: Any, arr: np.ndarray, dim_name: str, field_t: xr.DataArray) -> xr.DataArray:
        if isinstance(levels, xr.DataArray):
            return levels
        if arr.ndim == 1:
            return xr.DataArray(arr, dims=(dim_name,))
        # per-column levels share the field's non-vertical dims + dim_name
        spatial = [d for d in field_t.dims if d != dim_name]
        return xr.DataArray(arr, dims=spatial + [dim_name])

    def _kernel(self, field_block: np.ndarray, src_block: np.ndarray, dst_block: np.ndarray) -> np.ndarray:
        return self._interpolate(field_block, src_block, dst_block)


# ─── 3-D composition (FR-025) ─────────────────────────────────────────────────


def regrid_3d(
    field: xr.DataArray | xr.Dataset,
    source: Any,
    target: Any,
    src_levels: Any,
    dst_levels: Any,
    *,
    method: str = "bilinear",
    tension: float = 0.0,
    out_of_range: Literal["nan", "clip", "extrapolate"] = "nan",
    vertical_dim: str | None = None,
    **horizontal_options: Any,
) -> xr.DataArray:
    """One-call horizontal + vertical remapping of a 3-D (or 4-D) field.

    Horizontal regridding is applied to the spatial grid, then the vertical
    spline to the target levels. The vertical dimension is located by
    ``vertical_dim`` (or auto-detected against ``src_levels``); all other
    leading dimensions (time, ensemble) pass through untouched.
    """
    from .regridder import Regridder

    if not isinstance(field, xr.DataArray):
        raise AxisShapeError("regrid_3d expects an xarray DataArray field")
    rg_h = Regridder(source, target, method, **horizontal_options)
    src_arr = _levels_array(src_levels, "src_levels")
    vd = vertical_dim if vertical_dim is not None else _detect_vdim(field, src_arr)
    horizontal = rg_h(field)
    rg_v = VerticalRegridder(tension=tension, out_of_range=out_of_range)
    return cast(xr.DataArray, rg_v(horizontal, src_levels, dst_levels, vertical_dim=vd))


def _detect_vdim(field: xr.DataArray, src: np.ndarray) -> str:
    n_src = src.shape[-1]
    candidates = [str(d) for d in field.dims if field.sizes[str(d)] == n_src]
    if len(candidates) == 1:
        return candidates[0]
    for d in candidates:
        if d in field.coords and field.coords[d].dims == (d,):
            return d
    raise AxisShapeError(f"cannot auto-detect the vertical dim among {candidates}; pass vertical_dim")
