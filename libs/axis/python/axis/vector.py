# SPDX-License-Identifier: Apache-2.0
"""Rotation-aware paired u/v regridding (FR-021…FR-023).

``VectorRegridder`` compiles a coupled (W_u, W_v) operator between two grids.
Grid-frame rotation happens in the C++ engine: per-cell east-vector
orientation angles default to ``compute_rotation_angles`` (R8/FR-021) and may
be overridden with explicit ``src_alpha``/``dst_alpha`` arrays (FR-022). Only
the interpolation methods the rotation machinery can rotate correctly
(bilinear, nearest) are accepted; conservative orders raise
``AxisConfigError`` at fit (FR-023) — never a silent scalar fallback.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, cast

import numpy as np
import xarray as xr

from . import _core
from .errors import AxisConfigError, AxisShapeError
from .grid import Grid
from .regridder import _as_line_type
from .types import GridFamily, LineType, Method

logger = logging.getLogger("axis")

_VECTOR_METHODS = frozenset({Method.BILINEAR, Method.NEAREST})
_METHOD_MAP = {Method.BILINEAR: _core.Method.Bilinear, Method.NEAREST: _core.Method.NearestNeighbor}


def _as_method(method: Method | str) -> Method:
    try:
        m = Method(str(method).lower())
    except ValueError:
        raise AxisConfigError(f"unknown method {method!r}; choose from {', '.join(x.value for x in Method)}") from None
    if m not in _VECTOR_METHODS:
        raise AxisConfigError(
            f"vector regridding supports {', '.join(sorted(x.value for x in _VECTOR_METHODS))} only; "
            f"{m.value!r} has no correct grid-frame rotation (FR-023)"
        )
    return m


def _as_grid(obj: Any) -> Grid:
    return obj if isinstance(obj, Grid) else Grid(obj)


def _mesh_variant(grid: Grid, method: Method) -> str:
    if grid.family in (GridFamily.UGRID, GridFamily.ICON) and method is Method.BILINEAR:
        return "tri"
    return "poly"


def _norm_alpha(alpha: Any, n: int, name: str) -> np.ndarray | None:
    """Validate/flatten an explicit angle override to float64 (n,); None passes through."""
    if alpha is None:
        return None
    arr = np.asarray(alpha.data if isinstance(alpha, xr.DataArray) else alpha, dtype=np.float64).ravel()
    if arr.size != n:
        raise AxisShapeError(f"{name} has {arr.size} entries, grid has {n} cells")
    return arr


class _VectorApplier:
    """Picklable coupled SpMV functor shipping the (W_u, W_v) blobs to workers.

    Kernel contract: input ``uv`` of shape ``(*leading, 2, *shape_source)`` →
    output of the same layout with ``shape_target`` substituted. The component
    axis (u=0, v=1) sits immediately before the spatial dims.
    """

    __slots__ = ("bytes_u", "bytes_v", "shape_source", "shape_target", "key_u", "key_v")

    def __init__(
        self,
        *,
        bytes_u: bytes,
        bytes_v: bytes,
        shape_source: tuple[int, ...],
        shape_target: tuple[int, ...],
        key_u: str,
        key_v: str,
    ) -> None:
        self.bytes_u = bytes_u
        self.bytes_v = bytes_v
        self.shape_source = shape_source
        self.shape_target = shape_target
        self.key_u = key_u
        self.key_v = key_v

    def _matrices(self) -> tuple[_core.Matrix, _core.Matrix]:
        from .core import _WORKER_CACHE
        from .distributed import install_weights

        mu = _WORKER_CACHE.get(self.key_u)
        if mu is None:
            install_weights(self.key_u, self.bytes_u)
            mu = _WORKER_CACHE[self.key_u]
        mv = _WORKER_CACHE.get(self.key_v)
        if mv is None:
            install_weights(self.key_v, self.bytes_v)
            mv = _WORKER_CACHE[self.key_v]
        return cast(_core.Matrix, mu), cast(_core.Matrix, mv)

    def __call__(self, uv: np.ndarray) -> np.ndarray:
        n_si = len(self.shape_source)  # spatial dims on input
        n_so = len(self.shape_target)  # spatial dims on output
        if uv.shape[-n_si - 1] != 2:
            raise AxisShapeError(f"vector kernel expects a trailing component axis of size 2, got shape {uv.shape}")
        n_src = int(np.prod(self.shape_source))
        leading = uv.shape[:-n_si]  # (*leading, 2)
        n_other = int(np.prod(leading[:-1])) if leading[:-1] else 1
        # move the component axis to the front, then (n_other, n_src) per component
        comp_first = np.moveaxis(uv, -n_si - 1, 0)  # (2, *leading, *spatial)
        u = np.ascontiguousarray(comp_first[0].reshape(n_other, n_src))
        v = np.ascontiguousarray(comp_first[1].reshape(n_other, n_src))
        # engine wants (2*n_src, n_vars): u rows then v rows
        uv_flat = np.asfortranarray(np.concatenate([u.T, v.T], axis=0))
        mu, mv = self._matrices()
        out_u = np.array(_core.batch_apply(mu, uv_flat), dtype=uv.dtype).T  # (n_other, n_dst)
        out_v = np.array(_core.batch_apply(mv, uv_flat), dtype=uv.dtype).T
        rest = leading[:-1] + tuple(self.shape_target)
        return np.stack([out_u.reshape(rest), out_v.reshape(rest)], axis=-n_so - 1)


class VectorRegridder:
    """Coupled, rotation-aware horizontal vector interpolator.

    Example:
        >>> rg = VectorRegridder(cs_grid, target_grid, "bilinear")
        >>> u_out, v_out = rg(u_da, v_da)   # components stay paired
    """

    def __init__(
        self,
        source: Grid | xr.Dataset | xr.DataArray | dict[str, Any] | str,
        target: Grid | xr.Dataset | xr.DataArray | dict[str, Any] | str,
        method: Method | str = "bilinear",
        *,
        src_alpha: np.ndarray | xr.DataArray | None = None,
        dst_alpha: np.ndarray | xr.DataArray | None = None,
        periodic: bool | None = None,
        line_type: LineType | str | None = None,
    ) -> None:
        self.source = _as_grid(source)
        self.target = _as_grid(target)
        self.method = _as_method(method)
        self.periodic = self.source.periodic and self.target.periodic if periodic is None else bool(periodic)
        self.line_type = self.source.line_type if line_type is None else _as_line_type(line_type)
        self._uid = str(uuid.uuid4())
        self._dims_source = self.source.dims
        self._dims_target = self.target.dims
        self._shape_source = self.source.shape
        self._shape_target = self.target.shape
        self._n_src = self.source.n_cells
        self._n_dst = self.target.n_cells
        self._fit(src_alpha, dst_alpha)

    def _fit(self, src_alpha: np.ndarray | xr.DataArray | None, dst_alpha: np.ndarray | xr.DataArray | None) -> None:
        src_mesh = self.source.to_mesh(_mesh_variant(self.source, self.method))
        dst_mesh = self.target.to_mesh(_mesh_variant(self.target, self.method))
        sa = _norm_alpha(src_alpha, self._n_src, "src_alpha")
        da_ = _norm_alpha(dst_alpha, self._n_dst, "dst_alpha")
        config: dict[str, Any] = {
            "method": _METHOD_MAP[self.method],
            "periodic": self.periodic,
            "line_type": _core.LineType.GreatCircle if self.line_type == "great_circle" else _core.LineType.Cartesian,
        }
        t0 = time.perf_counter()
        self._W_u, self._W_v = _core.generate_vector_weights(src_mesh, dst_mesh, sa, da_, config)
        self._bytes_u = self._W_u.to_bytes()
        self._bytes_v = self._W_v.to_bytes()
        logger.info(
            "AXIS: vector fit %s %s%s->%s%s (%d->%d), nnz=%d, %.3f MB in %.3fs",
            self.method.value,
            self.source.family.value,
            self._shape_source,
            self.target.family.value,
            self._shape_target,
            self._n_src,
            self._n_dst,
            self.nnz,
            (len(self._bytes_u) + len(self._bytes_v)) / (1024 * 1024),
            time.perf_counter() - t0,
        )

    # ─── introspection ────────────────────────────────────────────────────────

    @property
    def nnz(self) -> int:
        return int(self._W_u.nnz) + int(self._W_v.nnz)

    def summary(self) -> str:
        return (
            f"axis.VectorRegridder(method={self.method.value!r}, "
            f"{self.source.family.value}{self._shape_source} -> {self.target.family.value}{self._shape_target}, "
            f"src={self._n_src} dst={self._n_dst} nnz={self.nnz})"
        )

    def __repr__(self) -> str:
        return self.summary()

    # ─── apply ────────────────────────────────────────────────────────────────

    def __call__(self, u: Any, v: Any, *, keep_attrs: bool = True) -> Any:
        if isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray):
            return self._transform_xarray(u, v, keep_attrs=keep_attrs)
        return self._transform_numpy(np.asarray(u), np.asarray(v))

    transform = __call__

    def _check_pair(self, u: np.ndarray, v: np.ndarray) -> None:
        if u.shape != v.shape:
            raise AxisShapeError(f"u and v must have matching shapes, got {u.shape} and {v.shape}")
        n_sd = len(self._shape_source)
        if u.ndim >= n_sd and tuple(u.shape[-n_sd:]) == tuple(self._shape_source):
            return
        if u.size == self._n_src:  # flat (n_cells,) — reshape to the grid layout
            return
        raise AxisShapeError(f"input shape {u.shape} does not match source grid {self._shape_source} ({self._n_src} cells)")

    def _reshape_src(self, a: np.ndarray) -> np.ndarray:
        n_sd = len(self._shape_source)
        if a.ndim >= n_sd and tuple(a.shape[-n_sd:]) == tuple(self._shape_source):
            return a
        return a.reshape(tuple(a.shape[: a.ndim - 1]) + tuple(self._shape_source)) if a.ndim else a.reshape(self._shape_source)

    def _applier(self) -> _VectorApplier:
        from .distributed import worker_key

        return _VectorApplier(
            bytes_u=self._bytes_u,
            bytes_v=self._bytes_v,
            shape_source=self._shape_source,
            shape_target=self._shape_target,
            key_u=worker_key(self._uid, self._bytes_u),
            key_v=worker_key(self._uid, self._bytes_v),
        )

    def _transform_numpy(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_pair(u, v)
        flat_in = u.ndim == 1 and u.size == self._n_src and len(self._shape_source) > 1
        u = self._reshape_src(u)
        v = self._reshape_src(v)
        n_sd = len(self._shape_source)
        uv = np.stack([u, v], axis=u.ndim - n_sd)  # (*leading, 2, *spatial)
        out = self._applier()(np.ascontiguousarray(uv))
        moved = np.moveaxis(out, -len(self._shape_target) - 1, 0)  # (2, *leading, *spatial_target)
        uo, vo = moved[0].copy(), moved[1].copy()
        if flat_in:
            uo = uo.reshape(uo.shape[: uo.ndim - len(self._shape_target)] + (self._n_dst,))
            vo = vo.reshape(vo.shape[: vo.ndim - len(self._shape_target)] + (self._n_dst,))
        return uo, vo

    def _transform_xarray(self, u: xr.DataArray, v: xr.DataArray, *, keep_attrs: bool) -> tuple[xr.DataArray, xr.DataArray]:
        for d, n in zip(self._dims_source, self._shape_source, strict=False):
            if d not in u.dims:
                raise AxisShapeError(f"u is missing source dimension {d!r} (has {u.dims})")
            if u.sizes[d] != n:
                raise AxisShapeError(f"source dim {d!r} size {u.sizes[d]} != grid size {n}")
        if u.dims != v.dims or u.shape != v.shape:
            raise AxisShapeError(f"u and v must share dims/shape, got {u.dims}{u.shape} and {v.dims}{v.shape}")
        # The component axis is a core dim so the kernel sees
        # (*leading, 2, *spatial) exactly and one apply_ufunc yields both.
        uv = xr.concat([u, v], dim="comp__axis")
        temp = [f"{d}__axis" for d in self._dims_target]
        applier = self._applier()
        self._provision_distributed(applier)
        out = xr.apply_ufunc(
            applier,
            uv,
            input_core_dims=[["comp__axis", *self._dims_source]],
            output_core_dims=[["comp__axis", *temp]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[u.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {"comp__axis": 2, **dict(zip(temp, self._shape_target, strict=False))},
                "allow_rechunk": True,
            },
        )
        out = out.rename(dict(zip(temp, self._dims_target, strict=False)))
        u_out = _attach(out.isel(comp__axis=0), self.target, u, keep_attrs)
        v_out = _attach(out.isel(comp__axis=1), self.target, v, keep_attrs)
        return u_out, v_out

    def _provision_distributed(self, applier: _VectorApplier) -> None:
        import sys

        dist = sys.modules.get("distributed") or sys.modules.get("dask.distributed")
        if dist is None:
            return
        try:
            client = dist.get_client()
        except Exception:
            return
        if client is None:  # pragma: no cover
            return
        from .distributed import provision_worker

        provision_worker(client, applier.key_u, applier.bytes_u)
        provision_worker(client, applier.key_v, applier.bytes_v)

    # ─── serialization (FR-027…FR-029) ───────────────────────────────────────

    def save_weights(self, path: str | os.PathLike[str]) -> None:
        """Persist the coupled (W_u, W_v) blobs in the native AXISW1 format."""
        from . import weights as _w

        _w.save_weight_set(
            path,
            kind="vector",
            settings={"method": self.method.value, "periodic": self.periodic, "line_type": self.line_type},
            source=self.source,
            target=self.target,
            matrices=[
                (self._W_u.n_src, self._W_u.n_dst, int(self._W_u.nnz)),
                (self._W_v.n_src, self._W_v.n_dst, int(self._W_v.nnz)),
            ],
            blobs=[self._bytes_u, self._bytes_v],
        )

    @classmethod
    def load_weights(cls, path: str | os.PathLike[str], *, source: Grid, target: Grid, **options: Any) -> VectorRegridder:
        obj = cls.__new__(cls)
        from . import weights as _w

        header, blobs = _w.load_weight_set(path, kind="vector", source=source, target=target)
        settings = {**_w.settings_from_header(header), **options}
        obj.source = _as_grid(source)
        obj.target = _as_grid(target)
        obj.method = _as_method(settings["method"])
        obj.periodic = bool(settings.get("periodic", obj.source.periodic and obj.target.periodic))
        obj.line_type = str(settings.get("line_type", obj.source.line_type))
        obj._uid = str(uuid.uuid4())
        obj._dims_source = obj.source.dims
        obj._dims_target = obj.target.dims
        obj._shape_source = obj.source.shape
        obj._shape_target = obj.target.shape
        obj._n_src = obj.source.n_cells
        obj._n_dst = obj.target.n_cells
        obj._bytes_u, obj._bytes_v = blobs
        obj._W_u = _core.Matrix.from_bytes(obj._bytes_u)
        obj._W_v = _core.Matrix.from_bytes(obj._bytes_v)
        return obj


def _attach(out: xr.DataArray, target: Grid, ref: xr.DataArray, keep_attrs: bool) -> xr.DataArray:
    from .regridder import _grid_coords

    attach = {}
    for name, (dims, values) in _grid_coords(target).items():
        if all(d in out.dims for d in dims) and np.asarray(values).shape == tuple(out.sizes[d] for d in dims):
            attach[name] = (dims, values)
    out = out.assign_coords(attach) if attach else out
    out.name = ref.name
    out.attrs = dict(ref.attrs) if keep_attrs else {}
    return out
