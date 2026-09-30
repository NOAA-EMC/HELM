# SPDX-License-Identifier: Apache-2.0
"""Fit-once / call-many horizontal regridding (FR-009…FR-019).

``Regridder`` compiles source→target weights eagerly in the constructor
(R2) and applies them to xarray, NumPy, or Dask-backed fields. There is no
unfitted state: construction either succeeds with weights ready, or raises
(fail-fast, FR-010).
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
from .core import _apply_weights_core
from .errors import AxisConfigError, AxisShapeError, AxisUnmappedError
from .grid import Grid
from .types import TRIANGULATED_METHODS, GridFamily, LineType, Method, Norm, Unmapped

logger = logging.getLogger("axis")

# method string / enum -> engine Method
_METHOD_MAP: dict[Method, Any] = {
    Method.BILINEAR: _core.Method.Bilinear,
    Method.BICUBIC: _core.Method.Bicubic,
    Method.PATCH: _core.Method.Patch,
    Method.NEAREST: _core.Method.NearestNeighbor,
    Method.CONSERVATIVE: _core.Method.Conservative,
    Method.CONSERVATIVE_2ND: _core.Method.Conservative2ndOrder,
}

_ALL_METHODS = frozenset(_METHOD_MAP)
_UNSTRUCTURED_LOCAL = frozenset({Method.BILINEAR, Method.NEAREST, Method.CONSERVATIVE, Method.CONSERVATIVE_2ND})
_POINTS_METHODS = frozenset({Method.BILINEAR, Method.NEAREST})

# Support matrix (data-model.md §2): which methods a family supports as a
# *source* and as a *target*. A pairing is fit-able iff the method is allowed
# on both ends; otherwise AxisConfigError is raised at construction (FR-010).
_SOURCE_OK: dict[GridFamily, frozenset[Method]] = {
    GridFamily.RECTILINEAR: _ALL_METHODS,
    GridFamily.CURVILINEAR: _ALL_METHODS,
    GridFamily.CUBED_SPHERE: _ALL_METHODS,
    GridFamily.UGRID: _UNSTRUCTURED_LOCAL,
    GridFamily.ICON: _UNSTRUCTURED_LOCAL,
    GridFamily.POINTS: _POINTS_METHODS,
}
_TARGET_OK = _SOURCE_OK


def _as_method(method: Method | str) -> Method:
    try:
        return Method(str(method).lower())
    except ValueError:
        raise AxisConfigError(f"unknown method {method!r}; choose from {', '.join(m.value for m in Method)}") from None


def _as_norm(norm: Norm | str) -> Norm:
    try:
        return Norm(str(norm).lower().replace("-", "_"))
    except ValueError:
        raise AxisConfigError(f"unknown norm {norm!r}; choose from {', '.join(n.value for n in Norm)}") from None


def _as_unmapped(unmapped: Unmapped | str) -> Unmapped:
    try:
        return Unmapped(str(unmapped).lower())
    except ValueError:
        raise AxisConfigError(f"unknown unmapped policy {unmapped!r}; choose from {', '.join(u.value for u in Unmapped)}") from None


def _as_line_type(line_type: LineType | str) -> str:
    try:
        return str(LineType(str(line_type).lower().replace("-", "_")).value)
    except ValueError:
        raise AxisConfigError(f"unknown line_type {line_type!r}; choose from {', '.join(t.value for t in LineType)}") from None


def _as_grid(obj: Any) -> Grid:
    """Wrap any accepted source/target descriptor into a Grid."""
    if isinstance(obj, Grid):
        return obj
    return Grid(obj)


class _Applier:
    """Picklable SpMV functor: the payload Dask ships to each worker.

    Holds plain arrays plus the serialized weight blob, reconstructing the
    engine ``Matrix`` lazily and caching it per worker (R10). Keeping this a
    module-level class — rather than a closure over the live Matrix — is what
    makes the dask path work under the process scheduler.
    """

    __slots__ = ("bytes", "dims_source", "shape_target", "skipna", "na_thres", "total_weights", "unmapped_mask", "key")

    def __init__(
        self,
        *,
        blob: bytes,
        dims_source: tuple[str, ...],
        shape_target: tuple[int, ...],
        skipna: bool,
        na_thres: float,
        total_weights: np.ndarray | None,
        unmapped_mask: np.ndarray | None,
        key: str,
    ) -> None:
        self.bytes = blob
        self.dims_source = dims_source
        self.shape_target = shape_target
        self.skipna = skipna
        self.na_thres = na_thres
        self.total_weights = total_weights
        self.unmapped_mask = unmapped_mask
        self.key = key

    def _matrix(self) -> _core.Matrix:
        from .core import _WORKER_CACHE

        mat = _WORKER_CACHE.get(self.key)
        if mat is None:
            # Lazy in-process provisioning (local schedulers, or a worker that
            # missed the driver's one-time client.run sync): idempotent —
            # parses once and records the sync counter (FR-033).
            from .distributed import install_weights

            install_weights(self.key, self.bytes)
            mat = _WORKER_CACHE[self.key]
        return cast(_core.Matrix, mat)

    def __call__(self, block: np.ndarray) -> np.ndarray:
        out = _apply_weights_core(
            block,
            self._matrix(),
            self.dims_source,
            self.shape_target,
            skipna=self.skipna,
            total_weights=self.total_weights,
            na_thres=self.na_thres,
        )
        mask = self.unmapped_mask
        if mask is not None:
            bad = np.asarray(mask).astype(bool)
            if bad.any():
                out = np.array(out, copy=True)
                out[..., bad.reshape(self.shape_target)] = np.nan
        return out


class Regridder:
    """A fitted source→target weight operator (read-only after construction).

    Example:
        >>> rg = axis.Regridder(source_grid, target_grid, "conservative")
        >>> out = rg(tas_da)          # DataArray | Dataset | ndarray | dask
    """

    def __init__(
        self,
        source: Grid | xr.Dataset | xr.DataArray | dict[str, Any] | str,
        target: Grid | xr.Dataset | xr.DataArray | dict[str, Any] | str,
        method: Method | str = "bilinear",
        *,
        norm: Norm | str = "frac_area",
        unmapped: Unmapped | str = "nan",
        skipna: bool = False,
        na_thres: float = 1.0,
        periodic: bool | None = None,
        line_type: LineType | str | None = None,
        src_mask: np.ndarray | xr.DataArray | None = None,
        dst_mask: np.ndarray | xr.DataArray | None = None,
    ) -> None:
        self.source = _as_grid(source)
        self.target = _as_grid(target)
        self.method = _as_method(method)
        self.norm = _as_norm(norm)
        self.unmapped = _as_unmapped(unmapped)
        self.skipna = bool(skipna)
        self.na_thres = float(na_thres)
        self.periodic = self.source.periodic and self.target.periodic if periodic is None else bool(periodic)
        self.line_type = self.source.line_type if line_type is None else _as_line_type(line_type)
        self._matrix: _core.Matrix | None = None
        self._bytes: bytes | None = None
        self._uid = str(uuid.uuid4())
        self._total_weights: np.ndarray | None = None
        self._unmapped_mask: np.ndarray | None = None
        self._fitted = False
        self._dims_source = self.source.dims
        self._dims_target = self.target.dims
        self._shape_source = self.source.shape
        self._shape_target = self.target.shape
        self._n_src = self.source.n_cells
        self._n_dst = self.target.n_cells
        self._fit(src_mask, dst_mask)

    # ─── fail-fast pairing validation (FR-010) ──────────────────────────────

    def _validate_pairing(self) -> None:
        m = self.method
        if m not in _SOURCE_OK[self.source.family]:
            raise AxisConfigError(
                f"method {m.value!r} cannot run from a {self.source.family.value} source; "
                f"supported source methods: {sorted(x.value for x in _SOURCE_OK[self.source.family])}"
            )
        if m not in _TARGET_OK[self.target.family]:
            raise AxisConfigError(
                f"method {m.value!r} cannot target a {self.target.family.value} grid; "
                f"supported target methods: {sorted(x.value for x in _TARGET_OK[self.target.family])}"
            )

    @staticmethod
    def _mesh_variant(grid: Grid, method: Method) -> str:
        """Polygon families need fan-triangulation for the local-stencil
        methods (bilinear/bicubic/patch); conservative/nearest use raw
        polygons. Every other family has a single canonical mesh."""
        if grid.family in (GridFamily.UGRID, GridFamily.ICON) and method in TRIANGULATED_METHODS:
            return "tri"
        return "poly"

    # ─── eager fit (R2) ──────────────────────────────────────────────────────

    def _fit(self, src_mask: np.ndarray | xr.DataArray | None, dst_mask: np.ndarray | xr.DataArray | None) -> None:
        self._validate_pairing()

        src_cell_mask = None
        if src_mask is not None:
            src_cell_mask = np.asarray(src_mask.data if isinstance(src_mask, xr.DataArray) else src_mask).ravel()
            if src_cell_mask.size != self._n_src:
                raise AxisShapeError(f"src_mask has {src_cell_mask.size} entries, source grid has {self._n_src} cells")
        dst_mask_arr = None
        if dst_mask is not None:
            dst_mask_arr = np.asarray(dst_mask.data if isinstance(dst_mask, xr.DataArray) else dst_mask).ravel().astype(np.int32)
            if dst_mask_arr.size != self._n_dst:
                raise AxisShapeError(f"dst_mask has {dst_mask_arr.size} entries, target grid has {self._n_dst} cells")

        src_mesh = self.source.to_mesh(self._mesh_variant(self.source, self.method), src_cell_mask)
        dst_mesh = self.target.to_mesh(self._mesh_variant(self.target, self.method))

        config: dict[str, Any] = {
            "method": _METHOD_MAP[self.method],
            "periodic": self.periodic,
            "line_type": _core.LineType.GreatCircle if self.line_type == "great_circle" else _core.LineType.Cartesian,
            "norm_type": _core.NormType.DstArea if self.norm is Norm.DST_AREA else _core.NormType.FracArea,
            # "nan" and "mask" both keep unmapped rows valid in the weights; the
            # difference is applied on output (NaN fill vs masked array).
            "unmapped": "error" if self.unmapped is Unmapped.ERROR else "mask",
        }
        if dst_mask_arr is not None:
            config["dst_mask"] = dst_mask_arr

        t0 = time.perf_counter()
        try:
            self._matrix = _core.generate_weights(src_mesh, dst_mesh, config)
        except RuntimeError as exc:
            if "unmapped" in str(exc).lower():
                raise AxisUnmappedError(str(exc)) from None
            raise
        self._bytes = self._matrix.to_bytes()
        from .distributed import weights_fingerprint

        self._fingerprint = weights_fingerprint(self._bytes)
        if self._matrix.has_unmapped_mask:
            self._unmapped_mask = np.asarray(self._matrix.unmapped_mask())
        if self.skipna:
            self._total_weights = np.array(_core.apply_weights(self._matrix, np.ones(self._n_src, dtype=np.float64))).ravel()
        self._fitted = True
        logger.info(
            "AXIS: fit %s %s%s->%s%s (%d->%d), nnz=%d, %.3f MB in %.3fs",
            self.method.value,
            self.source.family.value,
            self._shape_source,
            self.target.family.value,
            self._shape_target,
            self._n_src,
            self._n_dst,
            self.nnz,
            len(self._bytes) / (1024 * 1024),
            time.perf_counter() - t0,
        )

    # ─── introspection (FR-019) ──────────────────────────────────────────────

    @property
    def nnz(self) -> int:
        return int(self._matrix.nnz) if self._matrix is not None else 0

    @property
    def fitted(self) -> bool:
        return self._fitted

    def summary(self) -> str:
        size_mb = len(self._bytes) / (1024 * 1024) if self._bytes else 0.0
        return (
            f"axis.Regridder(method={self.method.value!r}, "
            f"{self.source.family.value}{self._shape_source} -> {self.target.family.value}{self._shape_target}, "
            f"src={self._n_src} dst={self._n_dst} nnz={self.nnz}, weights={size_mb:.3f} MB, "
            f"norm={self.norm.value!r}, unmapped={self.unmapped.value!r}, periodic={self.periodic}, "
            f"line_type={self.line_type!r}, skipna={self.skipna})"
        )

    def __repr__(self) -> str:
        if self._matrix is None:
            return f"<axis.Regridder (unfit, method={self.method.value!r})>"
        return self.summary()

    # ─── call (FR-013…FR-018) ────────────────────────────────────────────────

    def __call__(self, obj: Any, *, keep_attrs: bool = True) -> Any:
        return self.transform(obj, keep_attrs=keep_attrs)

    regrid = __call__

    def transform(self, obj: Any, *, keep_attrs: bool = True) -> Any:
        if not self._fitted or self._matrix is None:  # pragma: no cover
            raise RuntimeError("Regridder is not fitted")
        if isinstance(obj, xr.Dataset):
            ds_out = self._regrid_dataset(obj)
            ds_out.attrs = {**obj.attrs, **ds_out.attrs} if keep_attrs else {}
            return ds_out
        if isinstance(obj, xr.DataArray):
            da_out = self._regrid_dataarray(obj)
            da_out.attrs = {**obj.attrs, **da_out.attrs} if keep_attrs else {}
            return da_out
        if isinstance(obj, np.ndarray) or hasattr(obj, "__array__"):
            return self._regrid_ndarray(np.asarray(obj))
        raise TypeError(f"Regridder input must be DataArray/Dataset/ndarray, got {type(obj).__name__}")

    def _applier(self) -> _Applier:
        from .distributed import worker_key

        blob = self._bytes
        if blob is None:  # pragma: no cover
            raise RuntimeError("Regridder is not fitted")
        return _Applier(
            blob=blob,
            dims_source=self._dims_source,
            shape_target=self._shape_target,
            skipna=self.skipna,
            na_thres=self.na_thres,
            total_weights=self._total_weights,
            unmapped_mask=None if self.unmapped is Unmapped.ERROR else self._unmapped_mask,
            key=worker_key(self._uid, blob),
        )

    def _provision_distributed(self, applier_key: str) -> None:
        """One-time worker sync of the weight blob to an active cluster (FR-032).

        No-op unless a distributed client is running in this process; the
        driver-side guard makes repeat calls free. Local schedulers skip this
        and rely on the applier's lazy in-process install (FR-033).
        """
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

        if self._bytes is not None:
            provision_worker(client, applier_key, self._bytes)

    def _apply_numpy(self, block: np.ndarray) -> np.ndarray:
        return self._applier()(block)

    def _regrid_ndarray(self, arr: np.ndarray) -> np.ndarray:
        n_sd = len(self._shape_source)
        if arr.ndim == 1 and arr.size == self._n_src:
            return self._apply_numpy(arr.reshape((1, self._n_src))).reshape(self._shape_target)
        if arr.ndim < n_sd:
            raise AxisShapeError(f"input has {arr.ndim} dims, fewer than the {n_sd} source spatial dims {self._dims_source}")
        trailing = arr.shape[-n_sd:]
        if int(np.prod(trailing)) != self._n_src:
            raise AxisShapeError(f"trailing dims {trailing} do not match source grid {self._shape_source} ({self._n_src} cells)")
        return self._apply_numpy(arr)

    def _regrid_dataarray(self, da: xr.DataArray) -> xr.DataArray:
        for d, n in zip(self._dims_source, self._shape_source, strict=False):
            if d not in da.dims:
                raise AxisShapeError(f"input is missing source dimension {d!r} (has {da.dims})")
            if da.sizes[d] != n:
                raise AxisShapeError(f"source dim {d!r} size {da.sizes[d]} != grid size {n}")

        # Source and target can share dim names at different sizes; route the
        # output through temporary names, then rename back (xarray requires
        # exclude_dims to change a core dim's size in-place).
        temp_dims = [f"{d}__axis" for d in self._dims_target]
        applier = self._applier()
        self._provision_distributed(applier.key)
        out = xr.apply_ufunc(
            applier,
            da,
            input_core_dims=[list(self._dims_source)],
            output_core_dims=[temp_dims],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={"output_sizes": dict(zip(temp_dims, self._shape_target, strict=False)), "allow_rechunk": True},
        )
        out = out.rename(dict(zip(temp_dims, self._dims_target, strict=False)))
        out = self._attach_coords(out)
        if self.unmapped is Unmapped.MASK and self._unmapped_mask is not None:
            bad = np.asarray(self._unmapped_mask).astype(bool)
            if bad.any():
                # Unmapped cells are already NaN-filled by the applier; "mask"
                # additionally marks them with a boolean ``unmapped`` coord on
                # the target dims (xarray strips numpy masked arrays, so the
                # invalid-mark rides as a coordinate).
                spatial = bad.reshape(self._shape_target)
                shape = (1,) * (out.ndim - len(self._shape_target)) + self._shape_target
                out = out.assign_coords(unmapped=(tuple(self._dims_target), np.broadcast_to(spatial, shape).copy()))
        return out

    def _regrid_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        out_vars: dict[str, Any] = {}
        for name in ds.data_vars:
            da = ds[name]
            if all(d in da.dims for d in self._dims_source):
                out_vars[str(name)] = self._regrid_dataarray(da)
            else:
                out_vars[str(name)] = da
        res = xr.Dataset(out_vars)
        for c in ds.coords:  # carry non-spatial coords through
            if c not in res.coords and not set(ds[c].dims) & set(self._dims_source):
                res = res.assign_coords({c: ds[c]})
        return res

    def _attach_coords(self, out: xr.DataArray) -> xr.DataArray:
        attach = {}
        for name, (dims, values) in _grid_coords(self.target).items():
            if all(d in out.dims for d in dims) and np.asarray(values).shape == tuple(out.sizes[d] for d in dims):
                attach[name] = (dims, values)
        return out.assign_coords(attach) if attach else out

    # ─── serialization (FR-027…FR-029) ───────────────────────────────────────

    def _settings(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "norm": self.norm.value,
            "unmapped": self.unmapped.value,
            "skipna": self.skipna,
            "na_thres": self.na_thres,
            "periodic": self.periodic,
            "line_type": self.line_type,
        }

    def save_weights(self, path: str | os.PathLike[str]) -> None:
        """Persist the compiled weights in the native AXISW1 format (FR-027)."""
        from . import weights as _w

        if self._bytes is None or self._matrix is None:  # pragma: no cover
            raise RuntimeError("no weights to save")
        _w.save_weight_set(
            path,
            kind="scalar",
            settings=self._settings(),
            source=self.source,
            target=self.target,
            matrices=[(self._n_src, self._n_dst, int(self._matrix.nnz))],
            blobs=[self._bytes],
        )

    @classmethod
    def load_weights(cls, path: str | os.PathLike[str], *, source: Grid, target: Grid, **options: Any) -> Regridder:
        """Rebuild a fitted Regridder from a native weight file (FR-028).

        Generation-time settings are restored from the header (truthful
        summary, AC3); any keyword in ``options`` overrides them. The declared
        grids are fingerprint-validated before any data is touched (FR-029)."""
        from . import weights as _w

        header, blobs = _w.load_weight_set(path, kind="scalar", source=source, target=target)
        settings = {**_w.settings_from_header(header), **options}
        method = settings.pop("method")
        obj = cls.__new__(cls)
        obj._init_meta(source, target, method, **settings)
        obj._matrix = _core.Matrix.from_bytes(blobs[0])
        obj._bytes = blobs[0]
        from .distributed import weights_fingerprint

        obj._fingerprint = weights_fingerprint(blobs[0])
        obj._unmapped_mask = np.asarray(obj._matrix.unmapped_mask()) if obj._matrix.has_unmapped_mask else None
        if obj.skipna:
            obj._total_weights = np.array(_core.apply_weights(obj._matrix, np.ones(obj._n_src, dtype=np.float64))).ravel()
        obj._fitted = True
        return obj

    def to_esmf(self, path: str | os.PathLike[str]) -> None:
        """Export portable ESMF/SCRIP NetCDF with AXIS fingerprints (FR-028, SC-009)."""
        from . import weights as _w

        if self._matrix is None:  # pragma: no cover
            raise RuntimeError("not fitted")
        _w.write_esmf_file(
            path, self._matrix, source=self.source, target=self.target, method=self.method.value, norm=self.norm.value
        )

    @classmethod
    def from_esmf(cls, path: str | os.PathLike[str], *, source: Grid, target: Grid, **options: Any) -> Regridder:
        """Import an ESMF/SCRIP weight file (AXIS- or xESMF-produced, SC-009)."""
        from . import weights as _w

        matrix, attrs = _w.read_esmf_file(path, source=source, target=target)
        method = options.pop("method", attrs.get("axis_method", "bilinear"))
        if "axis_norm" in attrs:
            options.setdefault("norm", attrs["axis_norm"])
        obj = cls.__new__(cls)
        obj._init_meta(source, target, method, **options)
        obj._matrix = matrix
        obj._bytes = matrix.to_bytes()
        from .distributed import weights_fingerprint

        obj._fingerprint = weights_fingerprint(obj._bytes)
        obj._unmapped_mask = None
        obj._fitted = True
        return obj

    def _init_meta(self, source: Any, target: Any, method: Any, **options: Any) -> None:
        self.source = _as_grid(source)
        self.target = _as_grid(target)
        self.method = _as_method(method)
        self.norm = _as_norm(options.get("norm", "frac_area"))
        self.unmapped = _as_unmapped(options.get("unmapped", "nan"))
        self.skipna = bool(options.get("skipna", False))
        self.na_thres = float(options.get("na_thres", 1.0))
        periodic = options.get("periodic")
        self.periodic = self.source.periodic and self.target.periodic if periodic is None else bool(periodic)
        line_type = options.get("line_type")
        self.line_type = self.source.line_type if line_type is None else _as_line_type(line_type)
        self._uid = str(uuid.uuid4())
        self._total_weights = None
        self._dims_source = self.source.dims
        self._dims_target = self.target.dims
        self._shape_source = self.source.shape
        self._shape_target = self.target.shape
        self._n_src = self.source.n_cells
        self._n_dst = self.target.n_cells

    # ─── categorical remap (R15) ─────────────────────────────────────────────

    def regrid_categorical(
        self, da: xr.DataArray, categories: list[Any] | dict[Any, Any] | None = None, *, prefix: str = "fraction_"
    ) -> xr.Dataset:
        if not isinstance(da, xr.DataArray):
            raise TypeError("categorical input must be an xarray.DataArray")
        if categories is None:
            flat = np.asarray(da.values).ravel()
            uniq = np.unique(flat[~np.isnan(flat)]) if np.issubdtype(flat.dtype, np.floating) else np.unique(flat)
            mapping = {int(v): str(int(v)) for v in uniq}
        elif isinstance(categories, dict):
            mapping = categories
        else:
            mapping = {int(v): str(int(v)) for v in categories}
        out_vars = {f"{prefix}{name}": self((da == cat_id).astype(float), keep_attrs=False) for cat_id, name in mapping.items()}
        return xr.Dataset(out_vars)


# ─── target-coordinate helpers ────────────────────────────────────────────────


def _grid_coords(grid: Grid) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    """lon/lat coordinate arrays for a grid, keyed by name → (dims, values)."""
    p = grid._payload
    dims = tuple(str(d) for d in grid.dims)
    kind = p["kind"]
    out: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    if kind == "rect":
        out[dims[0]] = (dims[:1], p["lat"])
        out[dims[1]] = (dims[1:], p["lon"])
    elif kind in ("curv", "cs"):
        out["lat"] = (dims, p["lat"])
        out["lon"] = (dims, p["lon"])
    elif kind == "named":
        if p.get("projected"):
            out["x"] = (dims, p["x"])
            out["y"] = (dims, p["y"])
        elif "lon" in p and np.ndim(p["lon"]) == 1 and len(dims) == 2:
            out[dims[0]] = (dims[:1], p["lat"])
            out[dims[1]] = (dims[1:], p["lon"])
        elif "lon" in p:
            out["lat"] = (dims, np.asarray(p["lat"]))
            out["lon"] = (dims, np.asarray(p["lon"]))
    elif kind == "csr" and p.get("location") == "node":
        coords = p["node_coords"]
        out["lat"] = (dims, coords[:, 1])
        out["lon"] = (dims, coords[:, 0])
    return out
