# SPDX-License-Identifier: Apache-2.0
"""Thin ``.axis`` xarray accessor sugar over Grid/Regridder (R14).

``da.axis.to(target, method=...)`` regrids in one call; ``da.axis.fit(target,
method=...)`` returns a reusable :class:`~axis.Regridder` without applying it.
No independent logic lives here — everything delegates to the library.
"""

from __future__ import annotations

from typing import Any, cast

import xarray as xr

from .grid import Grid
from .regridder import Regridder


def _fit(
    obj: xr.DataArray | xr.Dataset, target: Grid | Regridder | xr.Dataset | xr.DataArray | dict[str, Any] | str, **kwargs: Any
) -> Regridder:
    if isinstance(target, Regridder):
        return target
    return Regridder(obj, target, **kwargs)


@xr.register_dataarray_accessor("axis")  # type: ignore[no-untyped-call]
class RegridDataArrayAccessor:
    """``da.axis`` — one-line regridding for DataArrays."""

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        self._obj = xarray_obj

    def to(self, target: Grid | Regridder | xr.Dataset | xr.DataArray | dict[str, Any] | str, **kwargs: Any) -> xr.DataArray:
        """Regrid to ``target`` (grid descriptor) or apply a pre-fitted Regridder."""
        if isinstance(target, Regridder):
            return cast(xr.DataArray, target(self._obj))
        return cast(xr.DataArray, _fit(self._obj, target, **kwargs)(self._obj))

    def fit(self, target: Grid | xr.Dataset | xr.DataArray | dict[str, Any] | str, **kwargs: Any) -> Regridder:
        """Return a fitted Regridder from this array's grid to ``target``."""
        return Regridder(self._obj, target, **kwargs)


@xr.register_dataset_accessor("axis")  # type: ignore[no-untyped-call]
class RegridDatasetAccessor:
    """``ds.axis`` — one-line regridding for Datasets."""

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        self._obj = xarray_obj

    def to(self, target: Grid | Regridder | xr.Dataset | xr.DataArray | dict[str, Any] | str, **kwargs: Any) -> xr.Dataset:
        """Regrid to ``target`` (grid descriptor) or apply a pre-fitted Regridder."""
        if isinstance(target, Regridder):
            return cast(xr.Dataset, target(self._obj))
        return cast(xr.Dataset, _fit(self._obj, target, **kwargs)(self._obj))

    def fit(self, target: Grid | xr.Dataset | xr.DataArray | dict[str, Any] | str, **kwargs: Any) -> Regridder:
        """Return a fitted Regridder from this dataset's grid to ``target``."""
        return Regridder(self._obj, target, **kwargs)
