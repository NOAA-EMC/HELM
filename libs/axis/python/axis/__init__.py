# SPDX-License-Identifier: Apache-2.0
"""AXIS — xESMF-class regridding on the Kokkos engine.

Public surface (``contracts/python-api.md``): ``import axis`` exposes exactly
the curated names in ``__all__``. The compiled engine lives at ``axis._core``
and is internal; the legacy factory/mesh/matrix surface is retired (FR-040).
"""
# ruff: noqa: I001

from . import _core as _core  # engine module — internal, deliberately not in __all__

# Register the .axis xarray accessor (import side-effect; reached via xarray).
from . import accessors  # noqa: F401

from .errors import (
    AxisCapabilityError,
    AxisConfigError,
    AxisError,
    AxisShapeError,
    AxisUnmappedError,
    AxisWeightMismatchError,
    GridError,
)
from .grid import Grid
from .regridder import Regridder
from .types import GridFamily, LineType, Method, Norm, Unmapped
from .vector import VectorRegridder
from .vertical import VerticalRegridder, regrid_3d

__all__ = [
    "Grid",
    "GridFamily",
    "Regridder",
    "VectorRegridder",
    "VerticalRegridder",
    "regrid_3d",
    "Method",
    "Norm",
    "Unmapped",
    "LineType",
    "AxisError",
    "GridError",
    "AxisConfigError",
    "AxisShapeError",
    "AxisWeightMismatchError",
    "AxisCapabilityError",
    "AxisUnmappedError",
]
