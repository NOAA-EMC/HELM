# SPDX-License-Identifier: Apache-2.0
"""Geometry-only point mapping to stable cell IDs."""

from __future__ import annotations

import atexit
import operator
import weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from . import axis_py


class PointLocationStatus(IntEnum):
    """Containment result for a point."""

    UNMAPPED = 0
    INTERIOR = 1
    BOUNDARY = 2
    # Match the spelling used by the C++ enum and other AXIS enums.
    Unmapped = UNMAPPED
    Interior = INTERIOR
    Boundary = BOUNDARY


@dataclass(frozen=True)
class PointLocationResult:
    """NumPy columns in the same order as the supplied point IDs."""

    point_ids: np.ndarray
    cell_ids: np.ndarray
    status: np.ndarray
    owner_ranks: np.ndarray
    owner_local_indices: np.ndarray


@dataclass(frozen=True)
class _Options:
    tolerance: float = 1e-10
    max_import_bytes: int = 512 * 1024 * 1024
    max_message_bytes: int = 8 * 1024 * 1024
    query_batch_size: int = 128


def _integer_ids(values: Any, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise ValueError(f"{name} must contain integers")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains an integer outside int64 range")
    result = np.ascontiguousarray(raw, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} must be unique")
    return result


def _coordinates(point_ids: np.ndarray, values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] not in (2, 3):
        raise ValueError("coordinates must have shape (n_points, 2 or 3)")
    if raw.shape[0] != point_ids.size:
        raise ValueError("point_ids and coordinates must have the same number of rows")
    if raw.dtype.kind not in "fiu":
        raise ValueError("coordinates must contain numeric values")
    result = np.asfortranarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("coordinates must be finite")
    return result


def _options(tolerance: Any, max_import_bytes: Any, max_message_bytes: Any, query_batch_size: Any) -> _Options:
    try:
        tolerance = float(tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tolerance must be a finite number in [0, 0.01)") from exc
    if not np.isfinite(tolerance) or tolerance < 0 or tolerance >= 0.01:
        raise ValueError("tolerance must be a finite number in [0, 0.01)")

    sizes = []
    for name, value, minimum in (
        ("max_import_bytes", max_import_bytes, 64),
        ("max_message_bytes", max_message_bytes, 24),
        ("query_batch_size", query_batch_size, 1),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer >= {minimum}")
        try:
            integer = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer >= {minimum}") from exc
        if integer < minimum or integer > np.iinfo(np.uintp).max:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        sizes.append(integer)
    return _Options(tolerance, *sizes)


def _result(columns: dict[str, np.ndarray]) -> PointLocationResult:
    return PointLocationResult(
        point_ids=columns["point_ids"],
        cell_ids=columns["cell_ids"],
        status=columns["status"],
        owner_ranks=columns["owner_ranks"],
        owner_local_indices=columns["owner_local_indices"],
    )


class PointLocator:
    """Map each local point independently to its containing mesh cell.

    The mesh must carry an explicit supported boundary model. Cell IDs must be
    nonnegative, unique, and ordered like ``mesh``'s cell rows.
    """

    def __init__(
        self,
        mesh: axis_py.Mesh,
        cell_ids: Any,
        *,
        tolerance: float = 1e-10,
        max_import_bytes: int = 512 * 1024 * 1024,
        max_message_bytes: int = 8 * 1024 * 1024,
        query_batch_size: int = 128,
    ) -> None:
        self._coordinate_system = mesh.coordinate_system
        ids = _integer_ids(cell_ids, "cell_ids")
        options = _options(tolerance, max_import_bytes, max_message_bytes, query_batch_size)
        self._native = axis_py._PointLocator(
            mesh, ids, options.tolerance, options.max_import_bytes, options.max_message_bytes, options.query_batch_size
        )

    def locate(self, point_ids: Any, coordinates: Any, *, coordinate_system: Any = None) -> PointLocationResult:
        ids = _integer_ids(point_ids, "point_ids")
        coords = _coordinates(ids, coordinates)
        system = self._coordinate_system if coordinate_system is None else coordinate_system
        return _result(self._native.locate(ids, coords, system))


_live_distributed_locators: weakref.WeakSet = weakref.WeakSet()


def _close_distributed_locators() -> None:
    for locator in list(_live_distributed_locators):
        locator.close()


def _collective_prepare(comm: Any, prepare: Any, what: str) -> Any:
    value = None
    error = None
    try:
        value = prepare()
    except Exception as exc:  # every rank reports local input errors together
        error = str(exc) or type(exc).__name__
    errors = comm.allgather(error)
    if any(message is not None for message in errors):
        details = "; ".join(f"rank {rank}: {message}" for rank, message in enumerate(errors) if message is not None)
        raise ValueError(f"{what} failed collectively ({details})")
    return value


class DistributedPointLocator:
    """Collectively map independently partitioned mesh and point rows.

    Pass a live mpi4py intracommunicator. Calls, including ``close``, must be
    made by all ranks in the communicator in the same order. Use as a context
    manager so its duplicated native communicator is released before MPI ends.
    mpi4py is imported only when this distributed class is constructed.
    """

    def __init__(
        self,
        mesh: axis_py.Mesh,
        cell_ids: Any,
        comm: Any,
        *,
        tolerance: float = 1e-10,
        max_import_bytes: int = 512 * 1024 * 1024,
        max_message_bytes: int = 8 * 1024 * 1024,
        query_batch_size: int = 128,
    ) -> None:
        from mpi4py import MPI

        if not MPI.Is_initialized() or MPI.Is_finalized():
            raise RuntimeError("DistributedPointLocator requires initialized MPI")

        def prepare() -> tuple[np.ndarray, _Options, int]:
            if comm is None or comm.Is_inter():
                raise ValueError("comm must be a live mpi4py intracommunicator")
            if not isinstance(mesh, axis_py.Mesh):
                raise ValueError("mesh must be an axis.Mesh")
            ids = _integer_ids(cell_ids, "cell_ids")
            options = _options(tolerance, max_import_bytes, max_message_bytes, query_batch_size)
            handle = operator.index(comm.py2f())
            return ids, options, handle

        ids, options, handle = _collective_prepare(comm, prepare, "distributed locator construction")
        self._comm = comm
        self._coordinate_system = mesh.coordinate_system
        self._native = axis_py._DistributedPointLocator(
            mesh, ids, handle, options.tolerance, options.max_import_bytes, options.max_message_bytes, options.query_batch_size
        )
        _live_distributed_locators.add(self)
        if not getattr(_close_distributed_locators, "registered", False):
            atexit.register(_close_distributed_locators)
            _close_distributed_locators.registered = True

    def locate(self, point_ids: Any, coordinates: Any, *, coordinate_system: Any = None) -> PointLocationResult:
        if self._native is None:
            raise RuntimeError("DistributedPointLocator is closed")
        system = self._coordinate_system if coordinate_system is None else coordinate_system

        def prepare() -> tuple[np.ndarray, np.ndarray, int]:
            if not isinstance(system, axis_py.CoordinateSystem):
                raise ValueError("coordinate_system must be an axis.CoordinateSystem")
            system_value = next((value for value, candidate in enumerate(axis_py.CoordinateSystem) if system == candidate), None)
            if system_value is None:
                raise ValueError("coordinate_system is not supported")
            ids = _integer_ids(point_ids, "point_ids")
            coords = _coordinates(ids, coordinates)
            return ids, coords, system_value

        ids, coords, system_value = _collective_prepare(self._comm, prepare, "distributed point validation")
        systems = self._comm.allgather(system_value)
        if any(value != systems[0] for value in systems[1:]):
            raise ValueError("distributed point validation failed collectively (coordinate_system differs across ranks)")
        return _result(self._native.locate(ids, coords, system))

    def close(self) -> None:
        if self._native is not None:
            self._native = None
            self._comm = None
            _live_distributed_locators.discard(self)

    def __enter__(self) -> DistributedPointLocator:
        if self._native is None:
            raise RuntimeError("DistributedPointLocator is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
