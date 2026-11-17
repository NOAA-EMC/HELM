# SPDX-License-Identifier: Apache-2.0
"""Geometry-only point mapping to stable cell IDs."""

from __future__ import annotations

import atexit
import operator
import uuid
import weakref
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from . import axis_py

_WORKER_LOCATORS: dict[tuple[str, int], PointLocator] = {}
_atexit_registered = False


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
    """NumPy or Dask columns in the same order as the supplied point IDs."""

    point_ids: Any
    cell_ids: Any
    status: Any
    owner_partitions: Any
    owner_local_indices: Any

    @property
    def owner_ranks(self) -> Any:
        """Compatibility alias for serial callers; distributed code uses partitions."""
        return self.owner_partitions


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
    if raw.dtype.kind == "u" and raw.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} contains an integer outside int64 range")
    # The native serial locator checks these values for NumPy calls. The Dask
    # path validates IDs in its lazy distributed validation graph.
    return np.ascontiguousarray(raw, dtype=np.int64)


def _coordinates(point_ids: np.ndarray, values: Any) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] not in (2, 3):
        raise ValueError("coordinates must have shape (n_points, 2 or 3)")
    if raw.shape[0] != point_ids.size:
        raise ValueError("point_ids and coordinates must have the same number of rows")
    if raw.dtype.kind not in "fiu":
        raise ValueError("coordinates must contain numeric values")
    # The native locator validates finite coordinates when each chunk executes.
    return np.asfortranarray(raw, dtype=np.float64)


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
        owner_partitions=columns.get("owner_partitions", columns.get("owner_ranks")),
        owner_local_indices=columns["owner_local_indices"],
    )


def _partition_cells(centroids: np.ndarray, cell_ids: np.ndarray, n_partitions: int) -> list[np.ndarray]:
    """Split cells into balanced, spatially coherent, deterministic shards."""
    centroids = np.asarray(centroids)
    cell_ids = np.asarray(cell_ids)
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError("centroids must have shape (n_cells, 3)")
    if cell_ids.ndim != 1 or cell_ids.shape[0] != centroids.shape[0] or cell_ids.dtype.kind not in "iu":
        raise ValueError("cell_ids must be a one-dimensional integer array matching centroids")
    if isinstance(n_partitions, (bool, np.bool_)):
        raise ValueError("n_partitions must be a positive integer")
    try:
        n_partitions = operator.index(n_partitions)
    except TypeError as exc:
        raise ValueError("n_partitions must be a positive integer") from exc
    if n_partitions < 1:
        raise ValueError("n_partitions must be a positive integer")

    shards: list[np.ndarray | None] = [None] * n_partitions

    def split(rows: np.ndarray, partition_ids: list[int]) -> None:
        if len(partition_ids) == 1:
            order = np.lexsort((rows, cell_ids[rows]))
            shards[partition_ids[0]] = rows[order]
            return

        n_left = len(partition_ids) // 2
        split_at = (len(rows) * n_left + len(partition_ids) - 1) // len(partition_ids)
        if rows.size:
            span = np.ptp(centroids[rows], axis=0)
            split_axis = int(np.argmax(span))
            order = np.lexsort((rows, cell_ids[rows], centroids[rows, split_axis]))
            rows = rows[order]
        split(rows[:split_at], partition_ids[:n_left])
        split(rows[split_at:], partition_ids[n_left:])

    split(np.arange(centroids.shape[0], dtype=np.int64), list(range(n_partitions)))
    return [shard if shard is not None else np.empty(0, dtype=np.int64) for shard in shards]


def _candidate_partitions(points: np.ndarray, bounds: np.ndarray) -> dict[int, np.ndarray]:
    """Group point row indices by the conservative bounds of candidate shards."""
    points = np.asarray(points, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    if bounds.ndim != 2 or bounds.shape[1] != 6:
        raise ValueError("bounds must have shape (n_partitions, 6)")
    candidates: dict[int, np.ndarray] = {}
    for partition, shard_bounds in enumerate(bounds):
        low = shard_bounds[:3]
        high = shard_bounds[3:]
        # Empty locators report inverted bounds. Ignore them, and avoid
        # routing malformed bounds that cannot conservatively contain points.
        if np.isnan(shard_bounds).any() or np.any(low > high):
            continue
        rows = np.flatnonzero(np.all((points >= low) & (points <= high), axis=1))
        if rows.size:
            candidates[partition] = rows
    return candidates


def _split_id_buckets(values: np.ndarray, n_buckets: int) -> tuple[np.ndarray, ...]:
    ids = _integer_ids(values, "point_ids")
    if ids.size and np.any(ids < 0):
        raise ValueError("point_ids must be nonnegative")
    return tuple(ids[ids % n_buckets == bucket] for bucket in range(n_buckets))


def _validate_id_bucket(*parts: np.ndarray) -> bool:
    nonempty = [np.asarray(part, dtype=np.int64) for part in parts if len(part)]
    if not nonempty:
        return True
    bucket_ids = np.concatenate(nonempty)
    if np.unique(bucket_ids).size != bucket_ids.size:
        raise ValueError("point_ids must be globally unique")
    return True


def _validated_id_buckets(*buckets: bool) -> bool:
    return all(buckets)


def _canonical_points(coordinates: np.ndarray, coordinate_system: Any) -> np.ndarray:
    """Return the xyz representation used by native locator bounds."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
        raise ValueError("coordinates must have shape (n_points, 2 or 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must contain only finite values")

    if coordinate_system == axis_py.CoordinateSystem.Cartesian3D:
        if coordinates.shape[1] == 3:
            return np.ascontiguousarray(coordinates)
        return np.column_stack((coordinates, np.zeros(coordinates.shape[0], dtype=np.float64)))

    if coordinate_system == axis_py.CoordinateSystem.SphericalDeg:
        factor = np.pi / 180.0
    elif coordinate_system == axis_py.CoordinateSystem.SphericalRad:
        factor = 1.0
    else:
        raise ValueError("coordinate_system is not supported")
    longitude = coordinates[:, 0] * factor
    latitude = coordinates[:, 1] * factor
    if np.any(np.abs(latitude) > np.pi / 2):
        raise ValueError("point locator: invalid point latitude")
    longitude = np.remainder(longitude, 2 * np.pi)
    cos_latitude = np.cos(latitude)
    return np.column_stack((cos_latitude * np.cos(longitude), cos_latitude * np.sin(longitude), np.sin(latitude)))


def _initialize_worker_locator(
    token: str,
    partition_id: int,
    shard: dict[str, np.ndarray],
    cell_ids: np.ndarray,
    coordinate_system_name: str,
    boundary_model_name: str,
    tolerance: float,
    query_batch_size: int,
) -> np.ndarray:
    coordinate_system = getattr(axis_py.CoordinateSystem, coordinate_system_name)
    boundary_model = getattr(axis_py.BoundaryModel, boundary_model_name)
    mesh = axis_py.make_ugrid_mesh(
        shard["node_coords"],
        shard["conn_offsets"],
        shard["conn_indices"],
        coordinate_system=coordinate_system,
        boundary_model=boundary_model,
    )
    locator = PointLocator(mesh, cell_ids, tolerance=tolerance, query_batch_size=query_batch_size)
    key = (token, partition_id)
    _WORKER_LOCATORS[key] = locator
    return np.asarray(locator._native.bounds(), dtype=np.float64)


def _release_worker_locators(token: str) -> None:
    for key in [key for key in _WORKER_LOCATORS if key[0] == token]:
        del _WORKER_LOCATORS[key]


def _worker_locate(
    token: str,
    partition_id: int,
    point_ids: np.ndarray,
    coordinates: np.ndarray,
    coordinate_system_name: str,
) -> dict[str, np.ndarray]:
    key = (token, partition_id)
    locator = _WORKER_LOCATORS.get(key)
    if locator is None:
        raise RuntimeError(f"point locator worker cache is missing partition {partition_id}")
    coordinate_system = getattr(axis_py.CoordinateSystem, coordinate_system_name)
    result = locator.locate(point_ids, coordinates, coordinate_system=coordinate_system)
    return {
        "cell_ids": np.asarray(result.cell_ids, dtype=np.int64),
        "status": np.asarray(result.status, dtype=np.int32),
        "owner_local_indices": np.asarray(result.owner_local_indices, dtype=np.int64),
    }


def _query_chunk(
    token: str,
    point_ids: np.ndarray,
    coordinates: np.ndarray,
    coordinate_system_name: str,
    worker_by_partition: dict[int, str],
    bounds_by_partition: np.ndarray,
    query_batch_size: int,
    ids_are_valid: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Route one aligned point chunk through its candidate worker locators."""
    from distributed import worker_client

    ids = _integer_ids(point_ids, "point_ids")
    if not ids_are_valid:
        raise RuntimeError("Dask point ID validation did not complete")
    if ids.size and np.any(ids < 0):
        raise ValueError("point_ids must be nonnegative")
    coords = _coordinates(ids, coordinates)
    system = getattr(axis_py.CoordinateSystem, coordinate_system_name)
    canonical = _canonical_points(coords, system)

    cell_ids = np.full(ids.size, -1, dtype=np.int64)
    statuses = np.full(ids.size, int(PointLocationStatus.UNMAPPED), dtype=np.int32)
    owner_partitions = np.full(ids.size, -1, dtype=np.int64)
    owner_local_indices = np.full(ids.size, -1, dtype=np.int64)
    if ids.size == 0:
        return ids, cell_ids, statuses, owner_partitions, owner_local_indices

    with worker_client() as client:
        for batch_start in range(0, ids.size, query_batch_size):
            batch_end = min(batch_start + query_batch_size, ids.size)
            batch_candidates = _candidate_partitions(canonical[batch_start:batch_end], bounds_by_partition)
            futures = {}
            for partition_id, batch_rows in batch_candidates.items():
                global_rows = batch_rows + batch_start
                futures[partition_id] = (
                    global_rows,
                    client.submit(
                        _worker_locate,
                        token,
                        partition_id,
                        ids[global_rows],
                        coords[global_rows],
                        coordinate_system_name,
                        workers=[worker_by_partition[partition_id]],
                        allow_other_workers=False,
                        pure=False,
                    ),
                )

            for partition_id, (rows, future) in futures.items():
                shard_result = future.result()
                matched = np.asarray(shard_result["cell_ids"], dtype=np.int64) >= 0
                if not np.any(matched):
                    continue
                matched_rows = rows[matched]
                matched_cell_ids = np.asarray(shard_result["cell_ids"], dtype=np.int64)[matched]
                replace = (cell_ids[matched_rows] < 0) | (matched_cell_ids < cell_ids[matched_rows])
                winners = matched_rows[replace]
                if not winners.size:
                    continue
                cell_ids[winners] = matched_cell_ids[replace]
                statuses[winners] = np.asarray(shard_result["status"], dtype=np.int32)[matched][replace]
                owner_partitions[winners] = partition_id
                owner_local_indices[winners] = np.asarray(shard_result["owner_local_indices"], dtype=np.int64)[matched][replace]

    return ids, cell_ids, statuses, owner_partitions, owner_local_indices


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
        self._native = axis_py._PointLocator(  # type: ignore[attr-defined]
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


class DistributedPointLocator:
    """Map Dask point arrays against a mesh sharded across active workers.

    Construction accepts one in-memory mesh and creates one serial locator per
    active Dask worker. ``locate`` keeps point rows lazy and routes each chunk
    only to workers whose conservative bounds contain its points. Compute the
    returned columns before calling ``close``.
    """

    def __init__(
        self,
        mesh: axis_py.Mesh,
        cell_ids: Any,
        *,
        client: Any = None,
        tolerance: float = 1e-10,
        query_batch_size: int = 128,
    ) -> None:
        try:
            import distributed
        except ImportError as exc:
            raise RuntimeError("DistributedPointLocator requires the optional 'distributed' dependency") from exc

        if client is None:
            try:
                client = distributed.get_client()
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError("DistributedPointLocator requires an active distributed.Client") from exc
        if not isinstance(client, distributed.Client):
            raise ValueError("client must be a distributed.Client")
        if not isinstance(mesh, axis_py.Mesh):
            raise ValueError("mesh must be an axis.Mesh")

        ids = _integer_ids(cell_ids, "cell_ids")
        if ids.size and (np.any(ids < 0) or np.unique(ids).size != ids.size):
            raise ValueError("cell_ids must be nonnegative and unique")
        options = _options(tolerance, 64, 24, query_batch_size)
        workers = sorted(client.scheduler_info().get("workers", {}))
        if not workers:
            raise ValueError("DistributedPointLocator requires at least one active Dask worker")

        self._client = client
        self._token = uuid.uuid4().hex
        self._closed = False
        self._coordinate_system = mesh.coordinate_system
        self._coordinate_system_name = mesh.coordinate_system.name
        self._query_batch_size = options.query_batch_size
        self._worker_by_partition = dict(enumerate(workers))
        self._bounds_by_partition = np.empty((len(workers), 6), dtype=np.float64)

        try:
            centroids = np.asarray(axis_py._mesh_cell_centroids(mesh), dtype=np.float64)  # type: ignore[attr-defined]
            shards = _partition_cells(centroids, ids, len(workers))
            boundary_model_name = mesh.boundary_model.name
            for partition_id, (worker, rows) in enumerate(zip(workers, shards, strict=True)):
                shard = axis_py._mesh_extract_cells(mesh, np.ascontiguousarray(rows, dtype=np.int64))  # type: ignore[attr-defined]
                shard_ids = np.ascontiguousarray(ids[rows], dtype=np.int64)
                future = client.submit(
                    _initialize_worker_locator,
                    self._token,
                    partition_id,
                    shard,
                    shard_ids,
                    self._coordinate_system_name,
                    boundary_model_name,
                    options.tolerance,
                    options.query_batch_size,
                    workers=[worker],
                    allow_other_workers=False,
                    pure=False,
                )
                self._bounds_by_partition[partition_id] = np.asarray(future.result(), dtype=np.float64)
                del shard, shard_ids, future
        except Exception:
            try:
                client.run(_release_worker_locators, self._token)
            except Exception:
                pass
            self._closed = True
            raise

        _live_distributed_locators.add(self)
        global _atexit_registered
        if not _atexit_registered:
            atexit.register(_close_distributed_locators)
            _atexit_registered = True

    def locate(self, point_ids: Any, coordinates: Any, *, coordinate_system: Any = None) -> PointLocationResult:
        if self._closed:
            raise RuntimeError("DistributedPointLocator is closed")
        system = self._coordinate_system if coordinate_system is None else coordinate_system

        try:
            import dask.array as da
            from dask import delayed
        except ImportError as exc:
            raise RuntimeError("DistributedPointLocator.locate requires the optional 'dask' dependency") from exc

        if not isinstance(system, axis_py.CoordinateSystem) or system not in tuple(axis_py.CoordinateSystem):
            raise ValueError("coordinate_system must be a supported axis.CoordinateSystem")
        mesh_is_spherical = self._coordinate_system in (
            axis_py.CoordinateSystem.SphericalDeg,
            axis_py.CoordinateSystem.SphericalRad,
        )
        query_is_spherical = system in (axis_py.CoordinateSystem.SphericalDeg, axis_py.CoordinateSystem.SphericalRad)
        if mesh_is_spherical != query_is_spherical:
            raise ValueError("coordinate_system must use the same coordinate family as the mesh")
        if not isinstance(point_ids, da.Array) or not isinstance(coordinates, da.Array):
            raise TypeError("point_ids and coordinates must be Dask arrays")
        if point_ids.ndim != 1 or point_ids.dtype.kind not in "iu":
            raise ValueError("point_ids must be a one-dimensional integer Dask array")
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3) or coordinates.dtype.kind not in "fiu":
            raise ValueError("coordinates must be a numeric Dask array with shape (n_points, 2 or 3)")
        if point_ids.shape[0] != coordinates.shape[0]:
            raise ValueError("point_ids and coordinates must have the same number of rows")
        if point_ids.chunks[0] != coordinates.chunks[0]:
            raise ValueError("point_ids and coordinates must have matching row chunks")
        if len(coordinates.chunks[1]) != 1 or coordinates.chunks[1][0] != coordinates.shape[1]:
            raise ValueError("the coordinate dimension must be contained in one Dask chunk")

        id_chunks = point_ids.to_delayed().ravel()
        coordinate_chunks = coordinates.to_delayed()
        bucket_chunks = [delayed(_split_id_buckets, pure=False)(id_chunk, len(self._worker_by_partition)) for id_chunk in id_chunks]
        id_buckets = []
        for bucket in range(len(self._worker_by_partition)):
            bucket_parts = [bucket_chunk[bucket] for bucket_chunk in bucket_chunks]
            id_buckets.append(delayed(_validate_id_bucket, pure=False)(*bucket_parts))
        ids_are_valid = delayed(_validated_id_buckets, pure=False)(*id_buckets)
        result_chunks: list[list[Any]] = [[], [], [], [], []]
        for chunk_index, chunk_size in enumerate(point_ids.chunks[0]):
            chunk_result = delayed(_query_chunk, pure=False)(
                self._token,
                id_chunks[chunk_index],
                coordinate_chunks[chunk_index, 0],
                system.name,
                self._worker_by_partition,
                self._bounds_by_partition,
                self._query_batch_size,
                ids_are_valid,
            )
            for column_index, dtype in enumerate((np.int64, np.int64, np.int32, np.int64, np.int64)):
                result_chunks[column_index].append(
                    da.from_delayed(chunk_result[column_index], shape=(int(chunk_size),), dtype=dtype)
                )

        columns = [da.concatenate(chunks) if len(chunks) > 1 else chunks[0] for chunks in result_chunks]
        return PointLocationResult(*columns)

    def close(self) -> None:
        if not self._closed:
            try:
                self._client.run(_release_worker_locators, self._token)
            except Exception:
                # A closed client has already lost access to its worker cache.
                pass
            self._closed = True
            _live_distributed_locators.discard(self)

    def __enter__(self) -> DistributedPointLocator:
        if self._closed:
            raise RuntimeError("DistributedPointLocator is closed")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
