# SPDX-License-Identifier: Apache-2.0
"""Distributed regridding of chunked spherical UGRID meshes."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import xarray as xr

from . import axis_py

_STENCIL_SIZES = {"nearest_neighbor": 1, "bilinear": 8, "bicubic": 64, "patch": 16}
_METHODS = {"conservative", "conservative2nd", "nearest_neighbor", "bilinear", "bicubic", "patch"}


@dataclass(frozen=True)
class MeshTile:
    """Contiguous cell rows and a worker-side loader returning UGRID arrays.

    The loader returns longitude/latitude nodes in degrees, CSR offsets and
    indices, and optionally an integer cell mask. Connectivity is tile local.
    Tile order defines global cell order.
    """

    size: int
    load: Callable[[], tuple[np.ndarray, ...]]


def _load(tile: MeshTile) -> tuple[np.ndarray, ...]:
    arrays = tile.load()
    if len(arrays) not in (3, 4, 5):
        raise ValueError("tile loader must return nodes, offsets, indices and optionally a mask and global node IDs")
    nodes = np.asfortranarray(arrays[0], dtype=np.float64)
    offsets = np.ascontiguousarray(arrays[1], dtype=np.int64)
    indices = np.ascontiguousarray(arrays[2], dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError("tile nodes must be finite longitude/latitude pairs")
    if offsets.shape != (tile.size + 1,) or indices.ndim != 1:
        raise ValueError("tile CSR arrays do not match its declared size")
    if offsets[0] != 0 or offsets[-1] != indices.size or np.any(np.diff(offsets) < 3):
        raise ValueError("invalid tile CSR offsets")
    if np.any(indices < 0) or np.any(indices >= len(nodes)):
        raise ValueError("tile connectivity references a missing node")
    mask = np.ones(tile.size, dtype=np.int32) if len(arrays) == 3 else np.ascontiguousarray(arrays[3], dtype=np.int32)
    if mask.shape != (tile.size,):
        raise ValueError("tile mask does not match its declared size")
    if len(arrays) == 5:
        ids = np.asarray(arrays[4])
        if ids.shape != (len(nodes),) or ids.dtype.kind not in "iu":
            raise ValueError("global node IDs must be a one-dimensional integer array matching nodes")
        return nodes, offsets, indices, mask, ids
    return nodes, offsets, indices, mask


def _inspect(tile: MeshTile, budget: int) -> tuple[list[float], int, int]:
    arrays = _load(tile)
    cost = sum(array.nbytes for array in arrays) * 4 + tile.size * 256
    if cost > budget:
        raise MemoryError(f"mesh tile estimated at {cost} bytes exceeds {budget} bytes")
    return axis_py._conservative_tile_bounds(axis_py.make_ugrid_mesh(*arrays[:4])), cost, int(np.diff(arrays[1]).max())  # type: ignore[attr-defined]


def _overlaps(source: list[float], target: list[float]) -> bool:
    return all(source[d] <= target[d + 3] and target[d] <= source[d + 3] for d in range(3))


def _points(arrays: tuple[np.ndarray, ...], method: str) -> np.ndarray:
    mesh = axis_py.make_ugrid_mesh(*arrays[:4])
    points = axis_py._stencil_query_points(mesh, {"method": method, "line_type": "great_circle"})  # type: ignore[attr-defined]
    return np.asarray(points, dtype=np.float32)


def _source_points(tile: MeshTile, method: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load(tile)
    rows = np.flatnonzero(arrays[3]) if method == "bicubic" else np.arange(tile.size)
    return _points(arrays, method)[rows], rows


def _target_points(tile: MeshTile, method: str) -> np.ndarray:
    return _points(_load(tile), method)


def _tile_candidates(
    source: tuple[np.ndarray, np.ndarray], target: np.ndarray, start: int, k: int
) -> tuple[np.ndarray, np.ndarray]:
    points, rows = source
    count = min(k, len(rows))
    distances = np.empty((len(target), count), dtype=np.float32)
    ids = np.empty((len(target), count), dtype=np.int64)
    global_rows = rows + start
    for j, point in enumerate(target):
        delta = points - point
        squared = np.sum(delta * delta, axis=1)
        order = np.lexsort((global_rows, squared))[:count]
        distances[j] = squared[order]
        ids[j] = global_rows[order]
    return distances, ids


def _merge_candidates(
    left: tuple[np.ndarray, np.ndarray], right: tuple[np.ndarray, np.ndarray], k: int
) -> tuple[np.ndarray, np.ndarray]:
    dist = np.concatenate((left[0], right[0]), axis=1)
    ids = np.concatenate((left[1], right[1]), axis=1)
    order = np.lexsort((ids, dist), axis=1)[:, :k]
    return np.take_along_axis(dist, order, axis=1), np.take_along_axis(ids, order, axis=1)


def _selected_rows(candidates: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return np.unique(candidates[1])


def _subset(arrays: tuple[np.ndarray, ...], rows: np.ndarray | None) -> tuple[np.ndarray, ...]:
    if rows is None:
        return arrays
    nodes, offsets, indices, mask = arrays[:4]
    lengths = offsets[rows + 1] - offsets[rows]
    selected = np.concatenate([indices[offsets[row] : offsets[row + 1]] for row in rows])
    used, local = np.unique(selected, return_inverse=True)
    return nodes[used], np.r_[0, np.cumsum(lengths)], local, mask[rows], *(a[used] for a in arrays[4:])


def _load_rows(tile: MeshTile, rows: np.ndarray | None) -> tuple[np.ndarray, ...]:
    return _subset(_load(tile), rows)


def _ring_rows(tile: MeshTile, frontier: np.ndarray) -> np.ndarray:
    arrays = _load(tile)
    if len(arrays) != 5:
        raise ValueError("second-order source tile loaders must supply global node IDs after the cell mask")
    _, offsets, indices, mask, ids = arrays
    hits = np.isin(ids[indices], frontier)
    return np.flatnonzero(np.logical_or.reduceat(hits, offsets[:-1]) & (mask != 0))


def _active_nodes(tile: MeshTile, rows: np.ndarray | None) -> np.ndarray:
    arrays = _load(tile)
    if len(arrays) != 5:
        raise ValueError("second-order source tile loaders must supply global node IDs after the cell mask")
    _, offsets, indices, mask, ids = arrays
    active = np.flatnonzero(mask) if rows is None else rows[mask[rows] != 0]
    if not active.size:
        return ids[:0]
    return np.unique(np.concatenate([ids[indices[offsets[row] : offsets[row + 1]]] for row in active]))


def _build(
    target: MeshTile, sources: tuple[tuple[np.ndarray, ...], ...], method: str, norm: str
) -> tuple[bytes, tuple[Any, ...] | None] | None:
    if not sources:
        return None
    nodes, offsets, indices, masks = [], [np.zeros(1, dtype=np.int64)], [], []
    node_base = edge_base = 0
    for xy, ptr, conn, mask, *_ in sources:
        nodes.append(xy)
        offsets.append(ptr[1:] + edge_base)
        indices.append(conn + node_base)
        masks.append(mask)
        node_base += len(xy)
        edge_base += len(conn)
    xy = np.concatenate(nodes)
    connectivity = np.concatenate(indices)
    if method == "conservative2nd":
        node_ids = np.concatenate([arrays[4] for arrays in sources])
        _, first, inverse = np.unique(node_ids, return_index=True, return_inverse=True)
        if not np.allclose(xy, xy[first[inverse]], atol=1e-10, rtol=0):
            raise ValueError("shared node IDs must have consistent coordinates")
        xy = xy[first]
        connectivity = inverse[connectivity]
    source = axis_py.make_ugrid_mesh(np.asfortranarray(xy), np.concatenate(offsets), connectivity, np.concatenate(masks))
    destination = axis_py.make_ugrid_mesh(*_load(target)[:4])
    config = {"method": method, "line_type": "great_circle", "unmapped": "ignore", "norm_type": norm}
    matrix = axis_py._generate_local_weight_block(source, destination, config)  # type: ignore[attr-defined]
    reconstruction = axis_py._pack_reconstruction(matrix) if matrix.has_linear_reconstruction else None  # type: ignore[attr-defined]
    return matrix.to_bytes(), reconstruction


def _apply(
    block: tuple[bytes, tuple[Any, ...] | None] | None,
    size: int,
    batch: tuple[int, ...],
    *chunks: np.ndarray,
    columns: np.ndarray | None = None,
) -> np.ndarray:
    if block is None:
        return np.zeros((*batch, size))
    payload, reconstruction = block
    matrix = axis_py.Matrix.from_bytes(payload)
    if reconstruction is not None:
        axis_py._restore_reconstruction(matrix, *reconstruction)  # type: ignore[attr-defined]
    if columns is not None:
        if not columns.size:
            return np.zeros((*batch, size))
        values = np.zeros((*batch, matrix.n_src))
        values[..., columns] = np.concatenate(chunks, axis=-1)
    else:
        values = np.concatenate(chunks, axis=-1).astype(np.float64, copy=False)
    flat = values.reshape(-1, values.shape[-1]).T
    return np.asarray(axis_py.batch_apply(matrix, np.asfortranarray(flat))).T.reshape((*batch, size))


def _block_columns(block: tuple[bytes, tuple[Any, ...] | None] | None) -> np.ndarray:
    if block is None:
        return np.empty(0, dtype=np.int64)
    payload, reconstruction = block
    matrix = axis_py.Matrix.from_bytes(payload)
    active = np.unique(np.asarray(matrix.factor_col))
    if reconstruction is None or not active.size:
        return active
    _, _, offsets, neighbors, *_ = reconstruction
    required = (neighbors[offsets[i] : offsets[i + 1]] for i in active)
    return np.unique(np.concatenate((active, *required)))


class DistributedRegridder:
    """Build reusable weight blocks on Dask workers for spherical UGRID tiles."""

    def __init__(
        self, client: Any, *, method: str = "conservative", memory_budget: int = 512 * 1024**2, norm: str = "dstarea"
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")
        if memory_budget < 1 or norm not in ("dstarea", "fracarea"):
            raise ValueError("expected a positive memory budget and dstarea or fracarea normalization")
        self._client = client
        self._budget = memory_budget
        self._norm = norm
        self._method = method
        self._blocks: list[tuple[Any, tuple[tuple[int, np.ndarray], ...], int, np.ndarray]] = []
        self._source_sizes: tuple[int, ...] = ()

    def fit(self, source: Sequence[MeshTile], destination: Sequence[MeshTile]) -> DistributedRegridder:
        source = tuple(tile for tile in source if tile.size != 0)
        destination = tuple(tile for tile in destination if tile.size != 0)
        if not source or not destination or any(tile.size < 0 for tile in (*source, *destination)):
            raise ValueError("source and destination must have positive total cell counts and no negative tile sizes")
        tiles = (*source, *destination)
        metadata = self._client.gather(self._client.map(_inspect, tiles, budget=self._budget, pure=False))
        source_meta, target_meta = metadata[: len(source)], metadata[len(source) :]
        offsets = np.cumsum([0, *(tile.size for tile in source)], dtype=np.int64)
        k = _STENCIL_SIZES.get(self._method)
        source_points = self._client.map(_source_points, source, method=self._method, pure=False) if k is not None else ()
        plans: list[tuple[MeshTile, tuple[tuple[int, np.ndarray | None], ...]]] = []
        pending = deque((str(i), tile, meta) for i, (tile, meta) in enumerate(zip(destination, target_meta, strict=True)))

        def split_target(index: str, target: MeshTile) -> bool:
            if target.size == 1:
                return False
            mid = target.size // 2
            children = [
                MeshTile(stop - start, partial(_load_rows, target, np.arange(start, stop)))
                for start, stop in ((0, mid), (mid, target.size))
            ]
            metadata = self._client.gather(self._client.map(_inspect, children, budget=self._budget, pure=False))
            for child_index in (1, 0):
                pending.appendleft((f"{index}.{child_index}", children[child_index], metadata[child_index]))
            return True

        while pending:
            index, target, (bounds, cost, _) = pending.popleft()

            selected: tuple[tuple[int, np.ndarray | None], ...]
            if k is None:
                selected = tuple((i, None) for i, (box, _, _) in enumerate(source_meta) if _overlaps(box, bounds))
            else:
                if cost + max(item[1] for item in source_meta) + 48 * target.size * k > self._budget:
                    if split_target(index, target):
                        continue
                    raise MemoryError(f"stencil discovery for destination tile {index} exceeds the memory estimate")
                query = self._client.submit(_target_points, target, self._method, pure=False)
                level = [
                    self._client.submit(_tile_candidates, points, query, int(offsets[i]), k, pure=False)
                    for i, points in enumerate(source_points)
                ]
                while len(level) > 1:
                    level = [
                        self._client.submit(_merge_candidates, level[i], level[i + 1], k, pure=False)
                        if i + 1 < len(level)
                        else level[i]
                        for i in range(0, len(level), 2)
                    ]
                rows = self._client.submit(_selected_rows, level[0], pure=False).result()
                owners = np.searchsorted(offsets[1:], rows, side="right")
                selected = tuple((int(i), rows[owners == i] - offsets[i]) for i in np.unique(owners))
            if self._method == "conservative2nd":
                closure = dict(selected)
                frontier = selected
                for _ring in range(2):
                    if not frontier:
                        break
                    node_parts = self._client.gather(
                        [self._client.submit(_active_nodes, source[i], rows, pure=False) for i, rows in frontier]
                    )
                    if sum(part.nbytes for part in node_parts) > self._budget:
                        raise MemoryError(f"destination tile {index} exceeds the neighbor query budget")
                    nodes = np.unique(np.concatenate(node_parts))
                    if not nodes.size:
                        break
                    hits: list[np.ndarray] = self._client.gather(self._client.map(_ring_rows, source, frontier=nodes, pure=False))
                    ring = tuple((i, ring_rows) for i, ring_rows in enumerate(hits) if ring_rows.size)
                    frontier = ring
                    for i, ring_rows in ring:
                        if i not in closure:
                            closure[i] = ring_rows
                        elif (existing_rows := closure[i]) is not None:
                            closure[i] = np.union1d(existing_rows, ring_rows)
                selected = tuple(sorted(closure.items()))
            estimate = cost + sum(
                source_meta[i][1] if rows is None else len(rows) * (256 + 96 * source_meta[i][2]) for i, rows in selected
            )
            estimate += 32 * target.size * sum(source[i].size if rows is None else len(rows) for i, rows in selected)
            if estimate > self._budget:
                if split_target(index, target):
                    continue
                raise MemoryError(
                    f"destination tile {index} for {self._method} needs an estimated {estimate} bytes (budget {self._budget})"
                )
            plans.append((target, selected))
        from distributed import wait

        blocks = []
        for target, selected in plans:
            shards = tuple(self._client.submit(_load_rows, source[i], rows, pure=False) for i, rows in selected)
            block = self._client.submit(_build, target, shards, self._method, self._norm, pure=False)
            blocks.append((block, selected, target.size))
        with self._client.as_current():
            wait([future for future, _, _ in blocks])
        ready = []
        for future, selected, size in blocks:
            if future.status != "finished":
                future.result()  # Raise worker errors without gathering successful blocks.
            columns = self._client.submit(_block_columns, future, pure=False).result()
            requests = []
            start = 0
            for i, rows in selected:
                count = source[i].size if rows is None else len(rows)
                left, right = np.searchsorted(columns, [start, start + count])
                local = columns[left:right] - start
                if local.size:
                    requests.append((i, local if rows is None else rows[local]))
                start += count
            ready.append((future, tuple(requests), size, columns))
        self._source_sizes = tuple(tile.size for tile in source)
        self._blocks = ready
        return self

    def apply(self, data: xr.DataArray, *, dim: str) -> xr.DataArray:
        """Regrid one cell dimension, preserving batch dimensions and coordinates."""
        import dask.array as da
        from dask import delayed

        if not self._blocks:
            raise RuntimeError("fit must complete before apply")
        if not isinstance(data.data, da.Array) or data.sizes[dim] != sum(self._source_sizes):
            raise ValueError("data must be Dask backed and match the fitted source cell dimension")
        ordered = data.transpose(..., dim)
        values = ordered.data.rechunk({ordered.ndim - 1: self._source_sizes})
        source_chunks = [
            values[..., start : start + size]
            for start, size in zip(np.cumsum([0, *self._source_sizes[:-1]]), self._source_sizes, strict=True)
        ]
        inputs = values.to_delayed()
        output = np.empty((*inputs.shape[:-1], len(self._blocks)), dtype=object)
        for batch_index in np.ndindex(inputs.shape[:-1]):
            shape = tuple(values.chunks[d][i] for d, i in enumerate(batch_index))
            for j, (block, selected, size, columns) in enumerate(self._blocks):
                chunks = [
                    source_chunks[i][..., rows].rechunk({ordered.ndim - 1: -1}).to_delayed()[(*batch_index, 0)]
                    for i, rows in selected
                ]
                result = delayed(_apply)(block, size, shape, *chunks, columns=columns)
                output[(*batch_index, j)] = da.from_delayed(result, shape=(*shape, size), dtype=np.float64)
        coords = {name: coord for name, coord in ordered.coords.items() if dim not in coord.dims}
        return xr.DataArray(
            da.block(output.tolist()), dims=ordered.dims, coords=coords, name=data.name, attrs=data.attrs
        ).transpose(*data.dims)
