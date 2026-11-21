# SPDX-License-Identifier: Apache-2.0
"""Stable point registrations and direct indexing for known grid geometries."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from os import PathLike
from typing import Any

import numpy as np

from .point_locator import _integer_ids


def _source_fingerprint(point_ids: Any, coordinates: Any) -> str:
    ids = np.ascontiguousarray(_integer_ids(point_ids, "point_ids"), dtype="<i8")
    xyz = np.ascontiguousarray(coordinates, dtype="<f8")
    if xyz.ndim != 2 or xyz.shape[0] != ids.size or xyz.shape[1] not in (2, 3) or not np.isfinite(xyz).all():
        raise ValueError("coordinates must contain finite point rows")
    h = hashlib.sha256(xyz.shape[1].to_bytes(1, "little"))
    h.update(ids.tobytes())
    h.update(xyz.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PointRegistration:
    """Layout-independent point IDs and cell IDs. Unmapped cells use -1."""

    point_ids: np.ndarray
    cell_ids: np.ndarray

    def __post_init__(self) -> None:
        points = _integer_ids(self.point_ids, "point_ids").copy()
        cells = _integer_ids(self.cell_ids, "cell_ids").copy()
        if points.size != cells.size or np.any(points < 0) or np.unique(points).size != points.size or np.any(cells < -1):
            raise ValueError("invalid registration IDs")
        points.flags.writeable = cells.flags.writeable = False
        object.__setattr__(self, "point_ids", points)
        object.__setattr__(self, "cell_ids", cells)

    def save(self, path: str | PathLike[str], *, coordinates: Any, grid_key: str, lookup_key: str) -> None:
        """Save a registration. Include geometry and stable cell IDs in grid_key."""
        with open(path, "wb") as file:
            np.savez(
                file,
                version=1,
                point_ids=self.point_ids,
                cell_ids=self.cell_ids,
                sources=_source_fingerprint(self.point_ids, coordinates),
                grid=grid_key,
                lookup=lookup_key,
            )

    @classmethod
    def load(
        cls, path: str | PathLike[str], *, point_ids: Any, coordinates: Any, grid_key: str, lookup_key: str
    ) -> PointRegistration:
        """Reload a registration after checking grid, lookup and source identity."""
        with np.load(path, allow_pickle=False) as file:
            if (
                file["version"].item() != 1
                or file["grid"].item() != grid_key
                or file["lookup"].item() != lookup_key
                or file["sources"].item() != _source_fingerprint(point_ids, coordinates)
                or not np.array_equal(file["point_ids"], point_ids)
            ):
                raise ValueError("registration does not match the grid, lookup settings or source coordinates")
            return cls(file["point_ids"], file["cell_ids"])

    def local_indices(self, owned_cell_ids: Any) -> np.ndarray:
        """Resolve stable IDs against the current partition, with -1 for missing cells."""
        ids = _integer_ids(owned_cell_ids, "owned_cell_ids")
        if np.any(ids < 0) or np.unique(ids).size != ids.size:
            raise ValueError("owned_cell_ids must be unique and nonnegative")
        result = np.full(self.cell_ids.size, -1, dtype=np.int64)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        matches = np.searchsorted(sorted_ids, self.cell_ids)
        rows = np.flatnonzero(matches < ids.size)
        rows = rows[sorted_ids[matches[rows]] == self.cell_ids[rows]]
        result[rows] = order[matches[rows]]
        return result


def _edges(values: Any) -> np.ndarray:
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be finite and strictly increasing")
    return edges


def _bins(edges: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.minimum(np.searchsorted(edges, values, side="right") - 1, len(edges) - 2)


def index_latlon_grid(coordinates: Any, longitude_edges: Any, latitude_edges: Any, *, periodic: bool = False) -> np.ndarray:
    """Return row-major cell indices for lon/lat degrees, or -1 outside the grid.

    Edges are ascending. At a shared edge the higher bin owns the point.
    The last outer edge is included unless longitude wraps periodically.
    """
    lon_edges, lat_edges = _edges(longitude_edges), _edges(latitude_edges)
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("coordinates must be finite longitude/latitude pairs")
    if lat_edges[0] < -90 or lat_edges[-1] > 90 or lon_edges[-1] - lon_edges[0] > 360:
        raise ValueError("invalid longitude or latitude edges")
    lon, lat = points.T
    if periodic:
        if not np.isclose(lon_edges[-1] - lon_edges[0], 360, atol=1e-10, rtol=0):
            raise ValueError("periodic longitude edges must span 360 degrees")
        lon = (lon - lon_edges[0]) % 360 + lon_edges[0]
    valid = (lon >= lon_edges[0]) & (lon <= lon_edges[-1]) & (lat >= lat_edges[0]) & (lat <= lat_edges[-1])
    rows = _bins(lat_edges, lat) * (len(lon_edges) - 1) + _bins(lon_edges, lon)
    return np.where(valid, rows, -1)


def index_cubed_sphere(xyz: Any, face_basis: Any, u_edges: Any, v_edges: Any) -> np.ndarray:
    """Return face-major row indices for a gnomonic cubed sphere.

    Six face bases have rows (outward normal, u axis, v axis) in storage order.
    Edges are ascending gnomonic ratios shared by the six faces. Ties use the
    first face and the higher bin; uncovered points return -1.
    """
    u, v = _edges(u_edges), _edges(v_edges)
    basis = np.asarray(face_basis, dtype=float)
    q = np.asarray(xyz, dtype=float)
    if basis.shape != (6, 3, 3) or not np.allclose(basis @ basis.transpose(0, 2, 1), np.eye(3)):
        raise ValueError("face_basis must provide six orthonormal face bases")
    if not np.allclose(np.sort(basis[:, 0] @ basis[:, 0].T, axis=1), [-1, 0, 0, 0, 0, 1]):
        raise ValueError("face normals must form three opposing axes")
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all() or np.any(np.max(abs(q), axis=1) == 0):
        raise ValueError("xyz must contain finite nonzero direction vectors")
    faces = np.argmax(q @ basis[:, 0].T, axis=1)
    local = np.einsum("pij,pj->pi", basis[faces], q)
    a = local[:, 1] / local[:, 0]
    b = local[:, 2] / local[:, 0]
    valid = (a >= u[0]) & (a <= u[-1]) & (b >= v[0]) & (b <= v[-1])
    rows = (faces * (len(v) - 1) + _bins(v, b)) * (len(u) - 1) + _bins(u, a)
    return np.where(valid, rows, -1)
