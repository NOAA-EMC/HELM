# SPDX-License-Identifier: Apache-2.0
"""WeightSet — portable serialization of fitted regridding operators.

Native format (``.axisw``, data-model §5)::

    magic(8B "AXISW1") | header_len(4B LE) | JSON header | frame*
    frame = len(8B LE) | WeightCache blob

The JSON header records the generation-time settings (method, norm,
unmapped, skipna, na_thres, periodic, line_type), the source/target grid
identity (family, shape, n_cells, fingerprint), and one frame per weight
matrix (one for scalar regridders, two for the coupled vector pair).

ESMF/SCRIP portability (SC-009): :func:`write_esmf_file` /
:func:`read_esmf_file` wrap the engine NetCDF IO and stamp the same
fingerprints as global attributes. Foreign files (xESMF/ESMF-produced) lack
the attributes and fall back to shape-only validation.

Every load path validates the declared grids **before any data is touched**
(FR-029, SC-006): a fingerprint or shape mismatch raises
:class:`AxisWeightMismatchError` whose message names the offending side and
the fix.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any

from . import _core  # noqa: E402  (module-level for typing; compiled, no cycle)
from .errors import AxisWeightMismatchError
from .grid import Grid

MAGIC = b"AXISW1\x00\x00"  # 8-byte field per data-model §5
VERSION = 1

#: generation-time settings persisted per regridder kind (data-model §5)
_SETTING_KEYS = {
    "scalar": ("method", "norm", "unmapped", "skipna", "na_thres", "periodic", "line_type"),
    "vector": ("method", "periodic", "line_type"),
}


def settings_from_header(header: dict[str, Any]) -> dict[str, Any]:
    keys = _SETTING_KEYS.get(header.get("kind", ""), ())
    return {k: header[k] for k in keys if k in header}


def _grid_meta(grid: Grid) -> dict[str, Any]:
    return {
        "family": grid.family.value,
        "shape": list(grid.shape),
        "n_cells": int(grid.n_cells),
        "fingerprint": grid.fingerprint,
    }


def pack(blobs: list[bytes]) -> bytes:
    out = bytearray()
    for b in blobs:
        out += struct.pack("<Q", len(b))
        out += b
    return bytes(out)


def unpack(payload: bytes, n: int) -> list[bytes]:
    blobs = []
    pos = 0
    for i in range(n):
        if pos + 8 > len(payload):
            raise AxisWeightMismatchError(f"weight file is truncated — expected frame {i + 1} of {n}")
        (size,) = struct.unpack_from("<Q", payload, pos)
        pos += 8
        if pos + size > len(payload):
            raise AxisWeightMismatchError(f"weight file is truncated inside frame {i + 1} of {n}")
        blobs.append(payload[pos : pos + size])
        pos += size
    return blobs


def save_weight_set(
    path: str | os.PathLike[str],
    *,
    kind: str,
    settings: dict[str, Any],
    source: Grid,
    target: Grid,
    matrices: list[tuple[int, int, int]],
    blobs: list[bytes],
) -> None:
    header = {
        "version": VERSION,
        "kind": kind,
        **settings,  # flat per data-model §5: method, norm, unmapped, skipna, na_thres, ...
        "source": _grid_meta(source),
        "target": _grid_meta(target),
        "matrices": [{"n_src": a, "n_dst": b, "nnz": c} for a, b, c in matrices],
    }
    hj = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        fh.write(struct.pack("<I", len(hj)))
        fh.write(hj)
        fh.write(pack(blobs))


def load_weight_set(path: str | os.PathLike[str], *, kind: str, source: Grid, target: Grid) -> tuple[dict[str, Any], list[bytes]]:
    """Read + validate a native weight file against the declared grids.

    Returns (header, blobs). Raises AxisWeightMismatchError on any identity
    mismatch before the caller touches data (FR-029/SC-006)."""
    raw = Path(path).read_bytes()
    if raw[:8] != MAGIC:
        raise AxisWeightMismatchError(
            f"{path} is not an AXIS weight file (magic {raw[:8]!r} != {MAGIC!r}); re-save with save_weights()"
        )
    if len(raw) < 12:
        raise AxisWeightMismatchError(f"{path} is truncated — no header")
    (hlen,) = struct.unpack_from("<I", raw, 8)
    if 12 + hlen > len(raw):
        raise AxisWeightMismatchError(f"{path} is truncated inside the header")
    try:
        header = json.loads(raw[12 : 12 + hlen])
    except json.JSONDecodeError as exc:
        raise AxisWeightMismatchError(f"{path} has a corrupt header: {exc}") from None
    if header.get("version", 0) > VERSION:
        raise AxisWeightMismatchError(f"{path} was written by a newer AXIS (format v{header['version']}); upgrade axis")
    if header.get("kind") != kind:
        raise AxisWeightMismatchError(
            f"{path} holds {header.get('kind')!r} weights, but a {kind!r} regridder was requested — use the matching load_weights"
        )
    _validate_grid(header, "source", source)
    _validate_grid(header, "target", target)
    blobs = unpack(raw[12 + hlen :], len(header["matrices"]))
    for meta, blob in zip(header["matrices"], blobs, strict=True):
        if meta["n_src"] != blob_n_src(blob) or meta["n_dst"] != blob_n_dst(blob):
            raise AxisWeightMismatchError(
                f"{path}: stored matrix shape ({meta['n_src']},{meta['n_dst']}) disagrees with the encoded blob"
            )
    return header, blobs


def blob_n_src(blob: bytes) -> int:
    from . import _core

    return int(_core.Matrix.from_bytes(blob).n_src)


def blob_n_dst(blob: bytes) -> int:
    from . import _core

    return int(_core.Matrix.from_bytes(blob).n_dst)


def _validate_grid(header: dict[str, Any], side: str, grid: Grid) -> None:
    meta = header[side]
    if meta["fingerprint"] != grid.fingerprint:
        raise AxisWeightMismatchError(
            f"{side} grid does not match the saved weights: file was generated for "
            f"{meta['family']}{tuple(meta['shape'])} (fingerprint {meta['fingerprint']}), "
            f"got {grid.family.value}{tuple(grid.shape)} (fingerprint {grid.fingerprint}). "
            f"Pass the original {side} Grid to load_weights(), or regenerate the weights."
        )
    if meta["n_cells"] != int(grid.n_cells):
        raise AxisWeightMismatchError(f"{side} grid cell count {grid.n_cells} != saved {meta['n_cells']}")


# ─── ESMF/SCRIP portable format (SC-009) ─────────────────────────────────────


def write_esmf_file(
    path: str | os.PathLike[str],
    matrix: _core.Matrix,
    *,
    source: Grid,
    target: Grid,
    method: str,
    norm: str = "frac_area",
) -> None:
    from . import _core

    if not getattr(_core, "HAVE_NETCDF", False):
        raise AxisWeightMismatchError("ESMF export unavailable — AXIS built without NetCDF support")
    _core.write_esmf(str(path), matrix)
    import netCDF4

    with netCDF4.Dataset(path, "a") as nc:
        nc.axis_source_fingerprint = source.fingerprint
        nc.axis_target_fingerprint = target.fingerprint
        nc.axis_method = method
        nc.axis_norm = norm
        nc.axis_version = VERSION


def read_esmf_file(path: str | os.PathLike[str], *, source: Grid, target: Grid) -> tuple[Any, dict[str, str]]:
    """Read an ESMF/SCRIP weight file and validate against the grids.

    AXIS-stamped files validate fingerprints; foreign (xESMF/ESMF) files lack
    the attributes and fall back to shape-only validation (SC-009)."""
    from . import _core

    if not getattr(_core, "HAVE_NETCDF", False):
        raise AxisWeightMismatchError("ESMF import unavailable — AXIS built without NetCDF support")
    matrix = _core.read_esmf(str(path))
    attrs: dict[str, str] = {}
    try:
        import netCDF4

        with netCDF4.Dataset(path) as nc:
            for a in ("axis_source_fingerprint", "axis_target_fingerprint", "axis_method", "axis_norm"):
                if a in nc.ncattrs():
                    attrs[a] = str(nc.getncattr(a))
    except ImportError:  # pragma: no cover — netCDF4 absent: engine read still works
        pass
    if "axis_source_fingerprint" in attrs and attrs["axis_source_fingerprint"] != source.fingerprint:
        raise AxisWeightMismatchError(
            f"source grid does not match {path}: file fingerprint {attrs['axis_source_fingerprint']}, grid fingerprint {source.fingerprint}. Pass the original source Grid."
        )
    if "axis_target_fingerprint" in attrs and attrs["axis_target_fingerprint"] != target.fingerprint:
        raise AxisWeightMismatchError(
            f"target grid does not match {path}: file fingerprint {attrs['axis_target_fingerprint']}, grid fingerprint {target.fingerprint}. Pass the original target Grid."
        )
    if int(matrix.n_src) != int(source.n_cells) or int(matrix.n_dst) != int(target.n_cells):
        raise AxisWeightMismatchError(
            f"weight matrix {matrix.n_src}x{matrix.n_dst} does not match grid sizes {source.n_cells}x{target.n_cells}. Pass the grids this file maps between."
        )
    return matrix, attrs
