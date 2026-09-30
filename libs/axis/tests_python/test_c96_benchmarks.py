# SPDX-License-Identifier: Apache-2.0
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from axis import _core

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

try:
    from fetch_c96 import fetch_c96_tiles

    C96_FILES = fetch_c96_tiles()
    _FETCH_ERROR = None
except Exception as exc:  # noqa: BLE001 - skip (not error) when tiles cannot be fetched
    C96_FILES = []
    _FETCH_ERROR = str(exc)

pytestmark = pytest.mark.skipif(
    len(C96_FILES) < 6,
    reason=f"C96_grid tile NetCDF files could not be fetched: {_FETCH_ERROR}"
    if _FETCH_ERROR
    else "C96_grid tile NetCDF files not found",
)


def get_c96_exact_mesh(ds_list):
    """Build a unified AXIS UnstructuredMesh from all 6 C96 tiles using exact supergrid corners."""
    node_coords_list = []
    conn_indices_list = []
    node_offset = 0

    for ds in ds_list:
        clon_t = ds["x"].values[0::2, 0::2]
        clat_t = ds["y"].values[0::2, 0::2]
        coords_t = np.column_stack([clon_t.ravel(), clat_t.ravel()])
        node_coords_list.append(coords_t)

        ni, nj = 96, 96
        nip1 = ni + 1
        i_grid, j_grid = np.meshgrid(np.arange(ni), np.arange(nj))
        bl = (i_grid + j_grid * nip1 + node_offset).ravel()
        br = ((i_grid + 1) + j_grid * nip1 + node_offset).ravel()
        tr = ((i_grid + 1) + (j_grid + 1) * nip1 + node_offset).ravel()
        tl = (i_grid + (j_grid + 1) * nip1 + node_offset).ravel()

        indices_t = np.column_stack([bl, br, tr, tl]).astype(np.int64).ravel()
        conn_indices_list.append(indices_t)
        node_offset += len(coords_t)

    node_coords = np.asfortranarray(np.concatenate(node_coords_list))
    conn_indices = np.concatenate(conn_indices_list)
    conn_offsets = np.arange(0, len(conn_indices) + 1, 4, dtype=np.int64)

    return _core.make_ugrid_mesh(node_coords, conn_offsets, conn_indices)


def test_c96_to_global_regular_conservative():
    """Verify conservative regridding from all 6 C96 tiles to a global 1-degree regular grid."""
    ds_list = [xr.open_dataset(f) for f in C96_FILES]

    mesh_src = get_c96_exact_mesh(ds_list)
    mesh_dst = _core.make_regular_mesh(360, 180, 0.0, -90.0, 1.0, 1.0)

    config = {
        "method": _core.Method.Conservative,
        "periodic": False,
        "line_type": _core.LineType.GreatCircle,
        "norm_type": _core.NormType.FracArea,
        "unmapped": _core.UnmappedAction.Ignore,
    }

    weights = _core.generate_weights(mesh_src, mesh_dst, config)
    src_const = np.ones(6 * 96 * 96, dtype=np.float64)
    out_const = np.array(_core.apply_weights(weights, src_const)).reshape((180, 360))

    # Verify global mass conservation on constant field
    assert np.allclose(out_const, 1.0, atol=1e-12), f"Max diff vs 1.0: {np.max(np.abs(out_const - 1.0))}"
    assert np.isnan(out_const).sum() == 0, "Output contains NaNs"


def test_global_regular_to_c96_conservative():
    """Verify conservative regridding from a global 1-degree regular grid to each C96 tile."""
    ds_list = [xr.open_dataset(f) for f in C96_FILES]

    mesh_src = _core.make_regular_mesh(360, 180, 0.0, -90.0, 1.0, 1.0)
    src_const = np.ones(180 * 360, dtype=np.float64)

    config = {
        "method": _core.Method.Conservative,
        "periodic": False,
        "line_type": _core.LineType.GreatCircle,
        "norm_type": _core.NormType.FracArea,
        "unmapped": _core.UnmappedAction.Ignore,
    }

    for t_idx in range(6):
        mesh_dst = get_c96_exact_mesh([ds_list[t_idx]])
        weights = _core.generate_weights(mesh_src, mesh_dst, config)
        out_const = np.array(_core.apply_weights(weights, src_const)).reshape((96, 96))

        assert np.allclose(out_const, 1.0, atol=1e-12), f"Tile {t_idx + 1} max diff vs 1.0: {np.max(np.abs(out_const - 1.0))}"
