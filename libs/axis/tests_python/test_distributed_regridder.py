# SPDX-License-Identifier: Apache-2.0
from functools import partial

import numpy as np
import pytest
import xarray as xr
from axis import DistributedRegridder, MeshTile


def cell(west):
    nodes = np.array([[west, 0], [west + 1, 0], [west + 1, 1], [west, 1]], dtype=float)
    return nodes, np.array([0, 4]), np.arange(4)


@pytest.mark.parametrize(("workers", "processes"), [(1, False), (2, True), (4, False)])
def test_distributed_identity_and_unmapped(workers, processes):
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")
    west, east, outside = (MeshTile(1, partial(cell, lon)) for lon in (179, -180, 10))
    data = xr.DataArray(da.from_array([[2.0, 3.0], [5.0, 7.0]], chunks=(1, 2)), dims=("time", "cell"))
    with distributed.Client(n_workers=workers, threads_per_worker=1, processes=processes, dashboard_address=None) as client:
        regridder = DistributedRegridder(client).fit([west, east], [east, west, outside])
        result = regridder.apply(data, dim="cell")
        assert isinstance(result.data, da.Array)
        np.testing.assert_allclose(result.compute(), [[3, 2, 0], [7, 5, 0]], atol=1e-10)
        np.testing.assert_allclose(regridder.apply(data * 2, dim="cell").compute(), result.compute() * 2)


def test_second_order_neighbor_ring_crosses_source_tiles():
    from axis.distributed_regridder import _active_nodes, _ring_rows

    def tagged_cell(west, ids, active=True):
        return (*cell(west), np.array([active], dtype=np.int32), np.array(ids))

    seed = MeshTile(1, partial(tagged_cell, 0, [10, 11, 12, 13]))
    neighbor = MeshTile(1, partial(tagged_cell, 1, [11, 14, 15, 12]))
    second_ring = MeshTile(1, partial(tagged_cell, 2, [14, 16, 17, 15]))
    masked = MeshTile(1, partial(tagged_cell, 1, [11, 14, 15, 12], False))
    first_nodes = _active_nodes(seed, None)
    np.testing.assert_array_equal(_ring_rows(neighbor, first_nodes), [0])
    assert not _ring_rows(second_ring, first_nodes).size
    assert not _ring_rows(masked, first_nodes).size
    np.testing.assert_array_equal(_ring_rows(second_ring, _active_nodes(neighbor, np.array([0]))), [0])


def test_destination_is_split_to_fit_budget():
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")

    def two_cells():
        first, second = cell(0)[0], cell(10)[0]
        return np.vstack((first, second)), np.array([0, 4, 8]), np.arange(8)

    source = [MeshTile(1, partial(cell, west)) for west in (0, 10)]
    data = xr.DataArray(da.from_array([2.0, 3.0], chunks=1), dims=("cell",))
    with distributed.Client(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as client:
        regridder = DistributedRegridder(client, memory_budget=1800).fit(source, [MeshTile(2, two_cells)])
        result = regridder.apply(data, dim="cell")
        assert result.data.chunks == ((1, 1),)
        np.testing.assert_allclose(result.compute(), [2, 3], atol=1e-10)


def test_second_order_weight_block_preserves_reconstruction():
    from axis import axis_py
    from axis.distributed_regridder import _apply, _build

    nodes = np.array([[x, y] for y in range(4) for x in range(4)], dtype=np.float64, order="F")
    indices = np.array(
        [node for j in range(3) for i in range(3) for node in (4 * j + i, 4 * j + i + 1, 4 * j + i + 5, 4 * j + i + 4)]
    )
    source = (nodes, np.arange(0, 37, 4), indices, np.ones(9, dtype=np.int32))

    def shifted_target():
        xy, offsets, conn = cell(1)
        return xy + 0.25, offsets, conn

    target = MeshTile(1, shifted_target)
    block = _build(target, ((*source, np.arange(16)),), "conservative2nd", "dstarea")
    assert block is not None and block[1] is not None
    target_nodes, target_offsets, target_indices = shifted_target()
    matrix = axis_py._generate_local_weight_block(
        axis_py.make_ugrid_mesh(*source),
        axis_py.make_ugrid_mesh(np.asfortranarray(target_nodes), target_offsets, target_indices, np.ones(1, dtype=np.int32)),
        {"method": "conservative2nd", "line_type": "great_circle", "unmapped": "ignore", "norm_type": "dstarea"},
    )
    values = np.arange(9, dtype=np.float64)
    np.testing.assert_allclose(_apply(block, 1, (), values), axis_py.apply_weights(matrix, values), atol=1e-12)


@pytest.mark.parametrize("method", ["nearest_neighbor", "bilinear", "bicubic", "patch", "conservative", "conservative2nd"])
def test_tiled_methods_match_native_reference(method):
    from axis import axis_py
    from axis.distributed_regridder import _subset

    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")
    n = 10
    nodes = np.asfortranarray([(x, y) for y in range(n + 1) for x in range(n + 1)], dtype=np.float64)
    lower = (np.arange(n)[:, None] * (n + 1) + np.arange(n)[None, :]).ravel()
    conn = (lower[:, None] + np.array([0, 1, n + 2, n + 1])).ravel()
    source = (nodes, np.arange(0, 4 * n * n + 1, 4), conn, np.ones(n * n, dtype=np.int32))
    tagged = (*source, np.arange(len(nodes)))
    tiles = [MeshTile(25, partial(_subset, tagged, np.arange(i, i + 25))) for i in range(0, n * n, 25)]

    def destination():
        xy, offsets, indices = cell(4)
        return np.asfortranarray(xy + [0.17, 4.27]), offsets, indices

    config = {"method": method, "line_type": "great_circle", "unmapped": "ignore", "norm_type": "dstarea"}
    matrix = axis_py._generate_local_weight_block(axis_py.make_ugrid_mesh(*source), axis_py.make_ugrid_mesh(*destination()), config)
    values = np.sin(lower * 0.314)
    reference = axis_py.apply_weights(matrix, values)
    data = xr.DataArray(da.from_array(values, chunks=12), dims=("cell",))
    with distributed.Client(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as client:
        locator = DistributedRegridder(client, method=method).fit(tiles, [MeshTile(1, destination)])
        np.testing.assert_allclose(locator.apply(data, dim="cell").compute(), reference, rtol=1e-8, atol=1e-10)


def test_apply_gathers_only_columns_used_by_weight_block():
    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")

    def two_cells():
        return np.vstack([cell(0)[0], cell(10)[0]]), np.array([0, 4, 8]), np.arange(8)

    data = xr.DataArray(da.from_array([2.0, np.nan], chunks=2), dims=("cell",))
    with distributed.Client(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as client:
        regridder = DistributedRegridder(client).fit([MeshTile(2, two_cells)], [MeshTile(1, partial(cell, 0))])
        _, requests, _, columns = regridder._blocks[0]
        np.testing.assert_array_equal(columns, [0])
        assert len(requests) == 1
        np.testing.assert_array_equal(requests[0][1], [0])
        np.testing.assert_allclose(regridder.apply(data, dim="cell").compute(), [2.0])
