# SPDX-License-Identifier: Apache-2.0
import axis
import numpy as np
import pytest


def test_registration_round_trip_and_remap(tmp_path):
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 3.0]])
    registration = axis.PointRegistration(np.array([9, 2, 1]), np.array([100, 300, -1]))
    path = tmp_path / "points.npz"
    registration.save(path, coordinates=coords, grid_key="grid-v1", lookup_key="containment")
    loaded = axis.PointRegistration.load(
        path, point_ids=registration.point_ids, coordinates=coords, grid_key="grid-v1", lookup_key="containment"
    )
    np.testing.assert_array_equal(loaded.local_indices([300, 100]), [1, 0, -1])
    with pytest.raises(ValueError, match="does not match"):
        axis.PointRegistration.load(
            path, point_ids=registration.point_ids, coordinates=coords + 1, grid_key="grid-v1", lookup_key="containment"
        )


def test_direct_grid_indices():
    coords = [[-181, -45], [181, 45], [180, 90], [0, 91]]
    np.testing.assert_array_equal(axis.index_latlon_grid(coords, [-180, 0, 180], [-90, 0, 90], periodic=True), [1, 2, 2, -1])
    basis = np.array(
        [
            [np.eye(3)[axis_] * sign, np.eye(3)[(axis_ + 1) % 3], np.eye(3)[(axis_ + 2) % 3]]
            for axis_ in range(3)
            for sign in (1, -1)
        ]
    )
    points = np.array([[1.0, -0.2, 0.2], [-1.0, 0.2, -0.2], [0.2, 0.2, 1.0]])
    np.testing.assert_array_equal(axis.index_cubed_sphere(points, basis, [-1, 0, 1], [-1, 0, 1]), [2, 5, 19])


def test_worker_shards_recover_after_cache_loss(tmp_path):
    from axis.point_locator import _release_worker_locators

    da = pytest.importorskip("dask.array")
    distributed = pytest.importorskip("distributed")
    path = tmp_path / "shard.npz"
    np.savez(
        path,
        node_coords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
        conn_offsets=np.array([0, 4]),
        conn_indices=np.arange(4),
        cell_ids=np.array([10]),
    )

    def load():
        with np.load(path) as data:
            return {k: data[k] for k in ("node_coords", "conn_offsets", "conn_indices")}, data["cell_ids"]

    with distributed.Client(n_workers=2, threads_per_worker=1, processes=False, dashboard_address=None) as client:
        with axis.DistributedPointLocator.from_shards(
            [load],
            coordinate_system=axis.CoordinateSystem.Cartesian3D,
            boundary_model=axis.BoundaryModel.CartesianStraight,
            client=client,
        ) as locator:
            client.run(_release_worker_locators, locator._token)
            result = locator.locate(da.from_array([2], chunks=1), da.from_array([[0.5, 0.5]], chunks=(1, 2)))
            np.testing.assert_array_equal(result.cell_ids.compute(), [10])
        with pytest.raises(ValueError, match="globally unique"):
            axis.DistributedPointLocator.from_shards(
                [load, load],
                coordinate_system=axis.CoordinateSystem.Cartesian3D,
                boundary_model=axis.BoundaryModel.CartesianStraight,
                client=client,
            )
