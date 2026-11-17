"""Point identity, geometry semantics, and NumPy ownership at the Python boundary."""

import gc

import axis
import numpy as np
import pytest


def point_mesh(scale=1.0):
    nodes = np.array([[0, 0], [scale, 0], [scale, scale], [0, scale]], dtype=float)
    return axis.make_ugrid_mesh(
        nodes,
        np.array([0, 4], dtype=np.int64),
        np.arange(4, dtype=np.int64),
        coordinate_system=axis.CoordinateSystem.Cartesian3D,
        boundary_model=axis.BoundaryModel.CartesianStraight,
    )


def test_preserves_coincident_points_and_result_memory():
    locator = axis.PointLocator(point_mesh(), [42], tolerance=0)
    # Sliced input exercises noncontiguous arrays and normal NumPy row ordering.
    coordinates = np.array([[0.5, 0.5], [0, 0], [0.5, 0.5], [0, 0], [2, 2]])[::2]
    result = locator.locate([90, 12, 8], coordinates, coordinate_system=axis.CoordinateSystem.Cartesian3D)
    del locator
    gc.collect()
    np.testing.assert_array_equal(result.point_ids, [90, 12, 8])
    np.testing.assert_array_equal(result.cell_ids, [42, 42, -1])
    np.testing.assert_array_equal(result.status, [axis.PointLocationStatus.Interior] * 2 + [axis.PointLocationStatus.Unmapped])
    np.testing.assert_array_equal(result.owner_ranks, [0, 0, -1])
    np.testing.assert_array_equal(result.owner_local_indices, [0, 0, -1])


def test_empty_input_and_invalid_ids():
    locator = axis.PointLocator(point_mesh(), [2])
    result = locator.locate([], np.empty((0, 2)), coordinate_system=axis.CoordinateSystem.Cartesian3D)
    assert result.cell_ids.shape == (0,)
    with pytest.raises(ValueError):
        locator.locate([1.2], [[0.5, 0.5]], coordinate_system=axis.CoordinateSystem.Cartesian3D)
    with pytest.raises(ValueError):
        locator.locate([1, 1], [[0.5, 0.5], [0.5, 0.5]], coordinate_system=axis.CoordinateSystem.Cartesian3D)
    with pytest.raises(ValueError):
        locator.locate([1], [[0.5]], coordinate_system=axis.CoordinateSystem.Cartesian3D)


def test_zero_tolerance_spherical_vertices():
    nodes = np.array([[10, 40], [12, 40], [12, 42], [10, 42]], dtype=float)
    mesh = axis.make_ugrid_mesh(
        nodes, np.array([0, 4], np.int64), np.arange(4, dtype=np.int64), boundary_model=axis.BoundaryModel.GreatCircle
    )
    result = axis.PointLocator(mesh, [42], tolerance=0).locate([7, 8, 9, 10], nodes)
    np.testing.assert_array_equal(result.cell_ids, [42] * 4)
    np.testing.assert_array_equal(result.status, [axis.PointLocationStatus.Boundary] * 4)


def test_python_oblique_scale_regression():
    scale = 1e6
    nodes = scale * np.array([[0, 0, 0], [1, 2, 3], [3, 3, 5], [2, 1, 2]], dtype=float)
    mesh = axis.make_ugrid_mesh(
        nodes,
        np.array([0, 4], np.int64),
        np.arange(4, dtype=np.int64),
        coordinate_system=axis.CoordinateSystem.Cartesian3D,
        boundary_model=axis.BoundaryModel.CartesianStraight,
    )
    result = axis.PointLocator(mesh, [99], tolerance=0).locate(
        [2], scale * np.array([[1.5, 1.5, 2.5]]), coordinate_system=axis.CoordinateSystem.Cartesian3D
    )
    assert result.cell_ids.tolist() == [99]


def test_native_cell_centroids_use_canonical_coordinates():
    cartesian = point_mesh()
    cartesian_centroids = axis.axis_py._mesh_cell_centroids(cartesian)
    np.testing.assert_allclose(cartesian_centroids, [[0.5, 0.5, 0.0]])

    spherical = axis.make_ugrid_mesh(
        np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float),
        np.array([0, 4], dtype=np.int64),
        np.arange(4, dtype=np.int64),
        boundary_model=axis.BoundaryModel.GreatCircle,
    )
    sphere_centroids = axis.axis_py._mesh_cell_centroids(spherical)
    np.testing.assert_allclose(np.linalg.norm(sphere_centroids, axis=1), [1.0])
    assert np.all(sphere_centroids[0] > 0.0)


def test_native_cell_extraction_remaps_connectivity_and_metadata():
    mesh = axis.make_ugrid_mesh(
        np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]], dtype=float),
        np.array([0, 4, 8], dtype=np.int64),
        np.array([0, 1, 2, 3, 1, 4, 5, 2], dtype=np.int64),
        coordinate_system=axis.CoordinateSystem.Cartesian3D,
        boundary_model=axis.BoundaryModel.CartesianStraight,
    )
    shard = axis.axis_py._mesh_extract_cells(mesh, np.array([1], dtype=np.int64))

    np.testing.assert_array_equal(shard["conn_offsets"], [0, 4])
    np.testing.assert_array_equal(shard["conn_indices"], [0, 1, 2, 3])
    np.testing.assert_array_equal(shard["node_coords"], [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]])
    shard_mesh = axis.make_ugrid_mesh(
        shard["node_coords"],
        shard["conn_offsets"],
        shard["conn_indices"],
        coordinate_system=mesh.coordinate_system,
        boundary_model=mesh.boundary_model,
    )
    result = axis.PointLocator(shard_mesh, [99]).locate(
        [1], [[1.5, 0.5]], coordinate_system=axis.CoordinateSystem.Cartesian3D
    )
    np.testing.assert_array_equal(result.cell_ids, [99])


def test_native_locator_bounds_include_tolerance():
    native = axis.axis_py._PointLocator(point_mesh(), np.array([1], dtype=np.int64), 1e-3, 64, 24, 1)

    bounds = native.bounds()

    assert bounds[0] <= -5e-4
    assert bounds[1] <= 0.0
    assert bounds[3] >= 1.0
    assert bounds[4] >= 1.0


def test_partition_cells_is_balanced_and_deterministic():
    from axis.point_locator import _partition_cells

    centroids = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0]], dtype=float)
    cell_ids = np.array([60, 10, 50, 20, 40, 30], dtype=np.int64)

    first = _partition_cells(centroids, cell_ids, 3)
    second = _partition_cells(centroids, cell_ids, 3)

    assert len(first) == 3
    assert [partition.size for partition in first] == [2, 2, 2]
    assert sorted(np.concatenate(first).tolist()) == list(range(6))
    for partition in first:
        np.testing.assert_array_equal(cell_ids[partition], np.sort(cell_ids[partition]))
    for first_partition, second_partition in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_partition, second_partition)


def test_serial_owner_ranks_aliases_owner_partitions():
    result = axis.PointLocator(point_mesh(), [42]).locate(
        [7], [[0.5, 0.5]], coordinate_system=axis.CoordinateSystem.Cartesian3D
    )

    np.testing.assert_array_equal(result.owner_partitions, [0])
    np.testing.assert_array_equal(result.owner_ranks, result.owner_partitions)
