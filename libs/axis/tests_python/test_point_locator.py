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
