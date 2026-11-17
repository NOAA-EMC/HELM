"""Run with mpiexec -n 1/2/4: independent Python point and mesh partitions."""

import gc

import axis
import numpy as np
from mpi4py import MPI


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    nodes, offsets, indices, cells = [], [0], [], []
    for cell in range(3):
        if cell % size == rank:
            start = len(nodes)
            nodes.extend([[cell, 0], [cell + 1, 0], [cell + 1, 1], [cell, 1]])
            indices.extend(range(start, start + 4))
            offsets.append(len(indices))
            cells.append(100 + cell)
    mesh = axis.make_ugrid_mesh(
        np.array(nodes, float).reshape(-1, 2),
        np.array(offsets, np.int64),
        np.array(indices, np.int64),
        coordinate_system=axis.CoordinateSystem.Cartesian3D,
        boundary_model=axis.BoundaryModel.CartesianStraight,
    )
    # At 4 ranks, ranks 2 and 3 intentionally own no query points.
    ids = np.array([p for p in range(9) if p % min(size, 2) == rank], np.int64)
    xy = np.array([[p % 3 + 0.5, 0.5] for p in ids], float).reshape(-1, 2)
    with axis.DistributedPointLocator(mesh, cells, comm, max_message_bytes=48, query_batch_size=2) as locator:
        result = locator.locate(ids, xy)
        np.testing.assert_array_equal(result.point_ids, ids)
        np.testing.assert_array_equal(result.cell_ids, 100 + ids % 3)
        np.testing.assert_array_equal(result.owner_ranks, (ids % 3) % size)
        # A Python-level validation error on one rank must fail collectively.
        bad_xy = [[0.5]] if rank == 0 else np.empty((0, 2))
        bad_ids = [123] if rank == 0 else []
        try:
            locator.locate(bad_ids, bad_xy)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid shape did not fail collectively")
        wrong_family = axis.CoordinateSystem.SphericalDeg if rank == 0 else axis.CoordinateSystem.Cartesian3D
        try:
            locator.locate(ids, xy, coordinate_system=wrong_family)
        except ValueError:
            pass
        else:
            raise AssertionError("coordinate-family mismatch did not fail collectively")
        assert locator.locate([], np.empty((0, 2))).cell_ids.size == 0
    gc.collect()
    np.testing.assert_array_equal(result.cell_ids, 100 + ids % 3)
    try:
        locator.locate(ids, xy)
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed locator accepted a query")

    # Spherical input units may differ by rank while the coordinate family
    # remains the same. The native locator converts each rank independently.
    sphere_nodes = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], float) if rank == 0 else np.array([[0, 0]], float)
    sphere_offsets = np.array([0, 4], np.int64) if rank == 0 else np.array([0], np.int64)
    sphere_indices = np.arange(4, dtype=np.int64) if rank == 0 else np.empty(0, np.int64)
    sphere_cells = np.array([55], np.int64) if rank == 0 else np.empty(0, np.int64)
    sphere_mesh = axis.make_ugrid_mesh(
        sphere_nodes,
        sphere_offsets,
        sphere_indices,
        coordinate_system=axis.CoordinateSystem.SphericalDeg,
        boundary_model=axis.BoundaryModel.GreatCircle,
    )
    with axis.DistributedPointLocator(sphere_mesh, sphere_cells, comm) as locator:
        units = axis.CoordinateSystem.SphericalDeg if rank % 2 == 0 else axis.CoordinateSystem.SphericalRad
        angle = 0.5 if rank % 2 == 0 else np.deg2rad(0.5)
        result = locator.locate([rank], [[angle, angle]], coordinate_system=units)
        np.testing.assert_array_equal(result.cell_ids, [55])
        np.testing.assert_array_equal(result.status, [axis.PointLocationStatus.Interior])
    if rank == 0:
        print(f"Python point locator: {size} MPI ranks passed", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
