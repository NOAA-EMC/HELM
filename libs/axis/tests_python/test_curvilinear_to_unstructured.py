import numpy as np
from axis import axis_py


def test_structured_grid_binding():
    cx = np.array([0.5, 1.5, 0.5, 1.5])
    cy = np.array([0.5, 0.5, 1.5, 1.5])

    grid = axis_py.StructuredGrid(2, 2, cx, cy)
    mesh = grid.to_unstructured()
    assert mesh.n_cells == 4
    assert mesh.n_nodes == 9
