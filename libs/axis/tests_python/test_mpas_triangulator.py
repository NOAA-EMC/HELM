import numpy as np
from axis import axis_py


def test_poly_triangulation():
    # Define a single pentagon cell centered at (0,0)
    # 5 nodes (vertices)
    node_coords = np.array([[0.0, 1.0], [1.0, 0.5], [0.5, -0.5], [-0.5, -0.5], [-1.0, 0.5]], dtype=np.float64)

    # 1 cell, 5 edges
    conn_raw = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    n_edges = np.array([5], dtype=np.int64)

    res = axis_py.triangulate_poly_cells(node_coords, conn_raw, n_edges)
    assert "conn_offsets" in res
    assert "conn_indices" in res
    # 5 edges triangle fan has max_edges - 2 = 3 triangles, total 3*3 = 9 indices
    assert len(res["conn_indices"]) == 9
