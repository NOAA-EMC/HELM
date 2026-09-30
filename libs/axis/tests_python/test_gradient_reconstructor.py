import numpy as np
from axis import axis_py


def test_gradient_reconstruct_linear_field():
    # 5 grid cell mesh (3D non-degenerate coords to prevent Cramer's rule determinant being 0 due to flat 2D layout)
    centroids = np.array(
        [
            [0.5, 0.5, 1.0],  # Cell 0
            [1.5, 0.5, 2.0],  # Cell 1
            [0.5, 1.5, 3.0],  # Cell 2
            [1.5, 1.5, 4.0],  # Cell 3
            [1.0, 1.0, 5.0],  # Cell 4
        ],
        dtype=np.float64,
    )

    # Linear field f(x, y, z) = 2x + 3y + 4z
    values = 2.0 * centroids[:, 0] + 3.0 * centroids[:, 1] + 4.0 * centroids[:, 2]

    # Adjacency CSR structure
    adj_offsets = np.array([0, 4, 8, 12, 16, 20], dtype=np.int64)
    adj_indices = np.array(
        [
            1,
            2,
            3,
            4,  # Cell 0 neighbors
            0,
            2,
            3,
            4,  # Cell 1 neighbors
            0,
            1,
            3,
            4,  # Cell 2 neighbors
            0,
            1,
            2,
            4,  # Cell 3 neighbors
            0,
            1,
            2,
            3,  # Cell 4 neighbors
        ],
        dtype=np.int64,
    )

    grad = axis_py.reconstruct_gradient(values, centroids, adj_offsets, adj_indices, False)

    # Verify reconstructed gradients (Expected x=2, y=3, z=4)
    assert grad.shape == (5, 3)
    for c in range(5):
        assert np.isclose(grad[c, 0], 2.0, rtol=1e-5)
        assert np.isclose(grad[c, 1], 3.0, rtol=1e-5)
        assert np.isclose(grad[c, 2], 4.0, rtol=1e-5)
