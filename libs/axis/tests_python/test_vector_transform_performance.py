import numpy as np
from axis import axis_py


def test_fused_vector_transform():
    src = axis_py.make_regular_mesh(4, 4, 0.0, -90.0, 90.0, 45.0)
    dst = axis_py.make_regular_mesh(2, 2, 0.0, -90.0, 180.0, 90.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    src_alpha = np.zeros(src.n_cells)
    dst_alpha = np.zeros(dst.n_cells)
    W_u, W_v = axis_py.generate_vector_weights(src, dst, src_alpha, dst_alpha, config)

    # 10 variables, cells x variables
    u = np.ones((src.n_cells, 10))
    v = np.ones((src.n_cells, 10))

    u_out, v_out = axis_py.vector_transform(W_u, W_v, u, v)
    assert u_out.shape == (dst.n_cells, 10)
    assert v_out.shape == (dst.n_cells, 10)
