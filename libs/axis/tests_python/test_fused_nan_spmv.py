import numpy as np
from axis import axis_py


def test_fused_nan_batch_apply():
    # Construct a simple regular mesh
    src = axis_py.make_regular_mesh(4, 4, 0.0, -90.0, 90.0, 45.0)
    dst = axis_py.make_regular_mesh(2, 2, 0.0, -90.0, 180.0, 90.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    matrix = axis_py.generate_weights(src, dst, config)
    matrix.to_csr()

    # Create source data with a NaN value
    src_data = np.ones((matrix.n_src, 2), dtype=np.float64)
    src_data[0, 0] = np.nan  # Inject NaN in variable 0

    # Ensure python binding works
    res = axis_py.nan_batch_apply(matrix, src_data, 1.0)
    assert res.shape == (matrix.n_dst, 2)


def test_fused_nan_correctness():
    # Construct a mesh
    src = axis_py.make_regular_mesh(10, 5, 0.0, -90.0, 36.0, 36.0)
    dst = axis_py.make_regular_mesh(5, 3, 0.0, -90.0, 72.0, 60.0)

    config = {"method": "bilinear", "unmapped": "ignore"}
    matrix = axis_py.generate_weights(src, dst, config)
    matrix.to_csr()

    # Source data with various NaNs
    src_data = np.ones((matrix.n_src, 3), dtype=np.float64)
    src_data[::3, 0] = np.nan
    src_data[1::4, 1] = np.nan
    src_data[::2, 2] = np.nan

    # 1. Manual baseline (the previous Python-side math)
    flat_data = src_data.T
    mask = np.isnan(flat_data)
    safe_data = np.where(mask, 0.0, flat_data)
    result_t = axis_py.batch_apply(matrix, np.asfortranarray(safe_data.T))
    result = result_t.T

    valid_mask = np.logical_not(mask).astype(np.float32)
    weights_sum_t = axis_py.batch_apply(matrix, np.asfortranarray(valid_mask.T))
    weights_sum = weights_sum_t.T

    total_weights = np.array(axis_py.apply_weights(matrix, np.ones(matrix.n_src))).flatten()

    with np.errstate(divide="ignore", invalid="ignore"):
        result /= weights_sum
        # total weights fraction threshold check (na_thres = 0.5)
        # weights_sum has shape (3, n_dst) because it is result.T (which was result_t.T, where result_t is shape (n_dst, 3))
        # wait! result is shape (3, n_dst), total_weights is shape (n_dst,)
        # Let's broadcast correctly:
        fraction_valid = weights_sum / total_weights[None, :]
        expected = np.where(fraction_valid < (1.0 - 0.5 - 1e-6), np.nan, result)

    # 2. C++ Fused execution
    actual_t = axis_py.nan_batch_apply(matrix, src_data, 0.5)
    actual = actual_t.T

    # Verify identical matching (both values and NaN locations)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, equal_nan=True)
