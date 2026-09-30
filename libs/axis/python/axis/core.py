# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np

from . import axis_py

# Worker-local cache for sparse weight matrices to optimize Dask performance
_WORKER_CACHE: dict[Any, Any] = {}


def _setup_worker_cache(key, obj):
    """Setup a shared object in the worker-local cache."""
    _WORKER_CACHE[key] = obj
    return True


def _sync_cache_from_worker_data(future_key, cache_key):
    """Retrieve dask future data from worker and save to local cache."""
    try:
        import dask.distributed

        worker = dask.distributed.get_worker()
        data = worker.data[future_key]
        _WORKER_CACHE[cache_key] = data
        return True
    except Exception:
        return False


def _apply_weights_core(
    data_block: np.ndarray,
    weights_matrix,
    dims_source: tuple,
    shape_target: tuple,
    skipna: bool = False,
    total_weights: np.ndarray | None = None,
    na_thres: float = 1.0,
    weights_key: str | None = None,
) -> np.ndarray:
    """
    Apply regridding weights to a NumPy data block.

    Handles both standard regridding and NaN-aware re-normalization
    using AXIS's C++ SpMV execution path.
    """
    # Cache retrieval
    if isinstance(weights_matrix, str):
        weights_key = weights_matrix
        weights_matrix = _WORKER_CACHE.get(weights_key)

    if weights_matrix is None:
        raise RuntimeError(f"Weights key '{weights_key}' not found in worker cache.")

    if isinstance(total_weights, str):
        tw_key = total_weights
        total_weights = _WORKER_CACHE.get(tw_key)
        if total_weights is None:
            raise RuntimeError(f"Total weights key '{tw_key}' not found in worker cache.")

    original_shape = data_block.shape
    n_source_dims = len(dims_source)
    spatial_shape = original_shape[len(original_shape) - n_source_dims :]
    other_dims_shape = original_shape[: len(original_shape) - n_source_dims]
    n_spatial = int(np.prod(spatial_shape))
    n_other = int(np.prod(other_dims_shape))

    # Reshape to 2D (other x spatial)
    flat_data = data_block.reshape(n_other, n_spatial)

    if n_spatial == 0 or n_other == 0:
        new_shape = other_dims_shape + shape_target
        return np.full(new_shape, np.nan, dtype=data_block.dtype)

    # AXIS batch_apply expects (n_src, n_vars).
    # flat_data is (n_vars, n_src), so we transpose to (n_src, n_vars)!
    flat_data_t = np.asfortranarray(flat_data.T)

    if not skipna:
        # Standard fast path: Apply weights directly in C++
        result_t = axis_py.batch_apply(weights_matrix, flat_data_t)
        result = result_t.T
    else:
        # NaN-aware re-normalization path using C++ fused solver
        if not weights_matrix.is_csr:
            weights_matrix.to_csr()
        result_t = axis_py.nan_batch_apply(weights_matrix, flat_data_t, na_thres)
        result = result_t.T

    new_shape = other_dims_shape + shape_target
    return result.reshape(new_shape).astype(data_block.dtype, copy=False)
