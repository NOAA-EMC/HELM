# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np

from . import _core

# Worker-local cache for sparse weight matrices to optimize Dask performance
_WORKER_CACHE: dict[Any, Any] = {}

# SC-005 observability: per-key count of weight (de)serialization events on
# this worker/process. Provisioning and lazy in-process install share it.
_WORKER_SYNC_COUNTS: dict[str, int] = {}


def _setup_worker_cache(key: Any, obj: Any) -> bool:
    """Setup a shared object in the worker-local cache."""
    _WORKER_CACHE[key] = obj
    return True


def _sync_cache_from_worker_data(future_key: str, cache_key: str) -> bool:
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
    weights_matrix: _core.Matrix | str,
    dims_source: tuple[str, ...],
    shape_target: tuple[int, ...],
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
    matrix: _core.Matrix | None = weights_matrix if isinstance(weights_matrix, _core.Matrix) else None
    if matrix is None:
        weights_key = weights_matrix if isinstance(weights_matrix, str) else weights_key
        matrix = _WORKER_CACHE.get(weights_key) if weights_key is not None else None

    if matrix is None:
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
        result_t = _core.batch_apply(matrix, flat_data_t)
        result = result_t.T
    else:
        # NaN-aware re-normalization path
        mask = np.isnan(flat_data)
        zero = flat_data.dtype.type(0)
        safe_data = np.where(mask, zero, flat_data)

        result_t = _core.batch_apply(matrix, np.asfortranarray(safe_data.T))
        result = result_t.T

        # Calculate weight sums of valid (non-NaN) inputs using float32 masks
        valid_mask = np.logical_not(mask).astype(np.float32)
        weights_sum_t = _core.batch_apply(matrix, np.asfortranarray(valid_mask.T))
        weights_sum = weights_sum_t.T

        with np.errstate(divide="ignore", invalid="ignore"):
            result /= weights_sum
            if total_weights is not None:
                fraction_valid = weights_sum / total_weights
                nan_val = result.dtype.type(np.nan)
                result = np.where(fraction_valid < (1.0 - na_thres - 1e-6), nan_val, result)

    new_shape = other_dims_shape + shape_target
    return result.reshape(new_shape).astype(data_block.dtype, copy=False)
