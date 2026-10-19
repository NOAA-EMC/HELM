# SPDX-License-Identifier: Apache-2.0
"""Dask-aware execution core: worker-local weight caches and array block helpers.

Provides the shared machinery behind ``Regridder``: a per-worker cache for
serialized AXIS matrices, validation of core-dimension chunking for blocked
apply, and a Dask worker plugin that installs weights once per worker instead
of shipping them with every task.
"""
from typing import Any

import numpy as np

from . import axis_py

# Worker-local cache for sparse weight matrices to optimize Dask performance
_WORKER_CACHE: dict[Any, Any] = {}


def _validate_core_dimension_chunks(data, core_dims, operation: str, input_name: str = "input") -> None:
    """Require each generalized-ufunc core dimension to be a single Dask chunk.

    The check relies only on xarray's ``chunksizes`` metadata and deliberately
    does not import Dask, so eager NumPy inputs keep Dask fully optional.
    """
    chunksizes = getattr(data, "chunksizes", None)
    if not chunksizes:
        return

    split_dimensions = {
        dim: tuple(chunksizes[dim])
        for dim in core_dims
        if dim in chunksizes and len(chunksizes[dim]) > 1
    }
    if not split_dimensions:
        return

    details = ", ".join(f"{dim}={chunks}" for dim, chunks in split_dimensions.items())
    rechunk = ", ".join(f"'{dim}': -1" for dim in split_dimensions)
    raise ValueError(
        f"AXIS {operation} requires each core dimension of {input_name} to be a single Dask chunk; "
        f"found {details}. Rechunk explicitly before regridding, for example "
        f"data.chunk({{{rechunk}}}), and keep independent batch dimensions chunked."
    )


def _setup_worker_cache(key, obj):
    """Setup a shared object in the worker-local cache."""
    _WORKER_CACHE[key] = obj
    return True


def _register_weight_plugin(client, plugin_name, weights_key, weights_bytes, total_key=None, total_weights=None):
    """Install serialized AXIS weights through Dask's worker lifecycle API."""
    from distributed.diagnostics.plugin import WorkerPlugin

    class AxisWeightPlugin(WorkerPlugin):
        """Worker lifecycle hook that materializes AXIS weights in the worker cache."""

        def __init__(self, name, serialized_matrix, sum_key, sums):
            """Store the cache key, serialized matrix, and optional skipna sums."""
            self.name = name
            self.serialized_matrix = serialized_matrix
            self.sum_key = sum_key
            self.sums = sums

        def setup(self, worker):
            """Deserialize the matrix (and sums) into this worker's cache."""
            from axis import axis_py
            from axis.core import _WORKER_CACHE

            _WORKER_CACHE[self.name] = axis_py.Matrix.from_bytes(self.serialized_matrix)
            if self.sum_key is not None:
                _WORKER_CACHE[self.sum_key] = self.sums

        def teardown(self, worker):
            """Drop this plugin's cache entries when the plugin is removed."""
            from axis.core import _WORKER_CACHE

            _WORKER_CACHE.pop(self.name, None)
            if self.sum_key is not None:
                _WORKER_CACHE.pop(self.sum_key, None)

    plugin = AxisWeightPlugin(weights_key, weights_bytes, total_key, total_weights)
    register_plugin = getattr(client, "register_plugin", None)
    if register_plugin is not None:
        register_plugin(plugin, name=plugin_name)
    else:
        client.register_worker_plugin(plugin, name=plugin_name)


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
        raise RuntimeError(
            f"AXIS weights '{weights_key}' are unavailable in this process. Supported schedulers are "
            "synchronous, threads, and Dask Distributed. The local multiprocessing scheduler is not supported; "
            "use threads or install AXIS with the 'distributed' extra and compute with a Dask Client."
        )

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
