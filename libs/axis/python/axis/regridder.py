# SPDX-License-Identifier: Apache-2.0
"""Public horizontal regridding API built on the compiled AXIS engine.

Defines :class:`Regridder`, which generates or loads interpolation weights and
applies them to NumPy, xarray, or Dask-backed fields, including distributed
execution where weights are installed once per worker.
"""

import logging
import time
import uuid
import weakref
from typing import Any, cast

import numpy as np
import xarray as xr

from . import axis_py
from .core import _WORKER_CACHE, _apply_weights_core, _register_weight_plugin, _setup_worker_cache, _validate_core_dimension_chunks
from .grid import (
    CurvilinearGrid,
    Geometry,
    RectilinearGrid,
    UnstructuredMesh,
    _get_mesh_info,
)

logger = logging.getLogger("axis")


class Regridder:
    """
    Exascale-ready spatial regridder wrapping AXIS's compiled Kokkos-parallel engine.

    Supports NumPy eager arrays, xarray Datasets/DataArrays, and Dask-backed lazy
    arrays on synchronous, threaded, or Dask Distributed schedulers.
    """

    source_grid_ds: xr.Dataset | None
    target_grid_ds: xr.Dataset | None

    def __init__(
        self,
        ds_in: Geometry | xr.Dataset | dict | None = None,
        ds_out: Geometry | xr.Dataset | dict | None = None,
        method: str = "bilinear",
        periodic: bool = False,
        unmapped: str = "ignore",
        skipna: bool = False,
        na_thres: float = 1.0,
        line_type: str = "great_circle",
        weights_file: str | None = None,
        src_mask: np.ndarray | xr.DataArray | None = None,
        dst_mask: np.ndarray | xr.DataArray | None = None,
        norm_type: str = "fracarea",
    ):
        """
        Initialize the regridder by generating spatial interpolation weights in C++,
        or by reloading pre-computed weights from a file.
        """
        self.method = method
        self.periodic = periodic
        self.unmapped = unmapped
        self.line_type = line_type
        self.skipna = skipna
        self.na_thres = na_thres
        self.norm_type = norm_type
        self._uid = str(uuid.uuid4())
        self._weights_matrix: axis_py.Matrix | None = None
        self._serialized_weights: bytes | None = None
        self._total_weights: np.ndarray | None = None
        self._registered_clients: weakref.WeakKeyDictionary[Any, str] = weakref.WeakKeyDictionary()

        self._method_map = {
            "bilinear": axis_py.Method.Bilinear,
            "nearest": axis_py.Method.NearestNeighbor,
            "bicubic": axis_py.Method.Bicubic,
            "patch": axis_py.Method.Patch,
            "conservative": axis_py.Method.Conservative,
            "conservative1storder": axis_py.Method.Conservative,
            "conservative2nd": axis_py.Method.Conservative2ndOrder,
            "conservative2ndorder": axis_py.Method.Conservative2ndOrder,
        }

        self._line_type_map = {
            "great_circle": axis_py.LineType.GreatCircle,
            "cartesian": axis_py.LineType.Cartesian,
        }

        if weights_file is not None:
            # Load pre-computed weights from file, completely bypassing weight generation
            with open(weights_file, "rb") as f:
                self._serialized_weights = f.read()
            self._weights_matrix = axis_py.Matrix.from_bytes(self._serialized_weights)

            if ds_in is not None and ds_out is not None:
                self._src_geom = ds_in if isinstance(ds_in, Geometry) else None
                self._dst_geom = ds_out if isinstance(ds_out, Geometry) else None
                from .grid import _get_mesh_info

                if isinstance(ds_in, (xr.Dataset, xr.DataArray)):
                    _, _, self._shape_source, self._dims_source, self._is_unstructured_src = _get_mesh_info(ds_in)
                    self.source_grid_ds = ds_in
                if isinstance(ds_out, (xr.Dataset, xr.DataArray)):
                    _, _, self._shape_target, self._dims_target, _ = _get_mesh_info(ds_out)
                    self.target_grid_ds = ds_out
        else:
            if ds_in is not None and ds_out is not None:
                self.fit(ds_in, ds_out, src_mask=src_mask, dst_mask=dst_mask)

    def fit(
        self,
        src: Geometry | xr.Dataset | dict,
        dst: Geometry | xr.Dataset | dict,
        src_mask: np.ndarray | xr.DataArray | None = None,
        dst_mask: np.ndarray | xr.DataArray | None = None,
    ) -> "Regridder":
        """
        Generate spatial interpolation weights (SpMV matrices) from source to destination.
        """
        self.close()
        self._uid = str(uuid.uuid4())
        t0 = time.perf_counter()
        logger.info("AXIS: Initiating high-performance weight matrix generation...")

        from .grid import GridFactory

        # 1. Normalize source and target geometries
        if isinstance(src, Geometry):
            self._src_geom = src
        elif isinstance(src, (xr.Dataset, xr.DataArray)):
            self._src_geom = GridFactory.from_xarray(src, method=self.method)
        elif isinstance(src, dict):
            lons = np.asarray(src.get("lon") if src.get("lon") is not None else src.get("lons"))
            lats = np.asarray(src.get("lat") if src.get("lat") is not None else src.get("lats"))
            if lons.ndim == 1:
                self._src_geom = RectilinearGrid(lons, lats)
            else:
                self._src_geom = CurvilinearGrid(lons, lats)
        else:
            raise TypeError(f"Unsupported source geometry container: {type(src)}")

        if isinstance(dst, Geometry):
            self._dst_geom = dst
        elif isinstance(dst, (xr.Dataset, xr.DataArray)):
            self._dst_geom = GridFactory.from_xarray(dst, method=self.method)
        elif isinstance(dst, dict):
            lons = np.asarray(dst.get("lon") if dst.get("lon") is not None else dst.get("lons"))
            lats = np.asarray(dst.get("lat") if dst.get("lat") is not None else dst.get("lats"))
            if lons.ndim == 1:
                self._dst_geom = RectilinearGrid(lons, lats)
            else:
                self._dst_geom = CurvilinearGrid(lons, lats)
        else:
            raise TypeError(f"Unsupported target geometry container: {type(dst)}")

        # 2. Re-extract dimensions and shapes
        if isinstance(src, (xr.Dataset, xr.DataArray)):
            _, _, self._shape_source, self._dims_source, self._is_unstructured_src = _get_mesh_info(src, self.method)
            self.source_grid_ds = src
        else:
            if isinstance(self._src_geom, RectilinearGrid):
                self._shape_source = (len(self._src_geom.lats), len(self._src_geom.lons))
                self._dims_source = ("lat", "lon")
            elif isinstance(self._src_geom, CurvilinearGrid):
                self._shape_source = self._src_geom.lons.shape
                self._dims_source = ("y", "x")
            else:
                self._shape_source = (len(cast(UnstructuredMesh, self._src_geom).coords),)
                self._dims_source = ("ncol",)
            self._is_unstructured_src = not hasattr(self._src_geom, "lons")
            self.source_grid_ds = None

        if isinstance(dst, (xr.Dataset, xr.DataArray)):
            _, _, self._shape_target, self._dims_target, _ = _get_mesh_info(dst, self.method)
            self.target_grid_ds = dst
        else:
            if isinstance(self._dst_geom, RectilinearGrid):
                self._shape_target = (len(self._dst_geom.lats), len(self._dst_geom.lons))
                self._dims_target = ("lat", "lon")
            elif isinstance(self._dst_geom, CurvilinearGrid):
                self._shape_target = self._dst_geom.lons.shape
                self._dims_target = ("y", "x")
            else:
                self._shape_target = (len(cast(UnstructuredMesh, self._dst_geom).coords),)
                self._dims_target = ("ncol",)
            self.target_grid_ds = None

        # 3. Compile weight mapping
        src_mesh = self._src_geom.to_mesh()
        dst_mesh = self._dst_geom.to_mesh()

        method_lower = self.method.lower()
        if method_lower not in self._method_map:
            raise ValueError(f"Unknown interpolation method: {self.method}. Choose from {list(self._method_map.keys())}")
        self._axis_method = self._method_map[method_lower]

        line_type_lower = self.line_type.lower()
        if line_type_lower not in self._line_type_map:
            raise ValueError(f"Unknown line type: {self.line_type}. Choose from {list(self._line_type_map.keys())}")
        self._axis_line_type = self._line_type_map[line_type_lower]

        norm_map = {
            "dstarea": axis_py.NormType.DstArea,
            "fracarea": axis_py.NormType.FracArea,
        }
        axis_norm = norm_map.get(self.norm_type.lower(), axis_py.NormType.FracArea)

        config = {
            "method": self._axis_method,
            "periodic": self.periodic,
            "line_type": self._axis_line_type,
            "norm_type": axis_norm,
            "unmapped": axis_py.UnmappedAction.Ignore if self.unmapped == "ignore" else axis_py.UnmappedAction.Error,
        }

        # Inject masks if provided
        if src_mask is not None:
            config["src_mask"] = np.asarray(src_mask, dtype=np.int32)
        if dst_mask is not None:
            config["dst_mask"] = np.asarray(dst_mask, dtype=np.int32)

        self._weights_matrix = axis_py.generate_weights(src_mesh, dst_mesh, config)
        self._serialized_weights = self._weights_matrix.to_bytes()

        # Compute total weights sum for NaN-aware re-normalization
        if self.skipna:
            self._total_weights = np.array(
                axis_py.apply_weights(self._weights_matrix, np.ones(self._weights_matrix.n_src))
            ).flatten()
        else:
            self._total_weights = None

        elapsed = time.perf_counter() - t0
        logger.info(
            f"AXIS: Weight generation compiled successfully in {elapsed:.4f} seconds. "
            f"Mapped: {self._weights_matrix.n_src} source nodes -> {self._weights_matrix.n_dst} target nodes. "
            f"Weight database size: {len(self._serialized_weights) / (1024 * 1024):.3f} MB."
        )

        return self

    def __repr__(self) -> str:
        """
        Return a rich textual summary of the Regridder's properties and compiled weights database.
        """
        if self._weights_matrix is None:
            return f"<axis.Regridder (unfit, method={self.method})>"

        n_src = self._weights_matrix.n_src
        n_dst = self._weights_matrix.n_dst
        serialized_size = len(self._serialized_weights) if self._serialized_weights else 0
        size_mb = serialized_size / (1024 * 1024)

        return (
            f"axis.Regridder\n"
            f"  Interpolation Method : {self.method}\n"
            f"  Source Grid Shape    : {self._shape_source} ({n_src} elements)\n"
            f"  Target Grid Shape    : {self._shape_target} ({n_dst} elements)\n"
            f"  Periodic Boundaries  : {self.periodic}\n"
            f"  Great Circle Lines   : {self.line_type == 'great_circle'}\n"
            f"  Weight Database Size : {size_mb:.3f} MB"
        )

    def to_file(self, filename: str) -> None:
        """
        Save the compiled sparse regridding weights to a binary file for future reuse.
        """
        if self._serialized_weights is None:
            raise RuntimeError("Cannot save weights: Regridder has not been compiled or fit.")
        with open(filename, "wb") as f:
            f.write(self._serialized_weights)

    def close(self) -> None:
        """Release this regridder's local cache entries and worker plugins."""
        weights_key = f"weights_{self._uid}"
        total_key = f"tw_{self._uid}_sum"
        _WORKER_CACHE.pop(weights_key, None)
        _WORKER_CACHE.pop(total_key, None)
        for client, plugin_name in list(self._registered_clients.items()):
            try:
                client.unregister_worker_plugin(plugin_name)
            except Exception:
                logger.debug("Could not unregister AXIS worker plugin %s", plugin_name, exc_info=True)
        self._registered_clients.clear()

    def __del__(self):
        """Best-effort close; cleanup must never mask shutdown or prior errors."""
        try:
            self.close()
        except Exception:
            # Cleanup must not mask interpreter shutdown or an earlier exception.
            pass

    def _prepare_distributed_weights(self, client, weights_key: str) -> None:
        """Register matrix and optional skipna sums on current and future workers."""
        weights_matrix = self._weights_matrix
        if weights_matrix is None:
            raise RuntimeError("Regridder must be fit before registering distributed weights.")
        if self.skipna and not weights_matrix.is_csr:
            weights_matrix.to_csr()
            self._serialized_weights = weights_matrix.to_bytes()
        if self._serialized_weights is None:
            self._serialized_weights = weights_matrix.to_bytes()

        total_key = f"tw_{self._uid}_sum" if self._total_weights is not None else None
        plugin_name = f"axis-weights-{self._uid}"
        if client not in self._registered_clients:
            _register_weight_plugin(
                client,
                plugin_name,
                weights_key,
                self._serialized_weights,
                total_key,
                self._total_weights,
            )
            self._registered_clients[client] = plugin_name

        # Explicit local schedulers used by the driver can share this process cache.
        _setup_worker_cache(weights_key, weights_matrix)
        if total_key is not None:
            _setup_worker_cache(total_key, self._total_weights)

    def to_esmf(self, filename: str) -> None:
        """
        Save the compiled sparse regridding weights to an ESMF/SCRIP-compliant NetCDF file.
        """
        if self._weights_matrix is None:
            raise RuntimeError("Cannot save weights: Regridder has not been compiled or fit.")
        if not hasattr(axis_py, "write_esmf"):
            raise RuntimeError("ESMF weight export is unavailable. AXIS was compiled without NetCDF support.")
        axis_py.write_esmf(filename, self._weights_matrix)

    @classmethod
    def from_esmf(cls, filename: str, src_grid: xr.Dataset, dst_grid: xr.Dataset, skipna: bool = False) -> "Regridder":
        """
        Load a Regridder instance from an ESMF-compliant NetCDF weight file.
        """
        if not hasattr(axis_py, "read_esmf"):
            raise RuntimeError("ESMF weight import is unavailable. AXIS was compiled without NetCDF support.")

        regridder = cls.__new__(cls)
        regridder.method = "esmf"
        regridder.periodic = False
        regridder.unmapped = "ignore"
        regridder.line_type = "great_circle"
        regridder.skipna = skipna
        regridder.na_thres = 1.0
        regridder.norm_type = "fracarea"
        regridder._uid = str(uuid.uuid4())
        regridder._registered_clients = weakref.WeakKeyDictionary()

        regridder._weights_matrix = axis_py.read_esmf(filename)
        regridder._serialized_weights = None

        # Extract coordinate information and dimensions
        _, _, regridder._shape_source, regridder._dims_source, regridder._is_unstructured_src = _get_mesh_info(src_grid)
        _, _, regridder._shape_target, regridder._dims_target, _ = _get_mesh_info(dst_grid)

        # Save original datasets for coordinate matching
        regridder.source_grid_ds = src_grid
        regridder.target_grid_ds = dst_grid

        # Compute total weights sum for NaN-aware re-normalization
        if skipna:
            import numpy as np

            regridder._total_weights = np.array(
                axis_py.apply_weights(regridder._weights_matrix, np.ones(regridder._weights_matrix.n_src))
            ).flatten()
        else:
            regridder._total_weights = None

        return regridder

    def transform(
        self,
        obj: xr.DataArray | xr.Dataset | np.ndarray,
        keep_attrs: bool = True,
    ) -> xr.DataArray | xr.Dataset | np.ndarray:
        """
        Apply spatial remapping to the input object (NumPy array, xarray.DataArray, or xarray.Dataset).
        """

        if not isinstance(obj, (xr.Dataset, xr.DataArray, np.ndarray)) and not hasattr(obj, "__array__"):
            raise TypeError("Input object must be an xarray.DataArray, xarray.Dataset, or numpy.ndarray")

        if self._weights_matrix is None:
            raise RuntimeError("Regridder must be fit to grids before calling transform().")

        res: xr.DataArray | xr.Dataset | np.ndarray
        if isinstance(obj, (xr.Dataset, xr.DataArray)):
            if isinstance(obj, xr.Dataset):
                res = self._regrid_dataset(obj)
            else:
                res = self._regrid_dataarray(obj)
            if keep_attrs:
                res.attrs = {**obj.attrs, **res.attrs}
            return res
        else:
            arr = np.asarray(obj)
            original_shape = arr.shape

            target_spatial_shape = self._shape_target

            n_spatial = self._weights_matrix.n_src

            # Case A: 1D flat spatial array
            if arr.ndim == 1 and len(arr) == n_spatial:
                res = np.array(axis_py.apply_weights(self._weights_matrix, arr))
                if len(target_spatial_shape) > 1:
                    return res.reshape(target_spatial_shape)
                return res

            # Case B: Standard 2D spatial grid matching source shape
            if arr.ndim == len(self._shape_source) and arr.shape == self._shape_source:
                res = np.array(axis_py.apply_weights(self._weights_matrix, arr.ravel()))
                return res.reshape(target_spatial_shape)

            # Case C: Multidimensional array (other_dims..., spatial_dims...)
            n_spatial_dims = len(self._shape_source)
            spatial_shape = original_shape[-n_spatial_dims:]
            other_dims_shape = original_shape[:-n_spatial_dims]

            assert int(np.prod(spatial_shape)) == n_spatial, (
                f"Trailing dimensions {spatial_shape} must match source spatial size {n_spatial}"
            )

            n_other = int(np.prod(other_dims_shape)) if other_dims_shape else 1
            flat_data = arr.reshape(n_other, n_spatial)
            flat_data_t = np.asfortranarray(flat_data.T)

            result_t = axis_py.batch_apply(self._weights_matrix, flat_data_t)
            result = result_t.T

            new_shape = other_dims_shape + target_spatial_shape
            return result.reshape(new_shape).astype(arr.dtype, copy=False)

    def __call__(
        self,
        obj: xr.DataArray | xr.Dataset | np.ndarray,
        keep_attrs: bool = True,
    ) -> xr.DataArray | xr.Dataset | np.ndarray:
        """
        Apply spatial remapping. Transparently handles NumPy, xarray, and remote Dask blocks.
        """
        return self.transform(obj, keep_attrs=keep_attrs)

    def _regrid_dataset(self, ds: xr.Dataset) -> xr.Dataset:
        """Remap all spatial variables inside a Dataset."""
        regridded_vars = {}
        for var in ds.data_vars:
            da = ds[var]
            if all(dim in da.dims for dim in self._dims_source):
                regridded_vars[var] = self._regrid_dataarray(da)
            else:
                regridded_vars[var] = da

        res = xr.Dataset(regridded_vars)
        # Inherit non-spatial global coordinates
        for c in ds.coords:
            if c not in res.coords and set(ds[c].dims).intersection(set(self._dims_source)) == set():
                res = res.assign_coords({c: ds[c]})
        return res

    def _regrid_dataarray(self, da_in: xr.DataArray) -> xr.DataArray:
        """Remap a single DataArray, including Dask-lazy execution paths."""
        # Detect if input array is backed by Dask
        is_dask = hasattr(da_in.data, "dask")

        input_core_dims = list(self._dims_source)
        temp_output_core_dims = [f"{d}_regridded" for d in self._dims_target]

        weights_arg: axis_py.Matrix | str | None = self._weights_matrix
        total_weights_arg: np.ndarray | str | None = self._total_weights
        weights_key_arg = None

        if is_dask:
            _validate_core_dimension_chunks(
                da_in,
                self._dims_source,
                "horizontal regridding",
                input_name="field",
            )

            # Register serialized weights through Dask's worker lifecycle API.
            try:
                import dask.distributed

                client = dask.distributed.get_client()
            except ImportError as exc:
                import dask

                if dask.config.get("scheduler") == "distributed":
                    raise RuntimeError(
                        "The distributed scheduler was requested, but Dask Distributed is not installed. "
                        "Install AXIS with `pip install 'axis-regrid[distributed]'` or `uv sync --extra distributed`."
                    ) from exc
                client = None
            except ValueError:
                client = None

            weights_key_arg = f"weights_{self._uid}"

            if client is not None:
                self._prepare_distributed_weights(client, weights_key_arg)
                weights_arg = weights_key_arg

                if self._total_weights is not None:
                    tw_key = f"tw_{self._uid}_sum"
                    total_weights_arg = tw_key
            else:
                # Synchronous and threaded schedulers execute in this process and can use the local cache.
                _setup_worker_cache(weights_key_arg, self._weights_matrix)
                weights_arg = weights_key_arg

                if self._total_weights is not None:
                    tw_key = f"tw_{self._uid}_sum"
                    _setup_worker_cache(tw_key, self._total_weights)
                    total_weights_arg = tw_key

        # Execute parallelized map-blocks SpMV using apply_ufunc
        out = xr.apply_ufunc(
            _apply_weights_core,
            da_in,
            kwargs={
                "weights_matrix": weights_arg,
                "dims_source": self._dims_source,
                "shape_target": self._shape_target,
                "skipna": self.skipna,
                "total_weights": total_weights_arg,
                "na_thres": self.na_thres,
                "weights_key": weights_key_arg,
            },
            input_core_dims=[input_core_dims],
            output_core_dims=[temp_output_core_dims],
            dask="parallelized",
            vectorize=False,
            output_dtypes=[da_in.dtype],
            dask_gufunc_kwargs={
                "output_sizes": dict(zip(temp_output_core_dims, self._shape_target, strict=False)),
                "allow_rechunk": False,
            },
        )

        # Rename temporary output dimensions to target dimension names
        rename_dict = {temp: orig for temp, orig in zip(temp_output_core_dims, self._dims_target, strict=False) if temp in out.dims}
        if rename_dict:
            out = out.rename(rename_dict)

        # Assign coordinates from target grid
        target_coords: dict[Any, Any] = {}
        if self.target_grid_ds is not None:
            for c in self.target_grid_ds.coords:
                c_dims = set(self.target_grid_ds.coords[c].dims)
                if c_dims.issubset(set(self._dims_target)):
                    target_coords[c] = self.target_grid_ds.coords[c]
        else:
            if isinstance(self._dst_geom, (RectilinearGrid, CurvilinearGrid)):
                if self._dst_geom.lons.ndim == 1:
                    target_coords["lat"] = self._dst_geom.lats
                    target_coords["lon"] = self._dst_geom.lons
                else:
                    target_coords["lat"] = (("y", "x"), self._dst_geom.lats)
                    target_coords["lon"] = (("y", "x"), self._dst_geom.lons)

        if target_coords:
            out = out.assign_coords(target_coords)
        out.name = da_in.name
        out.attrs = da_in.attrs.copy()
        return out

    def regrid_categorical(
        self,
        da_in: xr.DataArray,
        categories: list | dict | None = None,
        prefix: str = "fraction_",
    ) -> xr.Dataset:
        """
        Remap high-resolution categorical data (e.g., land-use classes) to coarse grid fractions.

        Parameters
        ----------
        da_in : xr.DataArray
            Categorical input DataArray containing integer class codes or IDs.
        categories : list or dict, optional
            If a list, computes fractions only for the specified category IDs.
            If a dict, maps category ID (key) to its corresponding string name (value)
            for the output variables.
            If None (default), automatically discovers all unique non-NaN category IDs.
        prefix : str, default "fraction_"
            Prefix to prepend to the output variable names (e.g. "fraction_forest").

        Returns
        -------
        xr.Dataset
            A dataset containing the fractional coverage for each category as separate variables.
        """
        import numpy as np

        if not isinstance(da_in, xr.DataArray):
            raise TypeError("Input categorical object must be an xarray.DataArray")

        # 1. Identify category values to extract
        if categories is None:
            # Drop NaNs and extract unique values
            flat_vals = da_in.values.ravel()
            unique_vals = np.unique(flat_vals[~np.isnan(flat_vals)])
            category_mapping = {int(val): str(int(val)) for val in unique_vals}
        elif isinstance(categories, dict):
            category_mapping = categories
        elif isinstance(categories, (list, tuple, np.ndarray)):
            category_mapping = {int(val): str(int(val)) for val in categories}
        else:
            raise TypeError("categories must be a list, dict, or None")

        # 2. Iterate and remap each category
        regridded_vars = {}
        for cat_id, cat_name in category_mapping.items():
            # Build binary indicator mask
            indicator = (da_in == cat_id).astype(float)

            # Carry over coordinates and metadata for exact spatial mapping
            indicator = indicator.copy(deep=True)

            # Remap via self
            fraction_da = self(indicator, keep_attrs=False)

            # Form clean variable name
            var_name = f"{prefix}{cat_name}"
            regridded_vars[var_name] = fraction_da

        return xr.Dataset(regridded_vars)
