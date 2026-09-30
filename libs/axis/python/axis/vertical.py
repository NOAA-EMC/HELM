# SPDX-License-Identifier: Apache-2.0
from typing import Any, cast

import numpy as np
import xarray as xr

from . import axis_py


class VerticalRegridder:
    """
    High-performance vertical spline interpolator parallelized with Kokkos on device.
    Supports standard eager NumPy arrays and lazy parallelized Dask distributed arrays.
    """

    def __init__(self, tension: float = 0.0):
        """
        Parameters
        ----------
        tension : float, default 0.0
            Tension parameter (0.0 = standard cubic spline, larger values enforce local linearity).
        """
        if tension < 0.0:
            raise ValueError("Tension parameter must be non-negative")
        self.tension = float(tension)

    def interpolate(
        self,
        src_field: np.ndarray | xr.DataArray,
        src_levels: np.ndarray | xr.DataArray,
        dst_levels: np.ndarray | xr.DataArray,
    ) -> np.ndarray | xr.DataArray:
        """
        Perform 1D or 2D vertical spline interpolation.
        Supports automatic coordinate alignment and lazy out-of-core Dask processing.
        """
        is_xarray = isinstance(src_field, xr.DataArray)

        # Cast input array to DataArray if passed as a raw NumPy array to leverage apply_ufunc
        if isinstance(src_field, xr.DataArray):
            da_field = src_field
        else:
            dims = [f"dim_{i}" for i in range(src_field.ndim - 1)] + ["lev"]
            da_field = xr.DataArray(np.asarray(src_field, dtype=np.float64), dims=dims)

        vert_dim = da_field.dims[-1]
        regridded_vert_dim = f"{vert_dim}_regridded"

        # Ensure coordinates are wrapped in DataArray
        if isinstance(src_levels, xr.DataArray):
            da_src_levels = src_levels
        else:
            src_levs_arr = np.asarray(src_levels, dtype=np.float64)
            if src_levs_arr.ndim == 1:
                da_src_levels = xr.DataArray(src_levs_arr, dims=[vert_dim])
            else:
                da_src_levels = xr.DataArray(src_levs_arr, dims=list(da_field.dims))

        if isinstance(dst_levels, xr.DataArray):
            da_dst_levels = dst_levels
            dst_vert_dim = dst_levels.dims[-1]
        else:
            dst_levs_arr = np.asarray(dst_levels, dtype=np.float64)
            dst_vert_dim = regridded_vert_dim
            if dst_levs_arr.ndim == 1:
                da_dst_levels = xr.DataArray(dst_levs_arr, dims=[dst_vert_dim])
            else:
                da_dst_levels = xr.DataArray(dst_levs_arr, dims=list(da_field.dims[:-1]) + [dst_vert_dim])

        # Define block-wise core execution kernel
        def _core_interpolate(field_block, src_lev_block, dst_lev_block):
            f_arr = np.asarray(field_block, dtype=np.float64)
            s_arr = np.asarray(src_lev_block, dtype=np.float64)
            d_arr = np.asarray(dst_lev_block, dtype=np.float64)

            original_shape = f_arr.shape
            n_src_lev = original_shape[-1]
            n_dst_lev = d_arr.shape[-1]

            flat_field = f_arr.reshape(-1, n_src_lev)

            # Guarantee contiguity
            if not flat_field.flags["C_CONTIGUOUS"]:
                flat_field = np.ascontiguousarray(flat_field)
            if not s_arr.flags["C_CONTIGUOUS"]:
                s_arr = np.ascontiguousarray(s_arr)
            if not d_arr.flags["C_CONTIGUOUS"]:
                d_arr = np.ascontiguousarray(d_arr)

            # Execute applying compiled core interpolators
            # C++ Core handles coordinate direction reversals natively without copies
            if s_arr.ndim == 1 and d_arr.ndim == 1:
                res_flat = axis_py.interpolate_vertical(flat_field, s_arr, d_arr, self.tension)
            else:
                flat_s = s_arr.reshape(-1, n_src_lev)
                flat_d = d_arr.reshape(-1, n_dst_lev)
                res_flat = axis_py.interpolate_vertical_varying(flat_field, flat_s, flat_d, self.tension)

            return res_flat.reshape(original_shape[:-1] + (n_dst_lev,))

        # Run block interpolation through apply_ufunc
        input_core_dims = [[vert_dim], [da_src_levels.dims[-1]], [da_dst_levels.dims[-1]]]
        output_core_dims = [[da_dst_levels.dims[-1]]]

        out = xr.apply_ufunc(
            _core_interpolate,
            da_field,
            da_src_levels,
            da_dst_levels,
            input_core_dims=input_core_dims,
            output_core_dims=output_core_dims,
            dask="parallelized",
            vectorize=False,
            output_dtypes=[da_field.dtype],
            dask_gufunc_kwargs={
                "output_sizes": {da_dst_levels.dims[-1]: da_dst_levels.shape[-1]},
                "allow_rechunk": True,
            },
        )

        # Pull global coordinate variables
        target_coords = {}
        for c in da_field.coords:
            if c != vert_dim and set(da_field.coords[c].dims).issubset(set(out.dims)):
                target_coords[c] = da_field.coords[c]

        target_coords[da_dst_levels.dims[-1]] = da_dst_levels
        out = out.assign_coords(target_coords)

        if is_xarray:
            assert isinstance(src_field, xr.DataArray)
            if da_dst_levels.dims[-1] != vert_dim:
                out = out.rename({da_dst_levels.dims[-1]: vert_dim})
            out.name = src_field.name
            out.attrs = src_field.attrs
            return out
        else:
            return out.values


def regrid_3d(
    ds: xr.Dataset,
    target_grid_2d: Any,
    target_levels_1d: np.ndarray,
    method_2d: str = "bilinear",
    tension_1d: float = 2.0,
    vertical_coord_name: str = "lev",
) -> xr.Dataset:
    """
    Perform 2D horizontal regridding and 1D vertical spline coordinate mapping in a single call.
    """
    from .regridder import Regridder

    # 1. Remap horizontally
    regridder_2d = Regridder(ds, target_grid_2d, method=method_2d)
    ds_h = cast(xr.Dataset, regridder_2d(ds))

    # 2. Remap vertically
    regridder_v = VerticalRegridder(tension=tension_1d)

    ds_out = xr.Dataset()
    for var in ds_h.data_vars:
        da = ds_h[var]
        if vertical_coord_name in da.dims:
            dims_order = [d for d in da.dims if d != vertical_coord_name] + [vertical_coord_name]
            da_reordered = da.transpose(*dims_order)

            src_levs = ds[vertical_coord_name]
            res_da = cast(xr.DataArray, regridder_v.interpolate(da_reordered, src_levs, target_levels_1d))

            ds_out[var] = res_da.transpose(*da.dims)
        else:
            ds_out[var] = da

    ds_out.attrs = ds.attrs
    return ds_out
