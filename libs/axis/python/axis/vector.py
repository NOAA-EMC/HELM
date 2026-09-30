# SPDX-License-Identifier: Apache-2.0
from typing import Any

import numpy as np
import xarray as xr

from . import axis_py
from .grid import CurvilinearGrid, Geometry, GridFactory, RectilinearGrid


def _to_geometry(obj: Geometry | xr.Dataset | xr.DataArray | dict) -> Geometry:
    """Normalize a Geometry, xarray container, or coordinate dict into a Geometry."""
    if isinstance(obj, Geometry):
        return obj
    if isinstance(obj, dict):
        return GridFactory.from_dict(obj)
    return GridFactory.from_xarray(obj)


class VectorRegridder:
    """
    Coupled horizontal vector interpolator (e.g. for winds and currents)
    accounting for grid-frame rotation.
    """

    def __init__(
        self,
        src: Geometry | xr.Dataset | dict,
        dst: Geometry | xr.Dataset | dict,
        method: str = "bilinear",
        src_alpha: np.ndarray | None = None,
        dst_alpha: np.ndarray | None = None,
        **kwargs: Any,
    ):
        self._src_geom = _to_geometry(src)
        self._dst_geom = _to_geometry(dst)

        src_mesh = self._src_geom.to_mesh()
        dst_mesh = self._dst_geom.to_mesh()

        n_src_cells = src_mesh.n_cells
        n_dst_cells = dst_mesh.n_cells

        if src_alpha is None:
            src_alpha = np.zeros(n_src_cells, dtype=np.float64)
        else:
            src_alpha = np.asarray(src_alpha, dtype=np.float64)

        if dst_alpha is None:
            dst_alpha = np.zeros(n_dst_cells, dtype=np.float64)
        else:
            dst_alpha = np.asarray(dst_alpha, dtype=np.float64)

        method_map = {
            "bilinear": axis_py.Method.Bilinear,
            "nearest": axis_py.Method.NearestNeighbor,
        }
        method_enum = method_map.get(method.lower(), axis_py.Method.Bilinear)

        config = {
            "method": method_enum,
            "periodic": kwargs.get("periodic", False),
            "line_type": axis_py.LineType.GreatCircle if kwargs.get("line_type") == "great_circle" else axis_py.LineType.Cartesian,
        }

        self._W_u, self._W_v = axis_py.generate_vector_weights(src_mesh, dst_mesh, src_alpha, dst_alpha, config)

    def transform(
        self, u: np.ndarray | xr.DataArray, v: np.ndarray | xr.DataArray
    ) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
        """
        Remap and rotate (u, v) wind or current components simultaneously.
        """
        is_xarray = isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray)

        u_arr = np.asarray(u.values if isinstance(u, xr.DataArray) else u, dtype=np.float64)
        v_arr = np.asarray(v.values if isinstance(v, xr.DataArray) else v, dtype=np.float64)

        # Determine source and target shapes
        src_shape: tuple[int, ...]
        if isinstance(self._src_geom, RectilinearGrid):
            src_shape = (len(self._src_geom.lats), len(self._src_geom.lons))
        elif isinstance(self._src_geom, CurvilinearGrid):
            src_shape = self._src_geom.lons.shape
        else:
            src_shape = (self._W_u.n_src // 2,)

        dst_shape: tuple[int, ...]
        if isinstance(self._dst_geom, RectilinearGrid):
            dst_shape = (len(self._dst_geom.lats), len(self._dst_geom.lons))
        elif isinstance(self._dst_geom, CurvilinearGrid):
            dst_shape = self._dst_geom.lons.shape
        else:
            dst_shape = (self._W_u.n_dst,)

        n_spatial = self._W_u.n_src // 2
        u_out: np.ndarray
        v_out: np.ndarray

        original_shape = u_arr.shape
        n_spatial_dims = len(src_shape) if u_arr.ndim > 1 else 1
        other_dims_shape = original_shape[:-n_spatial_dims] if u_arr.ndim > 1 else ()
        n_other = int(np.prod(other_dims_shape)) if other_dims_shape else 1

        flat_u = u_arr.reshape(n_other, n_spatial)
        flat_v = v_arr.reshape(n_other, n_spatial)

        # High-speed unified C++ remapping SpMV (handling 1D, 2D, and multidimensional arrays)
        u_out_t, v_out_t = axis_py.vector_transform(self._W_u, self._W_v, flat_u.T, flat_v.T)

        u_out = u_out_t.T.reshape(other_dims_shape + dst_shape)
        v_out = v_out_t.T.reshape(other_dims_shape + dst_shape)

        if is_xarray:
            assert isinstance(u, xr.DataArray) and isinstance(v, xr.DataArray)
            target_coords: dict[str, Any] = {}
            dims: tuple[str, ...]
            if isinstance(self._dst_geom, (RectilinearGrid, CurvilinearGrid)):
                if self._dst_geom.lons.ndim == 1:
                    target_coords["lat"] = self._dst_geom.lats
                    target_coords["lon"] = self._dst_geom.lons
                    dims = ("lat", "lon")
                else:
                    target_coords["lat"] = (("y", "x"), self._dst_geom.lats)
                    target_coords["lon"] = (("y", "x"), self._dst_geom.lons)
                    dims = ("y", "x")
            else:
                dims = ("ncol",)

            u_da = xr.DataArray(u_out, coords=target_coords, dims=dims, name=u.name, attrs=u.attrs)
            v_da = xr.DataArray(v_out, coords=target_coords, dims=dims, name=v.name, attrs=v.attrs)
            return u_da, v_da

        return u_out, v_out
