# SPDX-License-Identifier: Apache-2.0
"""Eager conservative transfer of layer means in an explicit physical measure."""
import numpy as np
from . import axis_py

class ConservativeVerticalRegridder:
    """Cache overlaps of shared/varying interfaces, with first or limited linear reconstruction.

    Interface differences are the carrier measure: volume/thickness for volume
    concentrations, or carrier mass for mixing ratios. No density is inferred.
    Arrays end in layer/interface dimension; leading geometry/mask dimensions
    must broadcast in the same column order as the field. xarray output retains
    field metadata; this initial implementation computes eager arrays.
    """
    def __init__(self, source_interfaces, destination_interfaces, method="limited_linear",
                 norm="dstarea", source_mask=None, destination_mask=None):
        """Select the reconstruction/normalization options and copy interface arrays.

        Interfaces are the cumulative physical measure per horizontal area and
        may be 1-D (shared by all columns) or broadcast over leading column
        dimensions. Masks mark participating source/destination layers.
        """
        if method not in {"first", "limited_linear"}:
            raise ValueError("method must be first or limited_linear")
        if norm not in {"dstarea", "fracarea"}:
            raise ValueError("norm must be dstarea or fracarea")
        self.method, self.norm = method, norm
        self._src = np.array(source_interfaces, dtype=np.float64, copy=True)
        self._dst = np.array(destination_interfaces, dtype=np.float64, copy=True)
        if self._src.ndim < 1 or self._dst.ndim < 1 or min(self._src.shape[-1], self._dst.shape[-1]) < 2:
            raise ValueError("At least two interfaces per grid are required")
        self._sm = None if source_mask is None else np.array(source_mask, dtype=np.int32, copy=True)
        self._dm = None if destination_mask is None else np.array(destination_mask, dtype=np.int32, copy=True)
        self._cache = {}
        self._last = None

    def _geometry(self, values, leading):
        if values.ndim == 1:
            return np.ascontiguousarray(values.reshape(1, -1))
        return np.ascontiguousarray(np.broadcast_to(values, leading + (values.shape[-1],)).reshape(-1, values.shape[-1]))

    def _state(self, field):
        a = np.asarray(field, dtype=np.float64)
        if a.ndim < 1 or a.shape[-1] != self._src.shape[-1]-1:
            raise ValueError("Source layer count differs from interfaces")
        leading = a.shape[:-1]
        if leading not in self._cache:
            nc = int(np.prod(leading)) if leading else 1
            if nc < 1:
                raise ValueError("At least one column is required")
            sm = np.ones(leading+(a.shape[-1],), np.int32) if self._sm is None else np.broadcast_to(self._sm, leading+(a.shape[-1],))
            dm = np.ones(leading+(self._dst.shape[-1]-1,), np.int32) if self._dm is None else np.broadcast_to(self._dm, leading+(self._dst.shape[-1]-1,))
            op = axis_py.make_vertical_conservative_operator(self._geometry(self._src, leading), self._geometry(self._dst, leading), nc,
                np.ascontiguousarray(sm.reshape(nc,-1)), np.ascontiguousarray(dm.reshape(nc,-1)))
            work = np.empty((nc,a.shape[-1]), np.float64) if self.method == "limited_linear" else None
            self._cache[leading] = (op, work)
        self._last = leading
        op, work = self._cache[leading]
        return np.ascontiguousarray(a.reshape(-1,a.shape[-1])), op, work, leading

    def remap(self, source):
        """Transfer layer means, returning destination layer means.

        xarray inputs yield a DataArray with surviving metadata and destination
        layer midpoints on the original vertical coordinate when available.
        """
        a, op, work, leading = self._state(source)
        result = op.apply(a, self.method, self.norm, work).reshape(leading+(self._dst.shape[-1]-1,))
        if hasattr(source, "dims"):
            import xarray as xr
            if not isinstance(source, xr.DataArray):
                raise TypeError("Use an ndarray or xarray DataArray")
            lev = source.dims[-1]
            coords = {name: value for name,value in source.coords.items() if lev not in value.dims}
            result = xr.DataArray(result, dims=source.dims, coords=coords, name=source.name, attrs=source.attrs)
            mid = .5*self._dst[...,:-1]+.5*self._dst[...,1:]
            if mid.ndim == 1:
                result = result.assign_coords({lev:mid})
            else:
                result = result.assign_coords(vertical_measure_midpoint=(source.dims,np.broadcast_to(mid,result.shape)))
        return result
    __call__ = remap

    def covered_source_integral(self, source):
        """Include the slope integral on partially covered source layers."""
        a, op, work, leading = self._state(source)
        return op.covered_source_integral(a,self.method,work).reshape(leading)

    def _diagnostic(self, name):
        if self._last is None:
            raise RuntimeError("Call remap or covered_source_integral before reading coverage")
        op, _ = self._cache[self._last]
        v = getattr(op,name)
        return v.reshape(self._last+(v.shape[-1],))
    @property
    def source_coverage(self):
        """Fraction of each source layer's measure used by the last transfer."""
        return self._diagnostic("source_coverage")
    @property
    def destination_coverage(self):
        """Fraction of each destination layer covered by source data."""
        return self._diagnostic("destination_coverage")
    @property
    def destination_covered_measure(self):
        """Covered physical measure of each destination layer, per column."""
        return self._diagnostic("destination_covered_measure")


def regrid_tracer_3d(tracer, carrier, horizontal_regridder, destination_interfaces,
                    vertical_dim="lev", method="limited_linear", require_full_coverage=True):
    """Conserve tracer content through a horizontal-then-vertical transfer.

    tracer and carrier are aligned xarray DataArrays. carrier is nonnegative
    volume or carrier mass per horizontal area in each source layer (zero=dry).
    The supplied horizontal operator must use first-order conservative dstarea
    weights and skipna=False. It transfers carrier and tracer*carrier separately.
    Destination interfaces are cumulative carrier measure, starting at zero;
    they must cover the resulting carrier column unless partial coverage is
    explicitly enabled. Return tracer, participating carrier, tracer_content, and coverage.
    Global preservation additionally requires a fully covered horizontal domain
    and the correct source/destination horizontal areas.
    """
    import xarray as xr
    if not isinstance(tracer,xr.DataArray) or not isinstance(carrier,xr.DataArray):
        raise TypeError("tracer and carrier must be xarray DataArrays")
    if getattr(horizontal_regridder,"method",None) not in {"conservative","conservative1storder"} or getattr(horizontal_regridder,"norm_type",None) != "dstarea" or getattr(horizontal_regridder,"skipna",True):
        raise ValueError("Horizontal transfer requires first-order conservative dstarea and skipna=False")
    if vertical_dim not in tracer.dims or vertical_dim not in carrier.dims:
        raise ValueError("Both inputs must contain the vertical layer dimension")
    tracer,carrier = xr.align(tracer,carrier,join="exact")
    tracer,carrier = xr.broadcast(tracer,carrier)
    # Preserve caller dimension order; inspect eager wet values before transfer.
    carrier = carrier.transpose(*tracer.dims)
    hv = np.asarray(carrier.values,dtype=np.float64);qv = np.asarray(tracer.values,dtype=np.float64)
    if not np.isfinite(hv).all() or (hv<0).any() or not np.isfinite(qv[hv>0]).all():
        raise ValueError("Carrier must be finite/nonnegative and wet tracer values finite")
    h = horizontal_regridder(carrier)
    content = horizontal_regridder(tracer.where(carrier>0,0)*carrier)
    dims = [d for d in h.dims if d != vertical_dim]+[vertical_dim]
    h,content = h.transpose(*dims),content.transpose(*dims)
    hvals = np.asarray(h.values,dtype=np.float64);mvals = np.asarray(content.values,dtype=np.float64)
    if not np.isfinite(hvals).all() or (hvals<0).any() or not np.isfinite(mvals).all():
        raise ValueError("Horizontal operator returned invalid carrier/content")
    q = np.divide(mvals,hvals,out=np.zeros_like(mvals),where=hvals>0)
    source_interfaces = np.concatenate([np.zeros(hvals.shape[:-1]+(1,)),np.cumsum(hvals,axis=-1)],axis=-1)
    operator = ConservativeVerticalRegridder(source_interfaces,destination_interfaces,method=method,norm="fracarea")
    values = operator.remap(q)
    srcfrac,dstfrac = operator.source_coverage,operator.destination_coverage
    dv = np.broadcast_to(np.abs(np.diff(operator._dst,axis=-1)),values.shape)
    if require_full_coverage and (not np.allclose(srcfrac[hvals>0],1,rtol=1e-12,atol=1e-12) or not np.allclose(dstfrac[dv>0],1,rtol=1e-12,atol=1e-12)):
        raise ValueError("Destination interfaces do not cover the same carrier column; enable partial coverage explicitly")
    coords = {name:value for name,value in h.coords.items() if vertical_dim not in value.dims}
    coords[vertical_dim] = np.arange(values.shape[-1])
    field = xr.DataArray(values,dims=dims,coords=coords,attrs=tracer.attrs)
    measure = xr.DataArray(operator.destination_covered_measure,dims=dims,coords=coords)
    return xr.Dataset({"tracer":field,"carrier":measure,"tracer_content":field*measure,
        "source_coverage":xr.DataArray(srcfrac,dims=[d if d!=vertical_dim else vertical_dim+"_source" for d in dims]),
        "destination_coverage":xr.DataArray(dstfrac,dims=dims,coords=coords),
        "covered_source_content":xr.DataArray(operator.covered_source_integral(q),dims=dims[:-1])},
        attrs={"vertical_measure":"cumulative carrier per horizontal area","vertical_method":method,
               "horizontal_method":"first_order_conservative_dstarea"})
