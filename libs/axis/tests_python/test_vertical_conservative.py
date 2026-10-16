# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import xarray as xr
from axis import ConservativeVerticalRegridder, regrid_tracer_3d

def test_cached_vertical_inventory():
    r=ConservativeVerticalRegridder([0,1,2,3],[0,.5,1.5,2.5,3])
    q=np.array([[0,1,1],[1,2,3.]])
    out=r(q)
    np.testing.assert_allclose((out*np.diff([0,.5,1.5,2.5,3])).sum(axis=-1),q.sum(axis=-1),rtol=1e-14)
    assert len(r._cache)==1
    r(q+1)
    assert len(r._cache)==1

def test_xarray_metadata_and_partial_slope_content():
    q=xr.DataArray([1.,2.,3.],dims=['lev'],name='tracer',attrs={'units':'kg/kg'})
    r=ConservativeVerticalRegridder([0,1,2,3],[1,1.5])
    out=r(q)
    assert out.name=='tracer' and out.attrs['units']=='kg/kg'
    np.testing.assert_allclose(out,[1.75]);np.testing.assert_allclose(r.covered_source_integral(q),.875)

class AreaConservativeTestTransfer:
    """Equal-area two-source-cell to one double-area-cell exact first-order oracle."""
    method='conservative';norm_type='dstarea';skipna=False
    def __call__(self,field):return field.mean('x').expand_dims(x=[0])

def test_tracer_workflow_transfers_carrier_and_content():
    # Unequal layer thickness: remapping concentration alone would give 3,
    # whereas transferring content/carrier gives 3.5 and conserves inventory.
    q=xr.DataArray([[2.,2.],[4.,4.]],dims=['x','lev'])
    h=xr.DataArray([[1.,1.],[3.,3.]],dims=['x','lev'])
    result=regrid_tracer_3d(q,h,AreaConservativeTestTransfer(),[0,1,4])
    np.testing.assert_allclose(result.tracer,3.5)
    # Destination cell area is 2 times each source cell area.
    np.testing.assert_allclose(2*result.tracer_content.sum(),(q*h).sum(),rtol=1e-14)
    with pytest.raises(ValueError):regrid_tracer_3d(q,h,AreaConservativeTestTransfer(),[0,1,5])

def test_tracer_workflow_explicit_partial_coverage():
    q=xr.DataArray([[2.,2.]],dims=['x','lev']);h=xr.ones_like(q)
    result=regrid_tracer_3d(q,h,AreaConservativeTestTransfer(),[-1,.5,3],require_full_coverage=False)
    np.testing.assert_allclose(result.tracer,2)
    np.testing.assert_allclose(result.carrier.sum(),2)
    np.testing.assert_allclose(result.tracer_content.sum(),4)
