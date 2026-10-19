# SPDX-License-Identifier: Apache-2.0
"""Dask integration checks for lazy horizontal and vertical AXIS operations."""

import sys

import numpy as np
import pytest
import xarray as xr

da = pytest.importorskip("dask.array")
import axis  # noqa: E402
import dask  # noqa: E402
from axis import VerticalRegridder  # noqa: E402


def test_horizontal_regrid_is_lazy_and_preserves_batch_chunks(dask_regrid_case):
    source_grid, target_grid, field = dask_regrid_case
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")

    result = regridder(field)

    assert hasattr(result.data, "dask")
    assert result.dims == ("time", "lat", "lon")
    assert result.data.chunks[0] == (1, 1)
    assert result.name == field.name
    assert result.attrs == field.attrs
    np.testing.assert_allclose(result.compute().values, 1.0)


def test_documented_lazy_workflow_smoke(dask_regrid_case):
    """Keep the README's construct-lazily, compute-explicitly flow executable."""
    source_grid, target_grid, field = dask_regrid_case
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")

    lazy_result = regridder(field)
    assert hasattr(lazy_result.data, "dask")
    result = lazy_result.compute(scheduler="threads")

    np.testing.assert_allclose(result.values, 1.0)
    regridder.close()


def test_horizontal_regrid_rejects_split_spatial_core_dimensions(dask_regrid_case):
    source_grid, target_grid, field = dask_regrid_case
    split_field = field.chunk({"lat": 2, "lon": 4})
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")

    with pytest.raises(ValueError, match="core dimension.*single Dask chunk") as error:
        regridder(split_field)

    assert "lat" in str(error.value)
    assert "data.chunk" in str(error.value)


@pytest.mark.parametrize("scheduler", ["synchronous", "threads"])
def test_horizontal_regrid_matches_eager_for_local_schedulers(dask_regrid_case, scheduler):
    source_grid, target_grid, field = dask_regrid_case
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")

    with dask.config.set(scheduler=scheduler):
        result = regridder(field).compute()

    np.testing.assert_allclose(result.values, 1.0)


def test_local_process_scheduler_reports_unsupported_mode(dask_regrid_case):
    source_grid, target_grid, field = dask_regrid_case
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")
    result = regridder(field)

    with pytest.raises(RuntimeError, match="multiprocessing scheduler is not supported"):
        result.compute(scheduler="processes")


def test_requested_distributed_scheduler_reports_missing_extra(dask_regrid_case, monkeypatch):
    source_grid, target_grid, field = dask_regrid_case
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")
    monkeypatch.setitem(sys.modules, "dask.distributed", None)

    with dask.config.set(scheduler="distributed"):
        with pytest.raises(RuntimeError, match="Dask Distributed is not installed"):
            regridder(field)


def test_vertical_regrid_is_lazy_and_matches_eager_output():
    values = np.broadcast_to(np.array([0.0, 2.0, 4.0]), (2, 3, 3)).copy()
    lazy_field = xr.DataArray(
        da.from_array(values, chunks=(1, 3, 3)),
        dims=("time", "column", "lev"),
        coords={"time": [0, 1], "column": [0, 1, 2], "lev": [0.0, 1.0, 2.0]},
        name="temperature",
        attrs={"units": "K"},
    )
    source_levels = xr.DataArray([0.0, 1.0, 2.0], dims=("lev",))
    target_levels = xr.DataArray([0.5, 1.5], dims=("target_lev",))
    regridder = VerticalRegridder()

    result = regridder.interpolate(lazy_field, source_levels, target_levels)
    eager = regridder.interpolate(lazy_field.compute(), source_levels, target_levels)

    assert isinstance(result, xr.DataArray)
    assert hasattr(result.data, "dask")
    assert result.dims == ("time", "column", "lev")
    assert result.name == lazy_field.name
    assert result.attrs == lazy_field.attrs
    np.testing.assert_allclose(result.compute().values, eager.values)


def test_vertical_regrid_supports_spatially_varying_levels():
    values = np.broadcast_to(np.array([0.0, 2.0, 4.0]), (2, 3)).copy()
    lazy_field = xr.DataArray(
        da.from_array(values, chunks=(1, 3)),
        dims=("column", "lev"),
        coords={"column": [0, 1], "lev": [0.0, 1.0, 2.0]},
        name="temperature",
    )
    source_levels = xr.DataArray(np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]), dims=("column", "lev"))
    target_levels = xr.DataArray(np.array([[0.5, 1.5], [0.5, 1.5]]), dims=("column", "target_lev"))
    regridder = VerticalRegridder()

    result = regridder.interpolate(lazy_field, source_levels, target_levels)
    eager = regridder.interpolate(lazy_field.compute(), source_levels, target_levels)

    assert hasattr(result.data, "dask")
    np.testing.assert_allclose(result.compute().values, eager.values)


def test_vertical_regrid_rejects_split_level_core_dimension():
    field = xr.DataArray(
        da.from_array(np.ones((2, 4)), chunks=(1, 2)),
        dims=("column", "lev"),
        coords={"lev": [0.0, 1.0, 2.0, 3.0]},
    )
    source_levels = xr.DataArray([0.0, 1.0, 2.0, 3.0], dims=("lev",))
    target_levels = xr.DataArray([0.5, 2.5], dims=("target_lev",))

    with pytest.raises(ValueError, match="core dimension.*single Dask chunk") as error:
        VerticalRegridder().interpolate(field, source_levels, target_levels)

    assert "lev" in str(error.value)
