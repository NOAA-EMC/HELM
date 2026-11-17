# SPDX-License-Identifier: Apache-2.0
"""Tests for AXIS weight setup under Dask Distributed worker lifecycles."""

import numpy as np
import pytest

dask_array = pytest.importorskip("dask.array")
distributed = pytest.importorskip("distributed")

import axis  # noqa: E402


@pytest.fixture
def local_distributed_client():
    cluster = distributed.LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    client = distributed.Client(cluster)
    try:
        yield cluster, client
    finally:
        client.close()
        cluster.close()


def test_distributed_weights_survive_worker_join_and_restart(dask_regrid_case, local_distributed_client):
    source_grid, target_grid, field = dask_regrid_case
    cluster, client = local_distributed_client
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")

    with client:
        result = regridder(field)
        np.testing.assert_allclose(result.compute().values, 1.0)

        cluster.scale(2)
        client.wait_for_workers(2)
        np.testing.assert_allclose(result.compute().values, 1.0)

        client.restart()
        client.wait_for_workers(2)
        np.testing.assert_allclose(result.compute().values, 1.0)
    regridder.close()


def test_distributed_weight_cache_isolated_per_regridder(dask_regrid_case, local_distributed_client):
    source_grid, target_grid, field = dask_regrid_case
    _, client = local_distributed_client
    first = axis.Regridder(source_grid, target_grid, method="bilinear")
    second = axis.Regridder(source_grid, target_grid, method="nearest")

    with client:
        first_result = first(field)
        second_result = second(field)
        np.testing.assert_allclose(first_result.compute().values, 1.0)
        np.testing.assert_allclose(second_result.compute().values, 1.0)
    first.close()
    second.close()


def test_dask_local_scheduler_works_without_distributed(dask_regrid_case, monkeypatch):
    source_grid, target_grid, field = dask_regrid_case
    regridder = axis.Regridder(source_grid, target_grid, method="bilinear")

    def no_client(*args, **kwargs):
        raise ValueError("No clients found")

    monkeypatch.setattr("dask.distributed.get_client", no_client)
    result = regridder(field).compute(scheduler="threads")

    np.testing.assert_allclose(result.values, 1.0)


def _great_circle_grid(nlon, nlat):
    """Regular lat-lon centers with explicit corners (great-circle cell edges)."""
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    lat = np.linspace(-60.0, 60.0, nlat)
    dlon, dlat = lon[1] - lon[0], lat[1] - lat[0]
    lons, lats = np.meshgrid(lon, lat)
    edges_lon = np.concatenate([lon - dlon / 2, [lon[-1] + dlon / 2]])
    edges_lat = np.concatenate([lat - dlat / 2, [lat[-1] + dlat / 2]])
    corner_lons, corner_lats = np.meshgrid(edges_lon, edges_lat)
    return axis.CurvilinearGrid(lons, lats, corner_lons=corner_lons, corner_lats=corner_lats, longitude_periodic=True)


def test_linear_conservative_rejects_serialization_paths(local_distributed_client, tmp_path):
    """Conservative2ndOrder must fail fast on paths that would drop linear reconstruction."""
    import dask.array as da
    import xarray as xr

    _, client = local_distributed_client
    source = _great_circle_grid(8, 5)
    target = _great_circle_grid(12, 7)
    field = xr.DataArray(da.ones((5, 8), chunks=(5, 8)), dims=("y", "x"), name="temperature")
    regridder = axis.Regridder(source, target, method="conservative2nd")

    with pytest.raises(ValueError, match="linear reconstruction"):
        regridder.to_file(str(tmp_path / "weights.bin"))
    with pytest.raises(ValueError, match="linear reconstruction"):
        regridder.to_esmf(str(tmp_path / "weights.nc"))
    with client:
        with pytest.raises(ValueError, match="linear reconstruction"):
            regridder(field).compute()

    # The local (threaded) scheduler still works: no serialization involved.
    result = regridder(field).compute(scheduler="threads")
    assert np.all(np.isfinite(result.values))
    regridder.close()
