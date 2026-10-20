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
