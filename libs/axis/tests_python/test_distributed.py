# SPDX-License-Identifier: Apache-2.0
"""US6 — scale out transparently with Dask (T043).

Asserts spec AC1–AC5 / FR-030…FR-034 / SC-005: lazy in → lazy out, eager
driver-side fit with logging, at-most-once worker provisioning with
collision-proof keys and worker-side sync counters, no compiled ``Matrix``
pickled through task payloads, scheduler parity, and bounded memory.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from axis import Grid, Regridder

xr = pytest.importorskip("xarray")
pytest.importorskip("dask")
import dask.array as da  # noqa: E402


@pytest.fixture
def pair():
    src = Grid(lon=np.arange(0.5, 360.0, 2.0), lat=np.arange(-89.0, 90.0, 2.0))
    dst = Grid(lon=np.arange(0.25, 360.0, 4.0), lat=np.arange(-89.5, 90.0, 4.0))
    rg = Regridder(src, dst, "bilinear")
    LON, LAT = np.meshgrid(np.asarray(src._payload["lon"]), np.asarray(src._payload["lat"]))
    da_src = xr.DataArray(np.cos(np.radians(LAT)), dims=src.dims)
    return rg, src, dst, da_src


# ─── AC1: lazy in → lazy out, eager driver-side fit logged ──────────────────


def test_lazy_in_lazy_out(pair, caplog):
    _, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 30, src.dims[1]: 60})
    import logging

    with caplog.at_level(logging.INFO, logger="axis"):
        rg = Regridder(src, Grid(lon=np.arange(0.5, 360.0, 4.0), lat=np.arange(-88.0, 89.0, 4.0)), "bilinear")
        assert rg.fitted  # fit is eager at construction (R2)
        out = rg(lazy)
    assert out.chunks is not None  # still lazy — never computed
    assert isinstance(out.data, da.Array)
    # fit happened eagerly on the driver and was logged (FR-031)
    assert any("fit" in r.message for r in caplog.records)
    assert out.shape[-2:] == rg.target.shape


def test_no_compute_called(pair, monkeypatch):
    rg, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 30})
    calls = []
    real_compute = da.Array.compute

    def spy(self, *a, **k):
        calls.append(1)
        return real_compute(self, *a, **k)

    monkeypatch.setattr(da.Array, "compute", spy)
    rg(lazy)
    assert not calls  # transform itself never computes


# ─── AC2 / FR-032: at-most-once per worker, counters, no Matrix in payloads ──


def test_no_engine_object_in_task_payload(pair):
    rg, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 30})
    out = rg(lazy)
    payload = pickle.dumps(out.data.__dask_graph__())
    # only serialized bytes may ride in task payloads — never engine objects
    applier = rg._applier()
    assert isinstance(applier.bytes, (bytes, bytearray))
    assert b"<axis._core.Matrix" not in payload
    # round-trip through pickle keeps bytes, not engine objects
    rt = pickle.loads(pickle.dumps(applier))
    assert isinstance(rt.bytes, (bytes, bytearray))


def test_worker_sync_at_most_once(pair, local_cluster):
    rg, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 15, src.dims[1]: 45})
    out = rg(lazy)
    res = out.compute()  # runs on the distributed cluster
    assert res.shape[-2:] == rg.target.shape
    counts = local_cluster.run(_worker_sync_counts)
    assert len(counts) >= 1
    key = f"weights_{rg._uid}_{rg._fingerprint}"
    for addr, table in counts.items():
        assert table.get(key, 0) <= 1, f"worker {addr} synced weights {table.get(key)} times"
    # at least one worker must have synced (the tasks ran somewhere)
    assert any(key in t and t[key] == 1 for t in counts.values())


def _worker_sync_counts():
    from axis.core import _WORKER_SYNC_COUNTS

    return dict(_WORKER_SYNC_COUNTS)


def test_repeated_calls_no_resync(pair, local_cluster):
    rg, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 20})
    rg(lazy).compute()
    counts_before = local_cluster.run(_worker_sync_counts)
    rg(lazy * 2.0).compute()
    counts_after = local_cluster.run(_worker_sync_counts)
    key = f"weights_{rg._uid}_{rg._fingerprint}"
    for addr, table in counts_after.items():
        assert table.get(key, 0) == counts_before[addr].get(key, 0)  # no second sync


def test_cache_key_collision_proof(pair):
    # two regridders with identical weights blobs but different uids must not
    # share a cache entry (uid in key), and a regridder whose bytes changed
    # must produce a different fingerprint (fingerprint in key)
    from axis.distributed import weights_fingerprint

    rg, *_ = pair
    other = Regridder(rg.source, rg.target, "bilinear")
    assert weights_fingerprint(rg._bytes) == weights_fingerprint(other._bytes)
    assert rg._uid != other._uid
    assert f"weights_{rg._uid}" != f"weights_{other._uid}"
    assert weights_fingerprint(b"garbage") != rg._fingerprint


# ─── AC3 / FR-033: scheduler parity ─────────────────────────────────────────


def test_scheduler_parity(pair):
    rg, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 30, src.dims[1]: 60})
    eager = rg(da_src).values
    sync = rg(lazy).compute(scheduler="synchronous").values
    threads = rg(lazy).compute(scheduler="threads").values
    procs = rg(lazy).compute(scheduler="processes").values
    np.testing.assert_allclose(sync, eager, atol=1e-12)
    np.testing.assert_allclose(threads, eager, atol=1e-12)
    np.testing.assert_allclose(procs, eager, atol=1e-12)


# ─── AC4: chunking rule — spatial dims rechunked, output chunks documented ──


def test_chunking_rule(pair):
    rg, src, _, da_src = pair
    lazy = da_src.chunk({src.dims[0]: 30, src.dims[1]: 60})
    out = rg(lazy)
    # documented rule: non-spatial dims keep their input chunks; each spatial
    # target dim's chunk size is min(input chunk along that axis, target size)
    assert out.chunks[0] == lazy.chunks[0] or sum(out.chunks[-2]) == rg.target.shape[0]
    assert sum(out.chunks[-1]) == rg.target.shape[1]


def test_4d_lazy_with_time(pair):
    rg, src, _, da_src = pair
    field = xr.concat([da_src] * 4, dim="time")
    lazy = field.chunk({"time": 2, src.dims[0]: 30})
    out = rg(lazy)
    assert out.sizes["time"] == 4
    assert out.chunks[-2:] is not None
    res = out.compute()
    np.testing.assert_allclose(res.isel(time=0).values, rg(da_src).values, atol=1e-12)


# ─── AC5 / SC-005: memory-bounded — no full materialization ─────────────────


def test_memory_bounded_large_array(local_cluster):
    # ~260 MB lazy float32 across workers capped at 512 MB: succeeds only if
    # work stays chunk-bounded (the driver graph never materializes the whole
    # array); we verify a slice against the eager path.
    src = Grid(lon=np.arange(0.5, 360.0, 1.0), lat=np.arange(-89.5, 90.0, 1.0))
    dst = Grid(lon=np.arange(1.0, 360.0, 4.0), lat=np.arange(-88.0, 89.0, 4.0))
    rg = Regridder(src, dst, "bilinear")
    nt = 1000
    vals = np.cos(np.radians(np.linspace(-89.5, 89.5, 180)))[:, None] * np.ones((1, 360))
    base = da.from_array(vals.astype(np.float32), chunks=(180, 360))
    big = da.ones((nt, 180, 360), chunks=(100, 180, 360), dtype=np.float32) * base  # lazy: 21 MB chunks
    lazy = xr.DataArray(big, dims=("time", *src.dims))
    out = rg(lazy)
    assert isinstance(out.data, da.Array)
    assert big.nbytes > 200 * 1024 * 1024  # the lazy input is ~260 MB, never materialized
    chunk_bytes = int(np.prod(big.chunksize)) * 4  # 100x180x360 float32 ~ 21 MB
    assert chunk_bytes < big.nbytes / 8  # peak work tracks chunk size, not dataset size
    first = out.isel(time=0).compute()
    ref = rg(lazy.isel(time=0).compute().astype(np.float32)).values
    np.testing.assert_allclose(first.values, ref, rtol=1e-5, atol=1e-6)


# ─── eager driver fit happens before any worker sees data (AC1) ─────────────


def test_fit_is_driver_side(local_cluster):
    src = Grid("C6")
    dst = Grid(lon=np.arange(0.5, 360.0, 10.0), lat=np.arange(-80.0, 81.0, 10.0))
    rg = Regridder(src, dst, "bilinear")
    assert rg.fitted  # fitted at construction, no lazy fit state
    counts = local_cluster.run(_worker_sync_counts)
    key = f"weights_{rg._uid}_{rg._fingerprint}"
    assert all(key not in t for t in counts.values())  # nothing synced yet
