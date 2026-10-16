#!/usr/bin/env python3
"""Measure strong and weak scaling for AXIS with local Dask Distributed workers.

The input is persisted on the workers before timed regridding. Reported compute
times therefore cover AXIS task scheduling, native application, and gathering
the output, but not initial input transfer or weight generation. Weight
generation and input persistence are timed separately.

Example::

    OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
      uv run --extra distributed --group test \
      python benchmarks/benchmark_dask_scaling.py --output benchmarks/results/dask_scaling.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

# AXIS initializes Kokkos from native code. Set conservative defaults before
# importing AXIS, while allowing callers to override them for other studies.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OMP_PROC_BIND", "false")

import dask.array as dsa
from dask import delayed
import numpy as np
import xarray as xr
from distributed import Client, LocalCluster, wait

import axis


def summarize(samples: list[float]) -> dict[str, float]:
    """Return robust timing statistics for a series of positive observations."""
    return {
        "median_seconds": float(statistics.median(samples)),
        "q25_seconds": float(np.quantile(samples, 0.25)),
        "q75_seconds": float(np.quantile(samples, 0.75)),
        "samples_seconds": samples,
    }


def make_grids(source_nx: int, source_ny: int) -> tuple[xr.Dataset, xr.Dataset]:
    """Build deterministic global regular grids at a 2:1 resolution ratio."""
    target_nx = max(8, source_nx // 2)
    target_ny = max(4, source_ny // 2)
    source = xr.Dataset(
        {
            "lon": ("lon", np.linspace(0.0, 360.0, source_nx, endpoint=False)),
            "lat": ("lat", np.linspace(-89.5, 89.5, source_ny)),
        }
    )
    target = xr.Dataset(
        {
            "lon": ("lon", np.linspace(0.0, 360.0, target_nx, endpoint=False)),
            "lat": ("lat", np.linspace(-89.0, 89.0, target_ny)),
        }
    )
    return source, target


def make_field(source: xr.Dataset, time_steps: int, phase_offset: float) -> xr.DataArray:
    """Create a lazy field without embedding a large NumPy array in the graph."""
    latitudes = source["lat"].values
    longitudes = source["lon"].values

    def cosine_slice(time_index, lat_values, lon_values, phase_step):
        lat = np.deg2rad(lat_values)[:, None]
        lon = np.deg2rad(lon_values)[None, :]
        return np.cos(lat) * np.cos(lon + time_index * phase_step)

    time_chunks = [
        dsa.from_delayed(
            delayed(cosine_slice)(index, latitudes, longitudes, phase_offset),
            shape=(latitudes.size, longitudes.size),
            dtype=np.float64,
        )
        for index in range(time_steps)
    ]
    lazy = dsa.stack(time_chunks, axis=0)
    return xr.DataArray(
        lazy,
        dims=("time", "lat", "lon"),
        coords={"time": np.arange(time_steps), "lat": source["lat"], "lon": source["lon"]},
        name="cosine_wave",
        attrs={"units": "1", "description": "smooth phase-shifted cosine field"},
    )


def time_weight_generation(source: xr.Dataset, target: xr.Dataset, repeats: int) -> tuple[axis.Regridder, dict]:
    """Warm native initialization, then measure reusable weight construction."""
    warmup = axis.Regridder(source, target, method="bilinear")
    warmup.close()

    samples = []
    retained = None
    for _ in range(repeats):
        start = time.perf_counter()
        regridder = axis.Regridder(source, target, method="bilinear")
        samples.append(time.perf_counter() - start)
        if retained is None:
            retained = regridder
        else:
            regridder.close()

    assert retained is not None
    return retained, summarize(samples)


def run_workload(
    regridder: axis.Regridder,
    source: xr.Dataset,
    time_steps: int,
    workers: int,
    repeats: int,
    phase_offset: float,
) -> dict:
    """Run one fixed worker-count and workload-size combination."""
    field = make_field(source, time_steps, phase_offset)
    cluster_start = time.perf_counter()
    cluster = LocalCluster(
        n_workers=workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
        memory_limit=0,
    )
    cluster_start_seconds = time.perf_counter() - cluster_start

    try:
        with Client(cluster) as client:
            client.wait_for_workers(workers)

            persist_start = time.perf_counter()
            worker_field = field.persist()
            wait(worker_field.data)
            input_persist_seconds = time.perf_counter() - persist_start

            graph_start = time.perf_counter()
            lazy_result = regridder(worker_field)
            graph_build_seconds = time.perf_counter() - graph_start
            task_graph_size = len(lazy_result.data.__dask_graph__())

            # Warm up native worker state and validate the computed values before
            # collecting timed observations.
            warm_result = lazy_result.isel(time=0).compute()
            eager_reference = regridder(
                xr.DataArray(
                    np.asarray(field.isel(time=0).compute().values),
                    dims=("lat", "lon"),
                    coords={"lat": source["lat"], "lon": source["lon"]},
                )
            )
            np.testing.assert_allclose(warm_result.values, eager_reference.values, rtol=1e-12, atol=1e-12)

            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                output = lazy_result.compute()
                samples.append(time.perf_counter() - start)
                if not np.isfinite(output.values).all():
                    raise RuntimeError("AXIS Dask scaling run produced non-finite output values")

            timing = summarize(samples)
            target_cells = int(regridder._weights_matrix.n_dst)
            source_cells = int(regridder._weights_matrix.n_src)
            median_seconds = timing["median_seconds"]
            return {
                "workers": workers,
                "threads_per_worker": 1,
                "time_steps": time_steps,
                "source_cells_per_step": source_cells,
                "target_cells_per_step": target_cells,
                "input_persist_seconds": input_persist_seconds,
                "graph_build_and_weight_registration_seconds": graph_build_seconds,
                "cluster_start_seconds": cluster_start_seconds,
                "task_graph_keys": task_graph_size,
                "compute": timing,
                "source_cells_per_second": float(time_steps * source_cells / median_seconds),
                "target_cells_per_second": float(time_steps * target_cells / median_seconds),
                "validation": "pass",
            }
    finally:
        cluster.close()


def add_scaling_metrics(rows: list[dict]) -> None:
    """Add speedup and efficiency relative to the matching one-worker workload."""
    for mode in ("strong", "weak"):
        baseline = next(row for row in rows if row["mode"] == mode and row["workers"] == 1)
        baseline_seconds = baseline["compute"]["median_seconds"]
        baseline_throughput = baseline["target_cells_per_second"]
        for row in rows:
            if row["mode"] != mode:
                continue
            workers = row["workers"]
            elapsed_ratio = baseline_seconds / row["compute"]["median_seconds"]
            if mode == "strong":
                row["speedup_vs_one_worker"] = elapsed_ratio
                row["parallel_efficiency"] = elapsed_ratio / workers
            else:
                # Weak scaling keeps per-worker work fixed. Ideal throughput
                # scales linearly while elapsed time remains nearly constant.
                row["throughput_scale_vs_one_worker"] = row["target_cells_per_second"] / baseline_throughput
                row["weak_scaling_efficiency"] = row["throughput_scale_vs_one_worker"] / workers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="1,2,4,8", help="comma-separated local worker counts")
    parser.add_argument("--strong-time-steps", type=int, default=128, help="fixed steps for strong scaling")
    parser.add_argument("--weak-steps-per-worker", type=int, default=32, help="steps per worker for weak scaling")
    parser.add_argument("--source-nx", type=int, default=720)
    parser.add_argument("--source-ny", type=int, default=360)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--weight-repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results/dask_scaling.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workers = sorted({int(value) for value in args.workers.split(",")})
    if not workers or workers[0] < 1 or args.repeats < 2 or args.weight_repeats < 2:
        raise SystemExit("workers must be positive, and timing/weight repeats must each be at least 2")
    if args.strong_time_steps < 1 or args.weak_steps_per_worker < 1:
        raise SystemExit("time-step counts must be positive")
    if args.source_nx < 8 or args.source_ny < 4:
        raise SystemExit("source grid must be at least 8 by 4")

    source, target = make_grids(args.source_nx, args.source_ny)
    regridder, weight_timing = time_weight_generation(source, target, args.weight_repeats)
    rows = []
    try:
        for worker_count in workers:
            strong = run_workload(
                regridder,
                source,
                args.strong_time_steps,
                worker_count,
                args.repeats,
                phase_offset=0.017,
            )
            strong["mode"] = "strong"
            rows.append(strong)

            weak = run_workload(
                regridder,
                source,
                args.weak_steps_per_worker * worker_count,
                worker_count,
                args.repeats,
                phase_offset=0.023,
            )
            weak["mode"] = "weak"
            rows.append(weak)
    finally:
        regridder.close()

    add_scaling_metrics(rows)
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "dask": __import__("dask").__version__,
            "distributed": __import__("distributed").__version__,
            "xarray": xr.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND"),
            "native_extension": str(axis.__file__),
        },
        "methodology": {
            "method": "bilinear",
            "execution": "single-node Dask Distributed, process workers, one native thread each",
            "source_shape": [args.source_ny, args.source_nx],
            "target_shape": [max(4, args.source_ny // 2), max(8, args.source_nx // 2)],
            "horizontal_core_chunks": "each complete source spatial dimension per task",
            "time_chunk": 1,
            "input": "persisted to workers before timed runs; transfer time reported separately",
            "compute_timing": "Dask compute wall time, including task scheduling, native regridding, and output gather",
            "warmups": "1 per workload size, excluded from timed samples",
            "repeats_per_point": args.repeats,
            "weight_generation": weight_timing,
            "strong_scaling": "fixed number of time steps while worker count increases",
            "weak_scaling": "time steps increase in proportion to worker count",
        },
        "results": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Weight generation median: {weight_timing['median_seconds']:.6f} s")
    print("mode workers steps compute_median_s q25_s q75_s target_cells_per_s speedup_or_scale efficiency")
    for row in rows:
        if row["mode"] == "strong":
            scale = row["speedup_vs_one_worker"]
            efficiency = row["parallel_efficiency"]
        else:
            scale = row["throughput_scale_vs_one_worker"]
            efficiency = row["weak_scaling_efficiency"]
        timing = row["compute"]
        print(
            f"{row['mode']} {row['workers']} {row['time_steps']} "
            f"{timing['median_seconds']:.6f} {timing['q25_seconds']:.6f} "
            f"{timing['q75_seconds']:.6f} {row['target_cells_per_second']:.1f} "
            f"{scale:.3f} {efficiency:.3f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
