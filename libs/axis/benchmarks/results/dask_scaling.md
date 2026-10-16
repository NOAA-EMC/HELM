# AXIS Dask scaling run

Run on 2026-10-15 with `libs/axis/.venv`, Python 3.14.7, Dask and Distributed
2026.7.1, xarray 2026.7.0, macOS arm64, 10 logical CPUs, and
`OMP_NUM_THREADS=1`. The exact samples and environment metadata are in
[`dask_scaling.json`](dask_scaling.json).

The benchmark used bilinear remapping from a 720 × 360 regular source grid to
a 360 × 180 target grid. Strong scaling kept 128 time steps fixed. Weak scaling
used 32 steps per worker. Each time step was one Dask chunk with both complete
horizontal dimensions. Five timed compute repetitions were collected per
point; the table reports medians. The timer covers distributed scheduling,
native regridding, and gathering the result. Cluster startup, input persistence,
and weight generation were recorded separately. One warm-up was excluded.

| Mode | Workers | Steps | Median compute (s) | Target cells/s (million) | Scale vs 1 worker | Efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Strong | 1 | 128 | 0.691 | 12.01 | 1.00× | 100% |
| Strong | 2 | 128 | 0.638 | 13.01 | 1.08× | 54% |
| Strong | 4 | 128 | 0.613 | 13.54 | 1.13× | 28% |
| Strong | 8 | 128 | 0.530 | 15.66 | 1.30× | 16% |
| Weak | 1 | 32 | 0.172 | 12.06 | 1.00× | 100% |
| Weak | 2 | 64 | 0.381 | 10.87 | 0.90× | 45% |
| Weak | 4 | 128 | 0.532 | 15.59 | 1.29× | 32% |
| Weak | 8 | 256 | 0.941 | 17.64 | 1.46× | 18% |

## Interpretation

On this single machine, adding workers improved strong-scaling throughput only
modestly: eight workers processed the fixed workload 1.30× as fast as one.
Efficiency fell as workers were added. Under weak scaling, eight workers
processed 1.46× the one-worker throughput for eight times the data, while
elapsed time increased from 0.172 s to 0.941 s. This workload therefore does
not show near-linear scaling; scheduling, per-task work, output transfer, and
single-host resource contention are material at this task size. The timing
quartiles in the JSON are broad at several points, so small differences between
worker counts should not be over-interpreted.

This is a local process-worker experiment, not a multi-node cluster result.
Input generation avoids embedding the full NumPy field in every Dask task
graph; input persistence and graph registration remain visible as separate
setup measurements. The benchmark checks a computed Dask slice against eager
AXIS output and rejects non-finite timed results.

Reproduce from `libs/axis` with:

```sh
OMP_NUM_THREADS=1 OMP_PROC_BIND=false \
  .venv/bin/python benchmarks/benchmark_dask_scaling.py \
  --workers 1,2,4,8 \
  --strong-time-steps 128 \
  --weak-steps-per-worker 32 \
  --source-nx 720 --source-ny 360 \
  --repeats 5 --weight-repeats 3 \
  --output benchmarks/results/dask_scaling.json
```
