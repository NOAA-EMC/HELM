# Using the AXIS Python Interface

AXIS (Arbitrary eXgrid Interpolation Solver) is a Kokkos-powered regridding
engine for Earth-system fields with a curated Python package (`import axis`)
that plugs into `xarray`, `cf-xarray`, and Dask. This README is the
task-oriented guide; the symbol-level reference lives in
[`python_api.md`](python_api.md), and the old→new symbol table in
[`migration-v1.md`](migration-v1.md).

**Contents**

- [Install](#install)
- [Five-minute tour](#five-minute-tour)
- [Building grids](#building-grids)
- [Regridding fields](#regridding-fields)
- [Saving and reusing weights](#saving-and-reusing-weights)
- [Vector (u/v) regridding](#vector-uv-regridding)
- [Vertical interpolation and 3-D](#vertical-interpolation-and-3-d)
- [Scaling out with Dask](#scaling-out-with-dask)
- [Command line](#command-line)
- [Error handling](#error-handling)
- [Where to go next](#where-to-go-next)

---

## Install

Requires Python **3.12+**. The package (`axis-regrid` on PyPI-style metadata)
ships a compiled `axis._core` extension built with CMake + nanobind, so
install from source in a checkout:

```bash
cd libs/axis
pip install . --no-build-isolation        # scikit-build-core drives CMake
```

Or, for development, build the extension in place and use `PYTHONPATH`:

```bash
cmake -B build -DAXIS_BUILD_PYTHON=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target _core -j
cp build/python/_core*.so python/axis/
export PYTHONPATH=$PWD/python
```

Run the test suite to verify a working install:

```bash
pip install pytest && python -m pytest tests_python -q
```

> **Note** — `axis._core` is the internal engine binding and deliberately not
> part of the public API. Everything you need is exported from `axis` itself.

---

## Five-minute tour

Three calls take you from two grids to a regridded field:

```python
import xarray as xr
import axis

src = axis.Grid(xr.open_dataset("analysis.nc"))   # 1. detect the source grid
dst = axis.Grid("C96")                            # 2. named cubed-sphere, no file
rg  = axis.Regridder(src, dst, "conservative")    # 3. fit (weights built eagerly)

out = rg(ds.tas)                                  # call many: DataArray in, out
```

Key properties of this shape:

- **`Grid` objects are immutable, hashable, and reusable** — build once, pass
  to many `Regridder`s.
- **Fitting happens at construction.** Impossible method/grid combinations
  raise at that point, never mid-run.
- **Calling is lazy and type-preserving.** `DataArray`→`DataArray`,
  `Dataset`→`Dataset`, `ndarray`→`ndarray`, dask-backed in → dask-backed out.

---

## Building grids

AXIS supports six horizontal grid families, auto-detected from CF metadata
(cf-xarray first, then heuristics, then UGRID attributes):

| Family | What it is | Typical construction |
|---|---|---|
| `RECTILINEAR` | 1-D `lat`/`lon` vectors | `axis.Grid(ds)` or `axis.Grid(lon=..., lat=...)` |
| `CURVILINEAR` | 2-D `lat`/`lon` matrices (CF bounds honored) | `axis.Grid(ds)` |
| `CUBED_SPHERE` | 6-tile files or named `C<N>` grids | `axis.Grid("C96")` |
| `UGRID` | MPAS / UGRID `mesh_topology` datasets | `axis.Grid(ds)` |
| `ICON` | raw node/offset/index connectivity | `axis.Grid.from_ugrid(nodes, offsets, indices)` |
| `POINTS` | sparse point cloud (obs/flight tracks) | `axis.Grid.from_points(lon, lat)` |

```python
g = axis.Grid(ds)              # auto-detect
g.family                       # -> axis.GridFamily.POINTS etc.
g.shape, g.n_cells             # logical dims and cell count
g.fingerprint                  # stable 16-hex id, used by weight files
```

Named grids come from the engine's registry — `C` (cubed-sphere), `F`
(regular Gaussian), `G` (NOAA GRIB), `N`/`O` (reduced Gaussian, ECMWF
octahedral), `R` (regular lat-lon):

```python
axis.Grid("O64")               # reduced Gaussian octahedral, T64
axis.Grid("F128")              # regular Gaussian, 128 latitudes
```

Explicit overrides are always available when auto-detection is ambiguous:
`axis.Grid(ds, lon=..., lat=..., bounds=...)`. A failure lists every pattern
searched and names the override to use (`GridError`).

Any grid works as **both source and target** of a `Regridder`.

---

## Regridding fields

```python
rg = axis.Regridder(
    src, dst,
    method="bilinear",        # bilinear | bicubic | patch | nearest |
                              # conservative | conservative2nd
    norm="frac_area",         # conservative only: frac_area | dst_area
    unmapped="nan",           # nan | mask | error
    skipna=False,             # renormalise weights over valid sources
    na_thres=1.0,             # max tolerated NaN fraction per output cell
    periodic=None,            # force dateline wrap for longitude
    line_type="great_circle", # great_circle | cartesian
    src_mask=None,            # boolean/0-1 mask: exclude masked source cells
    dst_mask=None,            # exclude masked destination cells (zeroed rows)
)
```

Apply it to anything:

```python
out_da  = rg(tas_dataarray)              # -> DataArray on dst's dims
out_ds  = rg(ds)                         # Dataset: regridded vars + passthrough
out_np  = rg(arr)                        # ndarray (..., *src_shape) -> (..., *dst_shape)
```

`rg.transform` and `rg.regrid` are aliases of `rg(...)`. Introspect the fitted
operator with `rg.nnz` (non-zero weights), `rg.fitted`, and `rg.summary()`.

There is also an xarray accessor for one-shot regrids:

```python
out = ds.tas.axis.to(dst, method="bilinear")      # one-shot
rg  = ds.tas.axis.fit(dst, method="bilinear")     # -> reusable Regridder
```

**Reuse across variables/time slices is free** — the weights are built once at
construction; every call is a pure SpMV against the cached matrix.

---

## Saving and reusing weights

Weight generation is the expensive step. Persist it:

```python
rg.save_weights("analysis_to_c96.axisw")   # native AXISW1 format
rg.to_esmf("analysis_to_c96.nc")           # portable ESMF/SCRIP NetCDF

rg2 = axis.Regridder.load_weights("analysis_to_c96.axisw", source=src, target=dst)
rg3 = axis.Regridder.from_esmf("analysis_to_c96.nc", source=src, target=dst)
```

Reloaded files are validated against the **grid fingerprints** of the
`source`/`target` you pass in, *before any data is touched*. A mismatch raises
`AxisWeightMismatchError` naming the offending side and how to fix it. ESMF
files produced by xESMF/ESMF lack AXIS fingerprints and fall back to
shape-only validation — results match xESMF within float tolerance.

The `VectorRegridder` has the same four methods
(`save_weights` / `load_weights` / `to_esmf` / `from_esmf`) and stores both
u- and v-matrices in one file.

---

## Vector (u/v) regridding

Wind components must be regridded *together* so direction survives the move
between frames (e.g. cubed-sphere tile edges):

```python
vrg = axis.VectorRegridder(src_cs, dst_ll, "bilinear")
u_out, v_out = vrg(u_in, v_in)
```

Coordinate-frame rotation is computed in the engine from grid geometry. Pass
`src_alpha=` / `dst_alpha=` (per-cell arrays) to override the orientation
angles. Only `bilinear` and `nearest` are valid for vectors; asking for a
conservative vector remap raises `AxisConfigError` at construction.

---

## Vertical interpolation and 3-D

Monotone cubic spline (Akima-style, with tension) along a vertical axis,
including per-column varying level arrays:

```python
vr = axis.VerticalRegridder(tension=2.0, out_of_range="nan")  # nan | clip | extrapolate
plev = vr(field, src_levels, dst_levels, vertical_dim="lev")
```

- `src_levels` / `dst_levels`: 1-D arrays, or arrays with a trailing level dim
  and leading column dims (per-column level heights, e.g. hybrid-sigma per
  model column).
- `out_of_range` controls target levels outside the source bracket: fill NaN
  (default), clamp to the boundary value, or linear extrapolation. Never a
  silent wrong number.
- Leading dims (time, ensemble) pass through; dask-backed fields stay lazy.

Horizontal + vertical in one call:

```python
out3d = axis.regrid_3d(field, src, dst, src_levels, dst_levels, method="bilinear")
```

---

## Scaling out with Dask

Lazy arrays remain lazy end-to-end — no user-side `.compute()` needed:

```python
import dask.distributed as dd
import axis, xarray as xr

client = dd.Client(dd.LocalCluster(n_workers=4, threads_per_worker=1))

big = xr.open_dataset("huge_field.nc", chunks={"time": 1})
rg  = axis.Regridder(axis.Grid(big), axis.Grid("C96"), "conservative")
out = rg(big.air)          # still lazy
res = out.compute()        # workers apply the cached weights per chunk
```

How it works: weights are generated once on the driver; on a distributed
cluster each worker receives the serialized matrix at most once (via
`client.run`), never re-pickled per task. On local process/thread schedulers a
thread-safe worker cache serves the same purpose. Peak worker memory is
bounded by chunk size, not field size.

---

## Command line

The install provides the `axis-regrid` executable (also `python -m axis.cli`):

```bash
# Regrid a file
axis-regrid regrid -s analysis.nc -t c96_grid.nc -o out.nc -m conservative -v tas

# Target can also be a named grid
axis-regrid regrid -s analysis.nc -t C96 -o out.nc -m bilinear

# Just produce the weights (portable ESMF or native)
axis-regrid weights -s analysis.nc -t c96_grid.nc -o w.nc --format esmf
axis-regrid weights -s analysis.nc -t c96_grid.nc -o w.axisw --format native

# Reuse precomputed weights from the Python API (or vice versa)
axis-regrid regrid -s analysis.nc --weights w.axisw -o out.nc

# Discover what's available
axis-regrid list methods
axis-regrid list grids
```

Useful flags (on both `regrid` and `weights`): `--norm frac-area|dst-area`,
`--unmapped nan|mask|error`, `--periodic`, `--skipna`, `--na-thres`,
`--line-type great-circle|cartesian`, `--debug` (full tracebacks).

Exit codes: `0` success, `1` runtime/IO/API error (diagnostic on stderr,
prefixed `axis-regrid: error:`), `2` usage error. A CLI-written weight file
loads in Python via `Regridder.from_esmf` / `load_weights`, and a
Python-written one is consumed by `--weights`.

---

## Error handling

Every exception derives from `axis.AxisError`, and every message names the
offending input plus a concrete fix:

| Exception | Raised when |
|---|---|
| `GridError` | coordinate/grid detection failed |
| `AxisConfigError` | unknown or structurally impossible method/config |
| `AxisShapeError` | field dims don't match the grid |
| `AxisWeightMismatchError` | reloaded weights don't match the given grids |
| `AxisCapabilityError` | optional engine feature (PROJ/NetCDF) not built |
| `AxisUnmappedError` | `unmapped="error"` policy was triggered |

All config arguments (`method`, `norm`, `unmapped`, `line_type`) accept plain
strings, so enum imports are optional.

---

## Where to go next

- [`python_api.md`](python_api.md) — full symbol-level API reference
- [`migration-v1.md`](migration-v1.md) — legacy (v1) → new API migration table
- [`getting_started.md`](getting_started.md) — C++ library build and concepts
- [`weight_generation.md`](weight_generation.md), [`apply.md`](apply.md),
  [`caching.md`](caching.md), [`masking.md`](masking.md) — engine internals
- Runnable example: [`../examples/example_regrid.py`](../examples/example_regrid.py)
- Tests double as usage docs: [`../tests_python/`](../tests_python/)
