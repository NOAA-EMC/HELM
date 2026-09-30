# Python API {#python_api}

AXIS ships a curated Python package (`import axis`) built on nanobind bindings
to the Kokkos engine. The public surface is exactly the names in
`axis.__all__`; the compiled engine lives at the internal `axis._core` module
and is not part of the user API.

> Migrating from the pre-redesign surface? See
> [migration-v1.md](migration-v1.md) for the old → new symbol table.

## Installation

Build AXIS with Python bindings enabled, then install the package:

```bash
cd libs/axis
cmake -B build \
  -DAXIS_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE=$(which python3) \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
pip install ./python
```

## Quick start

```python
import xarray as xr
import axis

# 1. Build reusable, immutable Grid objects (six families auto-detected)
src = axis.Grid(xr.open_dataset("analysis.nc"))
dst = axis.Grid("C96")                       # named cubed-sphere, no file needed

# 2. Fit once — weights are generated eagerly at construction
rg = axis.Regridder(src, dst, "conservative")

# 3. Call many — output type/laziness/dtype follow the input
tas = rg(xr.open_dataset("analysis.nc").tas)         # DataArray -> DataArray
ds_out = rg(xr.open_dataset("analysis.nc"))          # Dataset   -> Dataset
```

## `Grid`

A `Grid` is an immutable snapshot of a horizontal mesh. It auto-detects the
family (rectilinear, curvilinear, cubed-sphere, UGRID/MPAS, ICON, point cloud)
from any xarray container, or accepts explicit coordinates, a named-grid
string, or raw UGRID connectivity:

```python
axis.Grid(ds)                                   # auto-detect
axis.Grid(lon=lon_vector, lat=lat_vector)       # 1-D vectors -> rectilinear
axis.Grid(lon=lon2d, lat=lat2d, bounds=corners) # 2-D matrices -> curvilinear
axis.Grid("O96")                                # named grid: C, F, G, N, O, R families
axis.Grid.from_ugrid(node_coords, offsets, indices)
axis.Grid.from_points(lon, lat)                 # sparse point cloud
```

Read-only properties: `family`, `dims`, `shape`, `n_cells`, `periodic`,
`line_type`, `fingerprint`. A `Grid` is hashable and safe to reuse across
regridders.

## `Regridder` (scalar)

```python
rg = axis.Regridder(source, target, method="bilinear", *,
                    norm="frac_area", unmapped="nan", skipna=False,
                    na_thres=1.0, periodic=None, line_type=None,
                    src_mask=None, dst_mask=None)
```

`source`/`target` accept a `Grid`, `Dataset`, `DataArray`, `dict`, or named-grid
string. Structurally impossible method/grid pairings raise `AxisConfigError`
**at construction**, never at apply time.

Call it with a `DataArray`, `Dataset`, `ndarray`, or dask-backed array;
`transform` and `regrid` are aliases of `__call__`. Introspection: `nnz`,
`fitted`, `summary()`.

### Serialization

```python
rg.save_weights("w.axisw")    # native AXISW1 format
rg.to_esmf("w.nc")            # portable, xESMF-compatible ESMF NetCDF

rg = axis.Regridder.load_weights("w.axisw", source=src, target=dst)
rg = axis.Regridder.from_esmf("w.nc", source=src, target=dst)
```

Reloaded files are validated against the source/target grid fingerprints
before any data is touched; a mismatch raises `AxisWeightMismatchError` naming
the offending side.

## `VectorRegridder` (rotation-aware u/v)

```python
vrg = axis.VectorRegridder(src, dst, "bilinear", *, src_alpha=None, dst_alpha=None)
u_out, v_out = vrg(u, v)
```

Coordinate-frame rotation happens in the engine; `src_alpha`/`dst_alpha`
override the auto-computed per-cell orientation angles. Only `bilinear` and
`nearest` are accepted (conservative vector remapping raises `AxisConfigError`).

## `VerticalRegridder` / `regrid_3d`

```python
vrg = axis.VerticalRegridder(tension=0.0, out_of_range="nan")  # nan | clip | extrapolate
field_out = vrg(field, src_levels, dst_levels, vertical_dim="lev")

# Horizontal + vertical in one call:
out3d = axis.regrid_3d(field, src, dst, src_levels, dst_levels, method="conservative")
```

## Enums

String-coercible, so plain strings work everywhere:

```python
axis.Method    # bilinear, bicubic, patch, nearest, conservative, conservative2nd
axis.Norm      # frac_area, dst_area
axis.Unmapped  # nan, mask, error
axis.LineType  # great_circle, cartesian
```

## xarray accessor (sugar)

```python
ds.tas.axis.to(dst, method="bilinear")   # one-shot regrid
rg = ds.tas.axis.fit(dst, method="bilinear")  # reusable Regridder
```

## Dask / distributed

Lazy arrays stay lazy end-to-end — no user-side `.compute()`. On a distributed
cluster, each worker syncs a regridder's weights at most once via `client.run`;
on local schedulers a thread-safe worker cache avoids re-pickling.

## Errors

All derive from `axis.AxisError`; every message names the offending input and a
concrete fix:

| Exception | Raised for |
|---|---|
| `GridError` | coordinate detection failed |
| `AxisConfigError` | unknown/impossible method, norm, line_type |
| `AxisShapeError` | input shape/dim mismatch |
| `AxisWeightMismatchError` | reloaded weights don't match the grids |
| `AxisCapabilityError` | optional engine feature not built (PROJ/NetCDF) |
| `AxisUnmappedError` | `unmapped="error"` policy triggered |
