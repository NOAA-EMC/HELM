# Migration guide: legacy AXIS Python API → v2

This page is the single old → new mapping for the pre-redesign Python surface
retired by feature `001-axis-python-api-redesign` (FR-040). There are **no
compatibility shims**: every legacy name below is gone from `import axis` and
cannot be imported from its old submodule. Match your old call to the
replacement and update the import.

The compiled engine is now internal at `axis._core`; user code imports only the
curated names in `axis.__all__`.

---

## Symbol table

| Legacy (retired) | Replacement | Notes |
|---|---|---|
| `axis.Regridder(source, target, method=...)` (legacy signature) | `axis.Regridder(source, target, method=...)` (new) | Same call shape, new semantics: weights fit **eagerly at construction**; `source`/`target` may be `Grid`, `Dataset`, `DataArray`, `dict`, or a named-grid string. |
| `Regridder(method=...)` then `.fit(src, dst)` (instantiate-then-fit) | `Regridder(src, dst, method=...)` | Construction *is* the fit. `.transform(x)` / `.regrid(x)` remain aliases of `__call__(x)`. |
| `regridder.to_file(path)` | `regridder.save_weights(path)` | Native `AXISW1` format. |
| `Regridder(src, dst, weights_file=path)` | `Regridder.load_weights(path, source=g, target=g)` | Classmethod; pass `Grid` objects for `source`/`target`. |
| `regridder.to_esmf(path)` | `regridder.to_esmf(path)` | Unchanged name; now stamps AXIS fingerprints (still xESMF-loadable). |
| `Regridder.from_esmf(path, src, dst)` (positional) | `Regridder.from_esmf(path, source=g, target=g)` | Keyword-only `source`/`target` `Grid`s. |
| `axis.grid.GridFactory` | `axis.Grid` | The unified grid abstraction auto-detects family. |
| `axis.grid.Geometry` | `axis.Grid` | |
| `axis.grid.RectilinearGrid(lons, lats)` | `axis.Grid(lon=lons, lat=lats)` | 1-D `lon`/`lat` vectors → rectilinear. |
| `axis.grid.CurvilinearGrid(lons, lats)` | `axis.Grid(lon=lons, lat=lats)` | 2-D `lon`/`lat` matrices → curvilinear. |
| `axis.grid.UnstructuredMesh(coords, offsets, indices)` | `axis.Grid.from_ugrid(coords, offsets, indices)` | |
| `axis.grid.create_axis_mesh(ds)` | `axis.Grid(ds)` | Detection is identical, output is a reusable `Grid`. |
| `geometry.to_mesh()` | *(internal)* | Mesh construction is now hidden inside `Regridder`. |
| `axis.VectorRegridder(...)` (legacy) | `axis.VectorRegridder(source, target, method=...)` | New rotation-aware paired u/v; `rg(u, v)` returns `(u_out, v_out)`. |
| `axis.VerticalRegridder(...)` (legacy) | `axis.VerticalRegridder(*, tension=0.0, out_of_range="nan")` | Keyword-only; `rg(field, src_levels, dst_levels)`. |
| `axis.regrid_3d(...)` (legacy signature) | `axis.regrid_3d(field, source, target, src_levels, dst_levels, ...)` | One-call horizontal + vertical composition. |
| `ds.temperature.axis.regrid_to(other, method=...)` | `ds.temperature.axis.to(other, method=...)` | Accessor kept as thin sugar (R14); `regrid_to` → `to`, plus `fit` for a reusable regridder. |
| Raw `_core` re-exports: `Mesh`, `Matrix`, `make_regular_mesh`, `make_projected_mesh`, `make_ugrid_mesh`, `make_named_mesh`, `apply_weights`, `batch_apply`, `detect_tripolar_grid`, `generate_vector_weights` | *(internal only)* | Not part of the public surface. Use `Grid`/`Regridder`. |
| `_core.Method`, `_core.NormType`, `_core.UnmappedAction`, `_core.LineType` | `axis.Method`, `axis.Norm`, `axis.Unmapped`, `axis.LineType` | Public, string-coercible enums. |
| Legacy CLI `axis-regrid -s SRC -t DST -o OUT` (flat) | `axis-regrid regrid -s SRC -t DST -o OUT` | Subcommand form; see [CLI contract](../../specs/001-axis-python-api-redesign/contracts/cli.md). New `weights` and `list` subcommands. |

---

## Error types

Legacy code catching `ValueError` / `KeyError` / `TypeError` from grid or
method errors should catch the typed hierarchy (all derive from
`axis.AxisError`):

| Old | New |
|---|---|
| `ValueError` (bad method / norm / line_type) | `axis.AxisConfigError` |
| `KeyError` (missing coords) | `axis.GridError` |
| shape/size mismatch | `axis.AxisShapeError` |
| reloaded-weights grid mismatch | `axis.AxisWeightMismatchError` |
| capability not built (PROJ / NetCDF) | `axis.AxisCapabilityError` |
| `unmapped="error"` hit | `axis.AxisUnmappedError` |

Every message names the offending input and a concrete fix.

---

## Worked example

**Before (legacy):**

```python
import xarray as xr
from axis.grid import RectilinearGrid
import axis

ds_src = xr.open_dataset("analysis.nc")
ds_dst = xr.open_dataset("target.nc")

rg = axis.Regridder(method="conservative")
rg.fit(ds_src, ds_dst)
out = rg.transform(ds_src["tas"])
rg.to_file("w.bin")
rg2 = axis.Regridder(ds_src, ds_dst, weights_file="w.bin")
```

**After (v2):**

```python
import xarray as xr
import axis

ds_src = xr.open_dataset("analysis.nc")
src = axis.Grid(ds_src)                 # or Grid(lon=..., lat=...), Grid("C96"), ...
dst = axis.Grid(xr.open_dataset("target.nc"))

rg = axis.Regridder(src, dst, "conservative")   # fitted at construction
out = rg(ds_src["tas"])                  # __call__ (transform/regrid alias)
rg.save_weights("w.axisw")
rg2 = axis.Regridder.load_weights("w.axisw", source=src, target=dst)
```

---

## CLI

```bash
# regrid (subcommand now required)
axis-regrid regrid -s analysis.nc -t mpas_grid.nc -o out.nc -m conservative -v tas

# weights-only, portable ESMF (default) or native
axis-regrid weights -s analysis.nc -t mpas_grid.nc -o w.nc -m conservative
axis-regrid weights -s analysis.nc -t mpas_grid.nc -o w.axisw --format native

# list
axis-regrid list methods
axis-regrid list grids
```

Exit codes: `0` success · `1` runtime error (actionable message on stderr, no
traceback unless `--debug`) · `2` argument-parsing error.
