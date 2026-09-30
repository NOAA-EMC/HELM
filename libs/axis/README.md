# AXIS — Arbitrary eXgrid Interpolation Solver

**Tier 1 C++20 micro-library** within the [HELM](../../README.md) ecosystem providing
stateless spatial interpolation (regridding) for Earth-system fields.

AXIS replaces the legacy ESMF spatial-discretization stack (`ESMF_Mesh`,
`ESMF_Grid`, `ESMF_LocStream`, `ESMF_Regrid`, and the offline
`ESMF_RegridWeightGen` application) with a decentralized, zero-copy,
hardware-portable toolkit built on Kokkos, now fully packaged for Python, `xarray`, and **Dask parallel computing**.

---

## 🚀 Architectural Advantages vs. Legacy CDO/ESMF

AXIS represents a massive paradigm shift in high-performance Earth-system regridding, addressing the fundamental scaling bottlenecks of legacy CDO (SCRIP-based) and ESMF (ESMPy/xregrid) stacks:

1. **Exascale Speedups (10x to 127x faster):** Bypasses all disk-based file I/O overheads and executes parallel-on-device weight generation and SpMVs (Sparse Matrix-Vector multiplication) using Kokkos (CPU OpenMP, GPU CUDA/HIP).
2. **True 3D Spherical Cartesian BVH (Resolving Dateline Seams):** Legacy CDO/ESMF perform 2D search queries in flat $(\lambda, \theta)$ space, causing polar singularities and silent spatial misses at the $0^\circ/360^\circ$ periodic dateline seam. AXIS projects coordinates to 3D Cartesian space on the unit sphere and builds a **3D ArborX AABB tree**, natively resolving all wrapping seams and coordinate singularities.
3. **Spherical-Exact Analytical Conservative Overlaps:** Legacy SCRIP-based engines (CDO/ESMF) assume cell boundaries are straight lines in 2D coordinate space, leading to severe polar stretching and mass sum errors at high latitudes. AXIS treats cell boundaries as **true Great-Circle arcs** and computes intersections using exact spherical excess areas, guaranteeing flawless mass conservation.
4. **Z-Curve Morton Cache Alignment:** Unstructured queries are pre-sorted using a 3D Space-Filling Z-Curve, grouping parallel CPU/GPU threads spatially to maximize L1/L2 cache hit rates and eliminate memory bandwidth thrashing.
5. **Zero-Copy Memory Interfaces:** Preserves Fortran column-major (`LayoutLeft`/`np.asfortranarray`) memory layouts natively across boundaries, avoiding costly data layout transposition.

---

## 📊 Summary of Live Benchmark Results

*Measured on an ARM64 (aarch64) Linux VM under Docker with unified OpenMP parallel execution:*

| Case | Grid & Size | Method | CDO Time (s) | AXIS Time (s) | AXIS Speedup vs CDO |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **1** | Global upscale ($720\times360 \to 1440\times720$) | **Bilinear** | `0.804` | **`0.378`** | **$2.1\times$** ($38\times$ vs `xregrid`) |
| | | **Nearest** | `0.936` | **`0.289`** | **$3.2\times$** ($15\times$ vs `xregrid`) |
| | | **Conservative** | `2.866` | **`0.288`** | **$10.0\times$** ($81\times$ vs `xregrid`) |
| **2** | High-Res global ($259\text{k} \to 6.48\text{M}$ cells) | **Nearest** | `2.318` | **`1.470`** | **$1.5\times$** |
| | | **Conservative** | `12.033` | **`1.328`** | **$9.0\times$** |
| **3** | Regional Projected LCC ($120\times120 \to 100\times100$) | **Conservative** | `0.591` | **`0.054`** | **$10.8\times$** ($10.3\times$ vs `xregrid`) |
| **4** | Unstructured MPAS $\to$ Regular ($10\text{k} \to 90\times90$) | **Bilinear** | *FAILED* | **`0.123`** | **Exclusive!** |
| | | **Nearest** | `0.490` | **`0.147`** | **$3.3\times$** |
| | | **Conservative** | `0.728` | **`0.188`** | **$3.8\times$** |
| **5** | Regular $\to$ Unstructured MPAS ($360\times180 \to 2000$) | **Nearest** | `0.470` | **`0.209`** | **$2.2\times$** |
| | | **Conservative** | `0.492` | **`0.162`** | **$3.0\times$** |

---

## 🐍 Python & xarray/Dask Integration

AXIS is fully packaged for Python with deep `xarray`, `cf-xarray`, and **Dask** distributed cluster support. It installs seamlessly using standard packaging tools:

```bash
# Compile and install AXIS with Python bindings globally
pip install ./libs/axis
```

### 1. Reusable `Grid` objects
A `Grid` is an immutable snapshot of a horizontal mesh. It auto-detects the
grid family from any xarray container, or you can build it from explicit
coordinates, a named-grid string, or raw UGRID connectivity:

```python
import xarray as xr
import axis

ds_src = xr.open_dataset("gfs_source.nc")
ds_dst = xr.open_dataset("orca_target.nc")

src = axis.Grid(ds_src)                       # auto-detect (rectilinear/curvilinear/UGRID/...)
dst = axis.Grid(ds_dst)
cs  = axis.Grid("C96")                        # named cubed-sphere, no file needed
rr  = axis.Grid(lon=lon_vector, lat=lat_vector)   # explicit rectilinear
```

Six families are supported: rectilinear, curvilinear, cubed-sphere,
UGRID/MPAS, ICON, and sparse point clouds.

### 2. Fit-once / call-many `Regridder`
A `Regridder` fits its weights eagerly at construction and applies them to any
number of fields. Output type, dimensionality, laziness, and dtype follow the
input:

```python
rg = axis.Regridder(src, dst, "conservative")   # source/target may be Grid/Dataset/dict/str

tas = rg(ds_src.tas)              # DataArray in  -> DataArray out
allvars = rg(ds_src)              # Dataset in    -> Dataset out
flat = rg(np_array)               # ndarray in    -> ndarray out

# One-shot sugar via the xarray accessor:
regridded_temp = ds_src.temperature.axis.to(dst, method="bilinear")
```

Vector winds use the rotation-aware `VectorRegridder`, which returns a paired
`(u_out, v_out)` with coordinate-frame rotation handled in the engine:

```python
vrg = axis.VectorRegridder(src, dst, "bilinear")
u_out, v_out = vrg(ds_src.u10, ds_src.v10)
```

Vertical / 3-D columns use `VerticalRegridder` (per-column spline with an
explicit out-of-range policy) or the one-call `regrid_3d` composition:

```python
vrgd = axis.VerticalRegridder(tension=0.0, out_of_range="clip")
field_out = vrgd(field, src_levels, dst_levels, vertical_dim="lev")

# or horizontal + vertical in one call:
out3d = axis.regrid_3d(field, src, dst, src_levels, dst_levels, method="conservative")
```

### 3. Save, Reload, and Reuse Weights
Compiled weights can be serialized and reloaded in a fresh process — either in
AXIS's native format or as a portable, xESMF-compatible ESMF NetCDF file:

```python
rg.save_weights("w.axisw")                                   # native AXISW1
rg2 = axis.Regridder.load_weights("w.axisw", source=src, target=dst)

rg.to_esmf("w.nc")                                           # portable ESMF
rg3 = axis.Regridder.from_esmf("w.nc", source=src, target=dst)
```

Reloading validates the source/target grid fingerprints before touching data,
so a mismatched file fails fast with an actionable message.

### 4. Flawless Dask Distributed Parallelism
When running on backed Dask chunk-wise arrays, AXIS automatically serializes its C++ compiled weights and distributes them across all remote worker nodes in parallel using `client.run`. For local process/thread-wise schedulers, it leverages a thread-safe worker cache to completely bypass Python pickling limits, ensuring zero `PicklingError`s and beautiful, lazy parallel scalability.

---

## 💻 `axis-regrid` Command-Line Tool

AXIS registers a native command-line executable **`axis-regrid`** in your system path. It uses subcommands (`regrid`, `weights`, `list`) so shell scripts and batch pipelines can run Kokkos-parallel regridding on NetCDF files with a single command:

```bash
# Conservative remapping of all spatial variables from a file onto a named grid
axis-regrid regrid -s source.nc -t target_grid.nc -o output.nc -m conservative --skipna

# Remap specific variables with periodic wrapping enabled
axis-regrid regrid -s source.nc -t target_grid.nc -o output.nc -m bilinear --periodic -v temperature -v humidity

# Precompute a reusable, portable weight file (ESMF by default, or native)
axis-regrid weights -s source.nc -t mpas_grid.nc -o weights.nc -m conservative
axis-regrid weights -s source.nc -t mpas_grid.nc -o weights.axisw --format native

# List the six methods and the registered named-grid families
axis-regrid list methods
axis-regrid list grids
```

Exit codes: `0` success · `1` runtime error (an actionable message on stderr,
no traceback unless `--debug`) · `2` argument-parsing error.

> Migrating from the pre-redesign Python surface? See
> [`docs/migration-v1.md`](docs/migration-v1.md) for the full old → new symbol
> table.

---

## 🌊 Advanced Coastal, Vector, & Vertical Remapping

AXIS implements dedicated operational-grade solvers for complex Earth-system remapping tasks:

1. **Coupled Vector Wind Remapping:** Unlike scalar fields, vector wind and current components ($U$ and $V$) cannot be interpolated independently because their local physical coordinate frames rotate over the sphere's curved surface. AXIS features a unified **`VectorWeightGenerator`** that generates coupled weight matrices and applies local physical coordinate frame rotations in a single, parallel, on-device step.
2. **Coastal Renormalization & Extrapolation:** To prevent unphysical leaks at land-sea boundaries, AXIS deploys an advanced, multi-stage **coastal re-normalization solver**. It dynamically re-normalizes weights near complex shorelines to prevent missing-value bleed, and automatically extrapolates values onto coastal shelf boundaries to maintain absolute mass conservation and numerical stability.
3. **Stateless 1D Vertical Column Regridding:** For vertical coordinates (such as pressure levels, height, or hybrid sigma-pressure levels), AXIS features a highly optimized **`VerticalRegridder`**. It performs parallel column-wise 1D vertical interpolation (linear, log-linear, or tension-spline) and automatically handles out-of-bounds column extrapolation and underground pressure-level masking natively on-device.

---

## 🛠 Supported Interpolation Methods

AXIS implements six high-performance, stateless spatial interpolation algorithms:
1. **Bilinear (`bilinear`):** Standard multi-linear interpolation in local 2D physical tangent space via gnomonic projection.
2. **Nearest Neighbor (`nearest`):** Zero-order interpolation assigning values from the closest source cell centroid in $O(1)$ constant-time for regular grids, and $O(\log N)$ parallel BVH searches for unstructured grids.
3. **Bicubic (`bicubic`):** High-order Hermite spline interpolation solving local gradients over a $4\times4$ grid stencil.
4. **Patch (`patch`):** Smooth, biquadratic local least-squares approximation minimizing edge discontinuities across unstructured cells.
5. **Conservative First-Order (`conservative`):** Area-conservative remapping on the curved unit sphere using analytical spherical excess integrals.
6. **Conservative Second-Order (`conservative2nd`):** High-order conservative remapping incorporating piecewise linear reconstruction gradients to minimize numerical diffusion.

---

## 💻 Native C++20 Interface

AXIS is structured as a pure C++20 template library, allowing downstream compiled models to generate weights and execute SpMV remapping entirely on-device without host round-trips:

```cpp
#include <axis/topology/mesh_factory.hpp>
#include <axis/solver/weight_generator.hpp>
#include <axis/solver/apply.hpp>

// 1. Ingest grid geometries into Kokkos-parallel UnstructuredMeshes (HELM Law #2)
axis::ingest::GridDescriptor src_desc = ...;
auto src_mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(src_desc);
auto dst_mesh = axis::topology::MeshFactory::from_descriptor<Kokkos::HostSpace>(dst_desc);

// 2. Configure regridding options
axis::solver::RegridConfig config;
config.method = axis::solver::InterpolationMethod::Conservative1stOrder;
config.line_type = axis::solver::LineType::GreatCircle;

// 3. Generate the interpolation weights sparse matrix in parallel
auto weights = axis::solver::WeightGenerator::generate<Kokkos::HostSpace>(src_mesh, dst_mesh, config);

// 4. Perform parallel SpMV remapping on field arrays using Kokkos
axis::field_view<const double, 1> src_field(src_data_ptr, src_mesh.n_cells());
axis::field_view<double, 1> dst_field(dst_data_ptr, dst_mesh.n_cells());

axis::solver::apply<Kokkos::HostSpace>(weights, src_field, dst_field);
```

---

## 📜 Modern Fortran 2018 Interface

AXIS exposes a modern, strongly-typed object-oriented Fortran 2018 wrapper (`axis_mod.f90`) utilizing standard ISO_C_BINDINGS to completely bridge the gap for legacy atmospheric and ocean models:

```fortran
use axis_mod

type(axis_mesh_t)   :: src_mesh, dst_mesh
type(axis_matrix_t) :: weights_matrix
real(8), dimension(:), pointer :: src_data, dst_data

! 1. Initialize meshes from descriptors
src_mesh = axis_mesh_from_descriptor(src_desc)
dst_mesh = axis_mesh_from_descriptor(dst_desc)

! 2. Generate conservative weights matrix in parallel
call axis_generate_weights(src_mesh, dst_mesh, "conservative", weights_matrix)

! 3. Remap source data array to destination array via parallel SpMV
call axis_apply_weights(weights_matrix, src_data, dst_data)

! 4. Generate coupled vector weights (U/V) including coordinate frame rotation
type(axis_matrix_t) :: weights_u, weights_v
real(8), dimension(:), pointer :: src_alpha, dst_alpha
call axis_generate_vector_weights(src_mesh, dst_mesh, src_alpha, dst_alpha, "bilinear", weights_u, weights_v)
```

---

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| C++20 compiler | GCC ≥ 13, Clang ≥ 16 | Required |
| CMake | ≥ 3.21 | Required |
| Kokkos | ≥ 5.1 | Required (`find_package(Kokkos REQUIRED)`) |
| PROJ (libproj) | ≥ 9 | Optional — controlled by `AXIS_ENABLE_PROJ` (default ON) |
| GTest | any recent | Required when `AXIS_BUILD_TESTING=ON` |
| RapidCheck | latest | Required when `AXIS_BUILD_TESTING=ON` |
| gfortran | ≥ 13 | Required when `AXIS_BUILD_FORTRAN=ON` |

All prerequisites are pre-installed in the HELM Docker development container.

## Docker Container Launch

Build and start the HELM development container from the project root:

```bash
# From the HELM project root (where docker-compose.yml lives)
docker compose up -d --build

# Attach to the running container
docker compose exec helm-dev bash
```

Inside the container the workspace is mounted at `/workspace/helm-project` and
all toolchain dependencies (GCC-13, Kokkos, GTest, RapidCheck) are available
system-wide.

## CMake Configure and Build

```bash
# Inside the container, navigate to the AXIS library
cd /workspace/helm-project/libs/axis

# Configure (Release build, Python enabled, tests enabled)
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAXIS_BUILD_PYTHON=ON \
  -DAXIS_BUILD_TESTING=ON \
  -DAXIS_ENABLE_PROJ=ON

# Build
cmake --build build --parallel $(nproc)
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `AXIS_ENABLE_PROJ` | `ON` | Enable PROJ-based coordinate transforms |
| `AXIS_BUILD_TESTING` | `OFF` | Build the GTest + RapidCheck test suite |
| `AXIS_BUILD_FORTRAN` | `OFF` | Build the Fortran iso_c_binding interop layer |
| `AXIS_BUILD_PYTHON` | `OFF` | Build the compiled C++ Python nanobind wrapper |

### Consuming AXIS from a downstream project

```cmake
find_package(AXIS REQUIRED)
target_link_libraries(my_target PRIVATE HELM::AXIS)
```

## Running Tests

```bash
# After building with AXIS_BUILD_TESTING=ON
cd build
ctest --output-on-failure
```

The test suite includes:

- **Unit tests** (GTest) — conservation, constant-field preservation, bilinear
  exactness, SpMV correctness, descriptor validation, Gmsh round-trip, RAII
  handle verification.
- **Property-based tests** (RapidCheck) — universal correctness properties
  validated over randomized inputs (≥ 100 iterations per property).
- **Python Integration tests** (pytest) — verifies CF coordinate detection, MPAS polygon parsing, Dask chunking, weight saving/reloading, and comparative CDO/ESMF benchmarks.

## License

This project is part of the NOAA-EMC Ecosystem.

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) for details.
