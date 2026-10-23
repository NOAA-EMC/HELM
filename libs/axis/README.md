# AXIS — Arbitrary eXgrid Interpolation Solver

**Tier 1 C++20 micro-library** within the [HELM](../../README.md) ecosystem providing
stateless spatial interpolation (regridding) for Earth-system fields.

AXIS replaces the legacy ESMF spatial-discretization stack (`ESMF_Mesh`,
`ESMF_Grid`, `ESMF_LocStream`, `ESMF_Regrid`, and the offline
`ESMF_RegridWeightGen` application) with a decentralized, zero-copy,
hardware-portable toolkit built on Kokkos, packaged for Python and `xarray`, with optional **Dask parallel computing**.

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

## Python installation with pip or uv

AXIS includes a compiled C++ extension, so installing from this repository builds native code. Python dependencies are installed by pip or uv; CMake, a C++20 compiler, and an MPI C++ implementation must already be available. The build fetches Kokkos, KokkosKernels, and ArborX if they are not installed, so the first build needs network access. PROJ and the NetCDF C library are optional and enabled when found.

Fortran interfaces are optional. Python source builds disable both
`AXIS_BUILD_FORTRAN` and `HALO_BUILD_FORTRAN`, so `pip` and `uv` do not need a
Fortran compiler. Native CMake builds also default both interfaces to OFF.
Enable the interfaces explicitly with `-DAXIS_BUILD_FORTRAN=ON` and/or
`-DHALO_BUILD_FORTRAN=ON` when building for Fortran consumers; those builds
require a Fortran compiler.

### Prerequisites

- Python 3.10 or newer
- CMake 3.21 or newer
- A C++20 compiler
- MPI with a working C++ compiler wrapper (`mpicxx` or equivalent)
- Git and network access on the first build when native libraries are not installed

On macOS with Homebrew, install the native tools first:

```bash
brew install cmake open-mpi llvm libomp

export CC="$(brew --prefix llvm)/bin/clang"
export CXX="$(brew --prefix llvm)/bin/clang++"
export CXXFLAGS="-fopenmp=libomp"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
export LDFLAGS="-L$(brew --prefix libomp)/lib -Wl,-rpath,$(brew --prefix libomp)/lib"
```

On Debian or Ubuntu:

```bash
sudo apt-get install build-essential cmake git libopenmpi-dev openmpi-bin
```

### Install with pip

Run from the repository root. Use a virtual environment so the compiled package and Python dependencies stay together:

```bash
cd libs/axis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

To add optional NetCDF-backed xarray file I/O and/or Dask support:

```bash
python -m pip install '.[netcdf,dask]'
```

For an editable development install, use `python -m pip install -e '.[netcdf,dask]'` from `libs/axis`.

### Install with uv

The AXIS Python project lives in `libs/axis`:

```bash
cd libs/axis
uv sync
uv run python -c 'import axis; print("AXIS import succeeded")'
```

Add optional file I/O, Dask, or the development tools declared in `pyproject.toml`:

```bash
uv sync --extra netcdf --extra distributed --group test
uv sync --all-groups
```

After making C++ or binding changes, explicitly rebuild the editable installation:

```bash
uv pip install --python .venv/bin/python --reinstall-package axis-regrid -e .
```

Keep using the same Python environment that uv created by invoking commands with `uv run`.

### Check the installation

```bash
python -c 'import axis; axis.make_regular_mesh(4, 4, 0.0, 0.0, 1.0, 1.0); print("AXIS native mesh created")'
axis-regrid --help
```

If CMake reports that MPI is missing, install an MPI implementation and ensure `mpicxx` is on `PATH`, or set `MPI_CXX_COMPILER` to its full path. If CMake cannot fetch a dependency, install Kokkos, KokkosKernels, and ArborX in a prefix visible through `CMAKE_PREFIX_PATH`, or build from a network-enabled environment.

`pip install` and `uv sync` compile AXIS locally; they do not download a prebuilt platform wheel. Native wheel distribution needs a deliberate MPI/Kokkos runtime strategy for each supported platform.

AXIS integrates with `xarray`; Dask is optional and can be installed with the `dask` extra. The package does not require `cf-xarray` for its current Python API.

### 1. Basic Python Usage with xarray Accessor
AXIS registers a custom **`.axis`** accessor namespace on all xarray Datasets and DataArrays:

```python
import xarray as xr
import axis

# Load source and destination datasets
ds_src = xr.open_dataset("gfs_source.nc")
ds_dst = xr.open_dataset("orca_target.nc")

# Remap a DataArray instantly using the accessor
regridded_temp = ds_src.temperature.axis.regrid_to(ds_dst, method="bilinear")
```

### 2. Standard Regridder Object
You can instantiate a persistent `Regridder` object to reuse the same generated weights across multiple fields:

```python
regridder = axis.Regridder(ds_src, ds_dst, method="conservative")

# Apply to multiple variables
u_regridded = regridder(ds_src.u_wind)
v_regridded = regridder(ds_src.v_wind)
```

### 3. Save, Reload, and Reuse Weights
Bypass the weights generation phase entirely by saving compiled C++ weights to a file and reloading them:

```python
# Save weights to a binary file
regridder.to_file("my_weights.bin")

# Reload and reuse instantly in subsequent runs
fast_regridder = axis.Regridder(ds_src, ds_dst, weights_file="my_weights.bin")
result = fast_regridder(ds_src.temperature)
```

### 4. Dask execution

Dask is optional. Install local Dask array support with `python -m pip install '.[dask]'`; install the distributed scheduler with `python -m pip install '.[distributed]'`. For editable development, use `python -m pip install -e '.[dask]'` or `python -m pip install -e '.[distributed]'`. With uv, use `uv sync --extra dask` or `uv sync --extra distributed`.

AXIS keeps horizontal source spatial dimensions together as one core chunk for each task. Vertical interpolation similarly requires the complete source-level dimension and complete level-coordinate core dimensions in one chunk. Independent dimensions such as time can remain split into smaller chunks. AXIS rejects split core dimensions before creating the execution graph; explicitly rechunk them when the block size fits worker memory:

```python
field = field.chunk({"lat": -1, "lon": -1})
```

Supported schedulers are Dask's synchronous and threaded local schedulers, and Dask Distributed. The local multiprocessing scheduler is unsupported because native weight matrices are process-local. Distributed workers must have a compatible AXIS installation and native extension. AXIS registers serialized weights with a Dask worker lifecycle plugin, which reconstructs the native matrix on current and newly started workers. It uses `Client.register_plugin` in the current 2026.7.1 lock target, with a fallback to `Client.register_worker_plugin`; OMD's [worker lifecycle docs](https://github.com/dask/dask/blob/9dc535daa30d7d36d63d4a003b54342fbc7032db/docs/source/customize-initialization.rst#L112-L125) and [plugin registration example](https://github.com/dask/dask/blob/9dc535daa30d7d36d63d4a003b54342fbc7032db/docs/source/changelog.rst#L2768-L2785) informed this path. Call `regridder.close()` when finished to unregister that worker plugin and release local cached weights.

```python
from dask.distributed import Client
import axis

regridder = axis.Regridder(source_grid, target_grid, method="bilinear")
with Client() as client:
    lazy_result = regridder(field)  # graph construction does not compute the field
    result = lazy_result.compute()
    regridder.close()
```

For distributed use, construct the graph while the desired `Client` is active, as above. If Distributed is not installed, use the `distributed` extra. The current local `uv.lock` resolves Dask 2026.7.1; OMD's Dask 2026.8.0 documentation informed the lifecycle design but is not a tested compatibility claim.

---

## 💻 `axis-regrid` Command-Line Tool

AXIS registers a native command-line executable **`axis-regrid`** in your system path, allowing shell scripters and operational pipelines to run Kokkos-parallel, high-performance regridding on NetCDF files with a single command:

```bash
# Run conservative remapping on all spatial variables in a NetCDF file
axis-regrid -s source.nc -t target_grid.nc -o output.nc -m conservative --skipna

# Remap specific variables with periodic wrapping enabled
axis-regrid -s source.nc -t target_grid.nc -o output.nc -m bilinear --periodic -v temperature -v humidity
```

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
