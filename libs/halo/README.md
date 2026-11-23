# HALO: Hardware-Abstracted Link Operations

HALO is a Tier 1 C++20 micro-library within the [HELM](../../README.md) ecosystem. It provides RAII-wrapped MPI communication primitives with compile-time GPU-aware halo exchange via Kokkos, replacing the legacy ESMF VM class and halo exchange infrastructure.

## Features

- **RAII Resource Safety** — All MPI handles (`MPI_Comm`, `MPI_Request`, `MPI_Win`) are scope-bound C++ objects guaranteeing cleanup on all exit paths, including exceptions.
- **Compile-Time Memory Dispatch** — Template metaprogramming detects the Kokkos memory space at compile time and selects GPU-direct MPI or host-staged transfer paths.
- **Precomputed Exchange Plans** — Neighbor topology and buffer geometry are computed once and reused across timesteps.
- **Fortran C-Interop** — Full `iso_c_binding` interface for incremental adoption by legacy NUOPC/ESMF Fortran models.

## Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| CMake | 3.21+ | Build system generator |
| C++ Compiler | C++20 support | GCC 13+, Clang 16+, or NVHPC 23.1+ |
| MPI | MPI-3.0+ | OpenMPI, MPICH, or Cray MPICH |
| Kokkos | 4.3+ | Hardware portability layer |
| Google Test | 1.14+ | Optional — required when `BUILD_TESTING=ON` |
| RapidCheck | latest | Optional — required for property-based tests |
| Fortran Compiler | Fortran 2008 | Optional — gfortran 9+, ifort 2021+, or nvfortran 21.1+ |
| Spack | 0.21+ | Optional — for automated installation via `spack install halo` |

All prerequisites are pre-installed in the HELM project Docker container.

## Docker Container Launch

The recommended development environment is the HELM Docker container built from the project root:

```bash
# From the HELM project root directory:

# Build and start the container (detached)
docker compose up -d --build

# Enter the running container
docker compose exec helm-dev bash

# You are now at /workspace/helm-project inside the container
# HALO source is at /workspace/helm-project/libs/halo/
```

## CMake Configure and Build

All commands below assume you are inside the Docker container at `/workspace/helm-project/libs/halo/`.

### Basic Build

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20
cmake --build build --parallel $(nproc)
```

### Build with Tests Enabled

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_BUILD_TESTING=ON
cmake --build build --parallel $(nproc)
```

### Build with GPU-Aware MPI

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_GPU_AWARE_MPI=ON
cmake --build build --parallel $(nproc)
```

### Build with Fortran Interface

`HALO_BUILD_FORTRAN` defaults to `OFF`. C++ consumers and downstream Python
builds do not need a Fortran compiler. Set it to `ON` explicitly to build and
install the optional `HELM::HALO_Fortran` target and module files:

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_BUILD_FORTRAN=ON
cmake --build build --parallel $(nproc)
```

### Full Development Build (All Options)

```bash
cmake -B build -G Ninja \
  -DCMAKE_CXX_STANDARD=20 \
  -DHALO_BUILD_TESTING=ON \
  -DHALO_BUILD_FORTRAN=ON \
  -DHALO_GPU_AWARE_MPI=OFF
cmake --build build --parallel $(nproc)
```

## Running Tests

Tests require MPI and are executed via `ctest` with `mpirun`:

```bash
cd build
ctest --output-on-failure
```

To run tests with a specific number of MPI ranks:

```bash
mpirun -np 4 ./tests/halo_tests
```

Property-based tests (RapidCheck) run in single-rank mode with mocked MPI:

```bash
ctest --output-on-failure -R property
```

## CMake Targets

| Target | Description |
|---|---|
| `halo` | Main shared library |
| `HELM::HALO` | Namespace alias for downstream consumption |
| `HELM::HALO_Fortran` | Fortran interface library (when `BUILD_FORTRAN=ON`) |

## Consuming HALO in Downstream Projects

```cmake
find_package(HALO REQUIRED)
target_link_libraries(my_target PRIVATE HELM::HALO)
```

MPI and Kokkos include paths and link flags propagate transitively.

## Structured Halo Exchange API

The structured exchange API operates on multi-dimensional Kokkos views (rank 1–4) with automatic GPU-resident pack/unpack kernels. A `Structured_Halo_Plan` precomputes subview regions once; subsequent exchanges reuse the plan without recomputing slice boundaries.

### Creating a Plan and Exchanging

```cpp
#include <halo/structured_halo_plan.hpp>
#include <halo/exchange_structured.hpp>

// 2D periodic grid: 128×64 cells with halo width 2 in each dimension
const std::array<std::size_t, 2> extents = {132, 68}; // includes halos
const std::array<std::size_t, 2> halo_widths = {2, 2};

// Neighbor ranks for [west, east, south, north]
const std::array<int, 4> neighbors = {rank_west, rank_east, rank_south, rank_north};

halo::Structured_Halo_Plan<2> plan(extents, neighbors, halo_widths, comm);

// Allocate a Kokkos view matching the plan extents
Kokkos::View<double**, Kokkos::LayoutRight> field("field", 132, 68);

// Blocking exchange — packs, sends, receives, unpacks
halo::exchange_structured_blocking(plan, field);

// Async exchange — returns a handle for overlap with computation
auto handle = halo::exchange_structured_async(plan, field);
// ... compute interior while communication proceeds ...
handle.wait();  // completes exchange and unpacks halo regions
```

### Neighbor Collective Exchange

For symmetric topologies on hardware with topology-aware routing (e.g., HPE Cray Slingshot), the neighbor collective variant uses `MPI_Neighbor_alltoallv` for potentially better performance:

```cpp
// plan must be non-const (caches topology communicator on first call)
halo::exchange_neighbor_collective(plan, field);
```

Falls back to the standard Isend/Irecv path transparently when the MPI version is below 3.0 or the neighbor topology is asymmetric.

## Persistent Communication

For exchanges that repeat every timestep with the same buffers (common in NWP time-stepping loops), `Persistent_Halo_Handle` eliminates per-call MPI setup overhead by calling `MPI_Send_init`/`MPI_Recv_init` once:

```cpp
#include <halo/persistent_halo_handle.hpp>

// Bind the plan to a specific view (calls MPI_Send_init / MPI_Recv_init)
halo::Persistent_Halo_Handle persistent(plan, view);

// Timestep loop — reuses the same persistent requests
for (int step = 0; step < num_steps; ++step) {
    persistent.start();   // MPI_Startall
    persistent.wait();    // MPI_Waitall (+ device unpack if staged)
    // ... compute with updated halo data ...
}
// Destructor calls MPI_Request_free (RAII)
```

The handle is move-only. For device views without GPU-aware MPI, host staging buffers are allocated once at construction and reused across cycles.

## Diagnostics and Instrumentation

HALO provides an optional callback hook for measuring per-exchange timing, bytes transferred, and neighbor counts without modifying library source:

```cpp
#include <halo/diagnostics.hpp>

// Register a callback — invoked at exchange begin/end events
halo::Diagnostics::set_callback([](const halo::Exchange_Event& event) {
    if (event.phase == halo::Exchange_Event::Phase::end) {
        std::cout << "Rank " << event.local_rank
                  << " exchanged " << event.total_bytes << " bytes"
                  << " with " << event.neighbor_count << " neighbors"
                  << " in " << event.elapsed.count() << " ns"
                  << (event.is_async ? " (async)" : " (blocking)")
                  << "\n";
    }
});

// All subsequent exchange calls emit events to the callback
halo::exchange_structured_blocking(plan, field);

// Remove the callback — zero overhead when not registered
halo::Diagnostics::clear_callback();
```

When no callback is registered, `emit()` short-circuits on a boolean flag check with zero overhead (no timing calls, no allocation, no virtual dispatch). The diagnostics hook has no compile-time dependency on LOGS or any external library (Tier 1 isolation preserved).

## Spack Installation

HALO provides a Spack package for automated installation with dependency resolution:

```bash
# Basic installation
spack install halo

# With Fortran interface and tests
spack install halo +fortran +tests

# With GPU-aware MPI for CUDA systems
spack install halo +cuda +gpu_aware_mpi

# Verify installation
spack test run halo
```

### Spack Variants

| Variant | Default | Description |
|---|---|---|
| `+fortran` | ON | Build Fortran C-interop layer (halo_mod) |
| `+gpu_aware_mpi` | OFF | Enable GPU-aware MPI (pass device pointers directly) |
| `+tests` | OFF | Build test suite (GTest + RapidCheck) |
| `+openmp` | ON | Enable Kokkos OpenMP backend |
| `+cuda` | OFF | Enable Kokkos CUDA backend |
| `+hip` | OFF | Enable Kokkos HIP backend |
| `+serial` | ON | Enable Kokkos Serial backend |

Dependencies (MPI, Kokkos, optional GTest/RapidCheck) are resolved automatically. The package supports all Kokkos backends that HALO supports.

## Project Structure

```
libs/halo/
├── CMakeLists.txt              # Standalone build (produces HELM::HALO)
├── CHANGELOG.md                # Version history (semantic versioning)
├── cmake/                      # Exported CMake config templates
├── include/halo/              # Public C++ headers
│   ├── halo.hpp               # Umbrella header
│   ├── version.hpp            # Generated version macros (from version.hpp.in)
│   ├── communicator.hpp
│   ├── request_guard.hpp
│   ├── window_guard.hpp
│   ├── halo_plan.hpp
│   ├── halo_handle.hpp
│   ├── exchange.hpp
│   ├── environment.hpp
│   ├── structured_halo_plan.hpp    # Multi-dim structured exchange plan
│   ├── exchange_structured.hpp     # Structured blocking/async/collective
│   ├── persistent_halo_handle.hpp  # MPI persistent communication RAII
│   ├── diagnostics.hpp             # Instrumentation hook interface
│   ├── error_policy.hpp            # Error policy configuration
│   └── detail/                # Internal implementation headers
│       ├── pack_unpack.hpp    # GPU pack/unpack kernels
│       └── gpu_aware_probe.hpp # Runtime GPU-MPI detection
├── src/                       # Implementation files
│   ├── detail/
│   └── fortran/               # C-interop layer
├── spack/                     # Spack package definition
│   └── package.py
├── fortran/                   # Fortran halo_mod module
├── docs/                      # Extended documentation
│   ├── overlap_pattern.md     # Communication/computation overlap guide
│   └── thread_safety.md       # Thread safety model
├── tests/                     # C++ unit and property tests
└── tests_fortran/             # Fortran integration tests
```

## License

See [LICENSE](../../LICENSE) and [DISCLAIMER](../../DISCLAIMER) in the HELM project root.
