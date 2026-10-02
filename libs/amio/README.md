# AMIO — Asynchronous Multidimensional I/O

[![CI](https://github.com/bbakernoaa/amio/actions/workflows/ci.yml/badge.svg)](https://github.com/bbakernoaa/amio/actions/workflows/ci.yml)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
[![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen)](https://github.com/bbakernoaa/amio/actions/workflows/ci.yml)
[![C Standard](https://img.shields.io/badge/API-C99-blue)](include/amio/amio.h)
[![Fortran](https://img.shields.io/badge/Fortran-2003%20iso__c__binding-blue)](fortran/amio_mod.f90)
[![NOAA Disclaimer](https://img.shields.io/badge/NOAA-Disclaimer-yellow)](DISCLAIMER)

AMIO is a high-performance I/O engine for NOAA NWS Earth system models. It
provides a single C99/Fortran API that asynchronously reads and writes
multidimensional scientific data through pluggable backend drivers (NetCDF-4,
Zarr v3, GRIB2), hiding storage complexity from host model code.

## Key Capabilities

- **Asynchronous write** — `amio_write` deep-copies host data into a staging
  pool and returns immediately; serialization to disk happens on background
  worker threads.
- **Prefetch read** — `amio_read` returns data from a look-ahead staging
  buffer; upcoming timesteps are fetched in the background so reads are
  typically zero-latency.
- **Three backends** — NetCDF-4 (parallel HDF5 / MPI-IO), Zarr v3
  (TensorStore or NCZarr fallback), and GRIB2 (nceplibs-g2c) with
  composition-aware PDTs for aerosol, chemistry, and optical properties.
- **Mixed-format translation** — Read from one format, write to another in
  the same run with no external ETL.
- **Selective reads** — Bounding-box + stride descriptors fetch only the
  sub-region you need.
- **C99 + Fortran 2003** — Flat `extern "C"` API with `iso_c_binding`
  wrappers. No C++ leaks into the public surface.
- **MPI-aware** — Communicator splitting, parallel I/O, and dedicated I/O
  rank support.

## Quick Example (C)

```c
#include <amio/amio.h>

int main() {
    amio_core_handle core = NULL;
    amio_init("manifest.yaml", &core);

    // Write
    amio_dataset_handle wds = NULL;
    amio_open_dataset(core, "manifest.yaml", AMIO_MODE_WRITE, &wds);

    float temperature[100][200];
    /* ... fill temperature ... */

    amio_shape_t shape = { .rank = 2, .extents = {100, 200} };
    amio_io_handle io = NULL;
    amio_write(wds, "temperature", temperature, AMIO_DTYPE_F32, &shape, &io);
    // temperature[] is safe to overwrite NOW

    amio_flush(wds, 0);
    amio_close_dataset(wds);

    // Read back
    amio_dataset_handle rds = NULL;
    amio_open_dataset(core, "manifest.yaml", AMIO_MODE_READ, &rds);

    amio_view_handle view = NULL;
    amio_read(rds, "temperature", 0, NULL, &view);

    const void *data = NULL;
    size_t nbytes = 0;
    amio_view_data(view, &data, &nbytes);  // zero-copy pointer into staging
    // Use data[0..nbytes-1] ...

    amio_release_view(view);
    amio_close_dataset(rds);
    amio_finalize(core);
}
```

## Building

### Prerequisites

| Dependency | Required | Notes |
|-----------|----------|-------|
| CMake ≥ 3.20 | Yes | Build system |
| C++20 compiler | Yes | GCC 11+, Clang 14+ |
| Fortran 2003 compiler | Yes | gfortran 11+ |
| eckit ≥ 1.26 | Yes | Config, threading, factory |
| kokkos/mdspan | Yes | Header-only, std::mdspan reference impl |
| netCDF-c (parallel) | Yes | NetCDF-4 backend |
| nceplibs-g2c | Optional | GRIB2 backend |
| TensorStore | Optional | Zarr v3 (cloud + sharding) |
| MPI | Yes | Parallel HDF5, communicator splits |
| Catch2 v3 + RapidCheck | Tests only | Unit + property-based tests |

### CMake Configure + Build

```bash
cmake -S . -B build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMIO_BUILD_TESTING=ON \
  -DAMIO_BUILD_EXAMPLES=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Key CMake options:

| Option | Default | Description |
|--------|---------|-------------|
| `AMIO_BUILD_TESTING` | OFF | Build unit + PBT test suite |
| `AMIO_BUILD_EXAMPLES` | OFF | Build C/Fortran examples |
| `AMIO_BUILD_DOCS` | OFF | Build MkDocs documentation site |
| `AMIO_FORCE_NCZARR` | OFF | Skip TensorStore; use NCZarr fallback |
| `AMIO_HAS_TENSORSTORE` | Auto | Auto-detected; set OFF to force NCZarr |

### Docker (Development Container)

A full dev environment with all dependencies pre-built:

```bash
docker build --target deps -t amio:deps .
docker run --rm -v "$PWD":/workspace -w /workspace amio:deps bash
# Inside container:
cmake -S . -B build -GNinja -DAMIO_BUILD_TESTING=ON
cmake --build build -j$(nproc)
ctest --test-dir build
```

## Configuration

AMIO is configured through YAML manifest files:

```yaml
staging_pool:
  buffer_count: 16
  buffer_capacity_bytes: 52428800   # 50 MiB
worker_pool:
  threads: 4
prefetch:
  depth: 8
  read_timeout_s: 120
staging_timeout_ms: 30000
backend: netcdf4
path: /scratch/output.nc
data_model: classic
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
```

See [Configuration Guide](docs/site/user-guide/configuration.md) for the full
schema and per-backend options.

## Public API Surface

| Function | Purpose |
|----------|---------|
| `amio_init` | Create runtime (staging pool + workers) from manifest |
| `amio_finalize` | Drain workers and release all resources |
| `amio_open_dataset` | Open a file for reading or writing |
| `amio_close_dataset` | Flush pending writes and close |
| `amio_write` | Snapshot host data → async serialize to backend |
| `amio_read` | Get a prefetched view for a variable/timestep |
| `amio_view_data` | Retrieve data pointer + size from a read view |
| `amio_release_view` | Return view's staging buffer to the pool |
| `amio_flush` | Block until all pending writes complete |
| `amio_wait` | Block until a specific I/O operation completes |
| `amio_strerror` | Map error code to human-readable string |

All functions return `amio_status_t`. Check against `AMIO_OK`; use
`amio_strerror(rc)` for diagnostics.

## Supported Backends

| Backend | Library | Formats | Cloud | Lossless Codecs |
|---------|---------|---------|-------|-----------------|
| NetCDF-4 | netCDF-c + Parallel HDF5 | `.nc` | No | deflate, blosc |
| Zarr v3 | TensorStore or NCZarr | `.zarr` | S3, GCS, HTTPS | blosc, zstandard |
| GRIB2 | nceplibs-g2c | `.grib2` | No | libaec (AEC), JPEG2000 |

## Model Integration

AMIO is designed as a drop-in I/O layer for UFS, FV3, JEDI, and similar
Earth system models. The host initializes MPI, calls `amio_init` once per
run, then writes output fields and reads forcing/boundary data through the
timestep loop. See the
[Model Integration Guide](docs/site/user-guide/model-integration.md) for
patterns, Fortran examples, MPI setup, bounding-box reads, and performance
tuning.

## Downstream CMake Integration

After installing AMIO (`cmake --install build --prefix /opt/amio`), downstream
projects use standard `find_package`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(MyModel LANGUAGES C Fortran)

# Point to AMIO's install prefix if not in a system path
list(APPEND CMAKE_PREFIX_PATH "/opt/amio")

find_package(AMIO REQUIRED)

# C/C++ target
add_executable(my_model main.c)
target_link_libraries(my_model PRIVATE AMIO::amio_core)

# Fortran target
add_executable(my_model_f main.f90)
target_link_libraries(my_model_f PRIVATE AMIO::amio_fortran)
```

Exported targets:

| Target | Type | Provides |
|--------|------|----------|
| `AMIO::amio_core` | Shared (libamio.so) | C99 API + all backends |
| `AMIO::amio_fortran` | Static | iso_c_binding module + libamio link |
| `AMIO::public_headers` | Interface | C99 headers only (no link) |

No third-party headers (eckit, netCDF, TensorStore, g2c) leak through the
public interface. Downstream consumers only see `<amio/amio.h>`.

## Architecture

```
Host (Fortran/C) → C99 API → C-Boundary → Staging Pool → Worker Pool → Backend Driver → Storage
                                         ↕ Prefetch Queue ↕
```

- Write: synchronous snapshot into staging, async serialize on workers
- Read: background prefetch into staging, zero-copy view returned to host
- No C++ symbols cross the ABI; all state behind opaque handles

See [Architecture](docs/site/architecture.md) for diagrams and details.

## Testing

The test suite includes unit tests, driver integration tests, and
property-based tests (RapidCheck, 100 iterations each):

```bash
ctest --test-dir build --output-on-failure
```

Key test categories:
- `unit.*` — C-boundary, handle table, staging pool, worker pool, prefetch
- `pbt.*` — Correctness properties (round-trip fidelity, subset-equals-slice,
  view conservation, no-write-past-capacity, etc.)
- `unit.read_netcdf4`, `unit.read_grib2` — Driver integration with byte equality
- `integration.grib2_composition_roundtrip` — Composition PDT encode-decode
- `unit.e2e_netcdf_roundtrip` — Full public-API write→read round trip

## Documentation

Full documentation is built with MkDocs Material:

```bash
cmake --build build --target docs
# Open docs/_build/index.html
```

Or browse the source at `docs/site/`.

## License

See [LICENSE](LICENSE) and [DISCLAIMER](DISCLAIMER).

This project is part of the NOAA-EMC ecosystem.
