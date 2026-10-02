# AMIO - Asynchronous Multidimensional I/O {#mainpage}

## Overview

**AMIO** (Asynchronous Multidimensional I/O) is a high-performance I/O engine
designed for NOAA NWS Earth system models. It provides a unified C99 API surface
with Fortran 2003 bindings for writing and reading multidimensional scientific
data through pluggable backend drivers.

### Key Features

- **Asynchronous I/O**: Non-blocking write operations with staging pool and
  worker threads for overlapping computation with I/O.
- **Multiple Backends**: Pluggable drivers for NetCDF-4 (parallel HDF5),
  Zarr v3 (TensorStore or NCZarr fallback), and GRIB2 (nceplibs-g2c).
- **Unified API**: Single C99 API surface consumed by both C/C++ and Fortran
  applications through iso_c_binding wrappers.
- **MPI-Aware**: Communicator splitting and parallel I/O support for
  distributed-memory HPC applications.
- **Type-Safe Handles**: Opaque handle system with generation-counter
  validation prevents use-after-free and handle misuse.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Host Application                      │
│              (C / C++ / Fortran)                        │
├─────────────────────────────────────────────────────────┤
│              AMIO Public API (C99 FFI)                  │
│         amio.h / amio_mod (Fortran)                     │
├─────────────────────────────────────────────────────────┤
│                  C-Boundary Layer                       │
│    Handle Table │ Exception Bridge │ Config Loader      │
├─────────────────────────────────────────────────────────┤
│   Staging Pool  │  Worker Pool  │  Prefetch Queue       │
├─────────────────────────────────────────────────────────┤
│              Backend Factory (Plugin Registry)          │
├───────────────┬──────────────┬──────────────────────────┤
│  NetCDF-4     │   Zarr v3    │      GRIB2               │
│  Driver       │   Driver     │      Driver              │
│ (HDF5/pnetcdf)│(TensorStore/ │  (nceplibs-g2c)          │
│               │  NCZarr)     │                          │
└───────────────┴──────────────┴──────────────────────────┘
```

## Quick Start

### C API

```c
#include <amio/amio.h>

amio_core_handle core = NULL;
amio_status_t rc = amio_init("manifest.yaml", &core);
if (rc != AMIO_OK) {
    fprintf(stderr, "Init failed: %s\n", amio_strerror(rc));
    return 1;
}

amio_dataset_handle ds = NULL;
rc = amio_open_dataset(core, "dataset.yaml", AMIO_MODE_WRITE, &ds);

// Write data...
amio_flush(ds, 0);
amio_close_dataset(ds);
amio_finalize(core);
```

### Fortran API

```fortran
use amio_mod
type(c_ptr) :: core, dataset, io_handle
integer(c_int32_t) :: rc

rc = amio_init("manifest.yaml" // c_null_char, core)
rc = amio_open_dataset(core, "dataset.yaml" // c_null_char, &
                       AMIO_MODE_WRITE, dataset)

! Use convenience subroutines for type-safe writes
call amio_write_array(dataset, "temperature" // c_null_char, &
                      temp_field, io_handle, rc)

rc = amio_flush(dataset, 0_c_int64_t)
rc = amio_close_dataset(dataset)
rc = amio_finalize(core)
```

## API Reference

- @ref amio.h "C API Reference" — All public C entry points
- @ref amio_errors.h "Error Codes" — AMIO_ERR_* enumeration and amio_strerror
- @ref amio_types.h "Type Definitions" — Data types, shapes, and handles
- @ref amio_export.h "Export Macro" — AMIO_API visibility control
- @ref amio_mod "Fortran Module" — iso_c_binding wrapper with convenience routines

## Examples

Complete, compilable examples are provided in the `examples/` directory:

- **C Examples** (`examples/c/`):
  - `example_netcdf4_write.c` — Write a 3D temperature field to NetCDF-4
  - `example_zarr_write.c` — Write SST data to Zarr v3 (NCZarr mode)
  - `example_grib2_write.c` — Write geopotential height to GRIB2

- **Fortran Examples** (`examples/fortran/`):
  - `example_netcdf4_write.f90` — Write a 3D temperature field to NetCDF-4
  - `example_zarr_write.f90` — Write SST data to Zarr v3
  - `example_grib2_write.f90` — Write geopotential height to GRIB2

- **Manifest Files** (`examples/manifests/`):
  - `netcdf4_manifest.yaml` — Configuration for NetCDF-4 backend
  - `zarr3_manifest.yaml` — Configuration for Zarr v3 backend
  - `grib2_manifest.yaml` — Configuration for GRIB2 backend

## Building

```bash
mkdir build && cd build
cmake .. -DAMIO_BUILD_DOCS=ON -DAMIO_BUILD_EXAMPLES=ON
make
make docs    # Generate Doxygen HTML documentation
```

## License

See the LICENSE file in the project root.
