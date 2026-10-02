# Architecture

AMIO (Asynchronous Multidimensional I/O) is a C++20 library that decouples
scientific compute loops in NOAA NWS Earth system models from physical storage
architectures. Host applications interact with AMIO exclusively through a flat
C99 FFI surface and a Fortran 2003 `iso_c_binding` wrapper.

## Design Principles

1. **Synchronous Snapshot, Asynchronous Serialize** — A write call is
   *synchronous* with respect to capturing the host pointer (deep copy into
   staging pool), but *asynchronous* with respect to the backend write.
2. **Pointer-Swap Reads** — Read calls return a non-owning `Memory_View` that
   points into a staging pool buffer already populated by the prefetch queue.
3. **Strict ABI Isolation** — Public headers contain only C99 types. All C++
   dependencies are private to the `AMIO_Core` build.

## Layered Architecture

```mermaid
flowchart TB
    subgraph HOST["Host Application - Fortran 2003+ / C / C++"]
        FORTRAN["UFS / FV3 / JEDI Fortran kernels"]
        C_HOST["C / C++ host code"]
    end

    subgraph FFI["AMIO Public ABI - C99 plus iso_c_binding"]
        FORT_MOD["AMIO_Fortran_Module<br/>iso_c_binding wrappers"]
        C_API["AMIO_C_API<br/>extern C linkage, void* handles"]
    end

    subgraph CORE["AMIO_Core - libamio, C++20, private build"]
        BOUNDARY["C-Boundary<br/>handle table, error translation"]
        VIEW["Memory_View<br/>kokkos/mdspan"]
        STAGE["Staging_Pool"]
        PFQ["Prefetch_Queue"]
        WORKER["Worker_Pool<br/>eckit thread primitives"]
        FACTORY["Backend_Factory<br/>eckit::Factory"]
        CFG["Config_Loader<br/>eckit::YAMLConfiguration"]
        DIAG["Diag / Exception Bridge<br/>eckit::Exception to AMIO_ERR_*"]
    end

    subgraph DRIVERS["Backend Drivers - private, polymorphic"]
        NC4["NetCDF_Driver<br/>netCDF-cxx4 plus Parallel HDF5"]
        ZARR["Zarr_Driver<br/>TensorStore or NCZarr fallback"]
        G2["GRIB2_Driver<br/>nceplibs-g2c"]
    end

    subgraph STORAGE["Storage Layer"]
        LUSTRE["Lustre / GPFS via MPI-IO"]
        OBJ["S3 / GCS / HTTPS"]
        FILE["POSIX files"]
    end

    FORTRAN --> FORT_MOD
    FORT_MOD --> C_API
    C_HOST --> C_API
    C_API --> BOUNDARY
    BOUNDARY --> VIEW
    BOUNDARY --> CFG
    VIEW --> STAGE
    STAGE --> WORKER
    PFQ --> WORKER
    WORKER --> FACTORY
    FACTORY --> NC4
    FACTORY --> ZARR
    FACTORY --> G2
    NC4 --> LUSTRE
    NC4 --> FILE
    ZARR --> OBJ
    ZARR --> FILE
    G2 --> FILE
    DIAG -.surfaces.-> BOUNDARY
    WORKER -.exceptions.-> DIAG
```

## End-to-End Write Path

The write path is the core of AMIO's design. The key invariant is that the
host pointer is safe to reuse immediately after `amio_write` returns.

```mermaid
sequenceDiagram
    autonumber
    participant H as Host (Fortran/C)
    participant F as AMIO_Fortran_Module
    participant C as AMIO_C_API (C-Boundary)
    participant V as Memory_View (mdspan)
    participant S as Staging_Pool
    participant Q as Worker_Pool Queue
    participant W as Worker thread
    participant FX as Backend_Factory
    participant D as Backend_Driver
    participant ST as Storage layer

    H->>F: amio_write_f(handle, buf, shape, dtype)
    F->>C: amio_write via extern C
    C->>V: wrap raw pointer + shape into mdspan view
    C->>S: acquire buffer (capacity, timeout)
    alt no buffer in time
        S-->>C: backpressure
        C-->>F: AMIO_ERR_STAGING_BACKPRESSURE
        F-->>H: error, host pointer untouched
    else buffer acquired
        Note over V,S: Snapshot: host-to-host memcpy
        V->>S: deep copy (synchronous)
        C->>Q: enqueue WriteTask(buffer, dataset_id, var_id, seq)
        C-->>F: write_handle (Opaque_Handle), success
        F-->>H: returns within 10 ms, host may now mutate buf
    end
    Q->>W: dequeue (FIFO per dataset+variable)
    W->>FX: lookup Backend_Driver for dataset
    FX->>D: dispatch via eckit Factory
    W->>D: encode buffer + metadata
    D->>ST: serialize bytes (MPI-IO, HTTP, or file)
    ST-->>D: ack or error
    alt success
        D-->>W: ok
        W->>S: release buffer back to pool
    else failure
        D-->>W: throws eckit::Exception
        W->>C: record failure against write_handle
        W->>S: release buffer back to pool
    end
```

### Key Properties

- **Pointer release boundary** (between steps 6–7): when `amio_write` returns,
  no worker thread holds a pointer to the host buffer.
- **Order preservation**: FIFO discipline per `(dataset, variable)` with a
  sequence counter assigned at enqueue time.

## End-to-End Read Prefetch Path

```mermaid
sequenceDiagram
    autonumber
    participant H as Host
    participant C as AMIO_C_API
    participant PFQ as Prefetch_Queue
    participant W as Worker_Pool
    participant D as Backend_Driver
    participant ST as Storage
    participant S as Staging_Pool buffer

    Note over C,PFQ: open_read schedules N fetches ahead
    C->>PFQ: schedule t in [0, N-1]
    loop background prefetch
        PFQ->>W: dequeue fetch task
        W->>S: acquire buffer for timestep t
        W->>D: read t (optional bbox/strides)
        D->>ST: byte-range request
        ST-->>D: bytes
        D-->>W: decoded into Staging_Pool buffer
        W->>PFQ: mark t complete
    end

    H->>C: amio_read(handle, t)
    alt completed buffer for t
        C->>S: take ownership ref
        C-->>H: Memory_View (non-owning, points into S)
        C->>PFQ: schedule t+N (maintain look-ahead)
    else not completed
        C->>W: wait t with read_timeout
        alt completes in time
            W-->>C: ready
            C-->>H: Memory_View
        else timeout
            W-->>C: failure
            C-->>H: AMIO_ERR_TIMEOUT
        end
    end

    H->>C: amio_release_view(view_handle)
    C->>S: drop ref, return buffer to pool if last ref
```

## TensorStore Discovery and NCZarr Fallback

At CMake configure time, AMIO determines which Zarr backend to use:

```mermaid
flowchart TD
    A[CMake Configure] --> B{AMIO_FORCE_NCZARR=ON?}
    B -->|Yes| F[NCZarr fallback mode]
    B -->|No| C{find_package TensorStore}
    C -->|Found| D[TensorStore mode]
    C -->|Not found| E{netCDF-c has NCZarr?}
    E -->|Yes| F
    E -->|No| G[FATAL_ERROR: no Zarr backend]
```

## Memory Ownership Model

| Pointer / Handle | Allocated by | Owned by | Released when | Visible to host? |
|---|---|---|---|---|
| Host source pointer | Host application | Host | Host's discretion after `amio_write` returns | Yes |
| Memory_View over host pointer (write) | C-Boundary stack | C-Boundary | End of C-Boundary call | No |
| Staging_Pool buffer (write) | `amio_init` | Staging_Pool | Worker completes write task | No |
| Staging_Pool buffer (read) | `amio_init` | Staging_Pool | Last `Memory_View` ref released | No |
| Memory_View (read, returned to host) | C-Boundary | Host (via view_handle) | `amio_release_view` | Yes |
| Opaque_Handle (core, dataset, io, view) | Handle table | C-Boundary | Corresponding finalize/close/release | Yes |

## Build Target Topology

```mermaid
flowchart LR
    subgraph PUBLIC["Public (installed)"]
        PH["amio_public_headers<br/>(INTERFACE, C99-only)"]
        CORE["amio_core<br/>(SHARED, libamio.so)"]
        FORT["amio_fortran<br/>(STATIC)"]
    end

    subgraph PRIVATE["Private (not installed)"]
        DNC["driver_netcdf<br/>(STATIC)"]
        DZ["driver_zarr<br/>(STATIC)"]
        DG["driver_grib2<br/>(STATIC)"]
    end

    FORT --> PH
    FORT --> CORE
    CORE --> PH
    CORE -.PRIVATE.-> DNC
    CORE -.PRIVATE.-> DZ
    CORE -.PRIVATE.-> DG
```

Downstream consumers only see `AMIO::amio_core` (for C/C++) or
`AMIO::amio_fortran` (for Fortran). No third-party headers leak through
the public surface.
