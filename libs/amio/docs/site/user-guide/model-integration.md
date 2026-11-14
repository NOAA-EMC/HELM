# Model Integration Guide

This guide describes how to integrate AMIO into an Earth system model
(ESM) or coupler component for high-performance parallel I/O.

## Integration Pattern

AMIO is designed as a drop-in I/O backend for model output and input.
The typical integration pattern:

```mermaid
sequenceDiagram
    participant M as Model (Fortran/C)
    participant A as AMIO
    participant S as Storage

    Note over M,A: Initialization (once per run)
    M->>A: amio_init(manifest)
    M->>A: amio_open_dataset(write_config, WRITE)
    M->>A: amio_open_dataset(read_config, READ)

    Note over M,A: Timestep loop
    loop Every output timestep
        M->>A: amio_write(ds, "T", field, F32, shape)
        M->>A: amio_write(ds, "U", field, F32, shape)
        Note over M: Model continues computing immediately
    end

    loop Every input timestep
        M->>A: amio_read(ds, "forcing", t, bbox, &view)
        M->>A: amio_view_data(view, &ptr, &size)
        Note over M: Use ptr directly (zero-copy from staging)
        M->>A: amio_release_view(view)
    end

    Note over M,A: Finalization
    M->>A: amio_flush(ds, 0)
    M->>A: amio_close_dataset(ds)
    M->>A: amio_finalize(core)
```

## MPI Considerations

AMIO's NetCDF-4 backend uses parallel HDF5 with MPI-IO. The host
application must initialize MPI before calling `amio_init`:

```c
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
// ... later ...
amio_init("manifest.yaml", &core);
```

AMIO does not call `MPI_Init` or `MPI_Finalize` — that is the host's
responsibility. The `MPI_COMM_WORLD` communicator is used for parallel
I/O by default.

## Write Integration

### Snapshot Semantics

The key property of `amio_write` is **synchronous snapshot**: the host
data is deep-copied into a staging buffer before the call returns. This
means:

- The host can immediately reuse or deallocate its buffer
- No coordination is needed between the model's compute loop and I/O
- The actual disk write happens asynchronously on worker threads

### Minimal C Example

```c
#include <amio/amio.h>

void write_model_output(amio_dataset_handle ds,
                        const float *temperature,
                        int nx, int ny, int nz) {
    amio_shape_t shape = {0};
    shape.rank = 3;
    shape.extents[0] = nz;
    shape.extents[1] = ny;
    shape.extents[2] = nx;

    amio_io_handle io = NULL;
    amio_status_t rc = amio_write(ds, "temperature",
                                  temperature, AMIO_DTYPE_F32,
                                  &shape, &io);
    if (rc != AMIO_OK) {
        fprintf(stderr, "Write failed: %s\n", amio_strerror(rc));
    }
    // temperature buffer is safe to overwrite NOW
}
```

### Minimal Fortran Example

```fortran
subroutine write_model_output(ds, temperature, nx, ny, nz)
    use amio_mod
    use iso_c_binding
    type(c_ptr), intent(in) :: ds
    real(c_float), intent(in) :: temperature(nx, ny, nz)
    integer, intent(in) :: nx, ny, nz

    type(c_ptr) :: io_handle
    integer(c_int32_t) :: rc

    ! amio_write_array is a Fortran convenience wrapper
    call amio_write_array(ds, "temperature" // c_null_char, &
                          temperature, io_handle, rc)
    if (rc /= AMIO_OK) then
        print *, "Write failed: ", amio_strerror_f(rc)
    end if
end subroutine
```

## Read Integration

### Prefetch Look-Ahead

AMIO prefetches upcoming timesteps in the background. When your model
reads sequentially (t=0, 1, 2, ...), the data for t+N is already being
fetched while you process t. This hides I/O latency:

```
Time →
Model:   [compute t=0] [compute t=1] [compute t=2] ...
I/O:     [fetch t=0,1,2,3] [fetch t=4] [fetch t=5] ...
                           ↑ returned instantly (cached)
```

### Selective Reads (Bounding Box)

For regional models or domain decomposition, use a bounding box to read
only the sub-region you need:

```c
amio_bbox_t bbox = {0};
bbox.rank = 2;
bbox.offsets[0] = my_y_start;
bbox.offsets[1] = my_x_start;
bbox.extents[0] = my_ny;
bbox.extents[1] = my_nx;
bbox.strides[0] = 1;
bbox.strides[1] = 1;

amio_view_handle view = NULL;
rc = amio_read(ds, "sst", timestep, &bbox, &view);

const void *data = NULL;
size_t nbytes = 0;
rc = amio_view_data(view, &data, &nbytes);
// data contains only the requested sub-region (my_ny × my_nx)
// nbytes == my_ny * my_nx * sizeof(float)

amio_release_view(view);
```

Strided reads are also supported (`bbox.strides[d] > 1`) for
coarsening/subsampling without transferring the full resolution.

## Step Collection Migration

Step collections are intended for applications that need a catalogued,
complete logical model step with explicit schema and field presence. They
preserve the legacy writer path: migrate one dataset at a time, keep existing
legacy layouts for current consumers, and use a separate collection path for
the new layout. Do not infer completion from a NetCDF time dimension or a set
of Zarr arrays. A step is complete only after its immutable commit and
authoritative catalog head are published.

Step contributions use zero-based offsets and counts in C row-major order.
Each writer declares its local region and byte budget before copying field
data. A successful snapshot admission releases the source buffer; it does not
mean the payload has been written or made visible. Model time is metadata and
need not increase monotonically, while step IDs are strictly increasing and
never reused, including after failed attempts. Optional absent fields must be
declared as absent, and required fields must be present before the step can be
sealed.

The additive C evolution API supports synchronous POSIX NetCDF-4 and
TensorStore Zarr v3 lifecycles. NetCDF builds with parallel NetCDF expose
collective multi-rank submission through
`amio_begin_step_partitioned`: every rank supplies one global-coordinate box
per schema field, including zero-count boxes for empty contributors. At
end-step, nonempty boxes must cover each required field without gaps or
overlap; an optional field is either fully covered or absent on every rank.
All ranks must call partitioned begin, end, and cancel in the same order.
TensorStore Zarr uses the same collective region contract. AMIO derives a
regular shard grid from the submitted boxes, bounds each shard by the
configured payload budget (up to 64 MiB), and assigns each complete shard to
one producer rank. Region writes must remain aligned to that grid so two ranks
cannot update the same shard. All writers wait for their TensorStore writes
before the catalog head is published. The local file-driver path is qualified
at two and four ranks; cloud storage is not. Zarr publication is
visibility-only; stronger synchronization and durability waits currently
return `AMIO_ERR_UNSUPPORTED`.

The public direct path does not provide aggregation or GRIB2 collection
publication. Shared-shard Zarr output is supported for aligned direct
producer regions; arbitrary redistribution still belongs to the aggregation
work. C and Fortran wrappers and examples are available. The 100-step parallel NetCDF qualification covers uneven
writer partitions, empty contributors, different reader partitions,
cancellation, and native-reader comparisons. Keep production on the
established legacy path until the remaining M1 backend gates are complete.

Published step reads use a per-dataset 16 MiB bounded LRU cache. Cache identity
includes dataset UUID, step ID, schema digest, field name, and the exact
selection; schema and presence metadata are checked before cache allocation.
Reads larger than the cache budget still succeed without being cached. This
cache currently serves repeated public step-field reads; it is not yet wired
into the legacy timestep prefetch pipeline.

See [Step Completion and Storage Guarantees](./completion-guarantees.md) for
the POSIX publication levels and current TensorStore certification status.

Evolution dimensions use zero-based C order. Fortran descriptors use the same
logical order; reverse the Fortran logical dimension sequence when it differs
from the C-order storage layout. A successful `amio_put_field` snapshots the
source before return, so the model may reuse that buffer immediately. Check
every status and release operation, producer-step, schema, dataset, and context
handles through their distinct release calls.

For published persistent steps, each reader process calls
`amio_acquire_step_lease` before consumer work and
`amio_release_step_lease` when finished. Leases are process-local and do not
perform MPI communication. A lease keeps its dataset open;
`amio_close_dataset_ex` returns `AMIO_ERR_INVALID_POLICY` until all leases are
released. Step-field reads copy into caller-owned buffers, so they do not expose
borrowed views. The persistent API currently guarantees reader visibility only;
waiting for the durable milestone returns `AMIO_ERR_UNSUPPORTED`.

`amio_cancel_step` is available for an open step before publication. For
partitioned NetCDF output, all ranks call it collectively in the same order.
It fails accepted operations, consumes that step ID, keeps the step out of
complete-step discovery, and allows its producer handle to be released. The
legacy asynchronous API drains accepted writes through `amio_flush` or dataset
close; the persistent direct path completes backend writes during
`amio_end_step`.

## Mixed-Format Translation

AMIO supports reading from one format and writing to another in the
same run without external ETL:

```c
// Read from NetCDF-4
amio_dataset_handle src = NULL;
amio_open_dataset(core, "netcdf_source.yaml", AMIO_MODE_READ, &src);

// Write to Zarr v3
amio_dataset_handle dst = NULL;
amio_open_dataset(core, "zarr_target.yaml", AMIO_MODE_WRITE, &dst);

// Read a variable
amio_view_handle view = NULL;
amio_read(src, "temperature", 0, NULL, &view);

const void *data = NULL;
size_t nbytes = 0;
amio_view_data(view, &data, &nbytes);

// Write the same bytes to the target format
amio_shape_t shape = {.rank = 2, .extents = {ny, nx}};
amio_io_handle io = NULL;
amio_write(dst, "temperature", data, AMIO_DTYPE_F32, &shape, &io);

amio_release_view(view);
amio_flush(dst, 0);
```

Both datasets share the same staging pool and worker threads. No
intermediate copies or format conversions are needed.

## Error Handling Best Practices

```c
amio_status_t rc = amio_write(ds, var, data, dtype, &shape, &io);
switch (rc) {
    case AMIO_OK:
        break;  // success
    case AMIO_ERR_STAGING_BACKPRESSURE:
        // Staging pool full — model is producing faster than I/O
        // can drain. Options: increase buffer_count, or retry.
        break;
    case AMIO_ERR_BACKEND_FAILURE:
        // Storage error — disk full, network down, etc.
        break;
    default:
        fprintf(stderr, "Unexpected: %s\n", amio_strerror(rc));
}
```

## Configuration for Model Runs

A typical HPC model run uses:

```yaml
# manifest.yaml
staging_pool:
  buffer_count: 16           # 2× concurrent variables
  buffer_capacity_bytes: 52428800  # 50 MiB (fits one 3D field)
worker_pool:
  threads: 4                 # Dedicated I/O threads
prefetch:
  depth: 8                   # Look-ahead 8 timesteps
  read_timeout_s: 120        # 2 min timeout for slow storage
staging_timeout_ms: 30000    # 30s backpressure timeout
backend: netcdf4
path: /scratch/model_output.nc
data_model: classic
codec:
  active_codec: blosc
  lossless_allow_list:
    - blosc
```

## Thread Safety

- `amio_init` / `amio_finalize`: Call from a single thread (typically
  the main rank).
- `amio_write`: Thread-safe across different variables. Same variable
  writes are serialized by the worker pool's per-(dataset, variable)
  ordering.
- `amio_read`: Thread-safe for different variables. Same variable reads
  from multiple threads are serialized by the per-variable mutex.
- `amio_flush` / `amio_close`: Call from the same thread that opened
  the dataset.

## Performance Tuning

| Parameter | Effect | Guidance |
|-----------|--------|----------|
| `buffer_count` | Concurrency headroom | 2× max overlapping writes |
| `buffer_capacity_bytes` | Max single-field size | ≥ largest variable payload |
| `worker_pool.threads` | I/O parallelism | 2–4 for local disk, 4–8 for cloud |
| `prefetch.depth` | Read look-ahead | 4–16 depending on access pattern |
| `staging_timeout_ms` | Backpressure patience | 5000–30000 ms |
