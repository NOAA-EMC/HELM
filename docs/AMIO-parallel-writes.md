# AMIO asynchronous writes to a shared dataset

`amio_write_region()` snapshots a contiguous rank-local buffer and queues its
transfer into a specified region of a shared array. The C and Fortran bindings
use the same zero-based offsets and C row-major dimension order.

| Backend | Shared output across MPI ranks | Implementation |
| --- | --- | --- |
| NetCDF-4 | Supported with parallel netCDF/HDF5 | Collective hyperslab writes using global dimensions and local offsets |
| Zarr v3 / TensorStore | Supported | One rank creates metadata; ranks open the same array and update their regions |
| Zarr / NCZarr fallback | Unsupported | Multi-rank output opens and region submissions are rejected before writing |
| GRIB2 | Supported for complete messages | Rank-parallel encoding and explicit-offset collective MPI-IO; region writes unsupported |

## Collective contract

All participating ranks must open datasets in the same order using matching
manifests. NetCDF ranks must submit the same variable names, types, and global
shapes in the same order; local extents and offsets may differ. Each local extent
must be positive. Callers partition the global array into disjoint regions;
AMIO validates local bounds but does not perform a global overlap check.

Initialize MPI with `MPI_THREAD_MULTIPLE` when using background NetCDF or shared-file GRIB2 I/O. AMIO
checks the actual provided level through HELM HALO before opening the dataset.
HALO also owns Zarr communicator copies and propagates shared-store open failures.
Backend failures are reported through HELM LOGS and through the I/O status.

Wait for all previously submitted collective work on every rank before opening,
flushing, or closing datasets. Application threads must order collective
submissions consistently across ranks. Do not concurrently close/finalize a
handle while another application thread submits work through it.

The worker pool serializes each driver instance's writes. Collective NetCDF and GRIB2
writes share a process-local ordering domain so different worker schedules cannot
reorder variables across ranks. netCDF-C and NCZarr also share a process-local
library mutex because netCDF-C is not thread safe. These guards are independent
on each MPI rank: ranks still transfer their regions in parallel.

TensorStore performs chunk/shard transfers internally. Use a writable KvStore
with TensorStore's supported atomic update semantics. Disjoint regions may share
a shard; TensorStore must preserve its read-modify-write consistency. Cloud
credentials and filesystem/object-store permissions must be configured separately.
A Zarr dataset currently represents one array, so use one consistent variable
name and a separate dataset/store for each array.

## C example

Each rank writes two rows of a shared `[2 * nranks, nx]` array. All ranks use the
same output path; the Zarr manifest's `array_shape` must match this global shape.

```c
amio_shape_t local = {0};
local.rank = 2;
local.extents[0] = 2;
local.extents[1] = nx;

amio_write_region_t destination = {0};
destination.rank = 2;
destination.global_extents[0] = 2 * nranks;
destination.global_extents[1] = nx;
destination.offsets[0] = 2 * rank;

amio_io_handle pending = NULL;
amio_status_t rc = amio_write_region(dataset, "temperature", values,
    AMIO_DTYPE_F32, &local, &destination, &pending);
/* Check rc before continuing. values can now be reused after successful admission. */
/* Advance model computation while storage work proceeds on AMIO workers. */
rc = amio_wait(pending, 30000);
/* Check completion on every rank before collective lifecycle operations. */
rc = amio_flush(dataset, 30000);
```

For Fortran, `amio_mod` exports `amio_write_region_t` and the interoperable
`amio_write_region` function. Reverse Fortran array dimension order when populating
both the local shape and global destination, following the existing AMIO shape
convention. Pass `c_loc(values)` for a contiguous, interoperable host array.

## Completion and backpressure

The snapshot completes before submission returns, so the application owns no
borrowed buffer lifetime after admission. Snapshot copies and staging/queue
backpressure can block; the API does not promise constant-time submission.
`worker_pool.threads: 0` executes real synchronous writes instead of discarding
them. A rejected queue submission returns an error without a pending handle or a
retained staging buffer.

`amio_wait()` reaches a terminal state on backend success or failure.
`amio_flush()` waits for dataset writes and calls the backend flush operation.
`amio_close_dataset()` waits for all accepted writes before closing storage;
worker-pool draining does not silently time out. A failed write is not a
transactional rollback of earlier successful chunk, shard, or variable writes.

## GRIB2 complete-message parallel output

Open the `grib2` backend collectively on a common output `path`, then submit
complete F32 fields using the existing `amio_write()` API. Each rank can supply
a different grid and GRIB2 product configuration. The DRT must be supported by
the installed g2c codec build (AEC or lossless JPEG2000). For example:

```yaml
backend: grib2
path: shared-products.grb2
worker_pool: {threads: 4}
data_representation_template: libaec
grib2:
  parameter_category: 0
  parameter_number: 0
  type_of_first_fixed_surface: 100
  scaled_value_first_surface: 50000
```

Every participating rank submits **one complete field per round**, in the same
collective dataset order. Rank-local field shapes and compressed byte lengths
may differ. Messages appear in submission-round order, then communicator-rank
order within a round. Use one AMIO core per writer communicator, or externally
serialize collective work across cores. Dataset paths must match on all ranks;
product settings may differ. The open truncates the common file once using
`MPI_File_set_size`, preserving the existing write-mode replacement behavior.

Workers encode locally through g2c, then use HELM HALO to exchange encoded
message lengths and assign nonoverlapping byte ranges. `MPI_File_write_at_all`
writes each message directly at its assigned offset. The encoded byte buffer
remains owned by the worker until MPI completes; model computation can continue
after the synchronous snapshot returns. Process-local codec calls are guarded
without holding that guard across MPI collectives. Encoding and storage writes
still run in parallel across MPI ranks.

An encoding failure on any rank fails the round everywhere **before any bytes
for that round are emitted**. Further appends to that dataset fail until it is
closed. MPI-IO errors are likewise propagated and disable later appends, but an
I/O failure can leave a partial final round; there is no rollback. Successful
previous rounds remain a readable prefix. Message byte counts and cumulative
file offsets are checked for overflow. Flush uses collective `MPI_File_sync`,
and close collectively releases the file after accepted work has drained.

`amio_write_region()` remains unsupported for GRIB2: ranks holding fragments of
one compressed field must assemble the field on an encoding rank first, or use
complete tile messages with suitable grid metadata and consumer support.

## Validation

The new MPI integration test uses four AMIO workers per rank, uneven local slab
sizes, repeated writes, immediate source-buffer reuse, and an independent reader.
It verifies collective ordering across two active NetCDF outputs and concurrent
updates to one shared Zarr shard for TensorStore builds. A zero-worker MPI case
initializes through the public API and verifies synchronous NetCDF writes.
GRIB2, NCZarr, and TensorStore cases also open read datasets on rank zero while
the other ranks wait, preserving independent input access. A separate MPI test checks rejection of
insufficient thread support. Unit regressions cover asynchronous failure,
queue rejection, overflow/bounds checks, snapshot ownership, and zero-worker I/O.

Run `ctest -R 'integration.mpi_|unit.async_write' --output-on-failure` in an AMIO
build with testing enabled. The Zarr MPI test is registered only for TensorStore
builds; the default NCZarr CI build cannot certify shared Zarr output.

Local validation completed with 2- and 3-rank NetCDF round trips and 2- and
4-rank TensorStore Zarr round trips. The 12 targeted unit/integration checks and
C99/C++ header/symbol checks passed. The Fortran module compiled, and Doxygen
reported no documentation warnings for the documented interfaces. The Spack
closure check was skipped because Spack is not installed in this environment.

GRIB2 validation covers real AEC and JPEG2000 encoding, unequal message lengths,
multiple active output files, repeated asynchronous rounds, replacement of stale
files, independent message framing/decoding, and snapshot reuse. MPI tests also
check one-rank encoding failures, rejection of subsequent appends, inconsistent
paths/settings, failed file opens, and insufficient thread support.

The GRIB2 implementation passed 1-, 2-, and 4-rank round trips with AEC and
JPEG2000, along with the targeted MPI/serial regression suite and Doxygen checks.
