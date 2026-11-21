# Point-to-mesh containment

AXIS locates independent points in a surface mesh without regridding them into
cell values. Multiple points in one cell, including coincident points, retain
separate records. CECE is an example consumer: source properties, emissions,
vertical profiles and physics remain application responsibilities.

## API

Include `<axis/solver/point_locator.hpp>` for `axis::solver::PointLocator` or
`<axis/distributed/point_locator.hpp>` for
`axis::distributed::DistributedPointLocator`. Both accept a HostSpace
`UnstructuredMesh`, stable cell IDs in mesh owned-cell order, and optional
`PointLocationOptions`; the distributed constructor also takes `MPI_Comm`.
The locator retains its own geometry/index. Rebuild it when geometry or mesh
ownership changes. The input mesh may be released after construction.

```cpp
axis::distributed::DistributedPointLocator locator(
    owned_mesh, stable_cell_ids, comm);
auto mapping = locator.locate(axis::solver::PointSetView{
    point_ids, point_coordinates,
    axis::topology::CoordinateSystem::SphericalDeg});
```

Coordinates are non-owning `field_view<const double,2>` in LayoutLeft, shaped
`[local_points,2]` or `[local_points,3]`. Cartesian two-dimensional input implies
z=0. Spherical coordinates are longitude/latitude, degrees or radians; mesh and
point angular units may differ. The third spherical component is ignored but
must be finite if supplied. Longitudes are periodic; latitudes must be valid.

IDs are nonnegative signed 64-bit `axis::index_t`. Mesh cell IDs are globally
unique across owned cells; point IDs are globally unique across the current
query. These two ID spaces are independent. Do not supply mesh ghosts or repeat
a replicated point catalog on every rank. ID validation uses a partitioned
directory rather than gathering the global catalog.

Results are in local input order and contain `point_id`, `cell_id`,
`owner_rank`, `owner_local_index`, and `status` (`Interior`, `Boundary`, or
`Unmapped`). Unmapped host IDs and indices are -1. Local serial results use
owner rank 0. Mesh masks do not alter geometric containment. There is no
nearest-cell fallback. Overlapping cells and shared boundaries select the
smallest stable containing cell ID, independently of partition order.

The Python package provides an eager NumPy serial locator and a Dask Distributed
locator. UGRID meshes can declare their coordinate and boundary models when
they are built; the defaults preserve the existing spherical-degree,
unspecified-boundary behavior.

```python
import numpy as np
import axis

mesh = axis.make_ugrid_mesh(
    node_coords, conn_offsets, conn_indices,
    coordinate_system=axis.CoordinateSystem.Cartesian3D,
    boundary_model=axis.BoundaryModel.CartesianStraight,
)
locator = axis.PointLocator(mesh, cell_ids, tolerance=0.0)
mapping = locator.locate(point_ids, point_coordinates)
# mapping.cell_ids, mapping.status, mapping.owner_ranks, ... are NumPy arrays
```

For Python distributed mapping, install the optional `distributed` extra
(`axis-regrid[distributed]`). Give AXIS one complete in-memory mesh and let it
create one spatial shard per active Dask worker. Users do not choose a shard
count or construct shards. The complete mesh must fit in client memory during
setup; this path does not ingest or partition an out-of-core mesh. Each worker
keeps only its shard and local index.

```python
import dask.array as da
from distributed import Client

point_ids = da.from_array(source_ids, chunks=4096)
point_coordinates = da.from_array(source_coordinates, chunks=(4096, 2))
with Client() as client:
    locator = axis.DistributedPointLocator(mesh, cell_ids, client=client)
    mapping = locator.locate(point_ids, point_coordinates)
    cell_future, partition_future = client.compute([mapping.cell_ids, mapping.owner_partitions])
    mapped_cell_ids, owner_partitions = client.gather([cell_future, partition_future])
    locator.close()
```

The point IDs and coordinates must be Dask arrays with matching row chunks; the
coordinate dimension must fit in one chunk. IDs must be nonnegative, fit in
signed 64-bit values, and be globally unique within a query. The lazy graph
checks uniqueness in worker-count buckets without gathering the full point-ID
array to the client. Coordinates use the same supported Cartesian or spherical
formats as the C++ API. Result columns are lazy Dask arrays; compute them before
closing the locator. The result reports `owner_partitions` and a local cell
index within that partition. Stable IDs, rather than partition-local indices,
are the keys for comparing runs with different Dask layouts.

The Python distributed API does not import `mpi4py`. The C++
`axis::distributed::DistributedPointLocator` remains MPI based for compiled
consumers such as CECE.

A host rank/index belongs to the locator's current mesh ownership epoch. Only
use an owner-local index to access arrays on that owner rank, and translate it
through application cell ordering where necessary. Stable IDs, not these
indices, are the keys for comparing runs with different MPI layouts.

## Geometry contract

Cell corner connectivity is required. Center coordinates alone cannot establish
authoritative containment. Explicit `GeometryMetadata::boundary_model` selects:

* `CartesianStraight`: convex, coplanar polygons in Cartesian 3D.
* `GreatCircle`: convex spherical polygons contained in an open hemisphere,
  with minor great-circle edges. Polar and dateline-crossing cells are supported.
* `ConstantLatitude`: longitude/latitude rectangles, with longitude span less
  than 180 degrees; horizontal edges follow constant latitude.

Reject unsupported, nonconvex, degenerate, or unspecified boundary geometry.
Approximate/reconstructed corner provenance is retained by the caller's mesh;
containment is with respect to the supplied boundaries, not an inferred truth.
Tolerance defaults to `1e-10`: Cartesian length units or spherical radians.
It is an absolute geometric tolerance; values must be finite in `[0,0.01)`.
A candidate within tolerance of a boundary participates in stable-ID tie-breaking.
Containment and search bounds also include a small scale-relative floating-point
roundoff allowance, so exact boundaries remain candidates at zero user tolerance.

## C++ MPI and resource behavior

Construction and each query are collective, including empty ranks. Use the
same geometric coordinate family and options on all ranks; empty ranks may
omit boundary metadata. All MPI traffic is delegated to HALO on a duplicated
communicator. Call from the MPI initialization thread with the appropriate
MPI thread support for the selected Kokkos backend; do not query one locator
concurrently. Destroy locators before MPI finalization.

The implementation gathers only six bounds per rank. Queries and responses
stream in bounded peer rounds; points and mesh geometry are never globally
replicated. Defaults match PR #39: 128 queries per batch, 8 MiB per message,
512 MiB estimated retained directory budget. Messages are capped below MPI
integer count limits. Invalid inputs, inconsistent options, duplicate IDs and
budget excess fail collectively. Output size follows local point count;
there is no fixed source-count limit. Allocator exhaustion is not a recoverable
collective error guarantee.

ArborX indexes local cell boxes and processes bounded batches of queries. Each
query reduces candidates directly to its lowest matching stable cell ID.
Planar bounds include tolerance at acute
corners. Spherical bounds use conservative caps that include the accepted
tolerance region. Small cells can be pruned individually, and the combined
bounds let MPI and Dask skip disjoint partitions.

Bounds, local trees, retained geometry, outputs and allocator overhead are
outside the ID-directory budget; it is not a total RSS cap. ID directories
can be imbalanced by adversarial ID distributions and fail at the configured
budget. CPU results may be copied into application-owned device arrays.

## CECE example and validation

`example_cece_point_mapping` keeps source heights outside AXIS and maps eight
independent sources to one cell, with point ownership distinct from mesh
ownership. It does not change CECE's grid state, physics or latitude-band logic.
Build with `AXIS_BUILD_EXAMPLES=ON` and run at different MPI sizes:

```sh
mpiexec -n 4 ./build/examples/example_cece_point_mapping
ctest --test-dir build -R axis_point_locator --output-on-failure
```

Tests exercise real MPI at 1/2/4/8 ranks, independent partitions, coincident
points, stable boundary selection, dateline/polar geometry, empty inputs/ranks,
fragmented messages and collective failure paths. Field sampling, source
redistribution and emission deposition are separate operations, not part of
point containment.

### Verification record (2026-11-14)

On the PR #39 base `dbe8592`, GCC 14.2, Kokkos 5.1.1 and real OpenMPI 5.0.7:
all four locator MPI tests and the CECE example passed, including the numerical
review regressions (near poles, angular meridian tolerance, Cartesian scales
`1e-8` through `1e150`, and collinear initial polygon vertices). The larger
fixture uses 1,024 cells and 2,048 points with spatial and fragmented ownership.
Observed peak rank RSS was about 12 MiB; single-rank queries took about 0.012 s
for planar geometry and 0.095 s for spherical geometry. These small local,
oversubscribed runs are correctness/resource observations, not cluster scaling
measurements.

The configured full suite reported 272 passes, two skips and four unrelated
failures: three existing examples did not build against current APIs, and
`axis_distributed_8_ranks` failed its serial/distributed field comparison.
The latter was reproduced with both new locator object files removed from the
linked archive. Existing 1/2/4-rank regridding and 2x2/4x4 hybrid tests passed.
Property tests, Fortran, PROJ and NetCDF were disabled for this verification.

### Stack status

[PR #47](https://github.com/NOAA-EMC/HELM/pull/47) fixed the bicubic source
closure and has merged into `feature/paper-source`, the base branch of PR #39.
The point locator changes are stacked on that updated branch, so the bicubic
fix lands first while the C++ MPI locator remains available to CECE.


### Nearest cell centers

Install `axis-regrid[nearest]` and use `axis.NearestCellLocator(mesh, cell_ids)`
for repeated, batched nearest-center lookup on one host. Supply
`centers=` to use authoritative centers or Voronoi generating sites in the
mesh coordinate system. Each point keeps its input ID and receives one cell ID
and local cell index. Equal distances resolve to the smallest cell ID.
Nearest-center queries always map to a cell, including outside a regional
mesh. Use containment when outside points must remain unmapped.

### Reuse a registration

Construct `axis.PointRegistration(result.point_ids, result.cell_ids)` from
computed locator output. Save it with the original coordinates, a grid key
that fingerprints geometry and stable cell IDs, and a lookup key that
includes coordinate units, method and tolerance. Load checks those keys and
the ordered source IDs and coordinates before returning the mapping.
`local_indices(owned_cell_ids)` resolves it against the current partition.
Use `cell_ids == -1` to distinguish points that were originally unmapped
from points whose cells are owned elsewhere.

### Direct grid indexing

`axis.index_latlon_grid` accepts ascending longitude and latitude edges in
degrees. Enable `periodic=True` for a full longitude cycle.
`axis.index_cubed_sphere` accepts Cartesian directions, six face bases in
mesh storage order, and shared gnomonic u and v edges. Both return cell rows
that can index a stable cell ID array. Shared edges use the higher bin and
cube face ties use the first face. Use polygon containment when the
smallest-stable-cell-ID boundary rule or a non-gnomonic geometry is required.

### Load Dask partitions on workers

Install `axis-regrid[distributed]` and pass serializable loaders to
`DistributedPointLocator.from_shards`. Each loader opens storage available
to the workers and returns a compact CSR geometry dictionary and stable
cell IDs. The client receives bounds and validation results. A lookup can
rebuild a missing worker index using the same loader. The underlying mesh
and cell order must stay immutable for the life of the locator. Size
storage shards and worker assignments so the active indexes fit in memory.
