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

The Python package exposes the same serial and distributed interface. UGRID
meshes can declare their coordinate and boundary models when they are built;
the defaults preserve the existing spherical-degree, unspecified-boundary
behavior.

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

For distributed mapping, install the `mpi` extra (`axis-regrid[mpi]`) and pass
an mpi4py intracommunicator. Every rank calls construction, each query, and
`close` in the same order. Input conversion errors are reported collectively.
The context manager releases AXIS's duplicated communicator before MPI shuts
down:

```python
from mpi4py import MPI

with axis.DistributedPointLocator(mesh, owned_cell_ids, MPI.COMM_WORLD) as locator:
    mapping = locator.locate(local_point_ids, local_point_coordinates)
```

Python point and cell IDs must be one-dimensional, nonnegative, unique signed
64-bit integers. Coordinates are NumPy-compatible `[n_points,2 or 3]` arrays;
the wrapper accepts either row or column-major input and passes a Fortran-order
array to the C++ API. Result columns own their memory and remain valid after
the locator is closed or deleted. Importing AXIS or using the serial locator
does not import mpi4py.

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

## MPI and resource behavior

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

ArborX indexes local cell boxes. Planar bounds include tolerance at acute
corners. Spherical boxes currently conservatively cover the full unit sphere,
so spherical queries visit every nonempty rank and consider all its cells.
This preserves curved-edge/polar correctness but limits spherical throughput;
tighter spherical bounds are a future optimization, not a scaling claim.
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

### Stacked verification on PR #47 (2026-11-15)

The locator branch is now based on `c7ddcac`, the head of
[PR #47](https://github.com/NOAA-EMC/HELM/pull/47), which fixes the bicubic
source closure. On this stack the buildable suite passed: **273 passed, two
skipped, zero failures**, including the formerly failing eight-rank regridder,
all locator MPI sizes, and both hybrid layouts. The three obsolete examples
listed above were excluded because their existing source does not build.
Locator tests carry the `mpi` label so PR #47's new distributed CI step runs them.
