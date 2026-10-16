# HALO-backed distributed regridding

`axis::distributed::DistributedRegridder` is a CPU operator for uniquely owned
source and destination cells. **Production setup defaults to selective geometry
imports; it does not gather the global mesh on every rank.** HALO handles all
search routing, geometry transfers, field exchanges, and reverse reductions.
There is no new production dependency.

## Supported workflows

| Path | Distributed behavior |
|---|---|
| Nearest | Rank-pruned global nearest-centroid discovery and selective polygon import |
| Bilinear | Global nearest-eight centroid candidates for the native general algorithm |
| Bicubic / patch | Global nearest-sixteen centroid candidates for native stencil construction |
| First-order conservative | Distributed bounding-box intersection discovery; import candidate polygons |
| Unlimited linear conservative | Conservative candidates plus complete first/second vertex-neighbor rings |
| Limited linear conservative | Same geometry closure; reverse HALO minimum reduction establishes a common source-cell limiter |
| Vectors | Source-frame rotation, two common-frame scalar transfers, destination-frame rotation; componentwise linear limiter |
| Batched fields / levels | Column-major buffers; current application exchanges each field separately |
| Masks / extrapolation | Existing method-specific native rules, including distributed nearest-wet discovery |
| Vertical point / conservative | Own complete columns and run the existing vertical kernels locally on nonempty ranks |

Linear reconstruction retains its HostSpace, convex planar/great-circle cell,
and no-extrapolation restrictions. Constant-latitude first-order conservative
geometry uses periodic strip bounds, including dateline-crossing cells.

Both selective and replicated-reference distributed setup disable arithmetic
structured fast paths. Imported closures are not complete structured grids.
Compare against the **native general algorithms** with
`config.use_structured_fast_paths = false`; the default serial structured routes
can have different stencil and edge behavior. The usual coordinate conventions
and limitations of the native general algorithms still apply. In particular,
bilinear/bicubic/patch use their existing two-dimensional centroid conventions;
spherical nearest uses unit-sphere centroids. Equal-distance stencil ties retain
native ArborX behavior and should not be treated as a uniqueness guarantee.

## Ownership and usage

```cpp
#include <axis/distributed/regridder.hpp>

axis::distributed::DistributedOptions options;
options.max_import_bytes = 256ULL * 1024 * 1024;
options.max_message_bytes = 1ULL * 1024 * 1024;
options.query_batch_size = 128;

axis::distributed::DistributedRegridder op(
    owned_source_mesh, local_destination_mesh, config, communicator,
    global_source_node_ids, options);
op.apply(owned_source_values, local_destination_values);
```

Source cell IDs are rank-contiguous in owned-cell order. Source fields must use
that same order. Supply stable global node IDs for multi-rank linear
reconstruction: matching coordinates alone cannot establish shared topology.
Metadata must agree across source partitions; supplied areas must be present
on all nonempty source ranks or omitted on all of them. Empty ranks participate
in every collective operation.

Construction and application are collective. Source and destination buffers
must not overlap; vector outputs must also not overlap angle inputs. Invalid
extents and aliases fail collectively before writing output. Participating
source values and rotation angles must be finite. A single operator has mutable
workspaces and must not be applied concurrently from different threads.
Its numerical kernels can use OpenMP. Initialize MPI with at least
`MPI_THREAD_FUNNELED` for threaded kernels and call collective operations from
the MPI initialization thread under that mode. Distinct operators own isolated
communicators. Destroy operators before `MPI_Finalize`.

Communicators, plans, containers, and Kokkos workspaces have RAII ownership.
The caller's MPI communicator is borrowed and duplicated, not adopted.

`local_matrix()` indexes the compact dependency closure, not owned-source
indices. `requested_source_ids()` maps that closure to global cell IDs. Use
`op.apply()` to exchange values; the legacy `HaloPattern` apply overload does not
supply reconstruction halos.

`owned_source_fractions()` returns globally reduced conservative participating
fractions in owned-cell order. It is a coverage diagnostic. Partial higher-order
inventory also requires reconstruction moments; mean times covered area alone
is insufficient. Vector reconstruction is componentwise, not divergence
preserving. Vertical carrier/unit contracts are unchanged.

## Selective discovery and bounded transfers

1. Validate local meshes, ownership, metadata, and budget agreement collectively.
2. Build local ArborX trees for source centroids, wet centroids, and polygon
   bounds using the native centroid/bounding-box routines.
3. Gather **small rank summaries**, not geometry: centroid, wet-centroid, and
   polygon bounds. Memory for these summaries is proportional to rank count.
4. Process bounded destination query batches. Rank bounds prune disjoint
   intersection queries and nearest searches whose lower bound exceeds the
   current kth-best candidate distance. Candidate lists merge across owners;
   nearest stencils are not restricted to adjacent partitions.
5. Import selected polygons, node IDs, masks, and optional areas through HALO.
   Geometry messages have a configured cap. Responses and query work are
   streamed rather than allocating a global destination/source pair array.
6. For linear methods, hash source node IDs to distributed directory owners.
   Route node-to-cell incidence records through HALO, then request complete
   first rings and the second-ring fallback. No rank receives the full global
   node-incidence directory by design.
7. Build native weights against the imported closure; retain local destination
   rows and only the source values/stencil dependencies needed during apply.

`halo::bounded_peer_exchange` extends HALO with a count-checked variable-length
peer exchange. It checks receive size collectively before allocating payload
storage and uses existing packed HALO exchange plans. `Indexed_Gather_Plan`
reuses source dependency schedules during application; reverse sums reduce
coverage and reverse minima establish source limiters across destination ranks.

## Memory budgets and remaining scaling limits

Defaults are a **512 MiB estimated retained-import budget**, **8 MiB message
budget**, and 128 destination queries per batch. Geometry, directory, candidate,
and response growth are checked; excess causes a collective error. There is
**no automatic fallback to global geometry replication**.

These limits are not a total process RSS guarantee. Caller-owned meshes, local
search trees, matrices, numerical workspaces, allocation overhead, and temporary
buffers consume additional memory. Choose per-rank budgets for the node's rank
count and available memory. `geometry_statistics()` reports closure cells,
vertex incidences, estimated import bytes, maximum received geometry payload,
and directory incidences.

Memory now follows owned input plus actual destination dependencies and a
partitioned topology index, rather than an unconditional global mesh copy.
Poor spatial partitioning or a destination region requiring nearly every source
cell can still produce a large closure; the budget makes that case fail cleanly.
Further distribution of very large individual destination rows is not added.

Setup currently uses bounded peer rounds and collective validation. Broad rank
bounds can route searches to many ranks, and setup latency can grow with rank
count. Application still exchanges batch fields separately and performs
collective checks. This removes the replicated-geometry memory bottleneck; it
is **not a measured 1,000-core performance claim**. Fused field exchange and more
selective rank routing remain performance work. Accelerator communication and
distributed Python bindings are outside this implementation.

## Reference strategy

`options.geometry = GeometryStrategy::ReplicatedReference` explicitly selects
the original full-source-geometry correctness reference. It has an estimated
geometry budget check and should be used only on small validation problems.
Fields and weights remain distributed. Production callers should retain the
`Selective` default. The old `DistributedWeightGenerator` is a separate legacy
global-row adapter requiring complete source geometry; it is not selective
geometry discovery.

## Validation and scaling

The MPI integration target covers 42 planar/spherical, masked/unmasked,
normalization/limiter configurations, scalar/batch/vector application, repeated
fields, coverage, empty ranks, and collective errors. Additional tests cover
larger spatially partitioned closure sizes, selective/reference agreement,
insufficient budgets, fragmented 4 KiB geometry transfers, constant-latitude
dateline cells, remote wet extrapolation,
and whole-column vertical inventory and profiles.

```sh
ctest --test-dir build -R '^axis_distributed_' --output-on-failure
# Includes 1/2/4/8 rank runs and OpenMP-enabled 2x2 and 4x4 hybrid runs.
```

The executable reports actual Kokkos/OpenMP concurrency and checks requested
thread counts. The local process simulator exercises production HALO/AXIS
integration in isolated processes but is **not real MPI transport or a cluster
performance measurement**. Its earlier forced-single-thread setting was fixed;
previous environment-only hybrid results did not establish four-thread use.
Real MPI and cluster scaling validation remain required.

Build with `AXIS_BUILD_BENCHMARKS` and run:

```sh
mpiexec -n 8 ./axis_distributed_scaling 1024 1024 7 conservative 8 strong
mpiexec -n 8 ./axis_distributed_scaling 128 128 7 limited 8 weak
```

Arguments are global/per-rank x cells, y cells, repetitions, method, fields,
and strong/weak mode. Methods are `nearest`, `bilinear`, `bicubic`, `patch`,
`conservative`, `linear`, and `limited`. CSV reports the selective strategy,
node/rank/thread counts, sparse entries, setup and application timing, peak-rank
RSS on Linux, constant error, maximum rank closure size, estimated import bytes,
and received geometry message bytes. Application includes HALO communication
and reconstruction after two warmups; setup includes discovery and imports.
Input generation is outside setup timing. Weak mode uses fixed-size local
x-strips, so the fixture does not enlarge each rank's boundary merely by
increasing the global x extent. This Cartesian harness does not
replace the NOAA mesh suite or publication-quality cluster measurements.

Development checks on 2026-10-11 passed 83 local regression tests and the complete
integration suite at 1/2/4/8 simulated processes. The corrected hybrid run
reported and checked two threads at two processes and four threads at four
processes. On the
65,536-cell Cartesian limited reconstruction fixture with eight processes,
maximum rank closure was 8,832 cells (13.5% of the global mesh), estimated import
storage was 6,782,976 bytes, and maximum geometry payload was 10,240 bytes.
Constant error was `2.22e-16`. These are memory-locality/correctness observations,
not real MPI performance or proof of billion-cell operation.
