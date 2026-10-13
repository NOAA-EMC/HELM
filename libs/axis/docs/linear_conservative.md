# Linear conservative reconstruction

This implementation replaces the previous synthetic-coordinate correction in
`Conservative2ndOrder`. `Conservative1stOrder` is unchanged. No new runtime or
library dependencies are introduced.

## Use

```cpp
axis::solver::RegridConfig config;
config.method = axis::solver::InterpolationMethod::Conservative2ndOrder;
config.line_type = axis::solver::LineType::GreatCircle; // Cartesian for XY meshes
config.norm_type = axis::solver::NormType::DstArea;
config.extrap_method = axis::solver::ExtrapolationAction::None;
config.use_limiter = false; // true selects the limited linear reconstruction

auto matrix = axis::solver::WeightGenerator::generate(src_mesh, dst_mesh, config);
matrix.to_csr(); // recommended for repeated fields; sorts moments with weights
axis::solver::apply(matrix, source_cell_averages, destination_cell_averages);
```

Use finite, genuine cell averages as inputs for active cells. The same matrix can be reused for new
fields. Each apply computes that field's gradients and, when requested, its
limiter. `batch_apply` handles each variable independently, including strided
LayoutRight views. The masked apply overload preserves inactive output values.
Python's existing `use_limiter` configuration selects the two modes; its
second-order configuration disables extrapolation automatically.

## Reconstruction and conservation

Each source cell uses two linear coordinates centered on their **area means**.
For planar cells these are Cartesian coordinates. For great-circle spherical
cells they are projections of unit-sphere XYZ onto a local orthonormal tangent
basis. The sphere's area-mean XYZ is retained without normalization when
subtracting means.

The source reconstruction is `q_i + g0_i*b0 + g1_i*b1`. The mean of each basis
function over its source cell is zero. Planar first moments use translated
polygon formulas. Spherical vector moments use the boundary integral
`integral r dOmega = 0.5 * sum(edge_angle * unit_edge_normal)`.

For each overlap, setup stores its area weight and two normalized moments.
Application adds `weight*q_i + moment0*g0_i + moment1*g1_i`. DstArea divides
by full destination area; FracArea divides by covered destination area.
A complete, nonoverlapping destination tiling conserves the source integral
up to geometric and floating-point error, with either limiter setting.

**Partial source coverage requires a reconstructed-field integral.**
`sum(q_i*area_a[i]*frac_a[i])` describes the covered integral of the
piecewise-constant source field. It generally does not equal the covered
integral of a linear reconstruction: its overlap-moment contributions must
also be included. The existing fraction arrays continue to describe area
coverage, not gradient corrections. No row renormalization is used to hide
conservation errors.

## Gradient fit and limiting

Setup builds deterministic wet vertex-neighbor stencils, precomputes
inverse-distance-weighted 2D least-squares coefficients, and expands to a
second wet ring only when the first ring is rank deficient. If the expanded
stencil still lacks rank, the cell retains first order. The existing standalone
`GradientReconstructor` API is unchanged.

Apply evaluates gradients from neighbor-minus-cell field averages. Limited
mode scales each source gradient by one factor in [0,1]. Setup initially includes
source vertices and every overlap's area-mean reconstruction coordinates, then
retains a conservative hull of those controls.
Thus each overlap average stays within that source stencil's value range.
For positive overlap weights this bounds covered-area destination averages;
partially covered DstArea outputs additionally include zero dilution.
This is a remapped-average bound, not a guarantee of pointwise bounds at every
location inside a curved cell. Boundary cells and smooth extrema may be
strongly limited; unlimited mode is preferable when smooth-field accuracy
matters more than bounds.

The limiter depends on the destination overlaps. Reusing a matrix with new
fields is supported; changing either mesh requires new geometry setup.

## Implementation scope

- Setup currently requires **HostSpace** meshes. Apply uses Kokkos kernels;
  this change has been compiled and tested on CPU/OpenMP only.
- Supports convex planar XY cells and convex great-circle spherical cells
  contained in an open hemisphere. Spherical cells may have at most 32
  vertices; overlap storage has checked headroom. Clockwise rings are oriented
  during setup. Active degenerate or concave cells are rejected.
- Constant-latitude boundary metadata is rejected rather than silently
  replacing curved boundaries with great-circle arcs. That path needs its own
  analytic overlap moments before it can use this method.
- Source and destination masks are honored. Dry cells never enter fits or
  limiter ranges. Unmapped rows are zero or raise according to `unmapped`.
- Extrapolation must be `None`; synthesizing out-of-domain values does not
  preserve conservative integrals.
- Multi-rank/distributed application, coupled-vector generation, NaN-aware
  batch application, and ESMF scalar-weight export reject reconstruction
  operators explicitly. They need additional stencil exchange, reconstruction,
  or serialization support. Ordinary first-order use remains available.
- `factor_list`, `csr_values`, and `kk_crs_matrix` expose the **base area
  operator**. They omit gradient terms. Use `solver::apply` for the complete
  reconstruction. `has_linear_reconstruction()` identifies these operators.
- Existing ArborX bounding boxes and great-circle intersection predicates are
  reused. This change does not establish search completeness for every possible
  large or pathological spherical cell.

## Validation and cost

`test_linear_conservative.cpp` checks planar linear exactness, constant
preservation, full-domain conservation in both normalization modes, sharp-front
bounds, smooth refinement, seam/pole cases, masks, partial coverage, CSR moment
permutation, batch fields, large projected coordinates, and rank fallback.
`prop_conservative_2nd_order.cpp` adds randomized exactness, conservation, and
bounds checks using actual cell-average semantics.

Build with `AXIS_BUILD_TESTING=ON` and run the corresponding unit/property
executables. The focused benchmark builds with `AXIS_BUILD_BENCHMARKS=ON`:

```sh
OMP_NUM_THREADS=1 OMP_PROC_BIND=false ./axis_linear_benchmark 64
```

It measures seven fresh setup-plus-CSR samples and 101 application samples
following five warmups. It outputs median milliseconds and a checksum. This is
an engineering diagnostic, separate from the paper's acceptance protocol.
Single-field apply allocates an uninitialized two-component gradient workspace
and overwrites every element. `apply_linear_with_workspace<MemorySpace>` accepts
a caller-owned Kokkos View with shape `[n_src,2]` for explicit reuse. Concurrent
calls must use distinct workspaces. Batch apply allocates one workspace for the
whole batch and shares coefficient and moment traversal across tiles of eight
fields. CSR apply fuses the area and
moment contributions in one row kernel. Unlimited mode does not allocate
limiter control-point storage. CSR conversion retains both COO and CSR moments,
so memory growth exceeds the two extra coefficient arrays alone.

## Performance implementation

Host overlap generation processes destination rows in parallel. Source coverage
and limiter controls are assembled in deterministic destination/candidate order;
first-order great-circle coverage retains its original Morton ordering. Exceptions
are captured per row and rethrown on the caller thread. These HostSpace paths use
existing Kokkos parallelism, with no new dependency.

Planar clipping reuses two buffers within each destination row. Limited setup
prunes control points using a conservative hull predicate; uncertain nearly
collinear points are retained. Source vertices and overlap means both contribute
to this hull. Limiting still uses each field's own neighbor bounds.

Unlimited and limited kernels are specialized separately. The linear batch path
supports COO and CSR, zero-width batches, arbitrary batch widths, and strided
Kokkos subviews/layouts. The tiled implementation reduces launches and fences;
it does not cache field-dependent gradients inside a shared matrix.

CSR construction preserves sorted columns and aligned overlap moments. Grouped
COO rows are sorted within each row; arbitrary COO retains a full permutation
sort. The expanded scalar unlimited operator and unconditional limiter early
exits were not adopted because profiling did not show a consistent benefit.

The batch gradient workspace stores `2*n_src*n_vars` doubles. Parallel setup
also uses temporary row-local result storage proportional to destination count
and candidate pairs; this trades additional temporary memory for concurrency
and deterministic assembly. These allocations do not persist in the matrix.

### Setup allocation follow-up

Parallel overlap generation groups up to 128 consecutive destinations into each
work block. Linear setup reuses clipping buffers across the block and stores
limiter deltas only when the limiter is enabled. Final coverage sums and limiter
control insertion retain the original destination/candidate order.

Single-thread host execution bypasses parallel exception staging. First-order
spherical overlap setup accumulates directly into the final COO/coverage arrays
for a serial caller. Bilinear setup also writes directly to COO arrays for serial
or small workloads; the projection cache is reserved for its parallel path.
The overlap and shape-function algorithms, candidate order, and normalization
remain the same. Parallel work still forwards failures to the caller thread.
