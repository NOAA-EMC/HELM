# Grid conversion architecture

AXIS keeps one mesh storage format: `UnstructuredMesh<MemorySpace>`, whose
cell-to-node connectivity is CSR. Producers describe geometry and topology;
`make_unstructured()` validates their finished views in the views' owning
Kokkos memory space and move-adopts them. `MeshFactory` has a different role:
it selects and validates an input adapter based on a `GridDescriptor`
convention (CF, UGRID, GRIB, projected, named-grid, or rules). A direct caller
with already-built CSR should use `make_unstructured()`; a file/ingest
descriptor should use `MeshFactory`.

## Current providers

* `StructuredGrid` supports logically rectangular grids and explicit corners.
  Center-only conversion interpolates interior vertices and uses tensor-product
  one-sided linear extrapolation at nonperiodic outer edges and corners. For
  spherical coordinates, neighboring longitudes are locally unwrapped before
  interpolation. A one-cell axis has no slope information and is held constant
  in that direction; callers needing nondegenerate geometry for such a grid
  must supply corners. The no-argument conversion assumes a nonperiodic
  longitude domain. Periodic grids must declare their seam through
  `LongitudePeriodicity`; `GridDescriptor` carries the same explicit declaration
  for CF, GRIB, and projected inputs. No spacing-only periodicity heuristic is
  used. `RequireExplicit` rejects a center-only grid even after an earlier
  reconstruction. `set_rectilinear_bounds()` accepts authoritative 1-D bounds
  for a separable grid; otherwise `RectilinearMidpoint` validates separability
  and strict monotonicity before deriving interior midpoints and one-sided
  outer edges. Spherical rectilinear areas use the constant-latitude strip
  formula, not great-circle polygon area. Gaussian centers alone are
  insufficient: `GaussianLatLon` requires positive authoritative row weights
  summing to 2 (or explicit corners). It cumulatively partitions
  `mu=sin(latitude)` from +1 to -1, closes at both poles, and uses the same
  constant-latitude boundaries for its areas. Curvilinear input must be
  explicitly declared by the caller; the current tensor-product local
  reconstruction is approximate and rejects nonfinite, degenerate, folded,
  nonconvex, or center-excluding cells. Its result provenance remains
  `Approximate`.
* `ReducedGaussianGrid` accepts latitude boundaries and per-row longitude
  boundaries. Cell order is row-major within the declared ragged rows. In
  conforming mode, adjacent-row horizontal edges are split at the union of
  declared longitude boundaries; this can produce mixed-arity polygons. Areas
  use the exact spherical strip formula for constant-latitude edges. This is
  exact for the declared parallel/meridian boundaries, not for an all-great-
  circle interpretation. Nonconforming mode preserves source quads and may
  have hanging nodes. Named `O<N>` and `N<N>` grids derive strip boundaries
  from Gauss-Legendre weights and pass explicit 0–360° periodic row bounds to
  this provider, so unlike rows are never paired into guessed quadrilaterals.
* `MultiFaceGrid` assembles structured faces with explicit full-edge contacts,
  edge orientation/reversal, and optional vertex-equivalence classes. Face
  coordinates must already be expressed in one common coordinate system;
  equivalence checks validate shared coordinates (allowing a longitude-period
  wrap), but do not apply projections or rotations. Node IDs are deterministic
  in face order and local row-major order. Unequal face dimensions are
  supported when connected edge vertex counts match.
* UGRID, rule-based, and named-grid producers now finalize through the same CSR
  builder while retaining their existing geometry/topology generation.

The reconstructed perimeter is linear extrapolation of center coordinates,
not source-authoritative cell geometry. It is appropriate only where the caller
has declared the structured model. Curvilinear center-only reconstruction
remains an opt-in approximation, and generic spherical-area recomputation
interprets its edges as great-circle arcs; constant-latitude Gaussian and
reduced-Gaussian edges use their provider-specific strip-area model instead.
Conservative overlay does not yet consume the `ConstantLatitude` boundary
metadata: Gaussian strip areas are exact for their declared parallel/meridian
boundaries, but the current generic overlay clips its polygon representation
using Cartesian or great-circle edges. Conservative remapping across these
providers is therefore not yet guaranteed to conserve under that exact area
model.

## Limitations

The multi-face core supports full-edge contacts, not partial edge ranges; its
connected edge dimensions must match and there is no ESMF GridSpec Mosaic
parser yet. LLC/cubed-sphere layouts can be represented when their producer
supplies face corners in one common coordinate system and explicit contacts/
equivalence groups, but no grid-specific layout adapter is provided. CF and
GRIB descriptors now accept authoritative 1-D bounds,
Gaussian row weights, an explicit periodic seam, and a curvilinear-center
declaration. `ConventionKind::ReducedGaussian` also accepts explicit ragged
row offsets, per-row longitude boundaries, latitude boundaries, and the seam
declaration. Named regular Gaussian generation delegates to the quadrature-
based rule provider. Distributed face ownership and MPI halo assembly are not
added.

## Memory behavior

The generic builder validates CSR with Kokkos kernels and does not mirror
device views to the host. It then moves the views into the mesh, avoiding an
extra allocation/copy. Grid providers may still create staging views when
their source metadata is host-resident; callers that already hold finished
views in the target memory space retain the zero-copy path. The present
reduced-Gaussian input is host metadata, so conversion stages completed arrays
to the requested memory space before CSR validation.
