# Conservative vertical remapping

AXIS now includes a reusable finite-volume operator for **layer means**. It integrates overlaps between explicit source and destination interfaces. Select piecewise constant (`first`) or bounded, mean-preserving piecewise linear (`limited_linear`) reconstruction. The existing tension-spline profile interpolator remains a separate API for point values.

## Conserved quantity and interface units

- Interfaces represent the same physical cumulative measure on both grids. Their differences must equal the carrier amount associated with each layer.
- Thickness coordinates conserve concentration times thickness, proportional to tracer volume inventory for a column with fixed horizontal area. Use volume per horizontal area for volume concentrations.
- For mass mixing ratios, use **carrier mass per horizontal area** interfaces, including the appropriate density or pressure-to-mass conversion. Arbitrary geometric depth does not conserve mass mixing ratios when density varies.
- Source and destination interfaces may increase or decrease independently, be shared across columns, or vary by column. They must be finite and monotone. Repeated interfaces represent collapsed layers.
- Masks exclude layers; masked or collapsed layers may contain NaN. Wet positive-measure layer means must be finite. Geometry and masks are copied when the operator is constructed; construct a new operator if they change.

## Normalization and diagnostics

- `dstarea` divides integrated content by the full destination layer measure. Uncovered measure contributes zero, so a constant is reduced in a partly covered layer.
- `fracarea` divides by participating destination measure and preserves a constant wherever coverage is nonzero. Compute inventory using `destination_covered_measure`, not the full layer measure.
- Dry, masked, and unmapped outputs are zero. Coverage fractions explicitly identify missing participation.
- Both methods conserve the **covered reconstructed source integral** to floating-point roundoff. Full-column conservation requires matching wet source and destination domains.
- With limited linear reconstruction, a partially covered source layer contributes its mean term **plus its slope moment**. Use `covered_source_integral()`; multiplying the source mean by its covered width alone gives the wrong reference in this case.
- The limiter flattens end layers and layers adjacent to collapsed/masked layers. It bounds reconstructed endpoints by neighboring layer means and preserves nonnegative inputs over fully covered wet layers. Boundary flattening and limiting reduce local accuracy; no global second-order claim is implied.

## Python use

```python
import numpy as np
from axis import ConservativeVerticalRegridder

source_interfaces = np.array([0., 1., 2., 3.])
destination_interfaces = np.array([0., .5, 1.5, 2.5, 3.])
remap = ConservativeVerticalRegridder(
    source_interfaces, destination_interfaces,
    method="limited_linear", norm="dstarea")
q = np.array([[1., 2., 3.], [4., 4., 4.]])  # (column, layer)
q_destination = remap(q)
source_content = remap.covered_source_integral(q)
destination_content = (q_destination * np.diff(destination_interfaces)).sum(axis=-1)
np.testing.assert_allclose(destination_content, source_content, rtol=1e-14)
```

- Layer/interface is the last axis. Leading dimensions of interfaces and masks must broadcast to the field's leading dimensions in the same column order.
- One-dimensional interfaces share geometric overlaps. Multidimensional interfaces support column-varying or mixed shared/varying geometry.
- The wrapper caches the native operator and a slope workspace by field leading shape. Repeated fields reuse geometry and recompute field-dependent slopes.
- xarray DataArray output retains name, attributes, and nonvertical coordinates. Shared destination interfaces supply midpoint coordinates; varying interfaces supply a multidimensional `vertical_measure_midpoint` coordinate. This implementation is eager and does not provide distributed Dask execution.
- Python uses the existing CPU nanobind extension. No new production dependency is added.

## C++ use

```cpp
#include <axis/solver/vertical_conservative.hpp>
using Op = axis::solver::VerticalConservativeRegridder<Kokkos::HostSpace>;
using Order = axis::solver::VerticalConservativeOrder;
using Norm = axis::solver::VerticalConservativeNorm;
// Kokkos is initialized. Interfaces: (1, N+1) or (columns, N+1).
Op remap(source_interfaces, destination_interfaces, columns,
         source_mask, destination_mask);
Op::Field slopes("slopes", columns, source_layers);
remap.apply(source_layer_means, destination_layer_means,
            Order::LimitedLinear, Norm::DestinationMeasure, slopes);
auto covered_content = remap.covered_source_integral(
    source_layer_means, Order::LimitedLinear, slopes);
```

- Fields and masks have shape `(columns, layers)`; optional masks default to wet. Out-of-place application is required. Workspaces cannot alias either field.
- Native defaults are first order and full destination measure; the high-level Python default is limited linear and full destination measure.
- Setup and application use a monotone overlap sweep with at most `N_source + N_destination` entries per geometry row. Shared interfaces store one overlap row; masks and coverage remain column-specific. Limited reconstruction adds a linear source-layer slope pass. No timing or GPU speed claim has been measured for this addition.
- CUDA/HIP explicit instantiations follow existing AXIS configurations; this validation exercised only CPU OpenMP/Serial.

## Horizontal plus vertical tracer workflow

```python
from axis import Regridder, regrid_tracer_3d
horizontal = Regridder(source_grid, destination_grid,
                      method="conservative", norm_type="dstarea", skipna=False)
result = regrid_tracer_3d(
    tracer, carrier_per_area, horizontal, destination_carrier_interfaces,
    vertical_dim="lev", method="limited_linear")
```

- `carrier_per_area` is layer volume or carrier mass **per horizontal area**, aligned with the tracer DataArray. Zero carrier is dry.
- The helper remaps carrier and `tracer * carrier` separately with first-order conservative horizontal weights, then divides to recover the intermediate tracer means. It forms cumulative carrier interfaces starting at zero and applies conservative vertical remapping.
- Destination interfaces must be in that same cumulative carrier coordinate. A target depth grid with variable density needs an explicit physical conversion before this call; the helper does not infer it.
- Full vertical participation is checked by default. `require_full_coverage=False` explicitly permits partial participation and returns its coverage and stock diagnostics.
- Returned `carrier` is the participating destination measure, `tracer_content` is tracer times that measure, and `covered_source_content` includes partial-layer slope terms. Area-weight these quantities to compute domain inventory.
- Global conservation additionally requires correct horizontal areas and a fully covered wet horizontal domain. The helper checks the requested horizontal method/normalization, but cannot establish these domain properties from the callable alone. This is an eager sequential horizontal-then-vertical workflow, not a full three-dimensional intersection algorithm.

## Validation evidence (2026-10-10)

- Eleven native tests pass with one and four CPU threads: independently computed overlap values, both normalizations, masks/collapsed layers, geometry ownership, positivity/bounds, irregular and descending interfaces, mixed orientations, large coordinate origin, partial slope integrals, invalid inputs/alias rejection, workspace reuse, smooth refinement, and repeated remapping.
- After 1,000 round trips, absolute inventory drift is `4.66294e-15` (first order) and `1.4877e-14` (limited linear), for initial inventory `0.7`.
- Smooth quadratic layer-mean RMS at 8/16/32 source layers: first order `0.0241643 / 0.0117135 / 0.00577675`; limited linear `0.0194536 / 0.0072808 / 0.00266166`. Error reductions are approximately 19.5%, 37.8%, and 53.9% in this specific test, not general accuracy guarantees.
- All 16 native paper cases pass with both thread counts; reported inventory residual is zero for these small examples. Cases include volume concentration, carrier-mass mixing ratio, partial source layers, and masked NaN land layers.
- Five array-wrapper tests pass using an executable adapter calling the actual C++ operator. This checks Python shape handling, geometry caching, and diagnostics, **not the production nanobind extension**.
- Eleven existing vertical spline regression tests also pass. This feature does not change spline behavior.
- Production nanobind compilation and four pytest/xarray integration tests remain pending because those packages are unavailable in this environment. The integration tests include an analytic horizontal equal-area oracle; a real AXIS horizontal/vertical mesh run and NOAA/ESMC comparisons remain future validation.

## Background and source evidence

- MOM6 vertical reconstruction documentation describes reconstruction from layer means, including piecewise constant and piecewise linear choices: https://mom6.readthedocs.io/en/main/api/generated/pages/Vertical_Reconstruction.html . This implementation is independently written within AXIS; it does not add MOM6 or Compadre dependencies.
- MOM6 ALE documentation provides physical context for vertical remapping: https://mom6.readthedocs.io/en/main/api/generated/pages/ALE.html .
- Conservation interpretation and normalization deserve explicit reporting; see *Truly conserving with conservative remapping methods*, GMD 17 (2024): https://gmd.copernicus.org/articles/17/415/2024/ .
- Implementation: `include/axis/solver/vertical_conservative.hpp`, `src/solver/vertical_conservative.cpp`, `python/vertical_conservative_bindings.hpp`, and `python/axis/conservative_vertical.py`. Native executable evidence: `tests/test_vertical_conservative.cpp` and `paper/native/vertical_conservative_examples.cpp`.
