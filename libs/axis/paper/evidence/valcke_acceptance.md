# Supplemental native study acceptance

Status: **accepted within the recorded CPU, geometry and first-order method scope**.

## Executed experiments

| Bundle | Cases | Quality / timing records | Numerical checks | Maximum reproduction difference |
| --- | ---: | ---: | --- | ---: |
| [Valcke fields](../runs/valcke-converged/suite.json) | 48 | 96 / 96 | all pass | 1.3323e-15 |
| [Fine Atlantic Gulf Stream](../runs/gulfstream-converged/suite.json) | 1 | 2 / 2 | all pass | 0 |

All 196 records completed, with no unsupported or discarded cases. AXIS and
native ESMF through ESMC use shared spherical triangles, source averages and
matched settings. Reference integration, metrics, setup and repeated application
are native; Python orchestration and plotting use xarray. Timings were collected
serially with one thread/process, two warmups and ten repetitions, without
concurrent figure generation or reference validation.

The largest source-conservation residual is 4.3601e-11 percent. Maximum reported
destination-reference uncertainty is 1.1409e-7, below the 5e-7 threshold. Maximum
measured relative RMS misfit is 8.5792 percent across the 48-case suite and
3.3663 percent in the fine Atlantic case. These errors are measured first-order
results; numerical acceptance does not impose an accuracy cutoff on these fields.
No near-zero relative-error exclusions were needed in these experiments.

Exact full-precision statistics, suite hashes, environments and build metadata
are in [valcke_acceptance.json](valcke_acceptance.json).

## Reference validation and correction

Doubling the original quadrature orders revealed aliased sampling of the narrow
Gulf Stream jet, with a destination discrepancy of 0.001922. Initial bundles
`valcke-final` and `gulfstream-detail` are excluded from accepted claims. The
[failure record](validation/initial_order_failure.json) is retained.

The corrected integrator uses spherical-cap phase bounds to detect possible jet
intersections and forces spatial refinement before trusting rule agreement.
Four coarse global preflight fields agreed within the declared threshold.
Independent 12/24-point native rules then checked seven production Gulf Stream
references against the production 6/12-point rules: rectilinear, MPAS, all four
regional LCC direction/normalization combinations, and the fine Atlantic case.
All passed; the maximum absolute source/destination difference was
4.909193052071714e-8 versus a 5e-7 threshold.

See [production comparisons](validation/production_order_comparison.json),
[current reference build identities](validation/current_reference_builds.json),
and the [raw validation bundle](../runs/valcke-converged/validation/production-mesh-comparison.json).
Exact source variants, binaries, commands and comparison NetCDF files are saved.
Some combined local empirical estimates are smaller than the observed changes;
these estimates are not certified error enclosures. Extended-precision geometry
agreement alone does not validate float64 field integration.

## Publication products

The [figure gallery](figure_gallery.md) links the 27 supplemental comparison
figures and one further fine Atlantic spatial figure. Every figure has PDF,
SVG and 300 dpi PNG exports with hashed run inputs and generator identity.
The analytical-field overview, harmonic quality panels and fine Atlantic map
were visually reviewed for labels, color scales, clipping and numerical coverage.
All figure/table sidecars and required products passed publication validation.

[comparison_metrics.csv](../runs/valcke-converged/figures/comparison_metrics.csv)
retains the full-precision native percentage diagnostics. Standard dimensional
quality tables and performance plots accompany both bundles. The original
paper and smoke results remain separate and retain their existing acceptance.

## Environment and reproduction

The final run used `axis-paper-converged` with the same immutable
`cece/cece-dev:esmf` image, ID
`sha256:996a52f0312c1ca2f5eae99a953aa6b816e8e31d791bffae008e4a89b748beac`.
Native tools use GNU 13.3.0, ESMF 8.9.1, Kokkos 5.1.1, KokkosKernels macro
`50299`, ArborX 2.1.0 and NetCDF 4.9.2 on Linux aarch64. ArborX was restored
from its v2.1 source archive; its exact archive hash is retained because that
source distribution has no Git commit metadata. Python package versions and
native compiler/link commands are recorded, not inferred.

An interrupted corrected study and temporary tools were lost between turns.
The accepted experiment was rebuilt and rerun in workspace-backed storage.
A relative-path transport failure was retained under the build directory and
fixed by resolving runner input/output paths before Docker translation.
No scientific tolerance was loosened.

Use the commands in [the reproduction guide](valcke_figures.md) and
[README](../README.md). The new [frozen archive](../runs/valcke-archive/manifest.json)
contains the supplemental source/results plus the original paper, matched smoke,
regression and fine Atlantic evidence. Previously frozen archives are preserved.
