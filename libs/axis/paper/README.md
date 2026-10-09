# AXIS paper evidence suite

This optional suite prepares identical meshes with xarray, computes independent
native references, runs native AXIS and ESMF through ESMC, and generates figures,
tables, and manuscript evidence. Python and file I/O are outside benchmark
intervals. ESMF is linked only into the ESMC worker, never into core AXIS.

## Layout

- `native/`: fixture I/O, analytic references, remapping, metrics, and timing.
- `scripts/`: xarray preparation, orchestration, publication, and archival.
- `config/`: small smoke and larger paper profiles.
- `evidence/`: code citations, outline coverage, requirements, and acceptance.
- `runs/`, `data/`, `build*/`: ignored outputs; archive them separately.

## Build and run locally

Use Python 3.11+ and the pinned packages. The native build requires the normal
AXIS dependencies (Kokkos, KokkosKernels, ArborX, MPI), NetCDF-C, and an installed
ESMF whose `esmf.mk` matches the compiler/MPI installation.

```sh
python3 -m venv libs/axis/paper/.venv
libs/axis/paper/.venv/bin/python -m pip install -r libs/axis/paper/requirements.txt
export ESMFMKFILE=/path/to/installed/esmf.mk
cmake -S libs/axis -B libs/axis/paper/build \
  -DAXIS_BUILD_PAPER=ON -DAXIS_PAPER_ENABLE_ESMF=ON \
  -DAXIS_BUILD_TESTING=OFF -DAXIS_BUILD_PYTHON=OFF -DAXIS_BUILD_FORTRAN=OFF \
  -DAXIS_ENABLE_PROJ=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build libs/axis/paper/build --target \
  axis_paper_axis axis_paper_esmc axis_paper_reference axis_paper_metrics axis_paper_esmc_probe -j4
libs/axis/paper/build/paper/axis_paper_esmc_probe
libs/axis/paper/.venv/bin/python libs/axis/paper/scripts/prepare_cases.py \
  --profile libs/axis/paper/config/smoke.json --output libs/axis/paper/data/smoke
libs/axis/paper/.venv/bin/python libs/axis/paper/scripts/run_suite.py \
  --profile libs/axis/paper/config/smoke.json --cases libs/axis/paper/data/smoke \
  --bin-dir libs/axis/paper/build/paper --output libs/axis/paper/runs/smoke
```

Choose new case/run directories for each execution. Explicit ESMF enablement
fails configuration if the installed headers and symbols cannot compile/link.
With `AXIS_PAPER_ENABLE_ESMF=OFF`, AXIS/reference/metrics still build; requesting
ESMC in a profile records it as unavailable. With `AXIS_BUILD_PAPER=OFF`, none of
the paper dependencies or targets are introduced.

The smoke profile deliberately includes an ESMC-unsupported constant-latitude
case. A full smoke invocation therefore reports incomplete and exits 5. Use
`--allow-incomplete` for this diagnostic study; it preserves every status. For a
strictly complete matched study, create a new profile omitting `strip`, prepare
new fixtures, and run without that flag. Completion and numerical validation are
separate: inspect `suite.json` and each run's `metrics.numerical_validation`.

## Use cece-dev with installed ESMF

The validated image is `cece/cece-dev:esmf`. Run both engines inside the same
container. Mount this repository read-only, a writable results/build directory,
and an ArborX 2.1 source checkout. Set `ARBORX_SOURCE` to its absolute location.

```sh
PAPER_WORK=/tmp/axis-paper-work
mkdir -p "$PAPER_WORK"
docker run -d --name axis-paper \
  -v "$PWD":/work/HELM:ro -v "$PAPER_WORK":/out \
  -v "$ARBORX_SOURCE":/sources/ArborX:ro cece/cece-dev:esmf sleep infinity
docker exec axis-paper cmake -S /work/HELM/libs/axis -B /out/build \
  -DAXIS_BUILD_PAPER=ON -DAXIS_PAPER_ENABLE_ESMF=ON \
  -DAXIS_BUILD_TESTING=OFF -DAXIS_BUILD_PYTHON=OFF -DAXIS_BUILD_FORTRAN=OFF \
  -DAXIS_ENABLE_PROJ=OFF -DAXIS_FETCH_KOKKOS=OFF -DAXIS_FETCH_KOKKOSKERNELS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_ARBORX=/sources/ArborX -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
docker exec axis-paper cmake --build /out/build --target \
  axis_paper_axis axis_paper_esmc axis_paper_reference axis_paper_metrics axis_paper_esmc_probe -j4
libs/axis/paper/.venv/bin/python libs/axis/paper/scripts/prepare_cases.py \
  --profile libs/axis/paper/config/smoke.json --output "$PAPER_WORK/cases"
libs/axis/paper/.venv/bin/python libs/axis/paper/scripts/run_suite.py \
  --profile libs/axis/paper/config/smoke.json --cases "$PAPER_WORK/cases" \
  --bin-dir "$PAPER_WORK/build/paper" --output "$PAPER_WORK/smoke" \
  --container axis-paper --host-root "$PAPER_WORK" --allow-incomplete
```

The runner checks the declared mount, passes thread settings into `docker exec`,
and hashes the mounted native binaries. Native workers record Linux CPU/OS and
compiler details; Python host provenance remains separate. Container startup and
transport overhead do not enter native durations. Record the image digest in
`provenance.native_execution.image_id`; an image tag alone is not immutable.

## Geometry and references

Coordinates are degrees on a unit sphere; native spherical areas are steradians.
Masks use 1 for active and 0 for masked. Fixtures contain zero-based CSR node
indices, stable cell IDs, and parent IDs. ESMC receives local one-based indices.
Generic great-circle fixtures are shared literally by both engines. Rectilinear
cells are explicitly triangulated for these cases; their boundaries differ
slightly at different resolutions, so coverage near outer edges can be partial.
The independently defined covered-domain reference accounts for that difference.
Constant-latitude rectangles are a separate geometry class and remain unmatched
against the great-circle ESMC adapter.

`reference_geometry.hpp` independently intersects convex great-circle polygons
in a gnomonic chart and integrates area and vector area analytically. Rectangles
use exact longitude/latitude interval intersections and spherical measure
`d(lon) d(sin(lat))`. Valid input cells must be nonoverlapping, counterclockwise,
convex, and contained in an open hemisphere. Unsupported cells are rejected;
there is no engine-derived reference fallback. The implementation currently scans
source/destination bounding caps, so reference preparation is quadratic in cell
counts and is excluded from performance measurements.

Original-profile fields are cell averages: constant 1, smooth `2 + cos(lat)*cos(lon)`, signed
`cos(lat)*cos(lon)`, and the hemisphere indicator `x >= 0`. The signed smoke case
is symmetric around longitude 90 degrees and has a near-zero integral. The sharp
case crosses that hemisphere boundary. DstArea divides by full destination area;
FracArea divides by independently covered area. Masked sources are excluded.

Original-profile analytic references are evaluated with both `double` and `long double`; output records
precision differences, roundoff estimates, area uncertainty estimates, and areas
of discarded roundoff-sized overlaps. This is empirical precision evidence,
**not a formal interval error bound**. The recorded platform must actually offer
more than 53 mantissa bits for the extended-precision check to qualify. Formal
certification of ill-conditioned geometry remains outside supported claims.
The analytic approach replaces the planned adaptive quadrature for these exact
fields and restricted convex geometries; depth/subtriangle settings are retained
in policy identity but are not used by the analytic algorithm.

LCC uses pyproj to preserve projected cell corners. Real MPAS inputs use the
existing `libs/axis/data/mpas` cache; `--fetch` explicitly permits acquisition.
Original variable-arity polygons and separately named shared triangulated cases
remain distinct. The ESMC adapter accepts documented TRI/QUAD elements only.
Triangulation preserves parent IDs; reported parent averages aggregate child
values using full destination area for DstArea and independent covered area for FracArea. It does not reproduce the original
piecewise-constant parent representation exactly.

## Metrics and numerical acceptance

Native metrics report covered-area-weighted MAE, bias and RMS, unweighted maximum error,
row sums and constant errors, bounds violations, coverage/unmapped/failure counts,
engine-area conservation, and independent canonical-area bookkeeping. Conservation
uses source area times source fraction and, for FracArea, destination area times
destination fraction. Relative residuals are null near zero; absolute residuals
remain reported. A complete run can contain an unfavorable numerical result.

Smoke tolerances are recorded in the profile: constant/row errors `1e-10`,
conservation `1e-10 * F * A`, near-zero threshold `1e-12 * F * A`, reproduction
`1e-10 * F`, smooth RMS `0.1 * F`, where `F` is the field bound (at least 1) and
`A` the common covered area. Reference uncertainty must be below `1e-9 * F` for
smoke and `1e-10 * F` for paper. Changes require new profile/fixture identities.
A unit constant on a fully covered cell should return 1; a half-covered DstArea
cell should return 0.5 and a FracArea cell 1. No conservation or speedup result
is presumed favorable.

## Timing protocol

Quality and timing run in separate processes. Quality enables available optional
host overlap counters. Timing disables counters and records every measured
repetition; setup warm-ups and application warm-ups are excluded. The quality
record and timing record must agree on fixture, reference, options, and algorithm.

- Preparation: native mesh adaptation, recorded separately.
- Generation: AXIS weight generation or ESMC conservative regrid store.
- Finalization: AXIS CSR conversion plus first application, or ESMC first regrid.
- Setup: paired generation plus finalization for a fresh operator each repetition.
- Apply: a warmed existing operator, with destination overwrite.
- `repeated_workload:N`: N applications of the retained operator, excluding setup.
- `setup_and_reuse:N`: sum of measured generation, finalization, and N application
  intervals on the same fresh operator. It includes the extra first-use apply
  counted in finalization; it is not a contiguous wall-clock interval.
- `estimated_total:N`: median setup + N times median single apply, explicitly an
  estimate, separate from measured components.

Cleanup, initialization/finalization, Python, I/O, and references are excluded.
Kokkos fences bound AXIS timings; ESMC calls are synchronous and restricted to
one MPI process. Raw seconds are retained, summaries use median and IQR, and
reuse counts default to 1 and 100. Smoke uses 1 warm-up/3 repetitions; paper uses
2/10. Speedup eligibility requires identical cases/options, matched area/edge
conventions, and passing numerical/reproduction checks. Missing memory and setup
stage instrumentation are null with reasons. Results describe the recorded CPU
configuration; they establish no distributed or accelerator portability.

## Regenerate displays and manuscript evidence

Run from repository root, replacing `RESULTS` with the saved bundle directory:

```sh
python libs/axis/paper/scripts/make_figures.py --results RESULTS --output RESULTS/figures
python libs/axis/paper/scripts/make_tables.py --results RESULTS --output RESULTS/tables
python libs/axis/paper/scripts/evidence.py --results RESULTS --update
python libs/axis/paper/scripts/evidence.py --results RESULTS --check
python libs/axis/paper/scripts/evidence.py --results RESULTS --archive NEW_ARCHIVE_DIRECTORY
```

The five required products are workflow, source/reference/result/error maps,
performance panels, quality table, and configuration table. Figures export PDF,
SVG, and 300 dpi PNG; tables export full-precision CSV and rounded Markdown.
Each sidecar records input run hashes, script identity, argv, caption, and print
settings. Optional convergence requires three compatible resolutions, a common
domain/direction and adequate reference precision. Stage breakdowns require real
instrumentation; absence is reported explicitly.

Evidence updates preserve narrative outside managed blocks, resolve LLC to LCC,
and cite reviewed source paths/symbols/hashes plus saved measurements. The outline
inventory links each question to evidence or a stated limitation. Abstract and
conclusion claims must reflect actual measurement scope. Archives contain source,
dirty patches, inputs, configs, raw outputs, displays, and an integrity manifest.
See `evidence/smoke_acceptance.md` for the executed acceptance status.

## Code readability

Use descriptive names, small functions, and comments explaining units, numerical
assumptions, normalization, and timing boundaries. Avoid compressed statements.
Python dataset access goes through xarray; `netCDF4` is only its storage backend.
Native workers use NetCDF-C. Run `ruff format libs/axis/paper/scripts` and
`clang-format` with the paper directory's settings. Refresh citation hashes after
source changes before updating the manuscript.

For strict cross-engine acceptance, use `config/smoke-matched.json`. The broader
`config/smoke.json` retains geometry diagnostics with explicit ESMC exclusions.

## Figures following Valcke et al. (2022)

The supplemental `config/valcke-fields.json` runs the paper's sinusoid, harmonic,
vortex and Gulf Stream definitions with native adaptive cell-average references,
both directions and both DstArea/FracArea normalizations. See
[evidence/valcke_figures.md](evidence/valcke_figures.md) for figure correspondence,
actual meshes, percentage-misfit definitions and integration assumptions.
After the standard figure/table commands, run:

```sh
python libs/axis/paper/scripts/make_comparison_figures.py --results RESULTS --output RESULTS/figures
```

This adds mesh/field illustrations, per-case AXIS–ESMF quality panels and spatial
percentage-error maps. It preserves the original area-weighted error diagnostics
and identifies the differences from the example paper's original experiment.

To freeze the supplemental figures together with the original numerical study,
use the archive option for related bundles (each bundle is validated first):

```sh
python libs/axis/paper/scripts/evidence.py --results RESULTS --archive NEW_ARCHIVE_DIRECTORY \
  --related-results libs/axis/paper/runs/paper-final
```

The archive rewrites manuscript links to the primary and related result copies.
Previously frozen archives retain their original source and manuscript snapshots.


The completed supplemental figures are indexed in the
[figure gallery](evidence/figure_gallery.md). Numerical scope and reference
convergence are recorded in [supplemental acceptance](evidence/valcke_acceptance.md).
Generate the finer Atlantic product with `scripts/make_detail_figure.py` after
running `config/gulfstream-detail.json` through the same native pipeline.

For reproducible native execution in the installed Docker environment:

```sh
python libs/axis/paper/scripts/prepare_cases.py \
  --profile libs/axis/paper/config/valcke-fields.json --output CASES
AXIS_PAPER_CONTAINER_IMAGE=cece/cece-dev:esmf \
python libs/axis/paper/scripts/run_suite.py \
  --profile libs/axis/paper/config/valcke-fields.json --cases CASES \
  --bin-dir libs/axis/paper/build-container/build/paper \
  --container axis-paper-converged --host-root libs/axis/paper/build-container \
  --output RESULTS
```

`CASES` and `RESULTS` must resolve inside the shared host mount for container
execution. Use fresh directories; existing experiment bundles are immutable.
Repeat with `gulfstream-detail.json` and separate case/result directories for
the focused Atlantic experiment. The preparation step may require the existing
MPAS fixture described above; reference generation and benchmarking need no
network access.


## North Polar cap

`config/polar.json` adds a cap whose meshes include the North Pole itself,
periodic longitude rings and differently rotated great-circle boundaries.
Twelve cases cover constant/smooth/harmonic fields, both directions and both
normalizations. See [polar results](evidence/polar_acceptance.md). Run
`scripts/make_polar_figures.py` after the standard figure/table commands to
export pole-centered maps and quality plots with common normalization scales.
