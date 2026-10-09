# Native study acceptance

Status: **accepted within the recorded CPU and supported geometry scope**.

## Executed results

| Bundle | Complete / status records | Numerical checks | Timing reproduction maximum |
| --- | --- | --- | --- |
| [smoke-final](../runs/smoke-final/suite.json) | 36 / 36 | all pass | 4.4408920985006262e-16 |
| [paper-final](../runs/paper-final/suite.json) | 132 / 144 | all pass | 8.8817841970012523e-16 |

The paper study contains 66 quality and 66 timing records. Its 12 excluded
records are six original variable-arity MPAS cases in both modes: the ESMC
adapter supports documented TRI/QUAD mesh elements and reports these cases as
unsupported. Shared triangulations are measured separately. No failed case was
silently dropped. The strict matched smoke profile excludes the constant-latitude
strip from cross-engine comparisons; the auxiliary diagnostic profile records
that ESMC geometry exclusion explicitly.

All five required publication products were regenerated from saved native
outputs and their input/script hashes validated. The paper bundle also includes
eligible convergence curves. PDF/SVG/PNG maps, workflow, performance and
convergence exports were visually reviewed; maps have independent colorbar axes,
and performance panels separate mesh families. CSV tables preserve precision.

## Environment and commands

Native execution: `cece/cece-dev:esmf`, container `axis-paper-implementation`,
Linux aarch64, GNU 13.3.0, ESMF 8.9.1, Kokkos 5.1.1, KokkosKernels version macro `50299`,
ArborX 2.1.0, NetCDF 4.9.2, one MPI process. Host: Apple M4, 32 GiB RAM.
Python 3.14.7 uses xarray dataset access. Native records retain exact thread
settings, immutable image identity, compiler/link commands, runtime metadata,
binary hashes, source identity, raw observations and separate Python metadata.

Exact bundle identities and supplementary environment are recorded in
[acceptance_environment.json](acceptance_environment.json). Each native command
and artifact hash is also saved in the corresponding run record.

```sh
docker exec axis-paper-implementation cmake -S /work/HELM/libs/axis -B /out/build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
docker exec axis-paper-implementation cmake --build /out/build --target axis_paper_axis axis_paper_esmc axis_paper_reference axis_paper_metrics axis_paper_esmc_probe -j4

# PROFILE=smoke-matched or paper; CASES=matched-cases or paper-cases.
# Paper additionally requires --allow-incomplete for declared MPAS exclusions.
AXIS_PAPER_CONTAINER_IMAGE=cece/cece-dev:esmf python libs/axis/paper/scripts/run_suite.py \
  --profile libs/axis/paper/config/PROFILE.json \
  --cases /tmp/axis-paper-container/CASES \
  --bin-dir /tmp/axis-paper-container/build/paper \
  --container axis-paper-implementation --host-root /tmp/axis-paper-container \
  --output /tmp/axis-paper-container/OUTPUT
```

The strict smoke command exited zero. Paper exited zero with its explicit
`--allow-incomplete` policy. A deliberately missing ESMC executable produced
two unavailable records, retained both successful AXIS records and exited 5
without that policy. Appending a byte to a copied fixture was rejected as
`Stale case`. Core-only configuration exposes no paper or ESMF targets.
Formatting, Python error lint and `git diff --check` passed.

## Scientific review and limits

Regression cases exposed and fixed coastal normalization that had incorrectly
turned conservative DstArea rows into FracArea rows. Conservative rows now retain
their requested normalization. Strip adaptation now supplies spherical areas.
Independent reference refinement removes accounted roundoff slivers before
sharp-field clipping and records double versus long-double differences.
No acceptance tolerance was loosened.

Reference precision evidence is empirical, not a formal interval certificate.
The reference implementation supports the documented convex geometry scope.
Source inspection identifies Sutherland–Hodgman-style spherical clipping;
the historical Greiner–Hormann diagnostic string is retained and explained.
Measured setup-plus-reuse totals sum recorded native intervals from the same
fresh operator; they are not contiguous wall-clock measurements. No stage
breakdown, distributed scaling, accelerator performance or higher-order result
is inferred from these serial CPU studies.

The frozen archive is `../runs/paper-archive`; its manifest covers the exported
source snapshot and complete paper result bundle. Regeneration instructions are
in [README.md](../README.md).
