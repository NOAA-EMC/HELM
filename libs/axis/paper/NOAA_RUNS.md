# Running the expanded NOAA study

This extension requires the preceding linear-reconstruction/performance changes (`AXIS_regression_fixes.zip`). It changes paper-only code and adds no production dependencies. Use the paper README's existing ESMF/NetCDF/MPI/Python setup; ESMC is the native C interface to ESMF.

## 1. Build in the existing ESMF-enabled environment

Use a new build directory and set `ESMFMKFILE` to the installed version matching your compiler/MPI. Follow `README.md` for local or `cece/cece-dev:esmf` Docker execution. With the previously configured paper build:

```sh
cmake --build libs/axis/paper/build --target \
  axis_paper_axis axis_paper_esmc axis_paper_reference axis_paper_metrics \
  axis_paper_esmc_probe axis_paper_scalar_pilot axis_paper_capabilities -j1
libs/axis/paper/build/paper/axis_paper_esmc_probe
```

The probe now checks all five ESMC method constants. Native ESMC compilation has not been performed in this workspace. Keep configure/build logs and the ESMF version with the results.

## 2. Smoke suite, then the larger profile

```sh
python libs/axis/paper/scripts/prepare_cases.py \
  --profile libs/axis/paper/config/noaa-all-methods-smoke.json \
  --output libs/axis/paper/data/noaa-smoke
python libs/axis/paper/scripts/run_suite.py \
  --profile libs/axis/paper/config/noaa-all-methods-smoke.json \
  --cases libs/axis/paper/data/noaa-smoke \
  --bin-dir libs/axis/paper/build/paper \
  --output libs/axis/paper/runs/noaa-smoke
```

Use new output directories for each run. For the larger suite, replace the profile with `noaa-all-methods.json` and use new case/run directories. Its MPAS meshes require the existing data files; missing/unsupported cases remain in the inventory. For Docker, use the existing README's `--container`, `--host-root`, and `--container-root` routing arguments.

- `paired-quality.json`: accuracy on the same mapped wet destination population in both engines; use this for paired accuracy tables.
- `suite.json`: complete/incomplete/failed status, numerical failures, timing comparisons, and pairing labels.
- Per-worker `run.json`, `fields.nc`, `metrics.json`, `observations.csv`, and logs: retained raw evidence and provenance.
- `--allow-incomplete` tolerates unavailable cases but does not turn numerical failures into successful exit status.
- `conservation_checked=false` means the integral residual is diagnostic; do not publish it as a passed conservation test. Partially covered second-order source cells need the reconstructed covered-source integral.
- Existing figure scripts were built around the earlier conservative study. Review method labels and use the shared-population metrics when generating the all-method paper figures.

## 3. AXIS capability examples

```sh
libs/axis/paper/build/paper/axis_paper_scalar_pilot > scalar-pilot.csv
libs/axis/paper/build/paper/axis_paper_capabilities > capabilities.csv
```

The scalar pilot uses independent in-memory geometry and requires no reference files. It exercises all seven variants, constants/smooth fields, and source masks.

After applying `AXIS_vertical_fix.zip`, all four off-knot vertical affine-profile checks pass at a maximum error of `1.7763568394e-15`, and the capabilities executable exits successfully. The vector rotation example also passes. The vertical regression suite now includes independent cubic and positive-tension references and off-knot randomized properties.

The native scalar worker does not implement a direct ESMC vector or vertical comparison. `NOAA_WHITE_PAPER.md` distinguishes the AXIS examples, documented ESMC restrictions, and remaining adapter work. The current vector API rejects second-order conservative reconstruction.

## 4. Expanding another existing profile

```sh
python libs/axis/paper/scripts/expand_methods.py \
  --profile libs/axis/paper/config/valcke-fields.json \
  --output libs/axis/paper/config/noaa-valcke.json --id noaa-valcke
```

This makes method-specific case IDs, retains geometry/fields, and skips nonconservative `fracarea` variants because the interpolation track uses a full-cell transfer objective. It rejects already-expanded input rather than accidentally duplicating methods.
