# North Polar cap experiment

Status: **accepted within the recorded CPU and first-order spherical-triangle scope**.

## Geometry and protocol

The [polar profile](../config/polar.json) contains 12 cases: constant, smooth and
harmonic fields, both directions, and DstArea/FracArea normalization. Both engines
consume identical explicit great-circle triangles. Each mesh has periodic
longitude connectivity and one shared node at exactly 90°N. Outer vertices lie
at 60°N; the boundary between vertices consists of great-circle arcs.

The coarse mesh uses 24 sectors and eight radial intervals (360 triangles).
The fine mesh uses 48 sectors and 16 intervals (1,488 triangles), rotated 3.75°.
Their different outer polygon boundaries produce partial coverage, retained in
references and diagnostics. There is no land mask or extrapolation. Projection
is used for plotting only; remapping and area calculations are spherical.

Native AXIS and ESMC use the existing one-thread/process protocol, two warmups,
ten repetitions and reuse counts 1/100. Python dataset access uses xarray.

## Results

All [48 quality/timing records](../runs/polar-final/suite.json) completed and
passed numerical checks. Timing output exactly reproduced quality output.
The maximum source-conservation residual was 5.0595e-13 percent.

| Field | Forward RMS relative misfit | Reverse RMS relative misfit |
| --- | ---: | ---: |
| Smooth | 0.563605% | 0.093505% |
| Harmonic | 0.518418% | 0.123164% |

AXIS and ESMF agree to the shown digits for both normalizations. These are
unweighted per-cell percentage errors; area-weighted dimensional diagnostics
remain in the standard tables. The lower reverse errors reflect averaging onto
a coarser grid and do not establish a convergence rate. The harmonic field's
amplitude decreases toward the pole; the constant and smooth cases additionally
exercise connectivity and conservation there.

Every mesh has a single pole node, positive cell areas, manifold edge incidence
and disk Euler characteristic one. Four harmonic references were independently
recomputed with doubled Gauss orders, including both directions/normalizations.
The maximum source/destination difference was 1.1991e-14, below 5e-7. Reference
estimates remain empirical. Full results are in
[polar_acceptance.json](polar_acceptance.json),
[topology checks](../runs/polar-final/validation/topology.json) and
[reference comparisons](../runs/polar-final/validation/reference_comparison.json).

## Figures and interpretation

- [Actual polar connectivity](../runs/polar-final/figures/polar_meshes.png)
- [Harmonic spatial comparison, DstArea forward](../runs/polar-final/figures/polar_map_polar-harmonic-dstarea-forward.png)
- [Harmonic quality, DstArea](../runs/polar-final/figures/polar_quality_harmonic_dstarea.png)
- [Harmonic quality, FracArea](../runs/polar-final/figures/polar_quality_harmonic_fracarea.png)
- [Full-precision metrics CSV](../runs/polar-final/figures/polar_metrics.csv)

Fifteen comparison figures cover meshes, six quality panels and eight spatial
comparisons. PDF, SVG and 300 dpi PNG exports have hashed native inputs and
script identities. The required field map uses the pole-centered display.
Spatial maps share scales between engines. Quality panels share vertical limits
between DstArea and FracArea, so tiny conservation residuals can be compared
without independently rescaling the axes. Native edges are great circles;
straight display chords approximate them in the azimuthal equidistant view.
The North Pole is at the center; circles mark latitude and the cross marks 90°N.

## Reproduction and code

Mesh construction is in [polar_mesh.py](../scripts/polar_mesh.py); shared fixture
preparation is in [prepare_cases.py](../scripts/prepare_cases.py). Plotting is in
[make_polar_figures.py](../scripts/make_polar_figures.py).

```sh
python libs/axis/paper/scripts/prepare_cases.py --profile libs/axis/paper/config/polar.json --output CASES
python libs/axis/paper/scripts/run_suite.py --profile libs/axis/paper/config/polar.json --cases CASES --bin-dir BIN_DIR --output RESULTS
python libs/axis/paper/scripts/make_figures.py --results RESULTS --output RESULTS/figures
python libs/axis/paper/scripts/make_tables.py --results RESULTS --output RESULTS/tables
python libs/axis/paper/scripts/make_polar_figures.py --results RESULTS --output RESULTS/figures
```

Use the README's container options for Docker execution. Run the polar figure
command last so its pole-centered map replaces the generic longitude/latitude
required display. The [polar archive](../runs/polar-archive/manifest.json)
contains this study's results and current source; earlier studies retain their
separate archives. This experiment does not cover Antarctic, ice masks,
distributed execution or all production polar mesh types.
