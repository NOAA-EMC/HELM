# Valcke-style figures and added experiments

The example is Valcke, Piacentini and Jonville (2022), *Benchmarking Regridding
Libraries Used in Earth System Modelling*, doi:10.3390/mca27020031. Appendix A,
Figures A1–A4, supplies the four mathematical field definitions. The paper is
CC BY 4.0. The implementation cites that source in `native/analytical_fields.hpp`.

## Correspondence

| Example paper | AXIS publication product | Data and scope |
| --- | --- | --- |
| Figure 1: grids | `mesh_gallery` | Actual rectilinear, regional LCC and shared MPAS triangles |
| Figure 2: four functions | `analytical_fields` | Independently integrated sinusoid, harmonic, vortex and Gulf Stream source cell averages |
| Figures 9 and 13: conservative misfit comparisons | `quality_comparison_FIELD_NORM` | Mean/RMS/maximum percentage misfit, AXIS versus ESMC, with both directions and normalizations |
| Figure 10: source conservation | Fourth panel of every quality comparison | Common covered-domain residual, not an unqualified full-globe residual |
| Figures 7, 8, 12 and 15: spatial misfit | `comparison_map_FAMILY_FIELD_dstarea_forward` | Reference, both results, coverage, and both signed percentage-error maps |

These are new AXIS experiments with Valcke's analytical functions, not a
replication of the original paper's six grids, masks, methods or point sampling.
Only matched first-order conservative great-circle remapping is compared.
Original AXIS constant/smooth/step studies and their dimensional area-weighted
metrics remain available separately.

## Added experiment profile

`config/valcke-fields.json` defines 48 cases (four fields × three mesh pairs ×
two directions × two normalizations). Every case requests native AXIS and ESMC
quality and timing records. All source/destination connectivity is shared.

- Global rectilinear: 36×18 parents versus 72×36, with two great-circle triangles
  per parent, or 1,296 versus 5,184 native elements. Latitude ends at ±89.9°.
- Regional Atlantic LCC: 24×24 projected parents (1,152 triangles), centred on
  65°W, 40°N, projected square ±500 km; versus 32×32 longitude–latitude parents
  (2,048 triangles) over 73°W–57°W, 34°N–46°N. This region intersects the
  Gulf Stream test feature. Edges between inverse-projected corners are spherical.
- MPAS: the 15,360 shared centroid-fan triangles of `x1.2562`, versus the same
  1,296-element near-global rectilinear mesh.

There is no imposed land mask. Partial regional coverage is retained and shown.
The new profile is separate from the original `paper.json`; existing measurement
bundles are not overwritten or silently reinterpreted.

## Independent integration and diagnostics

`native/field_quadrature.hpp` integrates analytical values over spherical
polygons by triangulation and radial projection. A Duffy transformation gives a
square integration domain; independent tensor Gauss–Legendre rules with 6 and
12 nodes per dimension are compared. An empirical error estimate controls
geodesic subdivision of the largest-error subtriangle until the original
triangle’s global absolute budget is met, with declared depth/work limits. For the Gulf Stream, conservative spherical-cap bounds on its phase detect
triangles that may intersect the narrow jet. Such triangles are subdivided to
at most 0.1° edge length before quadrature-rule agreement is trusted; further
adaptive refinement resolves the clipped feature. This guards against both
rules missing the same jet. Source and destination integration uncertainties are saved. Geometry is independently evaluated in
double and extended precision; adaptive analytical-field evaluation, quadrature
and its accumulated error estimates use float64. Extended-precision geometry
agreement does not certify field integration accuracy. These are convergence estimates, not certified interval bounds.

The supplemental reference tolerance is `1e-7` times a conservative field bound
of five. Per-triangle integration budgets use one sixteenth of that tolerance;
source averages, destination averages and area agreement are checked before
native workers run. The original profile's geometry, conservation, constant and
reproduction tolerances are unchanged. Near-zero reference exclusions are
reported rather than assigning fabricated percentage errors.

Percentage diagnostics follow the paper's unweighted per-cell definitions;
they do not replace the existing covered-area-weighted dimensional errors.
The source conservation residual integrates the discrete source over common
coverage. Target conservation compares the remapped integral to the independent
continuous analytical reference over that same domain. Lmin/Lmax follow the
paper's algebraic formulas and are stored in the CSV/native metrics. Figure
captions identify these conventions and the symmetric-log treatment of exact
zero residuals.

## Reproduction

Prepare fixtures, then use the native run commands from `../README.md` with
`config/valcke-fields.json`. Generate the standard displays and then add the
comparison figures (replace `RESULTS` with the new saved bundle):

```sh
python libs/axis/paper/scripts/make_figures.py --results RESULTS --output RESULTS/figures
python libs/axis/paper/scripts/make_tables.py --results RESULTS --output RESULTS/tables
python libs/axis/paper/scripts/make_comparison_figures.py --results RESULTS --output RESULTS/figures
```

Comparison products export PDF, SVG, and 300 dpi PNG with run hashes, captions,
script identity and commands. `comparison_metrics.csv` preserves the data behind
all panels. Geographic cell polygons are used only for visualization; no
coastline dataset is substituted for the numerical coverage mask. Longitude
seam-crossing polygons are locally unwrapped and drawn with periodic copies.

The supplemental profile does not impose a new RMS accuracy cutoff on these
four fields: their relative/dimensional errors are measured findings. Passing
numerical validation means the implemented coverage, finiteness, row-sum,
conservation and timing-reproduction checks passed; it does not assert a
particular accuracy order or library superiority.

## Focused Atlantic spatial experiment

`config/gulfstream-detail.json` adds one matched Gulf Stream case over
85°W–10°W, 20°N–65°N. The 40×24 source parents and 120×72 destination parents
are split into 1,920 and 17,280 triangles respectively. Destination parent
spacing is 0.625° in both coordinates. This is a new native remapping experiment,
not interpolation or image enlargement of the coarse global result. The same
reference tolerance, float64 quadrature, conservative method, DstArea convention,
thread/rank configuration and ten native repetitions apply. Generate the
standard products first, then run:

```sh
python libs/axis/paper/scripts/make_detail_figure.py --results DETAIL_RESULTS --output DETAIL_RESULTS/figures
```


## Reference convergence review

An initial doubled-order calculation exposed aliased sampling of the narrow
Gulf Stream jet: the largest destination-reference difference was about
0.001922, exceeding the declared 5e-7 absolute threshold. Those initial bundles
(`valcke-final`, `gulfstream-detail`) are excluded from accepted quantitative
claims. The saved [failure record](validation/initial_order_failure.json)
retains the observed discrepancies and reference hashes.

After adding phase-bound detection and mandatory spatial subdivision, separate
6/12- and 12/24-point native workers agree on all four coarse global preflight
fields within 5e-7. The largest corrected Gulf Stream difference is
9.5423e-9; see the [convergence record](validation/refined_order_comparison.json)
and [build identities](validation/reference_builds.json). Some local combined
empirical estimates are smaller than observed differences. Consequently those
estimates are not asserted to enclose every cell's true error. Acceptance here
uses observed order convergence and the declared absolute threshold.

The xarray command `scripts/compare_reference_orders.py` preserves both passing
and failing comparisons. To reproduce the independent native variant, copy the
reference sources and change only `coarse = gauss(6), fine = gauss(12)` to
`coarse = gauss(12), fine = gauss(24)` in `field_quadrature.hpp`, compile using the
saved command, and run both workers on identical fixtures. The archived
validation folder retains both exact source variants and generated references.


The final production study additionally checks seven Gulf Stream references:
the global rectilinear and MPAS pairs, all four regional LCC direction and
normalization combinations, and the fine Atlantic case. All pass the same
5e-7 absolute threshold; the maximum observed difference is 4.9092e-8.
See [production order comparison](validation/production_order_comparison.json)
and [current build identities](validation/current_reference_builds.json).
Exact source variants, reference binaries, commands and all production
comparison NetCDF files are retained under
[the accepted validation bundle](../runs/valcke-converged/validation/production-mesh-comparison.json).

The earlier temporary build and an interrupted corrected study were lost
between turns. The final study was rebuilt and rerun in workspace-backed
storage using the same immutable ESMF image. Its dependency archive hash and
Python package lock are retained with validation. The original failed sampling
comparison remains as a historical JSON record; accepted production validation
has independently generated and retained raw references.
