# Figures following Valcke et al. (2022)

All figures below are generated from new native AXIS and ESMC experiments.
PDF, SVG, and 300 dpi PNG exports are available beside each image.

## The four analytical fields

![Sinusoid, harmonic, vortex and Gulf Stream cell averages](../runs/valcke-converged/figures/analytical_fields.png)

## Matched numerical comparison

![Harmonic percentage misfits and source conservation](../runs/valcke-converged/figures/quality_comparison_harmonic_dstarea.png)

The same four-panel comparison is available for every field with DstArea and
FracArea normalization. Both remapping directions appear separately. Source
conservation panels use the common covered domain; the separately retained
dimensional metrics use covered-area weighting. Coincident curves reflect the
actual measurements, and do not imply a universal accuracy guarantee.

## Focused Atlantic Gulf Stream experiment

![Reference, AXIS, ESMF, coverage and paired spatial errors](../runs/gulfstream-converged/figures/comparison_map_atlantic-gulfstream-dstarea-forward_atlantic_detail.png)

This figure uses a new 1,920 → 17,280 element experiment. It exposes spatial
errors from a first-order piecewise-constant source representation; the refined
destination grid does not turn that method into higher-order interpolation.

## Meshes and conservation

- [Mesh connectivity gallery](../runs/valcke-converged/figures/mesh_gallery.png)
- [Source conservation, DstArea](../runs/valcke-converged/figures/source_conservation_dstarea.png)
- [Source conservation, FracArea](../runs/valcke-converged/figures/source_conservation_fracarea.png)
- [Analytical target-integral difference, DstArea](../runs/valcke-converged/figures/target_conservation_dstarea.png)
- [Analytical target-integral difference, FracArea](../runs/valcke-converged/figures/target_conservation_fracarea.png)
- [Full numerical data behind the panels](../runs/valcke-converged/figures/comparison_metrics.csv)

The main supplemental folder contains 27 comparison figures; the focused
Atlantic experiment adds one further spatial figure. See
[valcke_figures.md](valcke_figures.md) for correspondence to the example paper,
mesh definitions, integration methods and exact reproduction commands.


## North Polar cap, including the pole

![North Polar harmonic comparison](../runs/polar-final/figures/polar_map_polar-harmonic-dstarea-forward.png)

The meshes include exactly 90°N, with a shared pole node and periodic longitude
connectivity. AXIS and ESMF produce closely matching results in all 12 cases.
See [polar acceptance and interpretation](polar_acceptance.md),
[mesh connectivity](../runs/polar-final/figures/polar_meshes.png), and
[quality comparison](../runs/polar-final/figures/polar_quality_harmonic_dstarea.png).
The polar DstArea and FracArea quality plots use identical vertical limits.
