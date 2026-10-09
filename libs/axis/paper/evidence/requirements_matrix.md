# Requirements and acceptance traceability

The suite implements native AXIS/ESMC computation and timing, native independent
references/metrics, xarray fixture preparation, publication exports, and manuscript
traceability. Executed numbers and hashes are recorded in `smoke_acceptance.md`.
Completion status and numerical validation are deliberately separate.

| Requirement | Implementation | Acceptance evidence |
| --- | --- | --- |
| FR-001 / SC-001: every outline question | `outline_inventory.json`; `scripts/evidence.py` | Inventory IDs equal original outline bullet IDs; source/run/product links checked |
| FR-002 / SC-002: three mesh families | `scripts/prepare_cases.py`; `config/paper.json` | Rectilinear, projected LCC, cached real MPAS, both directions; original/triangulated cases remain distinct |
| FR-003: analytical cell averages | `native/reference.cpp`; `native/reference_geometry.hpp` | Constant, smooth, signed near-zero-integral, hemisphere indicator; double/extended-precision differences and roundoff/area estimates |
| FR-004: numerical metrics | `native/metrics.cpp` | MAE/bias/RMS/max; engine and canonical covered-domain integrals; constants and rows; near-zero relative residual is null |
| FR-005: bounds and failures | `native/metrics.cpp`; core optional diagnostics | Overshoot/undershoot, unmapped/masked/nonfinite counts; overlap counters or null with reason; failed/unsupported records retained |
| FR-006 / SC-003: matched protocol | `scripts/run_suite.py`; `scripts/artifacts.py` | Shared fixture/reference/options, explicit matching key, area agreement and quality/timing linkage |
| FR-007: native setup/apply timing | `native/timing.hpp`; both workers | Fresh operators, first-use setup, warmed reuse, completed-work fences, raw repetitions, median/IQR; cleanup excluded |
| FR-008 / SC-004: reuse totals | `scripts/timing_summary.py`; worker observations | N=1/100 measured setup-and-reuse intervals, retained workloads and separately labeled estimates; speedups require passing matched evidence |
| FR-009: provenance | `native/provenance.hpp`; `scripts/provenance.py`; container transport | Real binary hashes, base revision/dirty patch, dependency versions, compiler commands, Linux runtime, Python host and explicit threads/image digest |
| FR-010 / SC-005: five displays | `scripts/make_figures.py`; `scripts/make_tables.py` | Workflow, maps, performance, quality/configuration tables; PDF/SVG/300 dpi PNG, full-precision CSV, sidecars |
| FR-011: conditional displays | `scripts/make_figures.py` | Compatible smooth-field ladders only; reference precision guard; stage timing omitted with reason when absent |
| FR-012: verified methods | `implementation_citations.json`; managed outline block | Exact hashes/base revision; distinguishes actual spherical half-space clipping from legacy Greiner–Hormann labels and analytic shortcuts |
| FR-013 / SC-006: measured claims | `scripts/evidence.py` | Numerical claims link complete run records; scope and exclusions explicit; no unmeasured portability or higher-order claim |
| FR-014: reproduction/archive | `README.md`; `scripts/archive.py` | Native/container commands, regenerated products, portable source snapshot and integrity manifest |
| FR-015: missing/failing inputs | Native exit codes; suite statuses/logs | Unsupported ESMC polygon/strip cases retained; missing executable and stale-hash checks recorded in acceptance |
| SC-007: executed reproduction | `config/smoke-matched.json`; `evidence/smoke_acceptance.md` | Strict matched native smoke suite, numerical reproduction comparison, all five displays and traceability |

## Supported scope and limits

- One CPU MPI process and recorded OpenMP settings. Higher-order, vector,
  distributed, accelerator and broad portability results are unmeasured.
- The ESMC adapter accepts documented triangle/quad elements. Original MPAS
  polygons are retained as unsupported ESMC cases; shared triangulations are
  explicitly different representations, including different cell-average fields.
- Analytic reference geometry requires convex nonoverlapping cells in an open
  hemisphere. Precision-agreement/roundoff estimates are empirical, not formal
  interval certificates. Reference bounding-cap scanning is quadratic and excluded
  from benchmark timings.
- Memory and setup-stage observations remain unavailable with reasons. No causal
  attribution to ArborX or clipping is made from whole-library timings.
- The initial planning sketches placed all run metadata in C++ records. The
  implemented split uses native case/reference/observation structures and atomic
  worker output, with complete run/status/matching JSON assembled and validated in
  Python. Implemented artifact layouts are reconciled in the contracts.

## Acceptance review

See `smoke_acceptance.md` for actual executed statuses, environment, commands,
export checks and hashes. The source audit corrected the distinction between
Sutherland–Hodgman-style great-circle clipping and legacy Greiner–Hormann wording.
Native source changes preserve conservative normalization during coastal
processing; strip adaptation supplies spherical areas. These changes alter prior
conservative masked behavior intentionally and require the recorded regression
cases. No acceptance tolerance was loosened.

Final acceptance: strict smoke **36/36** complete and passing; paper **132/144**
complete and passing, with **12 declared ESMC original-MPAS exclusions**. All
five required displays validate; the paper bundle also includes convergence.
Maximum timing reproduction difference is 8.9e-16. See the linked acceptance
report for native environment, commands, integrity rejection and scope limits.

## Manuscript drafting follow-through

Sections 1–4 of `specs/AXIS_JAMES_outline.md` now contain manuscript prose,
formulation and metric equations, actual mesh counts, ESMC settings, and native
timing definitions with direct source/result citations. `drafting_coverage.json`
preserves the original 99 questions and their evidence, including the 53 questions
answered by these sections. `outline_inventory.json` tracks the remaining 46
bullets; completed drafting questions have not been silently discarded. The
existing frozen archive retains its earlier manuscript snapshot.


## Valcke follow-through (T047–T048)

| User refinement | Implementation | Executed evidence |
| --- | --- | --- |
| Four Appendix A fields and independent integration | `native/analytical_fields.hpp`; `native/field_quadrature.hpp` | 49 cases, 196 native records, all numerical checks pass |
| Narrow-feature reference validation | Phase-bound sampling; `scripts/compare_reference_orders.py` | Seven production comparisons; maximum difference 4.9092e-8 < 5e-7 |
| Figures following the example paper | `scripts/make_comparison_figures.py`; `scripts/make_detail_figure.py` | 28 comparison figures, PDF/SVG/300 dpi PNG, full-precision CSV and hashed sidecars |
| Human readable code and xarray | Formatted native helpers and Python scripts; resolved container paths | Error lint, formatting and executed xarray workflow |

See [supplemental acceptance](valcke_acceptance.md) and the
[figure gallery](figure_gallery.md). The initial Gulf Stream reference failure
is retained explicitly and excluded from accepted claims.


## Polar follow-through (T049–T050)

The North Polar cap includes exactly 90°N, 360/1,488 cells and rotated outer
boundaries. All 48 native records pass; topology and four independent harmonic
reference comparisons pass. Fifteen comparison figures use pole-centered
spatial views and shared quality scales across normalizations. See
[polar acceptance](polar_acceptance.md) for results and reproducibility.
