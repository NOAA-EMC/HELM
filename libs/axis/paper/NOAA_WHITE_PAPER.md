# AXIS regridding: NOAA white-paper study

## 1. Purpose and operational use

- **Decision:** assess AXIS as a reusable regridding component for NOAA workflows, comparing accuracy, robustness, setup cost, and repeated application cost with native ESMC.
- **Examples:** transfer cell-mean tracers and fluxes between atmospheric/ocean meshes; rotate grid-relative velocity or wind components; interpolate vertical profiles; handle land/water masks and partial overlap.
- **Atmospheric chemistry objective:** preserve species inventory when transferring fields expressed as ppmv or µg/m³ across changing pressure, temperature, moisture, and layer geometry. Coordinate names and concentration units alone do not identify the conserved carrier.
- **Evidence standard:** report successful, failed, unsupported, and unavailable cases. Separate measured results from proposed experiments. Existing accepted first-order evidence remains historical evidence; it is not an all-method comparison.

## 2. Methods and comparison identities

| AXIS variant | ESMC comparison | Interpretation |
|---|---|---|
| Nearest neighbor | `NEAREST_STOD` | Same method family, source-to-destination direction |
| Bilinear | `BILINEAR` | Same family; implementation/stencils may differ |
| Bicubic | `PATCH` | Cross-method higher-order interpolation comparison |
| Patch | `PATCH` | Same family; recovery construction may differ |
| First-order conservative | `CONSERVE` | Same order and conservative objective |
| Unlimited second-order conservative | `CONSERVE_2ND` | Same order; reconstruction is implementation-specific |
| Limited second-order conservative | `CONSERVE_2ND` | Limited AXIS versus unlimited ESMC; explicitly distinguish |

- **Second-order conservative:** include both unlimited and limited AXIS paths in every compatible conservative case. Measure smooth-field accuracy and sharp-field extrema separately; the limiter trades accuracy near extrema for bounds control.
- **No bicubic equivalence claim:** the public ESMC method list supplies patch recovery, not a named bicubic method [1].
- **Representation:** the implemented suite measures the cell-mean transfer objective: both engines receive the same source cell averages and are assessed against independently integrated destination cell averages. This is a valid transfer experiment for interpolation methods, but it is not a point-interpolation convergence study. Add a separate point-value reference track before making point-interpolation order claims.
- **Centers:** ESMC receives prepared normalized spherical cell centers; AXIS uses its native center construction. Those conventions can differ. Record this as an implementation difference; do not call the sampled locations identical.
- **Configuration:** double precision, common great-circle geometry, masks, no extrapolation, ignore unmapped, one MPI rank. Use both destination-area and covered-area normalization for compatible conservative cases. ESMC version/API restrictions remain explicit [1].

## 3. Horizontal evaluation matrix

- **Smoke profile:** `config/noaa-all-methods-smoke.json`, 66 cases, 264 requested quality/timing worker executions across both engines. Includes constant, smooth, zero, sharp, source masks, destination masks, both conservative normalizations, seam, polar, rectilinear, and Lambert conformal conic geometry.
- **Larger profile:** `config/noaa-all-methods.json`, 252 cases, 1,008 requested quality/timing executions. Expands the existing paper geometry/field inventory across all seven variants; original and triangulated MPAS cases retain adapter restrictions rather than silently disappearing.
- **Mesh coverage:** retain rectilinear, LCC, and MPAS fixtures. LCC means Lambert conformal conic, not the ocean latitude–longitude–cap (LLC) mesh. The paper preparer still needs an explicit regional LLC fixture adapter before claiming LLC coverage in this suite.
- **Refinement and direction:** publish coarse-to-fine and fine-to-coarse errors at multiple mesh sizes; retain original versus shared triangulated representation as separate experiments. Original variable-arity MPAS is currently unsupported by the ESMC TRI/QUAD adapter.
- **Fields:** smooth analytic gradients/waves for convergence; zero crossings for relative-error handling; sharp interfaces for ringing; constant fields for reproduction and partial-coverage normalization. Retain independently integrated references and their uncertainty evidence.

## 4. Metrics and acceptance

- **Accuracy:** area-weighted RMS, mean absolute error, bias, maximum absolute error; relative metrics with explicit near-zero exclusions. Use `paired-quality.json` for comparisons on the intersection of both engines' mapped wet populations, and publish excluded/eligible counts.
- **Horizontal conservation:** first order checks covered-source versus destination integrals; second order checks this identity only when every active source cell is fully covered. For partially covered horizontal source cells, mean-times-covered-area omits the reconstructed slope contribution: report the residual as diagnostic and the conservation check as unavailable until that integral is exposed.
- **Vertical and chemical conservation:** use the operator's covered reconstructed source integral, including partial-layer slope moments, and the destination measure associated with the chosen normalization. Report species mass or moles, carrier participation, and absolute inventory residual; use relative residual only for a nonzero reference inventory. Area-weight per-column content for domain totals. A sum of concentration values is not an inventory check.
- **Interpolation integral drift:** report it as a diagnostic, not a conservation pass requirement. Do not dismiss a valid interpolation method because it does not conserve an area integral.
- **Constants and coverage:** apply an actual constant field to the complete operator, including reconstruction. Compare to one for mapped interpolation cells, or to the appropriate conservative fraction/normalization. Report masks, unmapped cells, unexpected mapping, nonfinite values, and partial reference exclusions.
- **Bounds:** report overshoot/undershoot and extrema for every method; characterize the limited-versus-unlimited tradeoff. A limiter validation should use the intended source reconstruction bounds, not assume every high-order method must be monotone.
- **Timing:** separate generation, first-use finalization, reused apply, and setup-plus-N-applications. Retain raw samples, medians, quartiles, reproducibility checks, thread/rank settings, compiler/library versions, and hashes. Cross-method ratios must retain their comparison labels.
- **Memory and scaling:** currently uninstrumented in this extension; measure isolated peak memory and CPU/GPU/MPI scaling on controlled hardware before adding claims.

## 5. AXIS capability examples

- **Vertical profiles:** uniform and column-varying vertical coordinates; affine profiles at off-knot destination levels; zero and positive tension. Add smooth thermocline/inversion application examples using the corrected natural spline.
- **Conservative vertical tracers:** explicit layer interfaces in volume/thickness or carrier-mass measure; first-order and bounded limited linear reconstruction; masks, collapsed layers, ascending/descending and column-varying coordinates; full and covered destination normalization. Report layer-mean error, coverage, covered reconstructed source inventory, destination inventory, positivity/extrema, and 1,000-round-trip drift. Use volume concentration and mass-mixing-ratio examples separately; geometric depth alone does not establish mass conservation when density varies.
- **Horizontal plus vertical:** demonstrate transfer of carrier and tracer content separately before vertical remapping. Check the area-weighted domain inventory and wet-domain coverage. The eager Python workflow is implemented with integration tests supplied; production nanobind/xarray and real AXIS mesh validation are pending. See `../docs/vertical_conservative.md` for the physical measure contract and limitations.
- **Vector:** analytic Earth-relative vector represented in varying source/destination grid rotations. Validate component error and vector magnitude/direction; then add a spatially varying vector transfer on a curvilinear mesh. AXIS's coupled rotation matrices are not automatically equivalent to ESMF's spherical vector transformation across distinct tangent planes.
- **ESMC vector scope:** a future direct vector worker must use a two-component ungridded field dimension, east/north components, spherical geometry, and an ESMC-supported nonconservative method. The current scalar worker keeps `vectorRegrid=false`; no native ESMC vector run has been implemented here. ESMC documents these vector restrictions [1].
- **Second-order vectors:** current `VectorWeightGenerator` explicitly rejects linear reconstruction. Do not advertise limited or unlimited second-order conservative vector support.
- **Masks:** include source and destination masks in paired horizontal cases, plus AXIS source-exclusion/destination-exclusion tests. Add coastal sentinel-leakage, narrow straits, no-wet-neighbor, and partial-coverage tracer cases before operational conclusions. Source values in land cells must not leak into wet destinations.

### 5.1 Vertical coordinates and physical measure

- **Implemented scope:** the native overlap operator accepts explicit finite, monotone interfaces, shared or varying by column, with either orientation. It operates in the supplied common cumulative measure. It does not identify coordinate types, parse CF `formula_terms`, infer density, or perform automatic coordinate conversion.
- **Interface requirement:** evaluate coordinate formulas at layer boundaries, not only at level centers. Resolve source and destination coordinates into a common physical measure before building overlaps. Surface pressure, terrain, density, and geometry changes can require rebuilding the cached operator.

| Input coordinate | Preparation for conservative transfer | Conservation interpretation |
|---|---|---|
| Pressure | Use pressure interfaces, or cumulative air mass per area | Under hydrostatic balance and constant gravity, layer total-air mass per area is `abs(Δp)/g`; mole fractions additionally need the appropriate air moles [6] |
| Log pressure | Recover physical pressure using the dataset's logarithm convention and reference pressure | Direct log-pressure overlaps preserve an integral in `d(log p)`, generally not species mass |
| Atmospheric sigma | Evaluate `p = p_top + sigma * (p_surface - p_top)` at interfaces [5] | Then construct pressure, mass, or mole measure as required by the tracer |
| Hybrid sigma-pressure | Evaluate `p = a*p0 + b*p_surface` or `p = ap + b*p_surface` per column [5] | Distinguish dimensionless `a` from pressure-valued `ap`; use interface coefficients |
| Altitude / height | Use physical height interfaces for volume concentration; integrate density for mass measure | Thickness conserves volume-based inventory per horizontal area; mass mixing ratios require carrier mass |
| Ocean sigma | Evaluate `z = eta + sigma*(depth + eta)` at interfaces [5] | Use physical thickness/volume, or density-weighted mass as appropriate |

- **Point profiles versus layer means:** log-pressure interpolation of point samples remains a distinct use case from finite-volume transfer. Changing the reconstruction coordinate changes the within-layer representation; merely supplying log-pressure interfaces does not make the overlap integral mass conservative.
- **Status:** the engine can consume these converted interfaces. Coordinate-specific convenience helpers, CF metadata interpretation, and named-coordinate application tests are proposed additions, not completed validation.

### 5.2 Chemical tracer units: ppmv and µg/m³

- **Physical distinction:** gas ppmv is a scaled mole fraction under the ideal-gas interpretation; µg/m³ is species mass per gas volume. Conversion depends on species molar mass, temperature, and pressure [7]. Use the corresponding layer mean and carrier amount, including its dry/moist or reference-volume convention.

| Input representation | Layer carrier per horizontal area | Species content per horizontal area |
|---|---|---|
| ppmv, dry-air basis | Dry-air moles `n_dry/A` | Species moles: `ppmv * 1e-6 * n_dry/A` |
| ppmv, moist-air basis | Total moist-air moles `n_moist/A` | Species moles: `ppmv * 1e-6 * n_moist/A` |
| µg/m³, actual-volume basis | Actual air volume `V/A` | Species mass: `C_ug * V/A`, in µg per m² |
| kg/kg mass mixing ratio | Air mass per area on the stated dry/moist basis | Species mass: mixing ratio times carrier mass per area |

- **Conservative workflow:** form carrier and species content on the source grid, transfer both horizontally with conservative destination-area weights, then remap vertically in the matching carrier measure. Recover destination concentration by division. Convert between species moles and mass using the species molar mass when comparing inventories across unit systems. The current generic helper can accept externally prepared mole or volume carriers; it does not inspect units or prepare these carriers automatically.
- **Ideal-gas conversion:** for gas-phase species, moist-air-basis ppmv and actual-volume concentration satisfy the following local relation, derived from the ideal gas law and the unit definitions [7, 8]:

  `C_ug_per_m3 = ppmv * p * M / (R * T)`

  Here `p` is absolute pressure in Pa, `T` is temperature in K, `M` is species molar mass in g/mol, and `R` is the universal gas constant in Pa·m³/(mol·K). The numerical ppmv factor `1e-6` and g-to-µg factor `1e6` cancel. If `M` is supplied in kg/mol, an additional factor of `1000` is required in this numerical formula.
- **Dry versus moist air:** for dry-air-basis ppmv converted to mass per actual moist-air volume, replace `p` by dry-air partial pressure `p - e`, where `e` is water-vapor partial pressure. Equivalently, `chi_moist = chi_dry * (1 - x_water)`, with water mole fraction `x_water = e/p`. The carrier must follow the same basis; total hydrostatic air mass alone does not establish a dry-air mole inventory. Moisture corrections are explicit reporting conventions [8].
- **Actual versus reference volume:** concentrations reported per standard/normal cubic meter require their declared reference temperature, pressure, and moisture convention. Convert to a consistent species content before transfer and reconstruct the requested output convention afterward. Do not assume a universal standard temperature or apply reference-condition concentration directly to actual layer volume.
- **Layer averaging:** multiplying a layer-mean mole fraction by a single midpoint `p/T` is not generally equivalent to integrating species content when conditions vary within the layer. Use known layer carrier amounts and a consistent reconstruction or quadrature. Different reconstruction measures can produce different local destination values while preserving the same total inventory.
- **Gas versus aerosol:** ppmv conversion requires a defined gas-phase species and molar mass. Bulk particulate matter in µg/m³ can be transferred as mass concentration using volume carriers; a generic PM mixture has no automatic ppmv interpretation.
- **Metadata and status:** retain species identity, molar-mass units, concentration units, dry/moist basis, volume reference conditions, carrier units, and the coordinate conversion used. Automatic unit-aware conversion and ambiguity checks are proposed; they are not present in the delivered API. No new dependency is needed for explicitly supplied numeric carrier/interface arrays.

### 5.3 Proposed coordinate and chemistry validation

- **Equivalent-coordinate cases:** construct one physical column represented through pressure, log pressure, sigma, hybrid sigma-pressure, and height/density. Convert each to the same carrier interfaces and check identical native remapping, coverage, and species inventory. Include terrain-varying surface pressure, both hybrid coefficient conventions, descending storage, and collapsed bottom layers.
- **Dual-unit cases:** represent a defined gas species in ppmv and µg/m³ under the same declared thermodynamic and moisture conditions. Compare source and destination inventories in common species-mass units. Require identical destination fields only when the carrier and reconstructed representation are also identical.
- **Thermodynamic and moisture cases:** vary temperature, pressure, and humidity by column/layer; exercise dry/moist ppmv and actual/reference-volume concentrations. Include analytic variable-state layer integrals to detect midpoint conversion errors and explicit unit-scale checks for Pa/hPa, K/°C, and g/mol versus kg/mol.
- **Failure and boundary cases:** reject missing species molar mass or reference conditions where required; invalid absolute temperature/pressure; inconsistent dry/moist basis; and insufficient wet-domain coverage. Test masked NaN land values and domain stock after repeated horizontal/vertical transfers.
- **Metrics:** publish species inventory residual, carrier inventory residual, destination concentration error against an independent reference, positivity/extrema, source/destination coverage, and conversion round-trip error. Separate coordinate/unit preprocessing cost from operator setup and reused application time.
- **Evidence boundary:** these cases are planned experiments. The native vertical results below establish the generic numerical operator's behavior, not coordinate-specific atmospheric conversion accuracy or a validated chemistry interface.

## 6. New evidence collected in this workspace

- **Scalar pilot:** 28/28 consistency checks passed: seven variants × constant/smooth × unmasked/source-masked. Uses a small spherical gnomonic triangular patch, a 4-to-7 refinement, actual AXIS operators, and independent cell-mean reference geometry. Pass means eligible output exists, finite error, and constant-response error below `1e-9`; it is not a full convergence or conservation certification.
- **Unmasked smooth pilot RMS:** nearest `3.9655313020e-4`; bilinear `4.6991383027e-4`; bicubic `5.2637415863e-5`; patch `5.2606505993e-5`; first order `3.5833593253e-4`; unlimited second order `1.2746508658e-4`; limited second order `2.7787769735e-4`. These are synthetic AXIS-only cell-mean-transfer errors, not native ESMC results or general rankings.
- **Existing capabilities:** all nine existing mask/vector/vertical tests passed. The new varying-grid rotation example has maximum component error `4.4408920985e-16`.
- **Resolved vertical defect:** all four new off-knot affine-profile examples failed the `1e-12` tolerance. Errors: uniform tension 0 `0.0109329446064`; uniform tension 2 `0.0401675289633`; varying tension 0 `0.0147594752187`; varying tension 2 `0.0542261641004`. Existing vertical tests sample source knots and therefore do not detect this issue. Before the fix, source inspection showed curvature-like coefficients passed to a first-derivative evaluator and positive tension unused in the solver. The vertical fix now uses consistent natural tension-spline coefficients and evaluation. All four off-knot examples pass with maximum error `1.7763568394e-15`; 11 vertical tests pass with one and four CPU threads, including independent cubic/positive-tension references, endpoint slopes, nonuniform/descending/varying levels, two-level columns, large/tiny tension, and smooth refinement.
- **Harness validation:** five Python routing/population tests passed; modified AXIS and metrics workers compiled as objects with a declaration-only NetCDF shim. This checks C++ syntax/types, not NetCDF linking or real worker I/O. ESMC worker compilation and execution remain pending.
- **Conservative vertical evidence:** eleven native tests pass with one and four CPU threads, and all 16 volume/mass/partial/masked paper cases pass. Inventory drift after 1,000 round trips is `4.66294e-15` first order and `1.4877e-14` limited linear (initial inventory `0.7`). Quadratic layer-mean RMS at 8/16/32 layers is `0.0241643 / 0.0117135 / 0.00577675` first order and `0.0194536 / 0.0072808 / 0.00266166` limited linear. This specific refinement test demonstrates improvement, not a general second-order rate at flattened boundaries. Five executable-backed Python array-wrapper tests pass; they do not validate production nanobind or xarray integration. No added dependencies, measured runtime comparison, native ESMC vertical comparison, or GPU validation is claimed.
- **Coordinate and chemistry status:** pressure/sigma/hybrid/height preparation and ppmv/µg/m³ conventions are specified in Sections 5.1–5.3. Their conversion helpers, metadata validation, and proposed atmospheric chemistry experiments have not been implemented or run. Generic overlap-engine support must not be presented as completed end-to-end support for each named coordinate or unit convention.
- **Execution limitation:** native ESMF, MPI, NetCDF and the Python paper I/O dependencies are unavailable here. The 66/252-case ESMC suites have been prepared, not executed. No speedup or ESMC accuracy claim is made from this extension.

## 7. Proposed figures and tables

- **Table 1:** methods, normalization, representation, capability restrictions, direct versus cross-method pairings.
- **Table 2:** per-case shared-population errors, eligible fraction, extrema, integral diagnostics, conservation-check availability, and status.
- **Table 3:** vertical coordinate conversion, carrier measure, concentration-unit/basis metadata, and implementation/validation status; include pressure, log pressure, sigma, hybrid sigma-pressure, altitude, ppmv, and µg/m³.
- **Figure 1:** smooth-field error versus resolution, separating cell-mean transfer and future point-interpolation tracks.
- **Figure 2:** sharp-interface transect and source-mask coastal example; compare first order, unlimited second order, and limited second order.
- **Figure 3:** setup, reused-apply, and end-to-end cost versus mesh size/application count, with raw variability and pairing labels.
- **Figure 4:** vertical profile curves/residuals and vector direction/magnitude example, with the vertical before/after regression evidence reported separately from the application case.
- **Figure 5:** conservative vertical layer means and cumulative tracer inventory for smooth, sharp, masked, and carrier-mass columns; first versus limited linear error under refinement, plus round-trip inventory drift. Keep point-profile interpolation and finite-volume transfer as separate panels.
- **Figure 6 (proposed):** one atmospheric species expressed in ppmv and µg/m³ across variable pressure/temperature/humidity; show recovered concentrations and the common species-mass inventory. Include a concentration-only transfer as a diagnostic example of inventory error, with all reference and moisture conventions stated.
- **Conclusion:** state demonstrated capabilities, specific operational tradeoffs, unresolved numerical issues, and the validation still needed for NOAA deployment.

## References and code evidence

- [1] ESMF Team. *ESMC Reference Manual*, method constants and `ESMC_FieldRegridStore`, including vector/fraction restrictions. https://earthsystemmodeling.org/docs/release/latest/ESMC_crefdoc/node5.html (accessed 2026-10-10; archive the tested ESMF version's manual for final publication).
- [2] ESMF Team. *Regridding overview*. https://earthsystemmodeling.org/regrid/ . Describes operator generation and application and version-dependent support tables.
- [3] MOM6 Team. *Vertical Reconstruction* and *ALE*. https://mom6.readthedocs.io/en/main/api/generated/pages/Vertical_Reconstruction.html and https://mom6.readthedocs.io/en/main/api/generated/pages/ALE.html . Background on reconstructing layer means and vertical remapping; AXIS implementation is independent.
- [4] *Truly conserving with conservative remapping methods*. GMD 17 (2024). https://gmd.copernicus.org/articles/17/415/2024/ . Background for explicitly reporting physical measure, normalization, and inventory diagnostics.
- [5] CF Conventions. *Appendix D: Dimensionless Vertical Coordinates* (version 1.7), atmosphere sigma, hybrid sigma-pressure, and ocean sigma definitions. https://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/apd.html (accessed 2026-10-10). Formulas must be evaluated at interfaces for layer remapping.
- [6] NOAA/NWS. *Temperature Gradients, Heights and Thicknesses*. https://www.weather.gov/source/zhu/ZHU_Training_Page/Miscellaneous/Heights_Thicknesses/thickness_temperature.htm (accessed 2026-10-10). Hydrostatic balance; the layer-mass relation in Section 5.1 is derived under the stated constant-gravity assumption.
- [7] US EPA. *On-line Tools for Site Assessment: Indoor Air Unit Conversion*. https://archive.epa.gov/athens/archive-athens/web/html/ia_unit_conversion.html (accessed 2026-10-10). Gas mole/volume ratios versus mass-per-volume units and dependence on species molar mass, temperature, and pressure.
- [8] Environment Agency. *Technical guidance and equations: pollution inventory reporting*, concentration conversion and moisture corrections, equations 12a–13. https://www.gov.uk/government/publications/pollution-inventory-reporting-general-and-sector-guidance/technical-guidance-and-equations-pollution-inventory-reporting (accessed 2026-10-10). Section 5.2 derives the Pa/K/g-per-mole form rather than copying the guidance's numerical unit convention.
- AXIS implementations: `src/solver/weight_generator.cpp`, `include/axis/detail/linear_geometry.hpp`, `src/solver/vector_regridder.cpp`, `src/solver/vertical_regridder.cpp`, `src/solver/vertical_conservative.cpp`, `python/axis/conservative_vertical.py`, and `src/detail/tspack.cpp`; public solver/configuration headers under `include/axis/solver/`. Use the archived patch/source hashes for the final study's exact code identity.
- This extension's executable evidence: `native/scalar_pilot.cpp`, `native/capability_examples.cpp`, `native/vertical_conservative_examples.cpp`, `../tests/test_vertical_conservative.cpp`, `scripts/methods.py`, `scripts/paired_metrics.py`, and the bundled pilot/test logs. ESMC pairings are encoded in the worker, not inferred from filenames.
