"""Valcke-style mesh, field, percentage-misfit and conservation comparisons.

Read only saved native worker outputs through xarray. Native metrics retain
both the original covered-area errors and the unweighted relative diagnostics
used here. This script adds products to the existing publication manifest.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import xarray as xr

from artifacts import load, save
from publication import COLORS, bundle, export, plt, sidecar
from matplotlib.collections import PolyCollection


FIELDS = ("sinusoid", "harmonic", "vortex", "gulfstream")
FAMILIES = ("rectilinear", "lcc", "mpas")


def dataset(path, group=None):
    with xr.open_dataset(path, group=group) as data:
        return data.load()


def material(run):
    fixture = run["_root"] / run["case"]["path"]
    return (
        dataset(fixture, "source"),
        dataset(fixture, "destination"),
        dataset(run["_folder"] / "fields.nc"),
        dataset(run["_root"] / run["reference"]["path"]),
    )


def geographic_polygons(mesh, values):
    """Unwrap each cell locally, duplicate seam-crossing cells, then axes clip.

    Merely plotting raw MPAS longitude vertices draws a spurious strip across
    the globe. Periodic copies preserve the two visible pieces at the seam.
    These are display polygons; native great-circle geometry is unchanged.
    """
    cells, colors = [], []
    for index, (start, end) in enumerate(zip(mesh.offsets.values[:-1], mesh.offsets.values[1:])):
        nodes = mesh.indices.values[start:end]
        longitude = np.rad2deg(np.unwrap(np.deg2rad(mesh.lon.values[nodes])))
        longitude -= 360 * np.floor((longitude.mean() + 180) / 360)
        polygon = np.column_stack((longitude, mesh.lat.values[nodes]))
        shifts = [0]
        if longitude.min() < -180:
            shifts.append(360)
        if longitude.max() > 180:
            shifts.append(-360)
        for shift in shifts:
            cells.append(polygon + (shift, 0))
            colors.append(values[index])
    return cells, np.ma.masked_invalid(colors)


def limits(mesh):
    if np.ptp(mesh.lon.values) > 300:
        return (-180, 180), (-90, 90)
    return (float(mesh.lon.min()), float(mesh.lon.max())), (
        float(mesh.lat.min()),
        float(mesh.lat.max()),
    )


def map_panel(ax, mesh, values, title, low, high, cmap="viridis", edges=False):
    cells, colors = geographic_polygons(mesh, values)
    artist = PolyCollection(
        cells,
        array=colors,
        cmap=cmap,
        clim=(low, high),
        edgecolors="#303030" if edges else "none",
        linewidths=0.15,
        rasterized=True,
        antialiaseds=edges,
    )
    ax.add_collection(artist)
    longitude, latitude = limits(mesh)
    ax.set(
        xlim=longitude, ylim=latitude, title=title, xlabel="Longitude (°)", ylabel="Latitude (°)"
    )
    ax.set_facecolor("#dddddd")
    ax.tick_params(labelsize=7)
    return artist


def case_label(run):
    family = {"rectilinear": "RLL", "lcc": "LCC", "mpas": "MPAS"}[run["family"]]
    direction = "reverse" if run["case_id"].endswith("reverse") else "forward"
    return f"{family} {direction}\n{run['source_cells']}→{run['destination_cells']}"


def relative_maps(axis_run, esmc_run, output, atlantic_detail=False):
    _, mesh, axis, reference = material(axis_run)
    esmc = dataset(esmc_run["_folder"] / "fields.nc")
    valid = (mesh["mask"].values != 0) & (reference.dst_covered.values > 0)
    truth = np.where(valid, reference.dst.values, np.nan)
    axis_values = np.where(valid & (axis.dst_frac.values > 0), axis.dst.values, np.nan)
    esmc_values = np.where(valid & (esmc.dst_frac.values > 0), esmc.dst.values, np.nan)
    relative_valid = valid & (np.abs(truth) > 5e-12)
    axis_error = np.full(truth.shape, np.nan)
    esmc_error = axis_error.copy()
    np.divide(100 * (axis_values - truth), np.abs(truth), out=axis_error, where=relative_valid)
    np.divide(100 * (esmc_values - truth), np.abs(truth), out=esmc_error, where=relative_valid)
    magnitude = max(np.nanmax(np.abs(axis_error)), np.nanmax(np.abs(esmc_error)), 1e-12)
    low = min(np.nanmin(truth), np.nanmin(axis_values), np.nanmin(esmc_values))
    high = max(np.nanmax(truth), np.nanmax(axis_values), np.nanmax(esmc_values))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.4), layout="constrained")
    for column, (values, label) in enumerate(
        ((truth, "Reference"), (axis_values, "AXIS"), (esmc_values, "ESMF"))
    ):
        field_artist = map_panel(axes[0, column], mesh, values, label, low, high)
    coverage = np.where(
        mesh["mask"].values != 0, reference.dst_covered.values / reference.dst_area.values, np.nan
    )
    coverage_artist = map_panel(axes[1, 0], mesh, coverage, "Independent covered fraction", 0, 1)
    for column, (values, label) in enumerate(
        ((axis_error, "AXIS misfit (%)"), (esmc_error, "ESMF misfit (%)")), 1
    ):
        error_artist = map_panel(
            axes[1, column], mesh, values, label, -magnitude, magnitude, "RdBu_r"
        )
    fig.colorbar(
        field_artist,
        ax=list(axes[0]),
        orientation="horizontal",
        fraction=0.06,
        pad=0.06,
        label="Cell-average field value (shared scale)",
    )
    fig.colorbar(coverage_artist, ax=axes[1, 0], orientation="horizontal", fraction=0.06, pad=0.06)
    fig.colorbar(
        error_artist,
        ax=list(axes[1, 1:]),
        orientation="horizontal",
        fraction=0.06,
        pad=0.06,
        label="Signed relative misfit (%) (shared scale)",
    )
    title = f"{axis_run['family'].upper()} · {axis_run['field']} · {axis_run['norm']} · {case_label(axis_run).splitlines()[0]}"
    if atlantic_detail:
        for ax in axes.flat:
            ax.set(xlim=(-85, -10), ylim=(20, 65))
        title += " · North Atlantic detail"
    fig.suptitle(title, fontsize=11)
    return export(
        fig,
        output,
        "comparison_map_" + axis_run["case_id"] + ("_atlantic_detail" if atlantic_detail else ""),
        [axis_run, esmc_run],
        "Matched great-circle cells and independently integrated analytical cell averages. "
        "Signed percentage errors use absolute reference denominators; near-zero references, masked and unmapped cells are gray. "
        "AXIS and ESMF share field/error limits. Geographic polygons approximate display edges only; longitude seam copies are clipped to the view.",
    )


def metric_panels(runs, output, field, norm):
    selected = [r for r in runs if r["field"] == field and r["norm"] == norm]
    cases = list(dict.fromkeys(r["case_id"] for r in selected))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), layout="constrained")
    metrics = (
        ("relative_mean_percent", "(a) Mean relative misfit (%)"),
        ("relative_rms_percent", "(b) RMS relative misfit (%)"),
        ("relative_max_percent", "(c) Maximum relative misfit (%)"),
        ("source_conservation_percent", "(d) Source conservation residual (%)"),
    )
    for ax, (key, title) in zip(axes.flat, metrics):
        for engine, marker in (("axis", "o"), ("esmc", "s")):
            values = []
            for case in cases:
                run = next(
                    (r for r in selected if r["case_id"] == case and r["engine"] == engine), None
                )
                values.append(run["metrics"].get(key) if run else np.nan)
            ax.plot(
                range(len(cases)),
                values,
                marker=marker,
                markersize=5,
                linestyle="-" if engine == "axis" else "--",
                color=COLORS[engine],
                label="AXIS" if engine == "axis" else "ESMF",
            )
        # Symlog shows exact zeros without substituting fabricated floor values.
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_title(title)
        ax.set_xticks(
            range(len(cases)),
            [case_label(next(r for r in selected if r["case_id"] == case)) for case in cases],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"First-order conservative · {field} · {norm}", fontsize=12)
    return export(
        fig,
        output,
        f"quality_comparison_{field}_{norm}",
        selected,
        "Valcke-style per-pair library comparison. Mean and RMS use unweighted per-cell absolute relative errors, unlike the separately retained covered-area-weighted dimensional metrics. "
        "Only completed matched quality records appear. Source conservation uses the common covered domain and normalization-correct areas. "
        "Symlog retains true zeros; its linear region is ±1e-12 percent. Near-zero reference exclusions are counted in native metrics.",
    )


def conservation_panels(runs, output, norm, target=False):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), layout="constrained")
    selected = [r for r in runs if r["norm"] == norm]
    key = "target_conservation_percent" if target else "source_conservation_percent"
    for ax, field in zip(axes.flat, FIELDS):
        field_runs = [r for r in selected if r["field"] == field]
        cases = list(dict.fromkeys(r["case_id"] for r in field_runs))
        for engine, marker in (("axis", "o"), ("esmc", "s")):
            records = [
                next(r for r in field_runs if r["case_id"] == case and r["engine"] == engine)
                for case in cases
            ]
            ax.plot(
                range(len(cases)),
                [r["metrics"][key] for r in records],
                marker=marker,
                color=COLORS[engine],
                linestyle="-" if engine == "axis" else "--",
                label="AXIS" if engine == "axis" else "ESMF",
            )
        ax.set_title(field.capitalize())
        ax.set_yscale("symlog", linthresh=1e-12)
        ax.set_ylabel("Residual (%)")
        ax.set_xticks(
            range(len(cases)),
            [case_label(next(r for r in field_runs if r["case_id"] == case)) for case in cases],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    name = ("target" if target else "source") + "_conservation_" + norm
    fig.suptitle(
        f"{'Analytical target' if target else 'Discrete source'} integral comparison · {norm}"
    )
    return export(
        fig,
        output,
        name,
        selected,
        "Four-field conservation comparison on the common covered domain. Source residual tests discrete integral preservation; "
        "target residual compares against continuous analytical truth and includes representation error. "
        "Both use normalization-correct areas. Symmetric-log axes retain exact zeros with linear region ±1e-12 percent.",
    )


def illustrations(runs, output):
    products = {}
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1), layout="constrained")
    used = []
    for ax, family in zip(axes, FAMILIES):
        run = next(
            r
            for r in runs
            if r["family"] == family and r["engine"] == "axis" and r["case_id"].endswith("forward")
        )
        source, _, data, _ = material(run)
        map_panel(
            ax,
            source,
            np.full(data.src.size, 0.05),
            f"{family.upper()} · {data.src.size:,} elements",
            0,
            1,
            "Greys",
            edges=True,
        )
        used.append(run)
    products["mesh_gallery"] = export(
        fig,
        output,
        "mesh_gallery",
        used,
        "Actual source connectivity of the added experiments, including explicit rectilinear/LCC triangles and shared MPAS centroid fans. No coastline or geographic boundary is a numerical mask.",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.5), layout="constrained")
    used = []
    for ax, field in zip(axes.flat, FIELDS):
        run = next(
            r
            for r in runs
            if r["family"] == "mpas"
            and r["engine"] == "axis"
            and r["field"] == field
            and r["case_id"].endswith("forward")
        )
        source, _, data, _ = material(run)
        artist = map_panel(
            ax,
            source,
            data.src.values,
            field.capitalize(),
            float(data.src.min()),
            float(data.src.max()),
        )
        fig.colorbar(
            artist,
            ax=ax,
            orientation="horizontal",
            fraction=0.06,
            pad=0.06,
            label="Source cell average",
        )
        used.append(run)
    products["analytical_fields"] = export(
        fig,
        output,
        "analytical_fields",
        used,
        "Four mathematical field definitions from Valcke et al. (2022), Appendix A, independently integrated over the actual MPAS source triangles. Each panel has its own declared color scale. This is not a replication of the paper's original grids or point sampling.",
    )
    return products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _, all_runs = bundle(args.results)
    runs = [
        r
        for r in all_runs
        if r["mode"] == "quality" and r["matching"] == "matched" and r["field"] in FIELDS
    ]
    if not runs:
        raise ValueError("No saved matched Valcke-field quality runs")
    products = illustrations(runs, output)
    for field in FIELDS:
        for norm in ("dstarea", "fracarea"):
            products[f"quality_comparison_{field}_{norm}"] = metric_panels(
                runs, output, field, norm
            )
        for family in FAMILIES:
            case = f"{family}-{field}-dstarea-forward"
            axis = next(r for r in runs if r["case_id"] == case and r["engine"] == "axis")
            esmc = next(r for r in runs if r["case_id"] == case and r["engine"] == "esmc")
            products["comparison_map_" + case] = relative_maps(axis, esmc, output)
    for norm in ("dstarea", "fracarea"):
        for target in (False, True):
            name = ("target" if target else "source") + "_conservation_" + norm
            products[name] = conservation_panels(runs, output, norm, target)
    case = "mpas-gulfstream-dstarea-forward"
    axis = next(r for r in runs if r["case_id"] == case and r["engine"] == "axis")
    esmc = next(r for r in runs if r["case_id"] == case and r["engine"] == "esmc")
    products["gulfstream_atlantic_detail"] = relative_maps(axis, esmc, output, atlantic_detail=True)
    # Full numerical data behind all panels, including both directions/norms.
    with (output / "comparison_metrics.csv").open("w", newline="") as stream:
        columns = ["case_id", "engine", "field", "norm", "source_cells", "destination_cells"]
        metric_keys = [
            "relative_mean_percent",
            "relative_rms_percent",
            "relative_max_percent",
            "source_conservation_percent",
            "target_conservation_percent",
            "lmin",
            "lmax",
            "relative_eligible",
            "relative_excluded_near_zero",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns + metric_keys)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {**{k: run[k] for k in columns}, **{k: run["metrics"].get(k) for k in metric_keys}}
            )
    sidecar(
        output / "comparison_metrics.csv",
        runs,
        "Native unweighted relative misfits, covered-domain conservation and extrema diagnostics behind the comparison panels.",
    )
    products["comparison_metrics"] = ["comparison_metrics.csv"]
    manifest_path = output / "artifacts.json"
    manifest = (
        load(manifest_path)
        if manifest_path.exists()
        else {
            "schema_version": 1,
            "products": {},
            "omissions": {},
            "missing_required": [],
            "status": "complete",
        }
    )
    manifest["products"].update(products)
    save(manifest_path, manifest)
    print(f"Generated {len(products) - 1} comparison figures in PDF/SVG/PNG")


if __name__ == "__main__":
    main()
