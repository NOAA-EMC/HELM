"""Export pole-centered views and matched native polar quality comparisons."""

import argparse
import csv
from pathlib import Path

import numpy as np

from artifacts import load, save
from publication import COLORS, bundle, export, plt, sidecar
from matplotlib.collections import PolyCollection
from make_comparison_figures import material, dataset


def polar_panel(axis, mesh, values, title, low, high, cmap="viridis", edges=False):
    """Use azimuthal equidistant display coordinates, with the pole at the origin."""
    radius = 90 - mesh.lat.values
    angle = np.deg2rad(mesh.lon.values)
    coordinates = np.column_stack((radius * np.sin(angle), -radius * np.cos(angle)))
    polygons = [
        coordinates[mesh.indices.values[start:end]]
        for start, end in zip(mesh.offsets.values[:-1], mesh.offsets.values[1:])
    ]
    artist = PolyCollection(
        polygons,
        array=np.ma.masked_invalid(values),
        cmap=cmap,
        clim=(low, high),
        edgecolors="#444444" if edges else "none",
        linewidths=0.2,
        rasterized=True,
        antialiaseds=edges,
    )
    axis.add_collection(artist)
    for latitude in (60, 70, 80):
        axis.add_patch(plt.Circle((0, 0), 90 - latitude, fill=False, color="gray", lw=0.4))
        axis.text(
            (90 - latitude) / np.sqrt(2),
            (90 - latitude) / np.sqrt(2),
            f"{latitude}°N",
            fontsize=6,
            color="#555555",
        )
    axis.plot(0, 0, "+", color="black", markersize=4)
    axis.set(xlim=(-32, 32), ylim=(-32, 32), aspect="equal", title=title)
    axis.set_xticks([])
    axis.set_yticks([])
    for x, y, text in ((0, -31, "0°"), (31, 0, "90°E"), (0, 31, "180°"), (-31, 0, "90°W")):
        axis.text(x, y, text, ha="center", va="center", fontsize=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    return artist


def spatial_comparison(axis_run, esmc_run, output, name=None):
    _, mesh, axis_data, reference = material(axis_run)
    esmc_data = dataset(esmc_run["_folder"] / "fields.nc")
    valid = reference.dst_covered.values > 0
    truth = np.where(valid, reference.dst.values, np.nan)
    values = [
        truth,
        np.where(valid, axis_data.dst.values, np.nan),
        np.where(valid, esmc_data.dst.values, np.nan),
    ]
    low = min(np.nanmin(value) for value in values)
    high = max(np.nanmax(value) for value in values)
    errors = []
    for value in values[1:]:
        error = np.full(truth.shape, np.nan)
        np.divide(
            100 * (value - truth), np.abs(truth), out=error, where=valid & (np.abs(truth) > 5e-12)
        )
        errors.append(error)
    magnitude = max(*(np.nanmax(np.abs(error)) for error in errors), 1e-12)
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7), layout="constrained")
    for column, (value, title) in enumerate(zip(values, ("Reference", "AXIS", "ESMF"))):
        field_artist = polar_panel(axes[0, column], mesh, value, title, low, high)
    coverage = reference.dst_covered.values / reference.dst_area.values
    coverage_artist = polar_panel(axes[1, 0], mesh, coverage, "Covered fraction", 0, 1)
    for column, (error, title) in enumerate(zip(errors, ("AXIS misfit (%)", "ESMF misfit (%)")), 1):
        error_artist = polar_panel(
            axes[1, column], mesh, error, title, -magnitude, magnitude, "RdBu_r"
        )
    for artist, selected_axes, label in (
        (field_artist, list(axes[0]), "Cell-average value (shared scale)"),
        (coverage_artist, axes[1, 0], "Covered fraction"),
        (error_artist, list(axes[1, 1:]), "Signed relative misfit (%) (shared scale)"),
    ):
        fig.colorbar(
            artist, ax=selected_axes, orientation="horizontal", fraction=0.05, pad=0.04, label=label
        )
    fig.suptitle(
        f"North Polar cap · {axis_run['field']} · {axis_run['norm']} · {axis_run['source_cells']}→{axis_run['destination_cells']} cells"
    )
    return export(
        fig,
        output,
        name or "polar_map_" + axis_run["case_id"],
        [axis_run, esmc_run],
        "North Polar cap including one shared pole node per mesh. Independent native cell-average reference, matched first-order conservative AXIS/ESMC results and signed percentage errors. Pole-centered azimuthal equidistant display only; native edges are great circles. Field and error scales are shared between engines. Gray indicates no eligible reference. Rotated outer great-circle boundaries give partial coverage.",
    )


def quality_comparison(runs, output, field, norm):
    selected = [run for run in runs if run["field"] == field and run["norm"] == norm]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), layout="constrained")
    quantities = (
        ("relative_mean_percent", "Mean relative misfit (%)"),
        ("relative_rms_percent", "RMS relative misfit (%)"),
        ("relative_max_percent", "Maximum relative misfit (%)"),
        ("source_conservation_percent", "Source conservation residual (%)"),
    )
    for axis, (key, title) in zip(axes.flat, quantities):
        for engine, marker in (("axis", "o"), ("esmc", "s")):
            records = [
                next(
                    run
                    for run in selected
                    if run["engine"] == engine and run["case_id"].endswith(direction)
                )
                for direction in ("forward", "reverse")
            ]
            axis.plot(
                [0, 1],
                [run["metrics"][key] for run in records],
                marker=marker,
                color=COLORS[engine],
                linestyle="-" if engine == "axis" else "--",
                label="AXIS" if engine == "axis" else "ESMF",
            )
        maximum = max(run["metrics"][key] for run in runs if run["field"] == field)
        axis.set_ylim(0, max(1.2 * maximum, 1e-14))
        if key == "source_conservation_percent":
            axis.set_yscale("symlog", linthresh=1e-12)
            axis.set_ylim(0, max(2 * maximum, 1e-12))
        axis.set_title(title)
        axis.set_xticks([0, 1], ["Forward\n360→1,488", "Reverse\n1,488→360"])
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle(f"North Polar cap · {field} · {norm}")
    return export(
        fig,
        output,
        f"polar_quality_{field}_{norm}",
        selected,
        "Matched native polar quality measurements. Mean/RMS/maximum use unweighted absolute percentage errors. Source conservation integrates over common coverage with normalization-correct areas. For each field, DstArea and FracArea panels use identical vertical limits, including the source-conservation scale. Tiny residual differences do not imply a practical accuracy advantage.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _, records = bundle(args.results)
    runs = [run for run in records if run["mode"] == "quality"]
    if len(runs) != 24 or any(
        run["family"] != "polar" or run["matching"] != "matched" for run in runs
    ):
        raise ValueError("Expected 12 matched polar AXIS/ESMC quality pairs")
    products = {}
    first = next(
        run
        for run in runs
        if run["engine"] == "axis" and run["case_id"] == "polar-constant-dstarea-forward"
    )
    source, destination, _, _ = material(first)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6), layout="constrained")
    for axis, mesh, title in (
        (axes[0], source, "Source · 360 cells"),
        (axes[1], destination, "Destination · 1,488 cells · rotated 3.75°"),
    ):
        polar_panel(axis, mesh, np.full(mesh.area.size, 0.1), title, 0, 1, "Greys", edges=True)
    products["polar_meshes"] = export(
        fig,
        output,
        "polar_meshes",
        [first],
        "Actual shared native connectivity includes periodic longitude rings and a single pole node. North Polar cap starts at 60°N; outer edges are great-circle segments. Azimuthal equidistant display uses straight chords for visualization only.",
    )
    for field in ("constant", "smooth", "harmonic"):
        for norm in ("dstarea", "fracarea"):
            products[f"polar_quality_{field}_{norm}"] = quality_comparison(
                runs, output, field, norm
            )
            if field != "constant":
                for direction in ("forward", "reverse"):
                    case_id = f"polar-{field}-{norm}-{direction}"
                    axis = next(
                        run for run in runs if run["case_id"] == case_id and run["engine"] == "axis"
                    )
                    esmc = next(
                        run for run in runs if run["case_id"] == case_id and run["engine"] == "esmc"
                    )
                    products["polar_map_" + case_id] = spatial_comparison(axis, esmc, output)
    # Replace the generic longitude/latitude display with a pole-centered
    # required field map, avoiding the longitude singularity at the pole.
    case_id = "polar-smooth-dstarea-forward"
    axis = next(run for run in runs if run["case_id"] == case_id and run["engine"] == "axis")
    esmc = next(run for run in runs if run["case_id"] == case_id and run["engine"] == "esmc")
    products["field_maps"] = spatial_comparison(axis, esmc, output, name="field_maps")
    columns = [
        "case_id",
        "engine",
        "field",
        "norm",
        "rms_error",
        "relative_mean_percent",
        "relative_rms_percent",
        "relative_max_percent",
        "source_conservation_percent",
        "target_conservation_percent",
    ]
    path = output / "polar_metrics.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run.get(key, run["metrics"].get(key)) for key in columns})
    sidecar(path, runs, "Full-precision native metrics for every matched polar quality record.")
    products["polar_metrics"] = [path.name]
    manifest_path = output / "artifacts.json"
    manifest = load(manifest_path)
    manifest["products"].update(products)
    save(manifest_path, manifest)
    print("Generated 15 polar comparison figures and full-precision CSV")


if __name__ == "__main__":
    main()
