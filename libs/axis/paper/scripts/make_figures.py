"""Reproduce paper figures exclusively from saved native results."""

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from publication import bundle, export, finish, plt, COLORS
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from artifacts import identity


def polygons(mesh):
    lon = mesh.lon.values
    lat = mesh.lat.values
    indices = mesh.indices.values
    offsets = mesh.offsets.values
    cells = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        k = indices[a:b]
        x = np.rad2deg(np.unwrap(np.deg2rad(lon[k])))
        cells.append(np.column_stack((x, lat[k])))
    return cells


def draw(ax, cells, values, title, vmin, vmax, cmap="viridis"):
    pc = PolyCollection(
        cells,
        array=np.ma.masked_invalid(values),
        cmap=cmap,
        clim=(vmin, vmax),
        edgecolors="none",
        rasterized=True,
    )
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_facecolor("#dddddd")
    return pc


def performance_display(runs, output):
    """Facet costs by family; group engine bars so print labels remain legible."""
    families = sorted({run["family"] for run in runs})
    fig, axes = plt.subplots(
        len(families), 3, figsize=(7.1, 2.6 * len(families)), squeeze=False, layout="constrained"
    )
    used = []
    for row, family in enumerate(families):
        candidates = [run for run in runs if run["family"] == family]
        field = (
            "constant"
            if any(run["field"] == "constant" for run in candidates)
            else candidates[0]["field"]
        )
        selected = [run for run in candidates if run["field"] == field]
        case_ids = list(dict.fromkeys(run["case_id"] for run in selected))
        labels = []
        for case_index, case_id in enumerate(case_ids):
            case_runs = [run for run in selected if run["case_id"] == case_id]
            first = case_runs[0]
            labels.append(
                f"{first['source_cells']}→{first['destination_cells']}\n{first['norm']}"
                + (" *" if any(run["matching"] != "matched" for run in case_runs) else "")
            )
            for run in case_runs:
                position = case_index + (-0.18 if run["engine"] == "axis" else 0.18)
                for column, key in enumerate(("setup:1", "apply:1")):
                    sample = run["timing"][key]
                    axes[row, column].bar(
                        position,
                        sample["median"],
                        width=0.32,
                        color=COLORS[run["engine"]],
                        yerr=[
                            [sample["median"] - sample["q25"]],
                            [sample["q75"] - sample["median"]],
                        ],
                        capsize=2,
                    )
                for count, marker in ((1, "o"), (100, "s")):
                    sample = run["timing"].get(f"setup_and_reuse:{count}")
                    if sample:
                        axes[row, 2].errorbar(
                            position,
                            sample["median"],
                            fmt=marker,
                            color=COLORS[run["engine"]],
                            markersize=4,
                            yerr=[
                                [sample["median"] - sample["q25"]],
                                [sample["q75"] - sample["median"]],
                            ],
                            capsize=2,
                        )
                used.append(run)
        for column, title in enumerate(
            ("Setup", "One application", "Setup + reuse\nCircles N=1; squares N=100")
        ):
            ax = axes[row, column]
            ax.set_title(title, fontsize=9)
            ax.set_yscale("log")
            ax.set_ylabel(f"{family}\nSeconds" if column == 0 else "Seconds")
            ax.set_xticks(range(len(case_ids)), labels, rotation=90, fontsize=7)
            ax.set_xlabel("Source → destination cells", fontsize=8)
    axes[0, 0].legend(
        handles=[Patch(facecolor=color, label=engine.upper()) for engine, color in COLORS.items()],
        fontsize=7,
    )
    return export(
        fig,
        output,
        "performance",
        used,
        "Native median/IQR on constant fields where available, otherwise the first available field; families are separate rows. "
        "Missing engine bars indicate unsupported cases, not zero cost. Asterisk marks unmatched geometry. "
        "Totals sum measured setup and N reuse intervals on one fresh operator; estimates remain separate in records.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    suite, runs = bundle(a.results)
    products = {}
    omitted = {}
    fig, ax = plt.subplots(figsize=(7.1, 2.5))
    ax.axis("off")
    labels = [
        "Mesh preparation",
        "ArborX candidates",
        "AXIS intersection\nand area weights",
        "Sparse assembly",
        "Repeated field\napplication",
    ]
    for i, text in enumerate(labels):
        ax.text(
            0.1 + i * 0.2,
            0.55,
            text,
            ha="center",
            va="center",
            transform=ax.transAxes,
            bbox={"boxstyle": "round", "facecolor": "#e8f2fa", "edgecolor": "#0072B2"},
            fontsize=8,
        )
        if i < 4:
            ax.annotate(
                "",
                xy=(0.21 + i * 0.2, 0.55),
                xytext=(0.18 + i * 0.2, 0.55),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->"},
            )
    ax.text(
        0.45,
        0.15,
        "Setup: preparation recorded separately; search/geometry/assembly build reusable weights",
        ha="center",
        transform=ax.transAxes,
        fontsize=8,
    )
    products["workflow"] = export(
        fig,
        out,
        "workflow",
        [],
        "AXIS general-mesh workflow; analytical rectangle shortcuts bypass candidate search.",
    )
    quality = [r for r in runs if r["mode"] == "quality"]
    selected = next((r for r in quality if r["engine"] == "axis" and r["field"] == "smooth"), None)
    if selected is None:
        selected = next(
            (r for r in quality if r["engine"] == "axis" and r["field"] != "constant"), None
        )
    if selected:
        root = selected["_root"]
        with xr.open_dataset(root / selected["case"]["path"], group="source") as f:
            srcmesh = f.load()
        with xr.open_dataset(root / selected["case"]["path"], group="destination") as f:
            dstmesh = f.load()
        with xr.open_dataset(selected["_folder"] / "fields.nc") as f:
            data = f.load()
        with xr.open_dataset(root / selected["reference"]["path"]) as f:
            ref = f.load()
        src = data.src.values.copy()
        src[srcmesh["mask"].values == 0] = np.nan
        reference_values = ref.dst.values.copy()
        reference_values[(ref.dst_covered.values <= 0) | (dstmesh["mask"].values == 0)] = np.nan
        dst = data.dst.values.copy()
        dst[data.dst_frac.values <= 0] = np.nan
        error = dst - reference_values
        lo = float(min(np.nanmin(src), np.nanmin(reference_values)))
        hi = float(max(np.nanmax(src), np.nanmax(reference_values)))
        lim = max(float(np.nanmax(np.abs(error))), 1e-15)
        fig = plt.figure(figsize=(7.1, 6.2), layout="constrained")
        grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.055))
        axes = np.array(
            [[fig.add_subplot(grid[row, column]) for column in range(2)] for row in range(2)]
        )
        pc = None
        for ax, mesh, values, title in [
            (axes[0, 0], srcmesh, src, "Source cell averages"),
            (axes[0, 1], dstmesh, reference_values, "Destination reference"),
            (axes[1, 0], dstmesh, dst, "AXIS result"),
        ]:
            pc = draw(ax, polygons(mesh), values, title, lo, hi)
        ec = draw(axes[1, 1], polygons(dstmesh), error, "AXIS minus reference", -lim, lim, "RdBu_r")
        fig.colorbar(pc, cax=fig.add_subplot(grid[0, 2]), label="Field value (shared)")
        fig.colorbar(ec, cax=fig.add_subplot(grid[1, 2]), label="Field difference")
        products["field_maps"] = export(
            fig,
            out,
            "field_maps",
            [selected],
            f"{selected['case_id']}: {selected['matching']}; gray denotes unmapped cells.",
        )
    else:
        omitted["field_maps"] = "No complete smooth-field quality run"
    timing = [run for run in runs if run["mode"] == "timing" and "setup:1" in run["timing"]]
    if timing:
        products["performance"] = performance_display(timing, out)
    else:
        omitted["performance"] = "No completed native timing records"
    # A resolution ladder must keep domains, fields, masks, and refinement
    # direction fixed. Family names alone do not establish compatibility.
    groups = {}
    for run in quality:
        definition = run.get("case_definition", {})
        source, destination = definition.get("source", {}), definition.get("destination", {})
        if not source.get("n") or not destination.get("n"):
            continue
        if (
            run["field"] != "smooth"
            or run["matching"] != "matched"
            or run["metrics"]["numerical_validation"] != "pass"
            or not run.get("reference_precision_verified")
        ):
            continue
        configuration = {
            "engine": run["engine"],
            "family": run["family"],
            "norm": run["norm"],
            "geometry": run["geometry"],
            "source": {key: value for key, value in source.items() if key not in {"n", "m"}},
            "destination": {
                key: value for key, value in destination.items() if key not in {"n", "m"}
            },
            "refinement_ratio": destination["n"] / source["n"],
            "aspect_ratio": source.get("m", source["n"]) / source["n"],
        }
        groups.setdefault(identity(configuration), []).append(run)
    ladders = []
    for records in groups.values():
        if len({run["source_cells"] for run in records}) < 3:
            continue
        if any(
            run["metrics"]["rms_error"] <= 0
            or run["reference_evidence"]["max_uncertainty"] >= 0.01 * run["metrics"]["rms_error"]
            for run in records
        ):
            continue
        ladders.append(sorted(records, key=lambda run: run["source_cells"]))
    if ladders:
        fig, ax = plt.subplots(figsize=(7.1, 3.5))
        used = []
        for ladder_index, records in enumerate(ladders):
            spacing = np.array([run["source_cells"] ** -0.5 for run in records])
            errors = np.array([run["metrics"]["rms_error"] for run in records])
            observed_rate = float(np.polyfit(np.log(spacing), np.log(errors), 1)[0])
            first = records[0]
            ratio = (
                first["case_definition"]["destination"]["n"]
                / first["case_definition"]["source"]["n"]
            )
            label = f"{first['engine']} / {first['family']} / ratio {ratio:g}; observed slope {observed_rate:.2f}"
            ax.loglog(
                spacing,
                errors,
                marker=("o", "s", "^", "D")[ladder_index % 4],
                linestyle="-" if ratio >= 1 else "--",
                color=COLORS[first["engine"]],
                label=label,
            )
            used.extend(records)
        ax.set_xlabel("Nominal source spacing: source cell count to power -1/2")
        ax.set_ylabel("Covered-area-weighted RMS error")
        ax.legend(fontsize=7)
        fig.tight_layout()
        products["convergence"] = export(
            fig,
            out,
            "convergence",
            used,
            "Observed smooth-field trends on compatible domain/direction ladders. Reference precision estimates are below 1% of included errors; slopes are empirical.",
        )
    else:
        omitted["convergence"] = (
            "Fewer than three compatible passing smooth-field resolutions with adequate reference precision"
        )
    stages = [
        run
        for run in runs
        if run["stage_timing"].get("reliable")
        and isinstance(run["stage_timing"].get("value"), dict)
    ]
    if stages:
        fig, ax = plt.subplots(figsize=(7.1, 3.5))
        for index, run in enumerate(stages):
            bottom = 0
            for phase, seconds in run["stage_timing"]["value"].items():
                if not np.isfinite(seconds) or seconds < 0:
                    raise ValueError("Invalid reliable stage observation")
                ax.bar(index, seconds, bottom=bottom, label=phase if index == 0 else None)
                bottom += seconds
        ax.set_xticks(range(len(stages)), [run["run_id"] for run in stages], rotation=80)
        ax.set_ylabel("Instrumented stage seconds")
        ax.legend()
        fig.tight_layout()
        products["stage_breakdown"] = export(
            fig,
            out,
            "stage_breakdown",
            stages,
            "Instrumented stage observations, separate from uninstrumented performance comparisons.",
        )
    else:
        omitted["stage_breakdown"] = (
            "No reliable stage timing instrumentation; no causal attribution"
        )
    return finish(out, products, omitted, ["workflow", "field_maps", "performance"])


if __name__ == "__main__":
    raise SystemExit(main())
