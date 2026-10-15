"""Reproduce paper figures exclusively from saved native results."""

import argparse
from pathlib import Path
import numpy as np
import xarray as xr
from publication import bundle, export, finish, plt, COLORS
from matplotlib.collections import PolyCollection
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter


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
    """Compare all method families at one common, largest rectilinear size."""
    selected = [
        run for run in runs
        if run["family"] == "rectilinear"
        and run["field"] == "smooth"
        and run["case_id"].startswith("rect-32-smooth-forward-")
    ]
    methods = (
        "nearest", "bilinear", "bicubic", "patch", "conservative_first_order",
        "conservative_second_order", "conservative_second_order_limited",
    )
    labels = (
        "Nearest neighbor", "Bilinear", "Bicubic", "Patch recovery", "First order",
        "Second order, unlimited", "Second order, limited",
    )
    lookup = {(run["options"]["method"], run["engine"]): run for run in selected}
    if any((method, engine) not in lookup for method in methods for engine in COLORS):
        raise ValueError("Missing timing records for the common rectilinear method comparison")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 4.0), sharey=True, layout="constrained")
    keys = ("setup:1", "apply:1", "setup_and_reuse:100")
    titles = ("Weight setup", "One apply", "Setup + 100 applies")
    used = []
    for column, (ax, key, title) in enumerate(zip(axes, keys, titles)):
        for index, (method, label) in enumerate(zip(methods, labels)):
            for engine, offset in (("axis", -0.14), ("esmc", 0.14)):
                run = lookup[(method, engine)]
                sample = run["timing"][key]
                ax.errorbar(
                    sample["median"], index + offset,
                    xerr=[[sample["median"] - sample["q25"]], [sample["q75"] - sample["median"]]],
                    fmt="o", color=COLORS[engine], markersize=4, capsize=2,
                    label=engine.upper() if index == 0 else None,
                )
                used.append(run)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=True))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title)
        ax.set_xlabel("Seconds (median and IQR)")
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_yticks(range(len(labels)), labels)
    axes[0].set_ylabel("Method")
    axes[0].legend(fontsize=7)
    return export(
        fig,
        output,
        "performance",
        used,
        "Native median and interquartile range for the smooth rectilinear forward case at source resolution 32 (2,048 source cells) and destination resolution 64 (8,192 cells). "
        "Panels show setup, one application, and setup plus 100 repeated applications. Bicubic is paired with ESMC PATCH; limited second order is paired with ESMC's unlimited second-order method.",
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
    # Keep one controlled rectilinear coarse-to-fine direction in the display.
    # For ESMC, collapse duplicate case pairings onto the native method family.
    groups = {"axis": {}, "esmc": {}}
    for run in quality:
        definition = run.get("case_definition", {})
        source, destination = definition.get("source", {}), definition.get("destination", {})
        if (
            run["field"] != "smooth"
            or run["family"] != "rectilinear"
            or run["norm"] != "dstarea"
            or run["geometry"] != "great_circle"
            or run["metrics"]["numerical_validation"] != "pass"
            or not run.get("reference_precision_verified")
            or source.get("n") not in {8, 16, 32}
            or destination.get("n") != 2 * source.get("n", 0)
            or "reverse" in run["case_id"]
        ):
            continue
        method = run["method_pairing"]["esmc_method"] if run["engine"] == "esmc" else definition["method"]
        groups[run["engine"]].setdefault(method, {}).setdefault(source["n"], run)
    method_labels = {
        "nearest": "Nearest neighbor",
        "bilinear": "Bilinear",
        "bicubic": "Bicubic",
        "patch": "Patch recovery",
        "conservative_first_order": "Conservative, first order",
        "conservative_second_order": "Conservative, second order",
        "conservative_second_order_limited": "Conservative, limited second order",
    }
    methods = (
        "nearest", "bilinear", "bicubic", "patch", "conservative_first_order",
        "conservative_second_order", "conservative_second_order_limited",
    )
    ladders = {engine: [] for engine in groups}
    for engine, by_method in groups.items():
        for method in methods:
            records = by_method.get(method, {})
            if set(records) != {8, 16, 32}:
                continue
            ladder = [records[size] for size in (8, 16, 32)]
            if any(
                run["metrics"]["rms_error"] <= 0
                or run["reference_evidence"]["max_uncertainty"] >= 0.01 * run["metrics"]["rms_error"]
                for run in ladder
            ):
                continue
            ladders[engine].append((method, ladder))
    if all(ladders.values()):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.5), sharey=True, layout="constrained")
        method_colors = {method: plt.get_cmap("tab10")(index) for index, method in enumerate(methods)}
        used = []
        for ax, engine in zip(axes, ("axis", "esmc")):
            for method, records in ladders[engine]:
                spacing = np.array([run["source_cells"] ** -0.5 for run in records])
                errors = np.array([run["metrics"]["rms_error"] for run in records])
                slope = float(np.polyfit(np.log(spacing), np.log(errors), 1)[0])
                label = f"{method_labels[method]} ({slope:.2f})"
                ax.loglog(
                    spacing, errors, marker="o", linewidth=1.2,
                    color=method_colors[method], label=label,
                )
                used.extend(records)
            ax.set_title(engine.upper())
            ax.set_xlabel("Nominal source spacing, N⁻¹ᐟ²")
            ax.grid(True, which="both", color="#dddddd", linewidth=0.55)
            ax.legend(fontsize=6.0, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        axes[0].set_ylabel("Covered-area-weighted RMS error")
        products["convergence"] = export(
            fig,
            out,
            "convergence",
            used,
            "Smooth analytic cell-mean transfer on matched rectilinear forward refinements (8→16, 16→32, and 32→64 grid dimensions). "
            "Each legend gives the empirical slope from these three levels; slopes are descriptive, not formal asymptotic order proofs. "
            "ESMC bicubic pairings use PATCH and its two second-order pairings collapse to the same ESMC second-order method.",
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
