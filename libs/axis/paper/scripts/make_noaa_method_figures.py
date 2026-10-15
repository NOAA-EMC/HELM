"""Create method-aware NOAA paper figures from validated native run bundles."""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from publication import bundle, export, finish, plt, COLORS
from matplotlib.collections import PolyCollection


METHODS = ("nearest", "bilinear", "bicubic", "patch")
CONSERVATIVE = (
    "conservative_first_order",
    "conservative_second_order",
    "conservative_second_order_limited",
)
METHOD_LABELS = {
    "nearest": "Nearest neighbor",
    "bilinear": "Bilinear",
    "bicubic": "Bicubic",
    "patch": "Patch recovery",
    "conservative_first_order": "First order",
    "conservative_second_order": "Second order, unlimited",
    "conservative_second_order_limited": "Second order, limited",
}


def get_run(runs, case_id, engine, mode="quality"):
    matches = [
        run for run in runs
        if run["case_id"] == case_id and run["engine"] == engine and run["mode"] == mode
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {engine} {mode} run for {case_id}; found {len(matches)}")
    return matches[0]


def load_group(run, group):
    path = run["_root"] / run["case"]["path"]
    with xr.open_dataset(path, group=group) as dataset:
        return dataset.load()


def polygons(mesh, polar=False):
    lon, lat = mesh.lon.values, mesh.lat.values
    indices, offsets = mesh.indices.values, mesh.offsets.values
    cells = []
    for start, stop in zip(offsets[:-1], offsets[1:]):
        nodes = indices[start:stop]
        if polar:
            angle = np.deg2rad(lon[nodes])
            radius = 90.0 - lat[nodes]
            x, y = radius * np.cos(angle), radius * np.sin(angle)
        else:
            x = np.rad2deg(np.unwrap(np.deg2rad(lon[nodes])))
            y = lat[nodes]
        cell = np.column_stack((x, y))
        cells.append(cell)
        if not polar:
            if np.min(x) < -180:
                cells.append(cell + np.array([360.0, 0.0]))
            if np.max(x) > 180:
                cells.append(cell - np.array([360.0, 0.0]))
    return cells


def expanded_values(mesh, values, polar=False):
    values = np.asarray(values)
    lon, lat = mesh.lon.values, mesh.lat.values
    indices, offsets = mesh.indices.values, mesh.offsets.values
    expanded = []
    for cell_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if polar:
            expanded.append(values[cell_index])
            continue
        nodes = indices[start:stop]
        x = np.rad2deg(np.unwrap(np.deg2rad(lon[nodes])))
        expanded.append(values[cell_index])
        if np.min(x) < -180:
            expanded.append(values[cell_index])
        if np.max(x) > 180:
            expanded.append(values[cell_index])
    return np.asarray(expanded)


def read_output(run):
    with xr.open_dataset(run["_folder"] / "fields.nc") as dataset:
        return dataset.load()


def read_reference(run):
    path = run["_root"] / run["reference"]["path"]
    with xr.open_dataset(path) as dataset:
        return dataset.load()


def draw_cells(ax, mesh, values, title, limit, cmap="RdBu_r", polar=False):
    collection = PolyCollection(
        polygons(mesh, polar=polar), array=np.ma.masked_invalid(expanded_values(mesh, values, polar=polar)), cmap=cmap,
        clim=(-limit, limit), edgecolors="none", rasterized=True,
    )
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_title(title)
    ax.set_xlabel("Polar stereographic x (degree-equivalent)" if polar else "Longitude (degrees)")
    ax.set_ylabel("Polar stereographic y (degree-equivalent)" if polar else "Latitude (degrees)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#e8e8e8")
    return collection


def method_error_figure(runs, output, size):
    case_prefix = f"rect-{size}-smooth-forward-"
    axes_runs = [get_run(runs, case_prefix + method, "axis") for method in METHODS]
    esmc_runs = [get_run(runs, case_prefix + method, "esmc") for method in METHODS]
    all_runs = axes_runs + esmc_runs
    mesh = load_group(axes_runs[0], "destination")
    reference = read_reference(axes_runs[0])
    truth = reference.dst.values
    covered = reference.dst_covered.values > 0
    errors = []
    for run in all_runs:
        field = read_output(run)
        valid = (field.dst_frac.values > 0) & covered & (mesh.mask.values != 0)
        errors.append(np.where(valid, field.dst.values - truth, np.nan))
    finite = np.concatenate([error[np.isfinite(error)] for error in errors])
    limit = max(float(np.quantile(np.abs(finite), 0.99)), 1e-12)

    fig, axes = plt.subplots(2, 4, figsize=(10.2, 5.6), layout="constrained")
    image = None
    for column, method in enumerate(METHODS):
        for row, (engine, run_index) in enumerate((("AXIS", column), ("ESMC", 4 + column))):
            display_method = "PATCH" if engine == "ESMC" and method in ("bicubic", "patch") else METHOD_LABELS[method]
            rms = run_rms(all_runs[run_index])
            image = draw_cells(
                axes[row, column], mesh, errors[run_index],
                f"{engine}: {display_method}\nRMS error {rms:.2e}", limit,
            )
    fig.colorbar(image, ax=axes, label="Mapped minus analytic cell mean", shrink=0.84, extend="both")
    fig.suptitle(f"Smooth-field error by interpolation method · {size}×{size} source", fontsize=12)
    return export(
        fig, output, "interpolation_method_errors", all_runs,
        "Signed error against independently integrated destination cell means. Shared symmetric color scale; values beyond the 99th-percentile scale are clipped. "
        "ESMC has no named bicubic method: its bicubic-column result is ESMC PATCH, so this is a cross-method comparison. Gray cells are masked or unmapped.",
    )


def run_rms(run):
    return float(run["metrics"]["rms_error"])


def method_transect_figure(runs, output, size, field="smooth"):
    prefix = f"rect-{size}-{field}-forward-"
    axis_runs = [get_run(runs, prefix + method, "axis") for method in METHODS]
    esmc_runs = [get_run(runs, prefix + method, "esmc") for method in METHODS]
    ref = read_reference(axis_runs[0])
    mesh = load_group(axis_runs[0], "destination")
    longitude = mesh.center_lon.values
    latitude = mesh.center_lat.values
    truth = ref.dst.values
    mid_lat = float(np.median(latitude))
    strip = np.abs(latitude - mid_lat) <= max(float(np.ptp(latitude)) / 12, 1e-6)
    x = np.rad2deg(np.unwrap(np.deg2rad(longitude)))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True, layout="constrained")
    for ax, engine, records in ((axes[0], "AXIS", axis_runs), (axes[1], "ESMC", esmc_runs)):
        if field == "smooth":
            ax.axhline(0, color="#222222", linewidth=1, label="Exact error = 0")
        else:
            ax.scatter(x[strip], truth[strip], marker="o", s=20, color="#222222", label="Analytic cell mean", zorder=5)
        for method, run in zip(METHODS, records):
            data = read_output(run)
            value = data.dst.values
            if field == "smooth":
                value = value - truth
            valid = strip & (data.dst_frac.values > 0)
            label = METHOD_LABELS[method]
            if engine == "ESMC" and method in ("bicubic", "patch"):
                label = "PATCH (bicubic pairing)" if method == "bicubic" else "PATCH"
            order = np.argsort(x[valid])
            ax.plot(x[valid][order], value[valid][order], marker=".", markersize=4, linewidth=1.05, label=label)
        ax.set_title(engine)
        ax.set_xlabel("Longitude (degrees)")
        ax.grid(True, color="#d8d8d8", linewidth=0.6)
    axes[0].set_ylabel("Destination cell mean")
    if field == "smooth":
        axes[0].set_ylabel("Destination error")
    axes[1].legend(fontsize=7, ncol=2, loc="best")
    field_label = "smooth field" if field == "smooth" else "sharp interface"
    fig.suptitle(f"Method behavior along a common {field_label} transect", fontsize=12)
    return export(
        fig, output, f"interpolation_{field}_transect", axis_runs + esmc_runs,
        f"Destination cell means along a narrow central latitude band for the {field_label}, compared with analytic cell means. AXIS and ESMC use the same meshes and reference; bicubic pairs with ESMC PATCH.",
    )


def conservative_figure(runs, output, size):
    prefix = f"rect-{size}-sharp-forward-"
    axis_runs = [get_run(runs, prefix + method, "axis") for method in CONSERVATIVE]
    esmc_runs = [get_run(runs, prefix + method, "esmc") for method in CONSERVATIVE]
    reference = read_reference(axis_runs[0])
    mesh = load_group(axis_runs[0], "destination")
    truth = reference.dst.values
    covered = reference.dst_covered.values > 0
    longitude = np.rad2deg(np.unwrap(np.deg2rad(mesh.center_lon.values)))
    latitude = mesh.center_lat.values
    strip = np.abs(latitude - np.median(latitude)) <= max(float(np.ptp(latitude)) / 12, 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True, layout="constrained")
    for ax, engine, records in ((axes[0], "AXIS", axis_runs), (axes[1], "ESMC", esmc_runs)):
        ax.scatter(longitude[strip], truth[strip], s=22, marker="o", color="#222222", label="Analytic cell mean", zorder=5)
        methods = CONSERVATIVE if engine == "AXIS" else CONSERVATIVE[:2]
        for method, run in zip(methods, records):
            data = read_output(run)
            valid = strip & covered & (data.dst_frac.values > 0)
            order = np.argsort(longitude[valid])
            ax.plot(longitude[valid][order], data.dst.values[valid][order], marker=".", linewidth=1.15, label=METHOD_LABELS[method])
        ax.set_title(engine)
        ax.set_xlabel("Longitude (degrees)")
        ax.grid(True, color="#d8d8d8", linewidth=0.6)
    axes[0].set_ylabel("Destination cell mean")
    axes[0].legend(fontsize=7, loc="best")
    axes[1].legend(fontsize=7, loc="best")
    fig.suptitle("Conservative methods across a sharp interface", fontsize=12)
    return export(
        fig, output, "conservative_sharp_transect", axis_runs + esmc_runs,
        "Sharp-field destination means on a shared central transect. The limited second-order result is compared with ESMC second order as limited versus un-limited, not an identical limiter. Inventory residuals and extrema are reported separately in the quality table.",
    )


def robustness_figure(runs, output):
    selected = []
    for case_id in ("seam-bicubic", "polar-bicubic"):
        try:
            selected.append(get_run(runs, case_id, "axis"))
        except ValueError:
            continue
    if len(selected) < 2:
        raise ValueError("Need completed AXIS seam and polar bicubic runs for the robustness figure")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), layout="constrained")
    error_arrays = []
    for ax, run in zip(axes, selected):
        mesh = load_group(run, "destination")
        ref = read_reference(run)
        data = read_output(run)
        valid = (data.dst_frac.values > 0) & (ref.dst_covered.values > 0) & (mesh.mask.values != 0)
        error = np.where(valid, data.dst.values - ref.dst.values, np.nan)
        error_arrays.append(error)
    limits = [
        max(float(np.quantile(np.abs(array[np.isfinite(array)]), 0.99)), 1e-12)
        for array in error_arrays
    ]
    for ax, run, error in zip(axes, selected, error_arrays):
        mesh = load_group(run, "destination")
        is_polar = "polar" in run["case_id"]
        panel_limit = limits[1 if is_polar else 0]
        image = draw_cells(
            ax, mesh, error,
            f"{'Polar cap' if is_polar else 'Dateline seam'}\n99th percentile ±{panel_limit:.1e}",
            panel_limit, polar=is_polar,
        )
        fig.colorbar(image, ax=ax, label="Signed cell-mean error", shrink=0.72, extend="both")
        if not is_polar:
            lon = mesh.lon.values
            lat = mesh.lat.values
            if np.ptp(lon) < 180:
                ax.set_xlim(float(np.min(lon)) - 2, float(np.max(lon)) + 2)
                ax.set_ylim(float(np.min(lat)) - 2, float(np.max(lat)) + 2)
    return export(
        fig, output, "seam_polar_robustness", selected,
        "AXIS bicubic signed errors in synthetic periodic dateline and polar-cap cases. Each panel has its own symmetric scale clipped at its 99th percentile; compare spatial structure within each panel, not color magnitude across panels.",
    )


def mpas_mesh_figure(runs, output, field):
    case_ids = [f"mpas-{field}-triangulated-{method}" for method in METHODS + CONSERVATIVE]
    records = [get_run(runs, case_id, "axis") for case_id in case_ids]
    source_mesh = load_group(records[0], "source")
    destination_mesh = load_group(records[0], "destination")
    reference = read_reference(records[0])
    source = read_output(records[0]).src.values
    labels = ["MPAS source cell means", "Analytic destination means"] + [
        METHOD_LABELS[method] for method in METHODS + CONSERVATIVE
    ]
    arrays = [source, reference.dst.values]
    for run in records:
        arrays.append(read_output(run).dst.values)
    finite = np.concatenate([array[np.isfinite(array)] for array in arrays])
    low, high = float(np.min(finite)), float(np.max(finite))
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 6.6), layout="constrained")
    meshes = [source_mesh, destination_mesh] + [destination_mesh] * len(records)
    for ax, mesh, values, label in zip(axes.flat, meshes, arrays, labels):
        image = PolyCollection(
            polygons(mesh), array=np.ma.masked_invalid(expanded_values(mesh, values)), cmap="viridis",
            clim=(low, high), edgecolors="none", rasterized=True,
        )
        ax.add_collection(image)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("auto")
        ax.set_facecolor("#e8e8e8")
        ax.set_title(label)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    field_label = "smooth field" if field == "smooth" else "sharp-interface field"
    fig.colorbar(image, ax=axes, label=f"Analytic {field_label}", shrink=0.86)
    fig.suptitle(f"{field_label.capitalize()} regridding on the MPAS x1.2562 mesh", fontsize=12)
    return export(
        fig, output, f"mpas_{field}_method_examples", records,
        f"Analytic {field_label} remapped from the public UCAR MPAS x1.2562 mesh (triangulated for the ESMC TRI/QUAD adapter) to a global rectilinear destination. This is a synthetic field on a realistic unstructured mesh, not an observed NOAA field. The panel shows AXIS variants; quantitative cross-engine accuracy is in the paired metrics table.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=32, help="Source rectilinear resolution for method examples")
    parser.add_argument("--robustness-results", help="Separate bundle containing seam and polar cases")
    parser.add_argument("--mpas-results", help="Separate bundle containing the MPAS field examples")
    parser.add_argument("--mpas-sharp-results", help="Separate bundle containing sharp-interface MPAS examples")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    _, runs = bundle(args.results)
    products, omitted = {}, {}
    required = [
        ("interpolation_method_errors", method_error_figure),
        ("conservative_sharp_transect", conservative_figure),
    ]
    for name, function in required:
        try:
            products[name] = function(runs, output, args.size)
        except ValueError as error:
            omitted[name] = str(error)
    for field in ("smooth", "sharp"):
        try:
            products[f"interpolation_{field}_transect"] = method_transect_figure(runs, output, args.size, field)
        except ValueError as error:
            omitted[f"interpolation_{field}_transect"] = str(error)
    try:
        _, robustness_runs = bundle(args.robustness_results) if args.robustness_results else (None, runs)
        products["seam_polar_robustness"] = robustness_figure(robustness_runs, output)
    except ValueError as error:
        omitted["seam_polar_robustness"] = str(error)
    try:
        _, mpas_runs = bundle(args.mpas_results) if args.mpas_results else (None, runs)
        products["mpas_smooth_method_examples"] = mpas_mesh_figure(mpas_runs, output, "smooth")
    except ValueError as error:
        omitted["mpas_smooth_method_examples"] = str(error)
    try:
        _, mpas_sharp_runs = bundle(args.mpas_sharp_results) if args.mpas_sharp_results else (None, runs)
        products["mpas_sharp_method_examples"] = mpas_mesh_figure(mpas_sharp_runs, output, "sharp")
    except ValueError as error:
        omitted["mpas_sharp_method_examples"] = str(error)
    finish(output, products, omitted, [])
    if omitted:
        print("Omitted figures:", omitted)
    print("Generated:", sorted(products))


if __name__ == "__main__":
    main()
