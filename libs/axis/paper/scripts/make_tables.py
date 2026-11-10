"""Export full-precision data and readable manuscript tables from native records."""

import argparse
import csv
from pathlib import Path
from publication import bundle, finish, sidecar


def table(out, name, rows, runs, caption, markdown_fields=None, details=None):
    fields = list(rows[0]) if rows else ["status"]
    path = out / (name + ".csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    sidecar(path, runs, caption)
    path = out / (name + ".md")

    def fmt(x):
        if x is None:
            return "unavailable"
        if isinstance(x, float):
            return f"{x:.5g}"
        return str(x).replace("|", "/").replace("\n", " ")

    display_fields = markdown_fields or fields
    text = [
        "| " + " | ".join(display_fields) + " |",
        "| " + " | ".join("---" for _ in display_fields) + " |",
    ]
    text.extend("| " + " | ".join(fmt(row[key]) for key in display_fields) + " |" for row in rows)
    if details:
        text.extend(["", *details])
    path.write_text("\n".join(text) + "\n")
    sidecar(path, runs, caption)
    return [name + ".csv", name + ".md"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    suite, runs = bundle(a.results)
    q = [r for r in runs if r["mode"] == "quality"]
    products = {}
    rows = []
    for r in q:
        row = {k: r[k] for k in ("case_id", "engine", "family", "field", "norm", "matching")}
        row["method"] = "conservative_first_order"
        row.update(
            {
                k: r["metrics"][k]
                for k in (
                    "mean_absolute_error",
                    "bias",
                    "rms_error",
                    "max_error",
                    "conservation_absolute",
                    "conservation_relative",
                    "constant_error",
                    "row_sum_error",
                    "overshoot",
                    "undershoot",
                    "eligible",
                    "masked",
                    "unmapped",
                    "negligible_coverage",
                    "nonfinite",
                    "numerical_validation",
                )
            }
        )
        row.update(
            {
                f"overlap_{key}": r["diagnostics"].get(key)
                for key in ("candidates", "invalid", "discarded", "zero_overlap")
            }
        )
        row["diagnostic_scope"] = r["diagnostics"]["availability_reason"]
        rows.append(row)
    if rows:
        products["quality_table"] = table(
            out,
            "quality",
            rows,
            q,
            "Errors are area-weighted on the common mapped domain; unmatched cases are identified. Null relative residuals have near-zero denominators.",
            markdown_fields=[
                "case_id",
                "engine",
                "norm",
                "matching",
                "rms_error",
                "max_error",
                "conservation_absolute",
                "overshoot",
                "undershoot",
                "numerical_validation",
            ],
        )
    config = []
    for r in q:
        config.append(
            {
                "case_id": r["case_id"],
                "engine": r["engine"],
                "source_cells": r["source_cells"],
                "destination_cells": r["destination_cells"],
                "geometry": r["geometry"],
                "algorithm": r["algorithm"],
                "norm": r["norm"],
                "extrapolation": "none",
                "compiler": r["build"]["compiler"],
                "flags": r["build"]["flags"],
                "native_os": r["build"].get("runtime_os"),
                "native_machine": r["build"].get("runtime_machine"),
                "preparation_host": r["provenance"]["hardware"],
                "kokkos_version": r.get("kokkos_version"),
                "esmf_version": r.get("esmf_version"),
                "netcdf_version": r["build"].get("netcdf_version"),
                "arborx_version": r.get("arborx_version"),
                "arborx_commit": r.get("arborx_commit"),
                "kokkoskernels_version": r.get("kokkoskernels_version"),
                "method": r["options"]["method"],
                "precision": r["options"]["precision"],
                "memory": r["memory"]["value"],
                "memory_reason": r["memory"]["reason"],
                "threads": r["timing_policy"]["threads"],
                "ranks": r["timing_policy"]["ranks"],
                "container": r["provenance"]["container_image"],
                "binary_sha256": r["binary_sha256"],
            }
        )
    build_details = [
        "## Software and execution environment",
        "",
        "The CSV retains all configuration fields; each run record retains complete compiler commands and source/input hashes.",
        "",
    ]
    for engine in sorted({run["engine"] for run in q}):
        run = next(run for run in q if run["engine"] == engine)
        build_details += [
            f"### {engine.upper()}",
            "",
            f"- Compiler: {run['build']['compiler']}; flags: `{run['build']['flags']}`.",
            f"- Native runtime: {run['build'].get('runtime_os')} / {run['build'].get('runtime_machine')}; {run['build'].get('logical_cpus')} visible logical CPUs.",
            f"- Threads/ranks: {run['timing_policy']['threads']}/{run['timing_policy']['ranks']}; container: `{run['provenance']['native_execution'].get('image_id', 'local')}`.",
            f"- Kokkos: {run.get('kokkos_version', 'not used')}; KokkosKernels: {run.get('kokkoskernels_version', 'not used')}; ArborX: {run.get('arborx_version', 'not used')}; ESMF: {run.get('esmf_version', 'not used')}.",
            f"- Memory: unavailable — {run['memory']['reason']}.",
            "",
        ]
    if config:
        products["configuration_table"] = table(
            out,
            "configuration",
            config,
            q,
            "Actual tested software, geometry, and hardware configuration; provenance records contain source patches, versions, and commands.",
            markdown_fields=[
                "case_id",
                "engine",
                "source_cells",
                "destination_cells",
                "geometry",
                "norm",
                "precision",
            ],
            details=build_details,
        )
    failures = [
        {k: r.get(k) for k in ("case_id", "engine", "mode", "status", "cause")}
        for r in suite["runs"]
        if r["status"] != "complete"
    ]
    if failures:
        table(
            out,
            "failures",
            failures,
            [],
            "Every requested unavailable, unsupported, or failed run is retained.",
        )
    return finish(out, products, {}, ["quality_table", "configuration_table"])


if __name__ == "__main__":
    raise SystemExit(main())
