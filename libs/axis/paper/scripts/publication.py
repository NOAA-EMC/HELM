"""Shared validated publication exports and provenance sidecars."""

import os
import json
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/axis-paper-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from artifacts import ROOT, checked_runs, digest, save

COLORS = {"axis": "#0072B2", "esmc": "#D55E00"}
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def bundle(results):
    return checked_runs(results)


def sidecar(path, runs, caption, **settings):
    save(
        str(path) + ".json",
        {
            "schema_version": 1,
            "artifact": path.name,
            "sha256": digest(path),
            "runs": [{k: r[k] for k in ("run_id", "case_id", "binary_sha256")} for r in runs],
            "run_record_hashes": {r["run_id"]: digest(r["_folder"] / "run.json") for r in runs},
            "argv": sys.argv,
            "generator_sha256": digest(Path(sys.argv[0]).resolve()),
            "generator_path": str(Path(sys.argv[0]).resolve().relative_to(ROOT)),
            "caption": caption,
            "settings": settings,
        },
    )


def export(fig, output, name, runs, caption):
    paths = []
    for ext in ("pdf", "svg", "png"):
        path = Path(output) / (name + "." + ext)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        sidecar(
            path,
            runs,
            caption,
            width_inches=float(fig.get_figwidth()),
            height_inches=float(fig.get_figheight()),
            dpi=300,
        )
        paths.append(path.name)
    plt.close(fig)
    return paths


def finish(output, products, omitted, required):
    missing = sorted(set(required) - set(products))
    save(
        Path(output) / "artifacts.json",
        {
            "schema_version": 1,
            "products": products,
            "omissions": omitted,
            "missing_required": missing,
            "status": "complete" if not missing else "incomplete",
        },
    )
    return 0 if not missing else 5


def validate_publications(results):
    """Verify all five required products and their links to saved run records."""
    root = Path(results).resolve()
    suite, runs = checked_runs(root)
    by_id = {run["run_id"]: run for run in runs}
    required = {"workflow", "field_maps", "performance", "quality_table", "configuration_table"}
    products = {}
    for directory in ("figures", "tables"):
        manifest_path = root / directory / "artifacts.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing publication manifest: {directory}")
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"Incomplete publication products: {directory}")
        for name, filenames in manifest["products"].items():
            for filename in filenames:
                path = root / directory / filename
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Publication path escapes bundle")
                metadata = json.loads(Path(str(path) + ".json").read_text())
                generator = ROOT / metadata["generator_path"]
                if not generator.is_file() or digest(generator) != metadata["generator_sha256"]:
                    raise ValueError(f"Stale publication generator: {generator}")
                if digest(path) != metadata["sha256"]:
                    raise ValueError(f"Stale publication: {filename}")
                for run_id, sha in metadata["run_record_hashes"].items():
                    if run_id not in by_id or digest(by_id[run_id]["_folder"] / "run.json") != sha:
                        raise ValueError(f"Stale publication input: {run_id}")
            products[name] = [f"{directory}/{filename}" for filename in filenames]
    if required - products.keys():
        raise ValueError(f"Missing required displays: {sorted(required - products.keys())}")
    return products
