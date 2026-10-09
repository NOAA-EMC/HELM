"""Versioned artifact identities and validation shared by all paper commands."""

import hashlib
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[4]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def relative(root, name):
    root = Path(root).resolve()
    p = (root / name).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f"Artifact escapes bundle: {name}")
    return p


def profile(path):
    p = load(path)
    if p.get("schema_version") != 1 or not p.get("cases"):
        raise ValueError("Unsupported/empty profile")
    for k in ("warmups", "repetitions", "threads"):
        if not isinstance(p[k], int) or p[k] < 1:
            raise ValueError(f"Invalid {k}")
    if p.get("ranks") != 1:
        raise ValueError("This study supports exactly one rank")
    if not p.get("reuse") or any(not isinstance(n, int) or n < 1 for n in p["reuse"]):
        raise ValueError("Invalid reuse counts")
    if len({c["id"] for c in p["cases"]}) != len(p["cases"]):
        raise ValueError("Duplicate case IDs")
    if not p.get("engines") or set(p["engines"]) - {"axis", "esmc"}:
        raise ValueError("Engines must be axis and/or esmc")
    if len(set(p["engines"])) != len(p["engines"]):
        raise ValueError("Duplicate engines")
    for name in (
        "reference",
        "area",
        "constant",
        "conservation",
        "near_zero",
        "reproduction",
        "smooth_rms",
    ):
        value = p["tolerances"][name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid tolerance: {name}")
    for case in p["cases"]:
        if case["field"] not in {
            "constant",
            "smooth",
            "zero",
            "sharp",
            "sinusoid",
            "harmonic",
            "vortex",
            "gulfstream",
        }:
            raise ValueError(f"Unsupported field: {case['field']}")
        if case["norm"] not in {"dstarea", "fracarea"}:
            raise ValueError("Invalid normalization")
        if case["geometry"] not in {"great_circle", "constant_latitude"}:
            raise ValueError("Invalid geometry")
    return p


def checked_runs(root):
    root = Path(root)
    suite = load(root / "suite.json")
    if suite["schema_version"] != 1:
        raise ValueError("Unsupported suite schema")
    runs = []
    for item in suite["runs"]:
        if item["status"] != "complete":
            continue
        folder = relative(root, item["path"])
        if digest(folder / "run.json") != item["record_sha256"]:
            raise ValueError(f"Stale run record {item['path']}")
        r = load(folder / "run.json")
        if r.get("schema_version") != 1 or r.get("status") != "complete":
            raise ValueError("Invalid completed run schema/status")
        if any(r.get(key) != item.get(key) for key in ("case_id", "engine", "mode")):
            raise ValueError("Run inventory identity differs from saved record")
        for name, sha in r["files"].items():
            if digest(relative(folder, name)) != sha:
                raise ValueError(f"Stale output {name}")
        for key in ("case", "reference"):
            material = relative(root, r[key]["path"])
            if digest(material) != r[key]["sha256"]:
                raise ValueError(f"Stale {key}")
        r["_folder"] = folder
        r["_root"] = root
        runs.append(r)
    by_id = {run["run_id"]: run for run in runs}
    for run in runs:
        if run["mode"] == "timing":
            quality = by_id.get(run["quality_run"])
            if (
                quality is None
                or quality["mode"] != "quality"
                or run["instrumentation"] != "disabled"
            ):
                raise ValueError("Timing run lacks matching uninstrumented/quality evidence")
            for key in ("case", "reference", "engine", "options", "comparison_key", "algorithm"):
                if run.get(key) != quality.get(key):
                    raise ValueError(f"Quality/timing mismatch: {key}")
    return suite, runs


def match_key(case):
    return identity(
        {k: case[k] for k in ("fixture_sha256", "field", "geometry", "norm", "reference_policy")}
    )
