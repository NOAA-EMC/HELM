"""Summarize retained native observations; no wrapper elapsed times enter these statistics."""

import csv
import numpy as np


def summarize(path):
    groups = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "complete":
                continue
            t = float(row["seconds"])
            if not np.isfinite(t) or t <= 0:
                raise ValueError("Invalid timing sample")
            groups.setdefault((row["phase"], int(row["applications"])), []).append(t)
    output = {}
    for (phase, n), values in groups.items():
        output[f"{phase}:{n}"] = {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "samples": len(values),
            "applications": n,
        }
    if "generate:1" in output and "operator_finalize:1" in output:
        # Paired setup observations, not the sum of independently selected quantiles.
        values = np.array(groups["generate", 1]) + np.array(groups["operator_finalize", 1])
        output["setup:1"] = {
            "median": float(np.median(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "samples": len(values),
            "applications": 1,
        }
        if "apply:1" in output:
            for phase, n in groups:
                if phase == "repeated_workload":
                    output[f"estimated_total:{n}"] = {
                        "seconds": output["setup:1"]["median"] + n * output["apply:1"]["median"],
                        "formula": "median(paired setup) + N * median(apply)",
                        "applications": n,
                    }
    return output


def comparisons(runs):
    """Compare only identical cases that both engines completed and validated."""
    grouped = {}
    for run in runs:
        if run["mode"] == "timing":
            grouped.setdefault(run["comparison_key"], {})[run["engine"]] = run
    output = []
    for key, engines in grouped.items():
        if set(engines) != {"axis", "esmc"}:
            continue
        axis, esmc = engines["axis"], engines["esmc"]
        eligible = all(
            run["matching"] == "matched" and run["metrics"]["numerical_validation"] == "pass"
            for run in (axis, esmc)
        )
        ratios = {}
        if eligible:
            for phase in ("setup:1", "apply:1", "setup_and_reuse:1", "setup_and_reuse:100"):
                if phase in axis["timing"] and phase in esmc["timing"]:
                    ratios[phase] = (
                        esmc["timing"][phase]["median"] / axis["timing"][phase]["median"]
                    )
        output.append(
            {
                "comparison_key": key,
                "case_id": axis["case_id"],
                "axis_run": axis["run_id"],
                "esmc_run": esmc["run_id"],
                "eligible": eligible,
                "reason": "Identical case/options and passing quality/reproduction checks"
                if eligible
                else "Unmatched geometry or failed numerical validation; no speedup claim",
                "esmc_seconds_over_axis_seconds": ratios,
            }
        )
    return output
