"""Run native references and separate quality/timing workers serially."""

import argparse
import os
import shutil
import subprocess
import sys
import xarray as xr
from pathlib import Path
from artifacts import digest, identity, load, profile, relative, save
from provenance import capture
from timing_summary import summarize, comparisons
from native_execution import NativeExecution


def invoke(argv, log, environment):
    with Path(log).open("w") as f:
        try:
            result = subprocess.run(
                list(map(str, argv)),
                stdout=f,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
            return result.returncode
        except OSError as e:
            f.write(str(e))
            return 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "cases", "bin-dir", "output"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument("--container", help="Existing Docker container with a shared writable mount")
    ap.add_argument("--host-root", help="Host directory mounted into the container")
    ap.add_argument("--container-root", default="/out")
    arguments = ap.parse_args()
    settings = profile(arguments.profile)
    cases = Path(arguments.cases).resolve()
    results_root = Path(arguments.output).resolve()
    results_root.mkdir(parents=True, exist_ok=False)
    (results_root / "fixtures").mkdir()
    (results_root / "references").mkdir()
    (results_root / "logs").mkdir()
    execution = NativeExecution(arguments.container, arguments.host_root, arguments.container_root)
    provenance = capture(results_root)
    provenance["native_execution"] = execution.provenance()
    binary_directory = Path(arguments.bin_dir).resolve()
    environment = dict(
        os.environ,
        OMP_NUM_THREADS=str(settings["threads"]),
        KOKKOS_NUM_THREADS=str(settings["threads"]),
        OMP_PROC_BIND="false",
    )
    provenance["resolved_thread_environment"] = {
        key: environment[key] for key in ("OMP_NUM_THREADS", "KOKKOS_NUM_THREADS", "OMP_PROC_BIND")
    }

    def run_native(arguments, log):
        if not Path(arguments[0]).is_file():
            Path(log).write_text(f"Requested native executable is unavailable: {arguments[0]}\n")
            return 3
        return invoke(execution.command(arguments, environment), log, environment)

    inventory = load(cases / "cases.json")
    if inventory["profile_sha256"] != identity(settings):
        raise ValueError("Profile differs from prepared cases")
    save(results_root / "profile.json", settings)
    save(results_root / "cases.json", inventory)
    suite = {
        "schema_version": 1,
        "profile": settings["id"],
        "provenance": provenance,
        "runs": [],
        "status": "running",
    }
    statuses = {
        2: "invalid_input",
        3: "unavailable",
        4: "unsupported",
        5: "failed",
        6: "invalid_reference",
    }
    for case in inventory["cases"]:
        if case["status"] != "prepared":
            for engine in settings["engines"]:
                suite["runs"].append(
                    {
                        "case_id": case["id"],
                        "engine": engine,
                        "status": case["status"],
                        "cause": case["cause"],
                    }
                )
            continue
        source = relative(cases, case["fixture"])
        if digest(source) != case["fixture_sha256"]:
            raise ValueError("Stale fixture")
        fixture = results_root / "fixtures" / source.name
        shutil.copy2(source, fixture)
        reference_path = (
            results_root
            / "references"
            / (
                identity({"fixture": case["fixture_sha256"], "policy": case["reference_policy"]})
                + ".nc"
            )
        )
        return_code = 0
        if not reference_path.exists():
            return_code = run_native(
                [
                    binary_directory / "axis_paper_reference",
                    "--case",
                    fixture,
                    "--output",
                    reference_path,
                ],
                results_root / "logs" / (case["id"] + "-reference.log"),
            )
        if return_code:
            for engine in settings["engines"]:
                suite["runs"].append(
                    {
                        "case_id": case["id"],
                        "engine": engine,
                        "status": statuses.get(return_code, "failed"),
                        "cause": "Reference worker failed; see logs",
                    }
                )
            save(results_root / "suite.json", suite)
            continue
        with xr.open_dataset(reference_path) as reference_data:
            reference_evidence = {
                "extended_precision_digits": int(reference_data.attrs["extended_precision_digits"]),
                "description": reference_data.attrs["precision_evidence"],
                "max_uncertainty": float(reference_data.uncertainty.max()),
                "max_precision_difference": float(reference_data.precision_difference.max()),
            }
        for engine in settings["engines"]:
            quality = None
            for mode in ("quality", "timing"):
                run_id = f"{case['id']}-{engine}-{mode}"
                folder = results_root / run_id
                command = [
                    binary_directory / f"axis_paper_{engine}",
                    "--case",
                    fixture,
                    "--reference",
                    reference_path,
                    "--output",
                    folder,
                    "--mode",
                    mode,
                    "--warmups",
                    settings["warmups"],
                    "--repetitions",
                    settings["repetitions"],
                    "--reuse",
                    ",".join(map(str, settings["reuse"])),
                ]
                log = results_root / "logs" / (run_id + ".log")
                return_code = run_native(command, log)
                record = {
                    "case_id": case["id"],
                    "engine": engine,
                    "mode": mode,
                    "status": statuses.get(return_code, "failed") if return_code else "complete",
                    "cause": None
                    if return_code == 0
                    else f"Worker exit {return_code}; see {log.name}",
                }
                if not return_code:
                    metric_command = [
                        binary_directory / "axis_paper_metrics",
                        "--case",
                        fixture,
                        "--reference",
                        reference_path,
                        "--run",
                        folder,
                    ]
                    if mode == "timing" and quality:
                        metric_command += [
                            "--compare",
                            results_root / quality["run_id"] / "fields.nc",
                        ]
                    return_code = run_native(
                        metric_command, results_root / "logs" / (run_id + "-metrics.log")
                    )
                    if return_code:
                        record.update(
                            status=statuses.get(return_code, "failed"), cause="Metric worker failed"
                        )
                    else:
                        native = load(folder / "native.json")
                        metrics = load(folder / "metrics.json")
                        compilation_database = binary_directory.parent / "compile_commands.json"
                        if compilation_database.exists():
                            worker_source = f"{engine}_worker.cpp"
                            native["build"]["effective_compile_commands"] = [
                                entry
                                for entry in load(compilation_database)
                                if Path(entry["file"]).name == worker_source
                            ]
                        flag_file = (
                            binary_directory
                            / "CMakeFiles"
                            / f"axis_paper_{engine}.dir"
                            / "flags.make"
                        )
                        if flag_file.exists():
                            native["build"]["effective_compile_flags"] = flag_file.read_text()
                        link_file = (
                            binary_directory
                            / "CMakeFiles"
                            / f"axis_paper_{engine}.dir"
                            / "link.txt"
                        )
                        if link_file.exists():
                            native["build"]["effective_link_command"] = (
                                link_file.read_text().strip()
                            )
                        if mode == "timing" and quality is None:
                            record.update(status="failed", cause="Missing matching quality pass")
                        else:
                            matching = (
                                "matched"
                                if case["geometry"] == "great_circle"
                                and metrics["max_relative_area_difference"] < 1e-8
                                else "unmatched"
                            )
                            reason = (
                                "Shared great-circle fixture/options; areas agree within 1e-8"
                                if matching == "matched"
                                else "Geometry or engine-area convention differs"
                            )
                            record.update(
                                native,
                                schema_version=1,
                                run_id=run_id,
                                case_id=case["id"],
                                field=case["field"],
                                family=case["family"],
                                case_definition=next(
                                    item for item in settings["cases"] if item["id"] == case["id"]
                                ),
                                norm=case["norm"],
                                geometry=case["geometry"],
                                source_cells=case["source_cells"],
                                destination_cells=case["destination_cells"],
                                matching=matching,
                                matching_reason=reason,
                                metrics=metrics,
                                provenance=provenance,
                                argv=execution.command(command, environment),
                                native_argv=list(map(str, command)),
                                metric_argv=execution.command(metric_command, environment),
                                metrics_binary_sha256=digest(
                                    binary_directory / "axis_paper_metrics"
                                ),
                                reference_binary_sha256=digest(
                                    binary_directory / "axis_paper_reference"
                                ),
                                binary_sha256=digest(binary_directory / f"axis_paper_{engine}"),
                                case={
                                    "path": str(fixture.relative_to(results_root)),
                                    "sha256": digest(fixture),
                                },
                                reference={
                                    "path": str(reference_path.relative_to(results_root)),
                                    "sha256": digest(reference_path),
                                },
                                reference_policy=case["reference_policy"],
                                reference_precision_verified=reference_evidence[
                                    "extended_precision_digits"
                                ]
                                > 53,
                                reference_evidence=reference_evidence,
                                reference_precision_reason="Analytic double/extended-precision agreement and roundoff estimate; not a formal interval certificate",
                                options={
                                    "method": "conservative_first_order",
                                    "precision": "float64",
                                    "normalization": case["norm"],
                                    "line_type": case["geometry"],
                                    "extrapolation": "none",
                                    "unmapped": "ignore",
                                    "mask_identity": case["fixture_sha256"],
                                },
                                comparison_key=identity(
                                    {
                                        "fixture": digest(fixture),
                                        "reference": digest(reference_path),
                                        "method": "conservative_first_order",
                                        "precision": "float64",
                                        "threads": settings["threads"],
                                        "ranks": 1,
                                        "normalization": case["norm"],
                                        "geometry": case["geometry"],
                                    }
                                ),
                                quality_run=quality["run_id"]
                                if quality and mode == "timing"
                                else None,
                                timing=summarize(folder / "observations.csv")
                                if mode == "timing"
                                else {},
                                timing_policy={
                                    "exclude": [
                                        "I/O",
                                        "Python",
                                        "reference",
                                        "mesh preparation",
                                        "cleanup",
                                    ],
                                    "setup": ["generate", "operator_finalize incl. first apply"],
                                    "apply": "existing operator with destination overwrite",
                                    "setup_and_reuse": "sum of generation, first-use finalization, and N native applies on the same fresh operator",
                                    "synchronization": "Kokkos fence or synchronous ESMC single PET",
                                    "threads": settings["threads"],
                                    "ranks": 1,
                                },
                                memory={
                                    "value": None,
                                    "reason": "No reliable isolated peak memory instrumentation",
                                },
                                stage_timing={
                                    "value": None,
                                    "reason": "Instrumentation disabled for performance; no causal attribution",
                                },
                            )
                            if mode == "timing" and native["algorithm"] == "unknown":
                                record["algorithm"] = quality["algorithm"]
                                record["algorithm_evidence"] = "linked quality run"
                            if mode == "timing":
                                expected_phases = ["generate:1", "operator_finalize:1", "apply:1"]
                                expected_phases += [
                                    f"{phase}:{count}"
                                    for count in settings["reuse"]
                                    for phase in ("repeated_workload", "setup_and_reuse")
                                ]
                                for phase in expected_phases:
                                    if (
                                        record["timing"].get(phase, {}).get("samples")
                                        != settings["repetitions"]
                                    ):
                                        raise ValueError("Incomplete native timing observations")
                            record["files"] = {
                                name: digest(folder / name)
                                for name in (
                                    "native.json",
                                    "fields.nc",
                                    "metrics.json",
                                    "observations.csv",
                                )
                            }
                            save(folder / "run.json", record)
                            record = {
                                "case_id": case["id"],
                                "engine": engine,
                                "mode": mode,
                                "status": "complete",
                                "path": run_id,
                                "record_sha256": digest(folder / "run.json"),
                            }
                            if mode == "quality":
                                quality = load(folder / "run.json")
                suite["runs"].append(record)
                save(results_root / "suite.json", suite)
    suite["status"] = (
        "complete" if all(r["status"] == "complete" for r in suite["runs"]) else "incomplete"
    )
    completed = [
        load(results_root / item["path"] / "run.json")
        for item in suite["runs"]
        if item["status"] == "complete"
    ]
    numerical_failures = [
        run["run_id"] for run in completed if run["metrics"]["numerical_validation"] != "pass"
    ]
    suite["comparisons"] = comparisons(completed)
    suite["numerical_validation"] = "fail" if numerical_failures else "pass"
    suite["numerical_failures"] = numerical_failures
    save(results_root / "suite.json", suite)
    print(
        f"{suite['status']}: {sum(r['status'] == 'complete' for r in suite['runs'])}/{len(suite['runs'])} runs in {results_root}; numerical validation {suite['numerical_validation']}"
    )
    return 0 if suite["status"] == "complete" or arguments.allow_incomplete else 5


if __name__ == "__main__":
    raise SystemExit(main())
