"""Audit source citations, connect outline questions to evidence, and freeze bundles."""

import argparse
import hashlib
import os
import re
from pathlib import Path

from artifacts import ROOT, checked_runs, digest, load, save
from archive import archive
from publication import validate_publications
from provenance import git

START = "<!-- AXIS-PAPER-EVIDENCE:BEGIN -->"
END = "<!-- AXIS-PAPER-EVIDENCE:END -->"
INVENTORY = ROOT / "libs/axis/paper/evidence/outline_inventory.json"


def outline_items(text):
    """Keep IDs stable across generated blocks and the LLC terminology correction."""
    section, items = "framing", []
    text = re.sub(re.escape(START) + r".*?" + re.escape(END), "", text, flags=re.S)
    for line in text.splitlines():
        if line.startswith("### "):
            section = line[4:].strip()
        if line.startswith("- "):
            normalized = line[2:].replace("regional LLC", "regional LCC")
            stable = hashlib.sha256((section + "\n" + normalized).encode()).hexdigest()[:16]
            items.append({"id": stable, "section": section, "text": normalized})
    return items


def check_sources():
    citations = load(ROOT / "libs/axis/paper/evidence/implementation_citations.json")
    marker_path = ROOT / "AXIS_PAPER_SOURCE.json"
    snapshot = load(marker_path) if marker_path.exists() else None
    for citation in citations["citations"]:
        if snapshot:
            if citation["revision"] not in snapshot.get(
                "reviewed_revisions", [snapshot["source_commit"]]
            ):
                raise ValueError("Citation revision is absent from the archived source lineage")
        else:
            git("cat-file", "-e", citation["revision"] + "^{commit}")
        path = ROOT / citation["path"]
        if (
            not path.is_file()
            or digest(path) != citation["sha256"]
            or citation["symbol"] not in path.read_text()
        ):
            raise ValueError(f"Stale source citation: {citation['path']}:{citation['symbol']}")
    return citations["citations"]


def supporting_runs(item, runs):
    """Select relevant measured scope; an empty selection remains explicitly limited."""
    text = (item["section"] + " " + item["text"]).lower()
    selected = [run for run in runs if run["mode"] == "quality"]
    for word, family in (("mpas", "mpas"), ("lcc", "lcc"), ("rectilinear", "rectilinear")):
        if word in text and sum(name in text for name in ("mpas", "lcc", "rectilinear")) == 1:
            selected = [run for run in selected if run["family"] == family]
    for word, field in (("constant", "constant"), ("smooth", "smooth"), ("sharp", "sharp")):
        if word in text:
            selected = [run for run in selected if run["field"] == field]
            break
    if any(
        word in text for word in ("performance", "runtime", "application cost", "timing", "amortiz")
    ):
        case_ids = {run["case_id"] for run in selected}
        selected = [run for run in runs if run["mode"] == "timing" and run["case_id"] in case_ids]
    return selected


def evidence_inventory(items, runs, citations, products, results):
    entries = []
    for item in items:
        selected = supporting_runs(item, runs)
        experimental = any(
            word in item["text"].lower()
            for word in (
                "result",
                "accuracy",
                "conserv",
                "cost",
                "misfit",
                "compar",
                "time",
                "field",
            )
        )
        limitations = [
            "Measured CPU configuration only; no higher-order, distributed or accelerator claim"
        ]
        if experimental and not selected:
            limitations.append(
                "No applicable completed run in this bundle; quantitative claim remains pending"
            )
        if any(run["metrics"]["numerical_validation"] != "pass" for run in selected):
            limitations.append(
                "Unfavorable numerical validations are retained; no accuracy guarantee"
            )
        if "mpas" in item["text"].lower():
            limitations.append(
                "Original polygons and shared triangulations are distinct; ESMC adapter accepts TRI/QUAD only"
            )
        entries.append(
            {
                **item,
                "status": "limited" if experimental else "supported",
                "evidence_type": "recorded experiments and inspected implementation"
                if experimental
                else "source/reproducibility inspection or editorial scope",
                "required_cases": sorted({run["case_id"] for run in selected}),
                "run_ids": [run["run_id"] for run in selected],
                "artifacts": sorted({path for paths in products.values() for path in paths}),
                "source_citations": [
                    {key: citation[key] for key in ("path", "symbol", "revision", "sha256")}
                    for citation in citations
                ],
                "limitations": limitations,
                "reproduction_command": "python libs/axis/paper/scripts/evidence.py --results RESULTS --check",
            }
        )
    return {
        "schema_version": 1,
        "bundle_sha256": digest(Path(results) / "suite.json"),
        "entries": entries,
    }


def relative_link(path, outline):
    return Path(os.path.relpath(Path(path).resolve(), outline.resolve().parent)).as_posix()


def evidence_block(outline, results, suite, runs, citations, products):
    quality = [run for run in runs if run["mode"] == "quality"]
    failures = [item for item in suite["runs"] if item["status"] != "complete"]
    lines = [
        START,
        "### Implementation and experiment evidence",
        "",
        "Regional terminology: the benchmark is Lambert conformal conic (LCC); earlier LLC labels are corrected here.",
        "",
        "**Verified formulation.** Inputs are float64 cell averages on a unit sphere. First-order weights are overlap area divided by full destination area (DstArea), or by covered destination area (FracArea). Masks exclude inactive cells; extrapolation is explicitly disabled. Conservation is assessed on the common covered domain, separately from analytical-field accuracy.",
        "",
        "**Verified geometry.** ArborX constructs spatial candidates. The general spherical kernel iteratively clips great-circle half-spaces in a Sutherland–Hodgman style; the legacy diagnostic label `host_spherical_greiner_hormann` does not establish use of the original linked-list Greiner–Hormann algorithm. Cartesian clipping and analytical rectangle/constant-latitude shortcuts are separate paths. The spherical kernel uses an inclusive signed-distance tolerance of -1e-14 and skips clip edges with squared normal length below 1e-30. These implementation tolerances do not prove exact geometry or arbitrary nonconvex support.",
        "",
        "**Assembly and reuse.** Native AXIS generation returns reusable weights, areas and fractions; CSR conversion and the first application are counted in setup. Repeated application calls KokkosSparse with destination overwrite. Native ESMC uses conservative store/regrid/release with GreatCircle lines, explicit DstArea/FracArea normalization, disabled extrapolation, and one MPI process. Timings exclude Python, transport, I/O and reference integration.",
        "",
        "**Independent truth.** Native references use analytic spherical area/vector-area integrals and independent gnomonic intersections, with double/extended-precision agreement and recorded roundoff estimates. They assume convex nonoverlapping cells within an open hemisphere. Empirical estimates are not formal interval certificates. Original MPAS polygons and shared centroid-fan triangulations retain separate identities and parent aggregation.",
        "",
        "Verified source citations (base revision plus exact working-tree SHA-256; archives retain the source snapshot):",
        "",
    ]
    for citation in citations:
        lines.append(
            f"- {citation['description']}: [`{citation['symbol']}`]({relative_link(ROOT / citation['path'], outline)}); SHA-256 `{citation['sha256']}`; base revision `{citation['revision']}`."
        )
    lines += [
        "",
        "### Answers supported by this recorded study",
        "",
        f"Recorded profile **{suite['profile']}**: {len(quality)} completed quality records and {len(failures)} incomplete/unsupported status entries. Overall numerical validation: **{suite.get('numerical_validation', 'unavailable')}**. Completion records and unfavorable numerical findings are separate; this scope does not establish universal accuracy or portability.",
        "",
        "1. **Accuracy and conservation:** the table below reports the measured error and conservation residual for each completed mesh/field/engine case, including unfavorable results.",
        "2. **ESMF comparison:** only shared great-circle fixtures with agreeing engine/reference areas are marked matched. Original variable-arity MPAS polygons remain unsupported by the TRI/QUAD ESMC adapter; transformed cases are identified explicitly.",
        "3. **Setup and reuse costs:** linked run records retain raw native repetitions, median/IQR, measured setup-and-reuse intervals at N=1 and N=100, and separately labeled estimates. Eligible speedup ratios are stored in `suite.json`; failed quality or unmatched geometry cannot support a speedup claim.",
        "",
        "| Case | Engine | Field | Match | RMS error | Absolute conservation residual | Validation |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in quality:
        metric = run["metrics"]
        link = relative_link(run["_folder"] / "run.json", outline)
        lines.append(
            f"| [{run['case_id']}]({link}) | {run['engine']} | {run['field']} | {run['matching']} | {metric['rms_error']} | {metric['conservation_absolute']} | {metric['numerical_validation']} |"
        )
    lines += [
        "",
        "| Requested case | Engine | Mode | Status | Cause |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in failures:
        cause = str(item.get("cause", "")).replace("|", "/")
        lines.append(
            f"| {item['case_id']} | {item['engine']} | {item.get('mode', 'reference')} | {item['status']} | {cause} |"
        )
    lines += ["", "### Publication products and remaining scope", ""]
    for product, paths in products.items():
        lines.append(
            f"- {product}: "
            + ", ".join(
                f"[{Path(path).suffix[1:]}]({relative_link(Path(results) / path, outline)})"
                for path in paths
            )
        )
    lines += [
        "",
        "Full-precision quality/configuration CSVs and artifact sidecars link every display to source run hashes and generating commands. The coverage inventory records every original outline bullet, its relevant case/run IDs, reviewed source, products, and limitations. Abstract/conclusion claims remain provisional until the numerical failures and unsupported scope have been reviewed. No higher-order, climate-fidelity, distributed scaling, memory, causal-stage or accelerator-portability finding is inferred from these CPU runs.",
        END,
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--outline", default="specs/AXIS_JAMES_outline.md")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--update", action="store_true")
    action.add_argument("--archive")
    parser.add_argument(
        "--related-results",
        action="append",
        default=[],
        help="Additional validated bundles to include with --archive",
    )
    arguments = parser.parse_args()
    if arguments.archive:
        archive(arguments.results, arguments.archive, arguments.related_results)
        return 0
    suite, runs = checked_runs(arguments.results)
    citations = check_sources()
    products = validate_publications(arguments.results)
    path = Path(arguments.outline)
    text = path.read_text()
    items = outline_items(text)
    inventory = load(INVENTORY)
    if {item["id"] for item in items} != {item["id"] for item in inventory["entries"]}:
        raise ValueError("Outline inventory is stale; review changed outline bullets")
    if arguments.update:
        inventory = evidence_inventory(items, runs, citations, products, arguments.results)
        save(INVENTORY, inventory)
        block = evidence_block(path, arguments.results, suite, runs, citations, products)
        text = text.replace("regional LLC", "regional LCC")
        if START in text:
            text = re.sub(
                re.escape(START) + r".*?" + re.escape(END), lambda match: block, text, flags=re.S
            )
        else:
            text = text.rstrip() + "\n\n" + block + "\n"
        path.write_text(text)
    elif inventory.get("bundle_sha256") != digest(Path(arguments.results) / "suite.json"):
        raise ValueError("Coverage inventory belongs to another scope; review with --update first")
    valid_ids = {run["run_id"] for run in runs}
    for entry in inventory["entries"]:
        if set(entry.get("run_ids", [])) - valid_ids or not entry.get("limitations"):
            raise ValueError(f"Invalid evidence entry: {entry['id']}")
    print(
        f"Verified {len(items)} outline entries, {len(runs)} complete run records, and all five required displays; numerical and scope limitations remain explicit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
