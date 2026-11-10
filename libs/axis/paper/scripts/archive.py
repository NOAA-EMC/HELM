"""Freeze complete inputs, code identity, raw observations, displays, and commands."""

import shutil
from pathlib import Path
from artifacts import ROOT, checked_runs, digest, save, load
from publication import validate_publications
from provenance import git


def archive(results, destination):
    results = Path(results).resolve()
    destination = Path(destination).resolve()
    if destination.is_relative_to(results) or destination.exists():
        raise ValueError("Archive must be a new directory outside results")
    checked_runs(results)
    validate_publications(results)
    citations = load(ROOT / "libs/axis/paper/evidence/implementation_citations.json")["citations"]
    for citation in citations:
        path = ROOT / citation["path"]
        if digest(path) != citation["sha256"] or citation["symbol"] not in path.read_text():
            raise ValueError(f"Stale archived implementation citation: {path}")
    shutil.copytree(results, destination / "results")
    source = destination / "source"
    source.mkdir()
    paths = set(git("ls-files").splitlines())
    paths.update(
        git("ls-files", "--others", "--exclude-standard", "--", "libs/axis/paper").splitlines()
    )
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "specs/004-axis-paper").rglob("*.md"))
    paths.add("specs/AXIS_JAMES_outline.md")
    for name in sorted(paths):
        p = ROOT / name
        if p.is_file() and p.is_relative_to(ROOT):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    # Keep manuscript links usable after moving results beside the source tree.
    outline = source / "specs/AXIS_JAMES_outline.md"
    import os

    original_prefix = Path(os.path.relpath(results, ROOT / "specs")).as_posix()
    archived_prefix = Path(os.path.relpath(destination / "results", outline.parent)).as_posix()
    outline.write_text(outline.read_text().replace(original_prefix + "/", archived_prefix + "/"))
    save(
        source / "AXIS_PAPER_SOURCE.json",
        {
            "schema_version": 1,
            "source_commit": git("rev-parse", "HEAD").strip(),
            "description": "Exact exported working-tree snapshot; original source.patch is under results",
            "reviewed_revisions": sorted({citation["revision"] for citation in citations}),
            "files": {
                str(path.relative_to(source)): digest(path)
                for path in source.rglob("*")
                if path.is_file()
            },
        },
    )
    save(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "source_commit": git("rev-parse", "HEAD").strip(),
            "files": {
                str(p.relative_to(destination)): digest(p)
                for p in destination.rglob("*")
                if p.is_file()
            },
        },
    )
