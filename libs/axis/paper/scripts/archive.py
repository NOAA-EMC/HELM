"""Freeze complete inputs, code identity, raw observations, displays, and commands."""

import shutil
from pathlib import Path
from artifacts import ROOT, checked_runs, digest, save, load
from publication import validate_publications
from provenance import git


def archive(results, destination, related_results=()):
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
    related = [Path(path).resolve() for path in related_results]
    if len({path.name for path in related}) != len(related):
        raise ValueError("Related result directories must have distinct names")
    for path in related:
        checked_runs(path)
        validate_publications(path)
        if destination.is_relative_to(path):
            raise ValueError("Archive cannot be placed inside a related result directory")
    shutil.copytree(results, destination / "results")
    for path in related:
        shutil.copytree(path, destination / "related_results" / path.name)
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
    text = outline.read_text().replace(original_prefix + "/", archived_prefix + "/")
    for path in related:
        original = Path(os.path.relpath(path, ROOT / "specs")).as_posix()
        archived = Path(
            os.path.relpath(destination / "related_results" / path.name, outline.parent)
        ).as_posix()
        text = text.replace(original + "/", archived + "/")
    outline.write_text(text)
    # Documentation outside the manuscript also links to result artifacts.
    # Rewrite only Markdown targets resolving inside explicitly bundled runs.
    import re

    result_locations = [(results, destination / "results")]
    result_locations.extend((path, destination / "related_results" / path.name) for path in related)
    # Keep documentation links to this archive portable as well.
    result_locations.append((destination, destination))
    for document in source.rglob("*.md"):
        original_document = ROOT / document.relative_to(source)

        def relocate_link(match):
            target = match.group(1)
            if "://" in target or target.startswith("#"):
                return match.group(0)
            path, separator, fragment = target.partition("#")
            resolved = (original_document.parent / path).resolve()
            for original_result, archived_result in result_locations:
                if resolved.is_relative_to(original_result):
                    replacement = archived_result / resolved.relative_to(original_result)
                    relative = Path(os.path.relpath(replacement, document.parent)).as_posix()
                    return "](" + relative + (separator + fragment if separator else "") + ")"
            return match.group(0)

        document.write_text(re.sub(r"\]\(([^)]+)\)", relocate_link, document.read_text()))
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
