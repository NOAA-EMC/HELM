"""Capture exact code and runtime environment; never infer missing versions."""

import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from artifacts import ROOT, digest, load


def git(*args):
    # Prefer the exported marker even when an archive happens to be extracted
    # inside an unrelated Git checkout.
    marker = ROOT / "AXIS_PAPER_SOURCE.json"
    if marker.exists():
        snapshot = load(marker)
        if args == ("rev-parse", "HEAD"):
            return snapshot["source_commit"] + "\n"
        if args[:1] == ("diff",):
            return ""
        if args[:1] == ("ls-files",):
            return "" if "--others" in args else "\n".join(snapshot["files"]) + "\n"
        raise ValueError(f"Unsupported git operation in archived snapshot: {args}")
    result = subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def capture(output):
    output = Path(output)
    patch = output / "source.patch"
    patch.write_text(git("diff", "--binary", "HEAD"))
    untracked = git("ls-files", "--others", "--exclude-standard").splitlines()
    versions = {}
    for name in ("numpy", "matplotlib", "xarray", "netCDF4", "pyproj"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "source_commit": git("rev-parse", "HEAD").strip(),
        "source_origin": "archived working-tree snapshot"
        if (ROOT / "AXIS_PAPER_SOURCE.json").exists()
        else "git working tree",
        "archive_snapshot_sha256": digest(ROOT / "AXIS_PAPER_SOURCE.json")
        if (ROOT / "AXIS_PAPER_SOURCE.json").exists()
        else None,
        "dirty_patch_sha256": digest(patch),
        "untracked_files": {p: digest(ROOT / p) for p in untracked if (ROOT / p).is_file()},
        "python": sys.version,
        "python_packages": versions,
        "hardware": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "argv": sys.argv,
        "thread_environment": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "KOKKOS_NUM_THREADS", "ESMF_NUM_THREADS")
        },
        "container_image": os.environ.get("AXIS_PAPER_CONTAINER_IMAGE"),
    }
