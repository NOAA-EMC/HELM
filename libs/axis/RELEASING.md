# AXIS wheel releases

The AXIS distribution is `axis-regrid`; import it as `axis`. Its source is
`libs/axis` relative to the HELM repository root.

## Wheel targets

The `AXIS wheels and releases` workflow builds and installs wheels for CPython
3.10–3.14 on Linux glibc 2.28+ (x86_64 and aarch64) and macOS 13+
(x86_64 and arm64). Each repaired wheel must import the native module, create
a mesh, pass pip's dependency check, and run the CLI before publication.
Windows, musl, PyPy, free-threaded Python, and GPU backends are outside this matrix.

Wheels contain serial CPU Kokkos and a bundled MPICH 4.2.3 runtime, built with
the socket device and without external fabric plugins. They disable native PROJ
and NetCDF support; install optional Python file I/O with the `netcdf` extra.
The MPI launcher is not shipped. These wheels target local Python regridding;
use a source build against the site's MPI/Kokkos for multi-node or GPU workloads.
Do not mix the bundled runtime with a different MPI implementation in one process.
Source builds retain their existing OpenMP and optional library defaults.

## Release procedure

1. Select a stable semantic version: PATCH for compatible fixes, MINOR for
   compatible features, MAJOR for incompatible public API changes. Before 1.0,
   document breaking changes and increment MINOR.
2. Update `project.version` in `pyproject.toml` and `project(AXIS VERSION ...)`
   in `CMakeLists.txt` together. Merge the change and this workflow into
   `feature/paper-source`, then wait for all four wheel jobs to pass.
3. Tag that exact commit and push the tag:

   ```bash
   git switch feature/paper-source
   git pull --ff-only
   git tag -a axis-v0.1.0 -m "AXIS 0.1.0"
   git push origin axis-v0.1.0
   ```

   Replace 0.1.0 with the selected version. Tags use the AXIS prefix to avoid
   coupling this library's version to other HELM components.

Tag builds verify stable SemVer, package/CMake/tag agreement, and ancestry on
`feature/paper-source`. All 20 wheels must pass before the release job creates
a draft, uploads wheels and SHA256SUMS, and publishes generated release notes.
No PyPI credentials are needed: distribution is through GitHub Releases.
A branch push, pull request, or manual run only creates workflow artifacts.

Download the matching wheel from the GitHub release and install it with
`python -m pip install ./axis_regrid-<version>-<python>-<abi>-<platform>.whl`.
For a failed build, fix the branch and issue a new version/tag; do not move
published tags. If upload fails after draft creation, inspect the draft before
retrying; the workflow fails on an existing release instead of overwriting it.
