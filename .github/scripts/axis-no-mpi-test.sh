#!/usr/bin/env bash
# Run in a clean image with build tools, Python, uv and no MPI installation.
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
if command -v mpicxx >/dev/null || command -v mpiexec >/dev/null; then
    echo 'MPI tools are present; use a clean MPI-free environment' >&2
    exit 1
fi
if find /usr/include /usr/local/include -name mpi.h -print -quit 2>/dev/null | grep -q . ||
   ldconfig -p 2>/dev/null | grep -E 'libmpi(_cxx)?\.so' >/dev/null; then
    echo 'MPI headers or runtime libraries are present' >&2
    exit 1
fi
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export CMAKE_ARGS="${CMAKE_ARGS:-} -DCMAKE_DISABLE_FIND_PACKAGE_MPI=TRUE -DAXIS_ENABLE_PROJ=OFF -DAXIS_ENABLE_NETCDF=OFF"
python3 -m venv "$scratch/pip"
"$scratch/pip/bin/python" -m pip install "$root/libs/axis[dask,distributed]" pytest
PATH="$scratch/pip/bin:$PATH" "$scratch/pip/bin/python" "$root/.github/scripts/axis-wheel-smoke.py"
PATH="$scratch/pip/bin:$PATH" "$scratch/pip/bin/python" -m pytest -q \
    "$root/libs/axis/tests_python/test_axis_dask.py" \
    "$root/libs/axis/tests_python/test_axis_dask_distributed.py"
python3 -m venv "$scratch/uv"
uv pip install --python "$scratch/uv/bin/python" "$root/libs/axis"
PATH="$scratch/uv/bin:$PATH" "$scratch/uv/bin/python" "$root/.github/scripts/axis-wheel-smoke.py"
