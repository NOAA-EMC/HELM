#!/usr/bin/env bash
# Rerun Table 4 (MPAS 10000 -> 90x90 regular) natively on macOS (arm64).
# Uses the axis-benchmark-env conda env and the build-macos _core .so
# (copied into python/axis/). Sources the fixed CCW-winding MPAS generator.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python
ENVBIN=/opt/homebrew/Caskroom/miniforge/base/envs/axis-benchmark-env/bin
PY=$ENVBIN/python
export PATH="$ENVBIN:$PATH"  # so python-cdo finds the cdo binary
OUT="$(pwd)/benchmarks/results"
mkdir -p "$OUT"

run() {
  local name="$1"; shift
  echo "=== $name ==="
  "$PY" -u benchmarks/compare_cdo.py "$@" > "$OUT/$name.log" 2>&1
  echo "    exit=$? -> $OUT/$name.log"
}

run t4_gc --src-size 10000 --grid-type mpas --dst-size 90x90 --dst-grid-type regular \
    --methods bilinear,nearest,conservative --field cosine
run t4_cart --src-size 10000 --grid-type mpas --dst-size 90x90 --dst-grid-type regular \
    --methods conservative --field cosine --line-type cartesian
