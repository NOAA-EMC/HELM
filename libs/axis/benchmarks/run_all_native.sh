#!/usr/bin/env bash
# Rerun every README benchmark table natively on macOS (arm64, Apple Silicon).
# Uses the axis-benchmark-env conda env (single libomp) and the build-macos
# _core .so (copied into python/axis/). Great-circle and cartesian passes
# run separately so conservative rows in each table fill independently.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=python
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
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

# Table 1: regular 720x360 -> 1440x720, cosine
run t1_gc --src-size 720x360 --dst-size 1440x720 --dst-grid-type regular \
    --methods bilinear,nearest,conservative --field cosine
run t1_cart --src-size 720x360 --dst-size 1440x720 --dst-grid-type regular \
    --methods conservative --field cosine --line-type cartesian

# Table 2: regular 720x360 -> 3600x1800, constant
run t2_gc --src-size 720x360 --dst-size 3600x1800 --dst-grid-type regular \
    --methods nearest,conservative --field constant
run t2_cart --src-size 720x360 --dst-size 3600x1800 --dst-grid-type regular \
    --methods conservative --field constant --line-type cartesian

# Table 3: LCC 120x120 -> 100x100 regular, conservative
run t3_gc --src-size 120x120 --dst-size 100 --grid-type lcc --dst-grid-type regular \
    --methods conservative --field cosine
run t3_cart --src-size 120x120 --dst-size 100 --grid-type lcc --dst-grid-type regular \
    --methods conservative --field cosine --line-type cartesian

# Table 4: MPAS 10000 -> 90x90 regular
run t4_gc --src-size 10000 --grid-type mpas --dst-size 90x90 --dst-grid-type regular \
    --methods bilinear,nearest,conservative --field cosine
run t4_cart --src-size 10000 --grid-type mpas --dst-size 90x90 --dst-grid-type regular \
    --methods conservative --field cosine --line-type cartesian

# Table 5: regular 360x180 -> MPAS 2000
run t5_gc --src-size 360x180 --dst-size 2000 --dst-grid-type mpas \
    --methods nearest,conservative --field cosine
run t5_cart --src-size 360x180 --dst-size 2000 --dst-grid-type mpas \
    --methods conservative --field cosine --line-type cartesian

echo "ALL DONE"
