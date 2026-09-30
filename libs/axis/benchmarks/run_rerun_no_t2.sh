#!/usr/bin/env bash
# Rerun README benchmark tables 1,3,4,5 (GC + cartesian) with the FRESH _core build.
# Table 2 is run separately after Docker memory is raised.
set -uo pipefail
cd /workspace/helm-project/libs/axis
export PYTHONPATH=python
OUT=/workspace/helm-project/libs/axis/benchmarks/results
mkdir -p "$OUT"

run() {
  local name="$1"; shift
  echo "=== $name ==="
  mamba run -n axis-bench python3 -u benchmarks/compare_cdo.py "$@" > "$OUT/$name.log" 2>&1
  echo "    exit=$? -> $OUT/$name.log"
}

# Table 1: regular 720x360 -> 1440x720, cosine
run t1_gc --src-size 720x360 --dst-size 1440x720 --dst-grid-type regular \
    --methods bilinear,nearest,conservative --field cosine
run t1_cart --src-size 720x360 --dst-size 1440x720 --dst-grid-type regular \
    --methods conservative --field cosine --line-type cartesian

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

echo "ALL DONE (t1,t3,t4,t5)"
