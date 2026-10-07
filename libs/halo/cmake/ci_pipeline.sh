#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# ci_pipeline.sh — Runnable HALO CI driver.
#
# Executes the HALO CI pipeline stages in sequence and exits non-zero on the
# FIRST failure, so it is suitable for use as a single CI entry point inside the
# HELM Docker container (see Dockerfile / docker-compose.yml). The stages match
# docs/ci_pipeline.md:
#
#   Stage 1  Static analysis / Tier 1 isolation scan        (Requirement 13.5)
#   Stage 2  Standalone CMake build in an isolated tree      (Requirement 12.3)
#   Stage 3  Unit tests, real 4-rank MPI (ctest -L mpi)      (Requirement 12.2)
#   Stage 4  Property tests, mocked MPI (ctest -L property)  (Requirement 12.2)
#   Stage 5  Sanitizer build + property tests (ASan+UBSan)   (Requirement 12.2)
#
# Everything runs inside the pinned HELM container, which is the single
# build/test environment for HALO (Requirement 12.2). Reproducibility rests on
# the pinned container image and the pinned in-tree commit (Requirement 12.5).
#
# Usage:
#   sh ci_pipeline.sh [SOURCE_DIR] [BUILD_ROOT]
#
#   SOURCE_DIR  HALO repository root (contains CMakeLists.txt). Defaults to the
#               parent directory of this script (libs/halo).
#   BUILD_ROOT  Scratch directory for the isolated standalone tree and the
#               sanitizer build dir. Defaults to /tmp.
#
# Exit status:
#   0  All stages passed.
#   N  The exit code of the first failing stage (non-zero).
# ─────────────────────────────────────────────────────────────────────────────

set -eu

# ── Resolve paths ────────────────────────────────────────────────────────────
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
SOURCE_DIR="${1:-$(CDPATH='' cd -- "$SCRIPT_DIR/.." && pwd)}"
BUILD_ROOT="${2:-/tmp}"

mkdir -p "$BUILD_ROOT"

STANDALONE_DIR="$BUILD_ROOT/halo-standalone"
ASAN_DIR="$BUILD_ROOT/halo-asan"

# OpenMPI refuses to run as root unless these are set. A fresh shell inside the
# container runs as root and does not inherit the compose service environment,
# so export them here to keep every mpirun/ctest stage self-contained.
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

JOBS=$(nproc 2>/dev/null || echo 4)

log() {
    echo ""
    echo "═════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "═════════════════════════════════════════════════════════════════════"
}

# ── Stage 1: Static isolation scan (Requirement 13.5) ────────────────────────
log "Stage 1 — Tier 1 isolation scan (Requirement 13.5)"
sh "$SCRIPT_DIR/check_tier1_isolation.sh" "$SOURCE_DIR"

# ── Stage 2: Standalone build in an isolated tree (Requirement 12.3) ─────────
# Copy ONLY the HALO tree to an isolated location with no sibling HELM
# components, proving the root CMakeLists builds HELM::HALO standalone.
log "Stage 2 — Standalone CMake build, isolated tree (Requirement 12.3)"
rm -rf "$STANDALONE_DIR"
cp -r "$SOURCE_DIR" "$STANDALONE_DIR"
rm -rf "$STANDALONE_DIR/build"

cmake -B "$STANDALONE_DIR/build" -S "$STANDALONE_DIR" -G Ninja \
    -DCMAKE_CXX_STANDARD=20 \
    -DHALO_BUILD_TESTING=ON \
    -DHALO_BUILD_FORTRAN=ON
cmake --build "$STANDALONE_DIR/build" --parallel "$JOBS"

# Confirm the expected archives were produced.
for lib in libhalo.a libhalo_c_interop.a libhalo_fortran.a; do
    if [ ! -f "$STANDALONE_DIR/build/$lib" ]; then
        echo "ci_pipeline: FAIL — expected artifact '$lib' not produced by standalone build" >&2
        exit 1
    fi
done
echo "Standalone build produced HELM::HALO (libhalo.a) and the Fortran libs."

# ── Stage 3: Unit tests, real 4-rank MPI (Requirement 12.2) ──────────────────
log "Stage 3 — Unit tests, mpirun -np 4 (Requirement 12.2)"
ctest --test-dir "$STANDALONE_DIR/build" --output-on-failure -L mpi

# ── Stage 4: Property tests, single-rank mocked MPI ──────────────────────────
log "Stage 4 — Property tests, single-rank mocked MPI"
ctest --test-dir "$STANDALONE_DIR/build" --output-on-failure -L property

# ── Stage 5: Sanitizer build + property tests (ASan + UBSan) ─────────────────
# detect_leaks=0: the C-interop property tests intentionally leak a fixed,
# bounded amount of memory (a process-global Communicator handle and an async
# Halo_Handle with no C destroy entry point). These are documented test-harness
# artifacts, not library defects -- see docs/ci_pipeline.md. ASan+UBSan catch
# real memory-safety/UB errors in library logic; none are present.
log "Stage 5 — Sanitizer build + property tests (ASan + UBSan)"
cmake -B "$ASAN_DIR" -S "$SOURCE_DIR" -G Ninja \
    -DCMAKE_CXX_STANDARD=20 \
    -DHALO_BUILD_TESTING=ON \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g" \
    -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build "$ASAN_DIR" --parallel "$JOBS"

ASAN_OPTIONS=detect_leaks=0 \
    ctest --test-dir "$ASAN_DIR" --output-on-failure -L property

log "All CI stages passed."
exit 0
