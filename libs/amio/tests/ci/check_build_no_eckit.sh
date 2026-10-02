#!/usr/bin/env bash
# check_build_no_eckit.sh -- CI gate: verifies AMIO builds without eckit in PREFIX_PATH.
# This test configures and compiles AMIO with HELM::CONF, HELM::HALO, HELM::LOGS
# present but eckit explicitly absent from CMAKE_PREFIX_PATH.
# Returns 0 (pass) when build succeeds; 1 (fail) otherwise.
#
# Validates: Requirements 12.2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMIO_ROOT="${SCRIPT_DIR}/../.."

# Use a dedicated build directory to avoid polluting the normal build.
BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/amio_no_eckit_build.XXXXXX")
trap 'rm -rf "$BUILD_DIR"' EXIT

# Construct a CMAKE_PREFIX_PATH that includes ONLY HELM libraries (no eckit).
# If HELM_PREFIX_PATH is set externally (e.g., by CI), use it directly.
# Otherwise, attempt to derive it from the current build's CMake cache.
if [ -z "${HELM_PREFIX_PATH:-}" ]; then
    # Try to extract the prefix path from the outer build's CMakeCache.txt.
    # The outer build directory is typically two levels up from tests/ci/.
    OUTER_BUILD_CACHE="${AMIO_ROOT}/build/CMakeCache.txt"
    if [ -f "$OUTER_BUILD_CACHE" ]; then
        HELM_PREFIX_PATH=$(grep -m1 '^CMAKE_PREFIX_PATH' "$OUTER_BUILD_CACHE" \
            | sed 's/^CMAKE_PREFIX_PATH[^=]*=//' || true)
    fi
fi

# Strip any eckit paths from the prefix path.
# This ensures eckit is genuinely absent even if HELM_PREFIX_PATH inadvertently contains it.
if [ -n "${HELM_PREFIX_PATH:-}" ]; then
    CLEAN_PREFIX_PATH=""
    IFS=';' read -ra PATH_ENTRIES <<< "$HELM_PREFIX_PATH"
    for entry in "${PATH_ENTRIES[@]}"; do
        case "$entry" in
            *eckit*|*ECKIT*) continue ;;
            *) CLEAN_PREFIX_PATH="${CLEAN_PREFIX_PATH:+${CLEAN_PREFIX_PATH};}${entry}" ;;
        esac
    done
    HELM_PREFIX_PATH="$CLEAN_PREFIX_PATH"
fi

echo "=== AMIO Build Verification (no-eckit prefix) ==="
echo "  AMIO source: ${AMIO_ROOT}"
echo "  Build dir:   ${BUILD_DIR}"
echo "  PREFIX_PATH: ${HELM_PREFIX_PATH:-<not set>}"
echo ""

echo "Configuring AMIO without eckit in CMAKE_PREFIX_PATH..."
if ! cmake -S "${AMIO_ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_PREFIX_PATH="${HELM_PREFIX_PATH:-}" \
    -DAMIO_BUILD_TESTING=OFF \
    -DAMIO_BUILD_DOCS=OFF \
    -DAMIO_BUILD_EXAMPLES=OFF \
    2>&1; then
    echo ""
    echo "FAIL: CMake configuration failed without eckit"
    exit 1
fi

echo ""
echo "Building AMIO..."
if ! cmake --build "${BUILD_DIR}" --parallel 2>&1; then
    echo ""
    echo "FAIL: Build failed without eckit in PREFIX_PATH"
    exit 1
fi

echo ""
echo "PASS: AMIO builds successfully without eckit in CMAKE_PREFIX_PATH"
exit 0
