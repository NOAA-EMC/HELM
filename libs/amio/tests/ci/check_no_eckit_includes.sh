#!/usr/bin/env bash
# check_no_eckit_includes.sh -- CI gate: verifies zero eckit includes in AMIO sources.
# Returns 0 (pass) when no eckit include is found; 1 (fail) otherwise.
#
# Validates: Requirements 12.1, 12.9
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMIO_ROOT="${SCRIPT_DIR}/../.."

# Directories to scan (only scan if they exist).
# tests/ is included: AMIO has no eckit integration at all, so no test source
# may include an eckit header either. The gate scripts themselves live in
# tests/ci/ and are excluded below so this script does not match its own grep.
SCAN_DIRS=()
for dir in "${AMIO_ROOT}/src" "${AMIO_ROOT}/include" "${AMIO_ROOT}/fortran" "${AMIO_ROOT}/tests"; do
    if [ -d "$dir" ]; then
        SCAN_DIRS+=("$dir")
    fi
done

if [ ${#SCAN_DIRS[@]} -eq 0 ]; then
    echo "WARN: No src/, include/, fortran/, or tests/ directories found under ${AMIO_ROOT}"
    echo "PASS: Nothing to scan."
    exit 0
fi

# Search for #include directives referencing eckit/ in source and header files.
# Pattern matches both #include <eckit/...> and #include "eckit/..."
# Excludes tests/ci/ (this gate's own scripts, which name eckit by design).
MATCHES=$(grep -rn --include='*.cpp' --include='*.hpp' --include='*.h' \
    --include='*.c' --include='*.f90' \
    --exclude-dir=ci \
    '#include.*eckit/' \
    "${SCAN_DIRS[@]}" 2>/dev/null || true)

if [ -n "$MATCHES" ]; then
    echo "FAIL: Found eckit #include directives in AMIO source files:"
    echo "$MATCHES"
    echo ""
    echo "All eckit includes must be removed. See Requirement 12.1."
    exit 1
fi

echo "PASS: No eckit #include directives found in src/, include/, fortran/, or tests/"
exit 0
