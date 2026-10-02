#!/usr/bin/env bash
# check_no_eckit_includes.sh -- CI gate: verifies zero eckit includes in AMIO sources.
# Returns 0 (pass) when no eckit include is found; 1 (fail) otherwise.
#
# Validates: Requirements 12.1, 12.9
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMIO_ROOT="${SCRIPT_DIR}/../.."

# Directories to scan (only scan if they exist).
SCAN_DIRS=()
for dir in "${AMIO_ROOT}/src" "${AMIO_ROOT}/include" "${AMIO_ROOT}/fortran"; do
    if [ -d "$dir" ]; then
        SCAN_DIRS+=("$dir")
    fi
done

if [ ${#SCAN_DIRS[@]} -eq 0 ]; then
    echo "WARN: No src/, include/, or fortran/ directories found under ${AMIO_ROOT}"
    echo "PASS: Nothing to scan."
    exit 0
fi

# Search for #include directives referencing eckit/ in source and header files.
# Pattern matches both #include <eckit/...> and #include "eckit/..."
MATCHES=$(grep -rn --include='*.cpp' --include='*.hpp' --include='*.h' \
    --include='*.c' --include='*.f90' \
    '#include.*eckit/' \
    "${SCAN_DIRS[@]}" 2>/dev/null || true)

if [ -n "$MATCHES" ]; then
    echo "FAIL: Found eckit #include directives in AMIO source files:"
    echo "$MATCHES"
    echo ""
    echo "All eckit includes must be removed. See Requirement 12.1."
    exit 1
fi

echo "PASS: No eckit #include directives found in src/, include/, or fortran/"
exit 0
