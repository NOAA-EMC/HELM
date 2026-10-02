#!/usr/bin/env bash
# check_no_eckit_recipe.sh -- CI gate: verifies AMIO declares no eckit dependency.
#
# Complements the three existing NO-ECKIT gates, which cover the *source* and
# *binary* surfaces:
#   check_no_eckit_includes.sh  -- no '#include <eckit/...>' in src/include/fortran
#   check_no_eckit_symbols.sh   -- no 'eckit::' symbols in the built libamio.so
#   check_build_no_eckit.sh     -- core reconfigures with eckit stripped
#
# None of those inspect the *dependency-declaration* surfaces, where a stale
# 'depends_on("eckit")' (spack recipe) or 'find_package(eckit)' (core CMake)
# can pull eckit into a build closure even though the compiled library never
# uses it. This gate closes that gap (Requirement 12.1, R12.8).
#
# Returns 0 (pass) when no eckit dependency declaration is found; 1 (fail)
# otherwise.
#
# Scope notes:
#   * The tests/ tree is EXCLUDED. Optional eckit-path tests guard their eckit
#     use behind 'if(eckit_FOUND)' and define AMIO_HAS_ECKIT only for those
#     targets; they are negative-path probes, not a library dependency. The gate
#     scripts themselves also mention eckit by name.
#   * Prose that NEGATES eckit ("never expose eckit", "MUST NOT declare an eckit
#     dependency") is not a dependency declaration and is not matched: the
#     patterns below require the 'depends_on("eckit' / 'find_package(eckit' call
#     forms, not the bare word.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AMIO_ROOT="${SCRIPT_DIR}/../.."

if [ ! -d "${AMIO_ROOT}" ]; then
    echo "WARN: AMIO root not found at ${AMIO_ROOT}"
    echo "PASS: Nothing to scan."
    exit 0
fi

fail=0

# 1) Spack recipe: any depends_on("eckit ...) or depends_on('eckit ...) is a
#    hard dependency declaration on eckit.
if [ -d "${AMIO_ROOT}/packages" ]; then
    RECIPE=$(grep -rnE "depends_on\([\"']eckit" "${AMIO_ROOT}/packages" 2>/dev/null || true)
    if [ -n "${RECIPE}" ]; then
        echo "FAIL: Found eckit dependency declarations in the spack recipe:"
        echo "${RECIPE}"
        echo ""
        echo "AMIO must not declare an eckit dependency. See Requirement 12.1."
        fail=1
    else
        echo "PASS: No eckit depends_on() declarations in packages/."
    fi
fi

# 2) Core CMake build files (exclude tests/): find_package(eckit) or a
#    target_link_libraries(... eckit ...) would reintroduce an eckit build edge.
CORE_CMAKE=$(
    find "${AMIO_ROOT}" \
        -path "${AMIO_ROOT}/tests" -prune -o \
        \( -name CMakeLists.txt -o -name '*.cmake' \) \
        -exec grep -nE "find_package\([[:space:]]*eckit|target_link_libraries\([^)]*[[:space:]]eckit[[:space:])]" {} + 2>/dev/null || true
)
if [ -n "${CORE_CMAKE}" ]; then
    echo "FAIL: Found eckit dependency declarations in core CMake build files:"
    echo "${CORE_CMAKE}"
    echo ""
    echo "The core library must not find_package or link eckit. See Requirement 12.1."
    fail=1
else
    echo "PASS: No eckit find_package()/target_link_libraries() in core CMake (tests/ excluded)."
fi

exit "${fail}"
