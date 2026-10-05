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
#   * AMIO has NO eckit integration anywhere, so every surface is scanned. The
#     only exclusions are this gate's own scripts (tests/ci/), which name eckit
#     by design, and the header-isolation probe, which lists eckit among
#     FORBIDDEN transitive headers (a negative assertion that must keep the name).
#   * Prose that NEGATES eckit ("never expose eckit", "MUST NOT declare an eckit
#     dependency", "no eckit #include directives") is not a dependency
#     declaration and is not matched: the patterns below require the
#     'depends_on("eckit' / 'find_package(eckit' / 'eckit::' code forms.
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

# 2) Every CMake build file (core AND tests, excluding this gate's own
#    tests/ci/ directory): find_package(eckit), an eckit link edge, or an
#    if(eckit_FOUND) guard would reintroduce an eckit build path.
CMAKE_HITS=$(
    find "${AMIO_ROOT}" \
        -path "${AMIO_ROOT}/tests/ci" -prune -o \
        \( -name CMakeLists.txt -o -name '*.cmake' \) \
        -exec grep -nE "find_package\([[:space:]]*eckit|target_link_libraries\([^)]*[[:space:]]eckit[[:space:])]|eckit_FOUND|AMIO_HAS_ECKIT" {} + 2>/dev/null || true
)
if [ -n "${CMAKE_HITS}" ]; then
    echo "FAIL: Found eckit build references in CMake files:"
    echo "${CMAKE_HITS}"
    echo ""
    echo "No CMake file may find_package, link, or conditionally enable eckit. See Requirement 12.1."
    fail=1
else
    echo "PASS: No eckit find_package()/link/eckit_FOUND/AMIO_HAS_ECKIT in any CMake file."
fi

# 3) No eckit C++ symbols anywhere in src/ or tests/ (code form only, so prose
#    that merely negates eckit is not flagged).
CODE_HITS=$(
    grep -rnE "eckit::" \
        --include='*.cpp' --include='*.hpp' --include='*.h' --include='*.c' \
        "${AMIO_ROOT}/src" "${AMIO_ROOT}/tests" "${AMIO_ROOT}/include" 2>/dev/null || true
)
if [ -n "${CODE_HITS}" ]; then
    echo "FAIL: Found eckit:: symbols in AMIO sources or tests:"
    echo "${CODE_HITS}"
    echo ""
    echo "AMIO has no eckit integration; remove these references. See Requirement 12.1."
    fail=1
else
    echo "PASS: No eckit:: symbols in src/, include/, or tests/."
fi

exit "${fail}"
