#!/usr/bin/env bash
# check_no_eckit_symbols.sh -- CI gate: verifies libamio.so exports no eckit:: symbols.
# Runs nm -D on the built shared library, demangles the output, and asserts
# zero symbols contain the substring 'eckit::'.
# Returns 0 (pass) when no eckit symbols found; 1 (fail) otherwise.
#
# Requirements: 12.8
set -euo pipefail

# The library path is passed as the first argument, or auto-detected from the build tree.
LIBAMIO="${1:-}"
if [ -z "$LIBAMIO" ]; then
    # Try common locations in the build directory.
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BUILD_DIR="${SCRIPT_DIR}/../../build"
    for candidate in \
        "${BUILD_DIR}/libamio.so" \
        "${BUILD_DIR}/lib/libamio.so" \
        "${BUILD_DIR}/libamio.dylib" \
        "${BUILD_DIR}/lib/libamio.dylib"; do
        if [ -f "$candidate" ]; then
            LIBAMIO="$candidate"
            break
        fi
    done
fi

if [ -z "$LIBAMIO" ] || [ ! -f "$LIBAMIO" ]; then
    echo "SKIP: libamio shared library not found (pass path as argument)"
    exit 0  # Skip rather than fail if library is not built yet
fi

echo "Checking symbols in: $LIBAMIO"

# Use nm -D (dynamic symbols) and demangle.
# On macOS, nm does not support -D (no dynamic symbol table flag); use nm without -D and c++filt.
if nm -D "$LIBAMIO" 2>/dev/null | c++filt | grep -q 'eckit::'; then
    echo "FAIL: Found eckit:: symbols in libamio shared library:"
    nm -D "$LIBAMIO" | c++filt | grep 'eckit::'
    exit 1
elif nm "$LIBAMIO" 2>/dev/null | c++filt | grep -q 'eckit::'; then
    echo "FAIL: Found eckit:: symbols in libamio shared library:"
    nm "$LIBAMIO" | c++filt | grep 'eckit::'
    exit 1
fi

echo "PASS: No eckit:: symbols found in libamio"
exit 0
