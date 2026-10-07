#!/bin/bash
# Generate API documentation: Doxygen XML → Moxygen Markdown → docs/site/api/
#
# Usage:
#   ./docs/generate-api.sh
#
# Prerequisites:
#   - doxygen (system package)
#   - npx / node (for moxygen)
#   - pip install mkdocs mkdocs-material (for site build)
#
# This script is also invoked by the CMake `docs` target.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== AMIO Documentation Pipeline ==="
echo ""

# Step 1: Run Doxygen to generate XML
echo "[1/3] Running Doxygen (XML output)..."
cd "${PROJECT_ROOT}"
doxygen docs/Doxyfile
echo "  XML output: docs/xml/"

# Step 2: Run Moxygen to convert XML to Markdown
echo "[2/3] Running Moxygen (XML → Markdown)..."
mkdir -p docs/site/api
npx moxygen docs/xml --output docs/site/api/api-%s.md --classes --pages
echo "  Markdown output: docs/site/api/"

# Step 3: Build MkDocs site
echo "[3/3] Building MkDocs site..."
cd docs
mkdocs build
echo "  Site output: docs/site/ (served from docs/_build/)"

echo ""
echo "=== Done! ==="
echo "To preview locally: cd docs && mkdocs serve"
