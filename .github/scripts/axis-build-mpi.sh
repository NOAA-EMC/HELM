#!/usr/bin/env bash
# Bundle a small MPICH runtime; avoid external fabric/provider plugins.
set -euo pipefail
project_dir=$(cd "$1" && pwd)
version=4.2.3
prefix=/tmp/axis-mpich
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
curl --fail --location --retry 3 \
  "https://www.mpich.org/static/downloads/${version}/mpich-${version}.tar.gz" \
  --output mpich.tar.gz
tar -xzf mpich.tar.gz
cd "mpich-${version}"
./configure --prefix="$prefix" --with-device=ch3:sock \
  --disable-fortran --disable-romio --disable-cxx \
  --enable-shared --disable-static
make -j3
make install
# auditwheel/delocate copies libmpi into each repaired wheel. The MPI launcher
# is intentionally not distributed; cluster builds should use the site's MPI.
cp COPYRIGHT "$prefix/MPICH-COPYRIGHT"
cp COPYRIGHT "$project_dir/python/axis/MPICH-COPYRIGHT"
