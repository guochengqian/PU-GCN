#!/usr/bin/env bash
set -euo pipefail

# Ubuntu/Debian:
#   sudo apt-get update && sudo apt-get install -y cmake build-essential libcgal-dev libboost-all-dev
# HPC clusters:
#   module load cgal boost cmake

cmake .
make -j"${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ ! -x ./evaluation ]; then
  echo "ERROR: ./evaluation was not built. Install CGAL/Boost development packages and rerun this script." >&2
  exit 1
fi

./evaluation Icosahedron.off Icosahedron.xyz
