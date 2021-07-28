#!/usr/bin/env bash
# sudo apt-get install libcgal-dev
#or module load cgal
cmake .
make -j32
./evaluation Icosahedron.off Icosahedron.xyz
