How to use:

1- run `cmake .`
2- run `make`
3- copy original `.off` mesh file to this directory
4- copy predicted points `.ply` of that same mesh
5- run `./evaluation mesh.off pred.ply`

results will be in _point2mesh file, use it for metric_calculation/evaluate.py
