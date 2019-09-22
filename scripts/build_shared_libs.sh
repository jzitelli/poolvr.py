#!/usr/bin/bash
project_root="${0%/*}/.."


echo '
building collisions.dll...
'
gfortran -v -shared -static -o $project_root"/poolvr/physics/collisions.dll" \
     $project_root"/poolvr/physics/collisions.f90"


echo '
building poly_solvers.dll...
'
# gcc -v -shared -static -o $project_root"/poolvr/physics/poly_solvers.dll" \
#     $project_root"/poolvr/physics/poly_solvers.c"
gfortran -v -shared -static -o $project_root"/poolvr/physics/poly_solvers.dll" \
     $project_root"/poolvr/physics/poly_solvers.f90"


rm *.mod
