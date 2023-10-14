#!/usr/bin/bash
project_root="${0%/*}/.."


echo '
building _collisions.dll...
'
gfortran -shared -fPIC -o $project_root"/poolvr/physics/_collisions.so" \
	 -ffree-line-length-none \
	 -v \
     $project_root"/poolvr/physics/collisions.f90"


echo '
building _poly_solvers.so...
'
gcc -shared -fPIC -o $project_root"/poolvr/physics/_poly_solvers.so" \
    -v \
    $project_root"/poolvr/physics/poly_solvers.c"


echo '
building _fpoly_solvers.so...
'
gfortran -shared -fPIC \
	 -v \
	 -o $project_root"/poolvr/physics/_fpoly_solvers.so" $project_root"/poolvr/physics/poly_solvers.f90"


rm *.mod
