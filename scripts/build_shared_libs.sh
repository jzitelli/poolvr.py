#!/usr/bin/bash
project_root="${0%/*}/.."


echo '
building collisions.dll...
'
gfortran -shared -static -o $project_root"/poolvr/physics/collisions.dll" \
	 -ffree-line-length-none \
	 -v \
     $project_root"/poolvr/physics/collisions.f90"


echo '
building poly_solvers.dll...
'
gfortran -shared -static \
	 -v \
	 -o $project_root"/poolvr/physics/poly_solvers.dll" $project_root"/poolvr/physics/poly_solvers.f90"


rm *.mod
