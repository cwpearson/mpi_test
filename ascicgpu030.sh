#! /bin/bash

yes | module clear -s
. $HOME/repos/Trilinos/cmake/std/atdm/load-env.sh Volta70-cuda-static-opt-rdc
module unload sems-openmpi
module load sems-openmpi/4.0.5 # needed for the --host syntax in mpirun

mpirun -np 8 -host ascicgpu030:4,ascicgpu032:4 \
./main-wrapper.sh