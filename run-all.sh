#! /bin/bash
# ./run-all.sh | grep 'PASS\|FAIL'

set -eou pipefail

run_cpu_test() {
    # mvapich2 makes a bunch of spam if you use non-GPU buffers :(
    export MV2_USE_CUDA=0
    # mvapich2 makes a bunch of spam if its build with CUDA but the above is set :(
    export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
    mpirun -n 4 $1 && echo PASS: mpirun -n 4 $1 || echo FAIL: mpirun -n 4 $1
}

run_gpu_test() {
    export MV2_USE_CUDA=1
    mpirun -n 4 $1 && echo PASS: mpirun -n 4 $1 || echo FAIL: mpirun -n 4 $1
}

# test that we can run something
run_cpu_test hostname

# rotate an integer around ranks
run_cpu_test ./main

# rotate with mpi-put
run_cpu_test ./one-sided