#! /bin/bash
# ./run-all.sh | grep 'PASS\|FAIL'


run_test() {
    mpirun -n 4 $1 && echo PASS: mpirun -n 4 $1 || echo FAIL: mpirun -n 4 $1
}

# test that we can run something
run_test hostname

# rotate an integer around ranks
run_test ./main

# rotate with mpi-put
run_test ./one-sided