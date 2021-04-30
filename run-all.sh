#! /bin/bash
# ./run-all.sh | grep 'PASS\|FAIL'

set -eou pipefail

host=`hostname`
cpu_flags=''
gpu_flags=''
one_node_flags='-n 4'
two_node_flags=''
impl=''

set -x
if [[ `mpirun --version | grep "Open MPI"` ]]; then
    impl="ompi"
fi
set +x

# Apr 30, 2021: OpenMPI 4.0.5 on ascicgpu doesn't seem to automatically know
# which network interface to use
if [[ "$host" =~ .*vortex.* ]]; then   # vortex
    echo $host matched vortex
elif [[ $host =~ .*ascicgpu.* ]]; then # ascicgpu
    echo $host matched ascicgpu
    if [[ $impl == "ompi" ]]; then     # ascicgpu + Open MPI
        two_node_flags="$two_node_flags \
        --mca btl_tcp_if_include 10.203.0.0/16 \
        -np 4 -host ascicgpu030:2,ascicgpu032:2"
    fi
fi

# print various flags for tests
echo "impl:           " $impl
echo "cpu_flags:      " $cpu_flags
echo "gpu_flags:      " $gpu_flags
echo "one_node_flags: " $one_node_flags
echo "two_node_flags: " $two_node_flags

run_cpu_1_test() {
    # mvapich2 makes a bunch of spam if you use non-GPU buffers :(
    export MV2_USE_CUDA=0
    # mvapich2 makes a bunch of spam if its build with CUDA but the above is set :(
    export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
    mpirun $cpu_flags $one_node_flags $1 \
      && echo PASS (1 node): mpirun $cpu_flags $one_node_flags $1 \
      || echo FAIL (1 node): mpirun $cpu_flags $one_node_flags $1
}

run_cpu_2_test() {
    # mvapich2 makes a bunch of spam if you use non-GPU buffers :(
    export MV2_USE_CUDA=0
    # mvapich2 makes a bunch of spam if its build with CUDA but the above is set :(
    export MV2_SUPPRESS_CUDA_USAGE_WARNING=1
    mpirun $cpu_flags $two_node_flags $1 \
      && echo PASS (2 node): mpirun $cpu_flags $two_node_flags $1 \
      || echo FAIL (2 node): mpirun $cpu_flags $two_node_flags $1
}


run_gpu_test() {
    export MV2_USE_CUDA=1
    mpirun -n 4 $1 && echo PASS: mpirun -n 4 $1 || echo FAIL: mpirun -n 4 $1
}

# test that we can run something
run_cpu_1_test hostname
run_cpu_1_test ./main
run_cpu_1_test ./one-sided

run_cpu_2_test hostname
run_cpu_2_test ./main
run_cpu_2_test ./one-sided