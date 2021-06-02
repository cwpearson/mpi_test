# mpi_test

Various standalone C++ MPI tests/examples/benchmarks.

If CUDA is detected, additional binaries can be built.

| name          | Kind      | Reqs.    | Ranks | Description |
|---------------|-----------|----------|-------|-------------|
|[hello-world][1] | Test      | MPI      | 1+    | an MPI hello-world |
|[one-sided`](https://github.com/cwpearson/mpi_test/blob/master/one_sided.cpp)    | Test      | MPI      | 2     | one-sided communication |
|[one-sided-gpu](https://github.com/cwpearson/mpi_test/blob/master/one_sided_gpu.cpp)| Test      | MPI+CUDA | 2     | one-sided communication on GPU buffer |
|[persistent](https://github.com/cwpearson/mpi_test/blob/master/persistent.cpp)   | Benchmark | MPI      | 2     | ping-pong time for persistent Send/Recv |
|[send-recv](https://github.com/cwpearson/mpi_test/blob/master/send_recv.cpp)    | Benchmark | MPI      | 2     | ping-pong time for Send/Recv |

[1]: https://github.com/cwpearson/mpi_test/blob/master/hello_world.cpp

## Build
```
mkdir build && cd build
cmake ..
make
```

## Run all tests

`run-all.sh` attempts to discover certain environments automatically.
You can always override the detected flags yourself if you want.

```
./run-all.sh | grep 'PASS\|FAIL'
```

If any tests fails, you can re-run them individually.

## Run individual tests

Execute any binary you want using `mpirun`, or whatever is appropriate for your platform.

## Examples

### Summit

1 node:  `jsrun -n 1 ./persistent`
2 nodes: `jsrun -n 2 -r 1 -a 1 ./persistent`


## Notes on specific platforms

Some Open MPIs use `long long` for their datatypes, which means we can't support ANSI C++ (`-ansi`).
