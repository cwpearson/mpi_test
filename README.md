# mpi_test

Various standalone C++ MPI tests/examples/benchmarks.

If CUDA is detected, additional binaries can be built.

| name               | Kind      | Reqs.    | Ranks | Description |
|--------------------|-----------|----------|-------|-------------|
|[hello-world][1]    | Test      | MPI      | 1+    | an MPI hello-world |
|[one-sided][2]      | Test      | MPI      | 2     | one-sided communication |
|[one-sided-gpu][3]  | Test      | MPI+CUDA | 2     | one-sided communication on GPU buffer |
|[persistent][4]     | Benchmark | MPI      | 2     | ping-pong time for persistent Send/Recv |
|[persistent-gpu][5] | Benchmark | MPI+CUDA | 2     | ping-pong time for persistent Send/Recv on GPU buffer|
|[send-recv][6]      | Benchmark | MPI      | 2     | ping-pong time for Send/Recv |
|[send-recv-gpu][7]  | Benchmark | MPI+CUDA | 2     | ping-pong time for Send/Recv on GPU buffer|

[1]: https://github.com/cwpearson/mpi_test/blob/master/hello_world.cpp
[2]: https://github.com/cwpearson/mpi_test/blob/master/one_sided.cpp
[3]: https://github.com/cwpearson/mpi_test/blob/master/one_sided_gpu.cpp
[4]: https://github.com/cwpearson/mpi_test/blob/master/persistent.cpp
[5]: https://github.com/cwpearson/mpi_test/blob/master/persistent_gpu.cpp
[6]: https://github.com/cwpearson/mpi_test/blob/master/send_recv.cpp
[7]: https://github.com/cwpearson/mpi_test/blob/master/send_recv_gpu.cpp

## Build
```
mkdir build && cd build
cmake ..
make
```

CMake will print the detected MPI environment.
Confirm it is what you expect.
For example:
```
-- MPI_VERSION:
-- MPI_CXX_COMPILER:            [...]/bin/mpicxx
-- MPI_CXX_COMPILE_OPTIONS:     -pthread
-- MPI_CXX_COMPILE_DEFINITIONS:
-- MPI_CXX_INCLUDE_DIRS:        [...]/include
-- MPI_CXX_LIBRARIES:           [...]/lib/libmpiprofilesupport.so;[...]/lib/libmpi_ibm.so
-- MPI_CXX_LINK_FLAGS:          -pthread
```
## Examples

### Summit

* 1 node:  `jsrun -n 1 ./persistent`
* 2 nodes: `jsrun -n 2 -r 1 -a 1 ./persistent`
* 2 nodes w/GPU: `jsrun --smpi="-gpu" -n 2 -r 1 -g 1 ./send-recv-gpu`

## Run all tests

`run-all.sh` attempts to discover certain environments automatically.
You can always override the detected flags yourself if you want.

```
./run-all.sh | grep 'PASS\|FAIL'
```

If any tests fails, you can re-run them individually.

## Run individual tests

Execute any binary you want using `mpirun`, or whatever is appropriate for your platform.


## Notes on specific platforms

Some Open MPIs use `long long` for their datatypes, which is not a part of ANSI C++ (`-ansi`).
