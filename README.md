# mpi_test

Various standalone C++ MPI tests/examples/benchmarks.

If CUDA is detected, additional binaries can be built.

| name              | Kind      | Reqs.    | Ranks | Description |
|-------------------|-----------|----------|-------|-------------|
|[hello-world][1]   | Test      | MPI      | 1+    | an MPI hello-world |
|[one-sided][2]    | Test      | MPI      | 2     | one-sided communication |
|[one-sided-gpu][3] | Test      | MPI+CUDA | 2     | one-sided communication on GPU buffer |
|[persistent][4]    | Benchmark | MPI      | 2     | ping-pong time for persistent Send/Recv |
|[send-recv][5]     | Benchmark | MPI      | 2     | ping-pong time for Send/Recv |

[1]: https://github.com/cwpearson/mpi_test/blob/master/hello_world.cpp
[2]: https://github.com/cwpearson/mpi_test/blob/master/one_sided.cpp
[3]: https://github.com/cwpearson/mpi_test/blob/master/one_sided_gpu.cpp
[4]: https://github.com/cwpearson/mpi_test/blob/master/persistent.cpp
[5]: https://github.com/cwpearson/mpi_test/blob/master/send_recv.cpp

## Build
```
mkdir build && cd build
cmake ..
make
```

CMake will print the MPI environment it is using. For example:
```
-- MPI_CXX_COMPILER:     [...]/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2/bin/mpicxx
-- MPI_CXX_INCLUDE_DIRS: [...]/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2/include
-- MPIEXEC_EXECUTABLE:   [...]/spectrum-mpi-10.3.1.2-20200121-p6nrnt6vtvkn356wqg6f74n6jspnpjd2/bin/mpiexec
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

* 1 node:  `jsrun -n 1 ./persistent`
* 2 nodes: `jsrun -n 2 -r 1 -a 1 ./persistent`


## Notes on specific platforms

Some Open MPIs use `long long` for their datatypes, which means we can't support ANSI C++ (`-ansi`).
