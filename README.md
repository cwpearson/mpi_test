# mpi_test

Various standalone C++ MPI tests/examples/benchmarks.

If CUDA is detected, additional binaries can be built.

| name          | Kind      | Reqs.    | Ranks | Description |
|---------------|-----------|----------|-------|-------------|
|`hello-world`  | Test      | MPI      | 1+    | an MPI hello-world |
|`one-sided`    | Test      | MPI      | 2     | one-sided communication |
|`one-sided-gpu`| Test      | MPI+CUDA | 2     | one-sided communication on GPU buffer |
|`persistent`   | Benchmark | MPI      | 2     | ping-pong time for persistent Send/Recv |

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

## Notes on specific platforms

Some Open MPIs use `long long` for their datatypes, which means we can't support ANSI C++ (`-ansi`).