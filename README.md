# mpi_test

Various standalone MPI binaries, either tests or examples depending on your perspective.

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

## Run Microbenchmarks

- `persistent` (`persistent.cpp`) ping-pong time for persistent communication.


## Notes on specific platforms

Some Open MPIs use `long long` for their datatypes, which means we can't support ANSI C++ (`-ansi`).