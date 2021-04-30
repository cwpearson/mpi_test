# mpi_test

Various standalone MPI binaries
* all pass `argc`/`argv` to `MPI_Init`

Adjust `Makefile` to match your environment, if needed
* uses `mpicxx` and a few simple flags by default

## Build
```
make
```

## Run all tests

Adjust `run-all.sh` to match your environment, if needed
```
./run-all.sh
```

If any tests fail, you can run them individually.

## Run individual tests

Execute any binary you want using `mpirun`, or whatever is appropriate for your platform.