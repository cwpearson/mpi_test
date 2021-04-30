# mpi_test

Various standalone MPI binaries, either tests or examples depending on your perspective.
The goal is to compile with no warnings with `-Wall -Wextra -Wshadow -pedantic` or similarly picky settings.

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

If any binaries fail, you can run them individually...

## Run individual tests

Execute any binary you want using `mpirun`, or whatever is appropriate for your platform.

## Notes on specific platforms

Some Open MPIs use `long long` for their datatypes, which means we can't support ANSI C++ (`-ansi`).