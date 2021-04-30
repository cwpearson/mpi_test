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