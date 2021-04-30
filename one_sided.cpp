#include <mpi.h>
#include <iostream>

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);


  MPI_Win win;
  int *a;
  {
  /* Make rank accessible in all processes */

    MPI_Aint size = sizeof(int);
    int disp_unit = sizeof(int);
    MPI_Win_allocate(size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &a, &win);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // expect our a to be set by the left
  int source;
  if (0 == rank) {
    source = size - 1;
  } else {
    source = rank - 1;
  }

  int target; // set the right
  if (rank == size - 1) {
    target = 0;
  } else {
    target = rank + 1;
  }

  // send our rank to the target window
  std::cout << "rank " << rank << " put to " << target << std::endl << std::flush;
  MPI_Put(&rank, 1, MPI_INT, target, 0, 1, MPI_INT, win);   

  MPI_Win_fence(0, win);

  int err = 0;

  if (*a != source) {
    std::cerr << "ERR: rank " << rank << " got " << *a << " expected " << source << std::endl;
    err = 1;
  }               

  MPI_Win_free(&win);
  MPI_Finalize();
  std::cout << "rank " << rank << "completed" << std::endl << std::flush;
  return err;
}
