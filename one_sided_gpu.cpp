#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

inline void checkCuda(cudaError_t result, const char *file, const int line) {
  if (result != cudaSuccess) {
    fprintf(stderr, "%s:%d: CUDA Runtime Error %d: %s\n", file, line, int(result), cudaGetErrorString(result));
    exit(-1);
  }
}
#define CUDA_RUNTIME(stmt) checkCuda(stmt, __FILE__, __LINE__);

int main(int argc, char **argv) {

  MPI_Init(&argc, &argv);

  int *a{};
  MPI_Win win;
  {
    MPI_Aint size = sizeof(int);
    int disp_unit = sizeof(int);
    CUDA_RUNTIME(cudaMalloc(&a, size));
    MPI_Win_create(a, size, disp_unit, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  }

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int target; // set the right
  if (rank == size - 1) {
    target = 0;
  } else {
    target = rank + 1;
  }

  // start exposure of window
  MPI_Win_fence(0, win);

  // send our rank to the target window
  std::cout << "rank " << rank << " put to " << target << std::endl << std::flush;
  MPI_Put(&rank, 1, MPI_INT, target, 0, 1, MPI_INT, win);   

  // end exposure of window
  MPI_Win_fence(0, win);

  int err = 0;          

  MPI_Win_free(&win);
  MPI_Finalize();
  std::cout << "rank " << rank << " completed" << std::endl << std::flush;
  return err;
}
