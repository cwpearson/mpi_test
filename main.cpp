#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from %s, rank %d out of %d processors\n", processor_name,
         world_rank, world_size);

  int dst_rank = (world_rank + 1) % world_size;
  int src_rank = world_rank - 1;
  if (src_rank < 0) {
    src_rank = world_size - 1;
  }
  float val = world_rank;
  MPI_Send(&val, 1, MPI_FLOAT, dst_rank, 0, MPI_COMM_WORLD);
  MPI_Status stat;
  MPI_Recv(&val, 1, MPI_FLOAT, src_rank, 0, MPI_COMM_WORLD, &stat);

  printf("rank %d  got %f \n", world_rank, val);

  // Finalize the MPI environment.
  MPI_Finalize();
}
