#include <mpi.h>
#include <cstdio>
#include <vector>

#include <sys/mman.h>
#include <errno.h>

const float sample_target = 200e-6;

struct Sample {
  double raw;
  double norm;
};

static Sample get_sample(int perSample, MPI_Request *sreq, MPI_Request *rreq, int rank, MPI_Comm comm) {
  Sample sample;
  MPI_Barrier(comm);
  double start = MPI_Wtime();
  for (int i = 0; i < perSample; ++i) {
    if (0 == rank) {
      MPI_Start(sreq);
      MPI_Wait(sreq, MPI_STATUS_IGNORE);
      MPI_Start(rreq);
      MPI_Wait(rreq, MPI_STATUS_IGNORE);
    } else if (1 == rank) {
      MPI_Start(rreq);
      MPI_Wait(rreq, MPI_STATUS_IGNORE);
      MPI_Start(sreq);
      MPI_Wait(sreq, MPI_STATUS_IGNORE);
    }
  }
  double stop = MPI_Wtime();
  sample.raw = stop-start;
  sample.norm = sample.raw / perSample;
  return sample;
}

int main(int argc, char **argv) {
  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size < 2) {
    printf("need at least 2 ranks!\n");
    exit(1);
  }

  int src = (rank + 1) % 2;
  int dst = (rank + 1) % 2;
  int tag = 0;
  MPI_Request sreq, rreq;
  int numIters = 100;


  std::vector<size_t> sweep{
    1,
    64,
    128,
    256,
    512,
    1 * 1024,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
  };

  if (0 == rank) {
    printf("bytes,min,max,avg,med\n");
  }

  for (size_t bytes : sweep) {
    std::vector<double> samples(numIters);
    char *buf = new char[bytes];
    if (mlock(buf, bytes)) {
      perror("error locking memory");
    }

    MPI_Send_init(buf, bytes, MPI_BYTE, dst, tag, MPI_COMM_WORLD, &sreq);
    MPI_Recv_init(buf, bytes, MPI_BYTE, src, tag, MPI_COMM_WORLD, &rreq);

    // try to reach 200us / sample
    int perSample = 1;
    for (int i = 0; i < 10; ++i) {
      double sample = get_sample(perSample, &sreq, &rreq, rank, MPI_COMM_WORLD).raw;
      // estimate number of measurements per sample
      int guess = sample_target / sample + /*rounding*/0.5;
      // close half the distance to this estimate
      perSample += (guess - perSample) * 0.5;
      if (perSample < 1) perSample = 1;
      MPI_Bcast(&perSample, 1, MPI_INT, 0, MPI_COMM_WORLD);
    } 

    if (0 == rank) {
      fprintf(stderr, "sample averaged over %d iterations\n", perSample);
    }

    for (int i = 0; i < numIters; ++i) {
      samples[i] = get_sample(perSample, &sreq, &rreq, rank, MPI_COMM_WORLD).norm;
    }

    // each sample is the max time observed
    MPI_Allreduce(MPI_IN_PLACE, samples.data(), numIters, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    // bubble sort
    bool changed = true;
    while (changed) {
      changed = false;
      for (int i = 0; i < numIters - 1; ++i) {
        if (samples[i] > samples[i+1]) {
          double tmp = samples[i+1];
          samples[i+1] = samples[i];
          samples[i] = tmp;
          changed = true;
        }
      }
    }

    // average
    double avg = 0;
    for (int i = 0; i < numIters; ++i) {
      avg += samples[i];
    }
    avg /= numIters;

    if (0 == rank) {
      printf("%lu,%e,%e,%e,%e\n", bytes, samples[0], samples[numIters-1], avg, samples[numIters/2]);
    }

    if (munlock(buf, bytes)) {
      perror("error unlocking memory");
    }
    delete[] buf;
  }

  MPI_Finalize();
  return 0;
}
