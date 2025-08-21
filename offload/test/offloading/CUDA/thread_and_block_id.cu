// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp 
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

__global__ void fill(int *A) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  A[tid] = 42;
}

int main(int argc, char **argv) {
  int NThreads = 128;
  int NBlocks = 512;
  int Size = sizeof(int) * NThreads * NBlocks;
  int *Ptr = (int *)calloc(1, Size);
  int *DevPtr;
  cudaMalloc(&DevPtr, Size);
  cudaMemcpy(DevPtr, Ptr, Size, cudaMemcpyHostToDevice);
  printf("DevPtr %p\n", DevPtr);
  // CHECK: DevPtr [[DevPtr:0x.*]]
  fill<<<NBlocks, NThreads>>>(DevPtr);
  cudaMemcpy(Ptr, DevPtr, Size, cudaMemcpyDeviceToHost);

  for (int I = 0; I < NBlocks * NThreads; ++I) {
    if (Ptr[I] == 42)
      continue;
    printf("Error at %i: %i vs %i\n", I, Ptr[I], 42);
    return 1;
  }
  return 0;
}
