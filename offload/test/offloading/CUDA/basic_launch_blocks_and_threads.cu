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

#include "cuda_runtime_api.h"

__global__ void square(int *A) {
  __scoped_atomic_fetch_add(A, 1, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

int main(int argc, char **argv) {
  int DevNo = 0;
  int *Ptr;
  cudaMalloc(&Ptr, 4);
  printf("Ptr %p\n", Ptr);
  // CHECK: Ptr [[Ptr:0x.*]]
  square<<<7, 6>>>(Ptr);
  cudaMemcpy(&I, Ptr, sizeof(int), cudaMemcpyDeviceToHost);
  printf("I: %i\n", I);
  // CHECK: I: 42
}
