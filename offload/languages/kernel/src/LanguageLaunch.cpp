//===------ LanguageLaunch.cpp - Language (CUDA/HIP) launch api -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageLaunch.h"

#include <cstdio>

extern "C" {

struct CallConfigurationTy {
  dim3 GridSize;
  dim3 BlockSize;
  size_t SharedMemory;
  void *Stream;
};

static thread_local CallConfigurationTy CC = {};

/// Push call configuration for kernel launch
unsigned llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                   size_t __shared_memory, void *__stream) {
  CallConfigurationTy &Kernel = CC;
  Kernel.GridSize = __grid_size;
  Kernel.BlockSize = __block_size;
  Kernel.SharedMemory = __shared_memory;
  Kernel.Stream = __stream;
  return 0;
}

/// Pop call configuration for kernel launch
unsigned llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                  size_t *__shared_memory, void *__stream) {
  CallConfigurationTy &Kernel = CC;
  *__grid_size = Kernel.GridSize;
  *__block_size = Kernel.BlockSize;
  *__shared_memory = Kernel.SharedMemory;
  *((void **)__stream) = Kernel.Stream;
  return 0;
}

} // extern "C"
