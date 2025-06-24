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

/// Internal kernel launch implementation
ol_result_t llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                 dim3 BlockDim, void *KernelArgsPtr,
                                 size_t DynamicSharedMem, void *Stream,
                                 LLVMOffloadKernelArgsTy *LOKA) {
  auto Device = olKGetDefaultDevice();
  ol_kernel_handle_t Kernel = olKGetKernel(KernelID);

  ol_kernel_launch_size_args_t LaunchSizeArgs;
  LaunchSizeArgs.NumGroupsX = GridDim.x;
  LaunchSizeArgs.NumGroupsY = std::max(GridDim.y, 1u);
  LaunchSizeArgs.NumGroupsZ = std::max(GridDim.z, 1u);
  LaunchSizeArgs.GroupSizeX = BlockDim.x;
  LaunchSizeArgs.GroupSizeY = std::max(BlockDim.y, 1u);
  LaunchSizeArgs.GroupSizeZ = std::max(BlockDim.z, 1u);
  LaunchSizeArgs.DynSharedMemory = DynamicSharedMem;
  LaunchSizeArgs.Dimensions =
      1 + !!(GridDim.y * BlockDim.y > 1) + !!(GridDim.z * BlockDim.z > 1);

  ol_queue_handle_t Queue = Stream ? reinterpret_cast<ol_queue_handle_t>(Stream)
                                   : olKGetDefaultQueue();

  ol_result_t Result;
  if (LOKA)
    Result = olLaunchKernel(Queue, Device, Kernel, LOKA->Args, LOKA->Size,
                            &LaunchSizeArgs, nullptr);
  else
    Result = olLaunchKernel(Queue, Device, Kernel, KernelArgsPtr, size_t(-1),
                            &LaunchSizeArgs, nullptr);

  return Result;
}

#define LLVM_STYLE_LAUNCH(SUFFIX, PER_THREAD_STREAM)                           \
unsigned __llvmLaunchKernel##SUFFIX(const char *KernelID, dim3 GridDim,     \
                                    dim3 BlockDim, void *KernelArgsPtr,     \
                                    size_t DynamicSharedMem, void *Stream) {\
  auto *LOKA = reinterpret_cast<LLVMOffloadKernelArgsTy *>(KernelArgsPtr);  \
  ol_result_t Result =                                                      \
      llvmLaunchKernelImpl(KernelID, GridDim, BlockDim, KernelArgsPtr,      \
                           DynamicSharedMem, Stream, LOKA);                 \
  return Result ? Result->Code : 0;                                         \
}

LLVM_STYLE_LAUNCH(, false);
LLVM_STYLE_LAUNCH(_spt, true);
LLVM_STYLE_LAUNCH(_ptsz, true);

} // extern "C"
