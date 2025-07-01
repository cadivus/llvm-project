//===------ LanguageLaunch.h - Header for LanguageLaunch.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LANGUAGE_LAUNCH_H
#define LLVM_LANGUAGE_LAUNCH_H

#include "ExportedAPI.h"
#include "Types.h"
#include "OffloadAPI.h"

#include <cstdint>
#include <cstddef>
#include <algorithm> // for std::max

extern "C" {

struct LLVMOffloadKernelArgsTy {
  size_t Size;
  void *Args;
  void *_; // Reserved for future use
};

/// Push call configuration for kernel launch
unsigned llvmPushCallConfiguration(dim3 __grid_size, dim3 __block_size,
                                   size_t __shared_memory, void *__stream);

/// Pop call configuration for kernel launch
unsigned llvmPopCallConfiguration(dim3 *__grid_size, dim3 *__block_size,
                                  size_t *__shared_memory, void *__stream);

/// Internal kernel launch implementation
ol_result_t llvmLaunchKernelImpl(const char *KernelID, dim3 GridDim,
                                 dim3 BlockDim, void *KernelArgsPtr,
                                 size_t DynamicSharedMem, void *Stream,
                                 LLVMOffloadKernelArgsTy *LOKA);

}

#endif // LLVM_LANGUAGE_LAUNCH_H
