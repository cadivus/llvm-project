/*===---- hip_runtime.cpp - HIP runtime api implementations ----------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#include "hip_runtime.h"

#include "OffloadAPI.h"

#define LANGUAGE hip

#include "LanguageRuntime.cpp"

#include "LanguageRegistration.cpp"

#include "LanguageLaunch.cpp"

// Must be last as it introduces alises for some definitions from above.
#include "LanguageAliases.h"

extern "C" hipError_t hipLaunchKernel(const char *KernelID, dim3 GridDim,
                                      dim3 BlockDim, void **KernelArgsPtr,
                                      size_t DynamicSharedMem, void *Stream) {
  return olKConvertResult(llvmLaunchKernelImpl(KernelID, GridDim, BlockDim,
                                               KernelArgsPtr, DynamicSharedMem,
                                               Stream, nullptr));
}
