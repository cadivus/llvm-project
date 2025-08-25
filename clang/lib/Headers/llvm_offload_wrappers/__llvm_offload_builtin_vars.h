/*===--- llvm_offload_builtin_vars.h - Kenrel language built-in variables --===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 *
 * This is a modified copy of __cuda_builtin_vars.h to be used by CUDA and HIP
 * when compiled via llvm offloading. The main difference to the original is
 * that we use gpuintrin.h to get portable behavior rather than nvvm intrinsics.
 *
 */

#ifndef __LLVM_OFFLOAD_BUILTIN_VARS_H
#define __LLVM_OFFLOAD_BUILTIN_VARS_H

#include <gpuintrin.h>

// Forward declares from vector_types.h.
struct uint3;
struct dim3;

// The file implements built-in CUDA/HIP variables using __declspec(property).
// https://msdn.microsoft.com/en-us/library/yhfk0thd.aspx
// All read accesses of built-in variable fields get converted into calls to a
// getter function which in turn calls the appropriate builtin to fetch the
// value.
//
// Example:
//    int x = threadIdx.x;
// IR output:
//  %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3
// PTX output:
//  mov.u32     %r2, %tid.x;

#define __LLVM_OFFLOAD_DEVICE_BUILTIN(FIELD, INTRINSIC)                        \
  __declspec(property(get = __fetch_builtin_##FIELD)) unsigned int FIELD;      \
  static inline __attribute__((always_inline))                                 \
  __attribute__((device)) unsigned int __fetch_builtin_##FIELD(void) {         \
    return INTRINSIC;                                                          \
  }

#if __cplusplus >= 201103L
#define __DELETE = delete
#else
#define __DELETE
#endif

// Make sure nobody can create instances of the special variable types.  nvcc
// also disallows taking address of special variables, so we disable address-of
// operator as well.
#define __LLVM_OFFLOAD_DISALLOW_BUILTINVAR_ACCESS(TypeName)                    \
  __attribute__((device)) TypeName() __DELETE;                                 \
  __attribute__((device)) TypeName(const TypeName &) __DELETE;                 \
  __attribute__((device)) void operator=(const TypeName &) const __DELETE;     \
  __attribute__((device)) TypeName *operator&() const __DELETE

struct __cuda_builtin_threadIdx_t {
  __LLVM_OFFLOAD_DEVICE_BUILTIN(x, __gpu_thread_id_x());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(y, __gpu_thread_id_y());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(z, __gpu_thread_id_z());
  // threadIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __LLVM_OFFLOAD_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_threadIdx_t);
};

struct __cuda_builtin_blockIdx_t {
  __LLVM_OFFLOAD_DEVICE_BUILTIN(x, __gpu_block_id_x());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(y, __gpu_block_id_y());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(z, __gpu_block_id_z());
  // blockIdx should be convertible to uint3 (in fact in nvcc, it *is* a
  // uint3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __LLVM_OFFLOAD_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockIdx_t);
};

struct __cuda_builtin_blockDim_t {
  __LLVM_OFFLOAD_DEVICE_BUILTIN(x, __gpu_num_threads_x());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(y, __gpu_num_threads_y());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(z, __gpu_num_threads_z());
  // blockDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __LLVM_OFFLOAD_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_blockDim_t);
};

struct __cuda_builtin_gridDim_t {
  __LLVM_OFFLOAD_DEVICE_BUILTIN(x, __gpu_num_blocks_x());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(y, __gpu_num_blocks_y());
  __LLVM_OFFLOAD_DEVICE_BUILTIN(z, __gpu_num_blocks_z());
  // gridDim should be convertible to dim3 (in fact in nvcc, it *is* a
  // dim3).  This function is defined after we pull in vector_types.h.
  __attribute__((device)) operator dim3() const;
  __attribute__((device)) operator uint3() const;

private:
  __LLVM_OFFLOAD_DISALLOW_BUILTINVAR_ACCESS(__cuda_builtin_gridDim_t);
};

#define __LLVM_OFFLOAD_BUILTIN_VAR                                             \
  extern const __attribute__((device)) __attribute__((weak))
__LLVM_OFFLOAD_BUILTIN_VAR __cuda_builtin_threadIdx_t threadIdx;
__LLVM_OFFLOAD_BUILTIN_VAR __cuda_builtin_blockIdx_t blockIdx;
__LLVM_OFFLOAD_BUILTIN_VAR __cuda_builtin_blockDim_t blockDim;
__LLVM_OFFLOAD_BUILTIN_VAR __cuda_builtin_gridDim_t gridDim;

//__attribute__((device)) const int warpSize = __gpu_num_lanes();

#undef __LLVM_OFFLOAD_DEVICE_BUILTIN
#undef __LLVM_OFFLOAD_BUILTIN_VAR
#undef __LLVM_OFFLOAD_DISALLOW_BUILTINVAR_ACCESS
#undef __DELETE

#endif /* __LLVM_OFFLOAD_BUILTIN_VARS_H */
