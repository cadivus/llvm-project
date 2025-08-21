/*===---- language_runtime.h - Kernel language runtime api declarations ----===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#include <cstddef>
#include <cstdint>

enum Error_t : uint32_t {
  Success = 0,
};

enum MemcpyKind {
  MemcpyHostToHost = 0,
  MemcpyHostToDevice = 1,
  MemcpyDeviceToHost = 2,
  MemcpyDeviceToDevice = 3,
  MemcpyDefault = 4
};

/// Malloc, with type template overlay.
///{
Error_t Malloc(void **Dev_Ptr, size_t Size);

template <class T> static inline Error_t Malloc(T **dev_Ptr, size_t Size) {
  return ::Malloc((void **)dev_Ptr, Size);
}

Error_t HostAlloc(void **Ptr, size_t Size, unsigned int Flags);

template <class T>
static inline Error_t HostAlloc(T **Ptr, size_t Size, unsigned int Flags) {
  return ::HostAlloc((void **)Ptr, Size, Flags);
}

Error_t MallocHost(void **Ptr, size_t Size);

template <class T> static inline Error_t MallocHost(T **Ptr, size_t Size) {
  return ::MallocHost((void **)Ptr, Size);
}
///}

/// Free, no type template necessary.
Error_t Free(void *Dev_Ptr);

/// Memcpy, with type template overlay.
///{
Error_t Memcpy(void *Dst, const void *Src, size_t Size, MemcpyKind Kind);

template <class T>
static inline Error_t Memcpy(T *Dst, const T *Src, size_t Size,
                             MemcpyKind Kind) {
  return ::Memcpy((void *)Dst, (const void *)Src, Size, Kind);
}
///}

Error_t HostFree(void *Ptr);

///

#if defined(__AMDGPU__) || defined(__NVPTX__)
#include <gpuintrin.h>

__device__ inline __attribute__((convergent)) void __syncthreads() {
  __gpu_sync_threads();
}

#define __LLVM_OFFLOAD_DEVICE_BUILTIN(FIELD, OFFSET)                           \
  __declspec(property(get = __get_##FIELD,                                     \
                      put = __put_##FIELD)) unsigned int FIELD;                \
  __device__ inline __attribute__((always_inline)) T __get_##FIELD(void)       \
      const {                                                                  \
    return Vec[OFFSET];                                                        \
  }                                                                            \
  __device__ inline __attribute__((always_inline)) T __put_##FIELD(T V) {      \
    return Vec[OFFSET] = V;                                                    \
  }

template <class T, int Size> struct BaseVector {
  using VT = float __attribute__((ext_vector_type(Size)));
  VT Vec;

  __device__ __host__ BaseVector() = default;
  __device__ __host__ BaseVector(std::initializer_list<T> List) {
    auto It = List.begin();
    for (int I = 0, E = List.size(); I < E; ++I, ++It)
      Vec[I] = *It;
  }

  template <typename... Args>
  __device__ __host__ BaseVector(Args... args) : BaseVector({args...}) {}

  __device__ __host__ T &operator[](int Idx) { return Vec[Idx]; }
  __device__ __host__ const T &operator[](int Idx) const { return Vec[Idx]; }

  __LLVM_OFFLOAD_DEVICE_BUILTIN(x, 0);
  __LLVM_OFFLOAD_DEVICE_BUILTIN(y, 1);
  __LLVM_OFFLOAD_DEVICE_BUILTIN(z, 2);
  __LLVM_OFFLOAD_DEVICE_BUILTIN(w, 3);
};

#define __VECTOR_DEF_IMPL(TY, SIZE)                                            \
  using TY##SIZE = BaseVector<TY, SIZE>;                                       \
                                                                               \
  template <typename... Args>                                                  \
  __device__ __host__ TY##SIZE make_##TY##SIZE(Args... args) {                 \
    return TY##SIZE(args...);                                                  \
  }

#define __VECTOR_DEF(TY)                                                       \
  __VECTOR_DEF_IMPL(TY, 1)                                                     \
  __VECTOR_DEF_IMPL(TY, 2)                                                     \
  __VECTOR_DEF_IMPL(TY, 3)                                                     \
  __VECTOR_DEF_IMPL(TY, 4)                                                     \
  __VECTOR_DEF_IMPL(TY, 8)                                                     \
  __VECTOR_DEF_IMPL(TY, 16)

__VECTOR_DEF(float)
__VECTOR_DEF(double)
__VECTOR_DEF(int8_t)
__VECTOR_DEF(int16_t)
__VECTOR_DEF(int32_t)
__VECTOR_DEF(int64_t)
__VECTOR_DEF(uint8_t)
__VECTOR_DEF(uint16_t)
__VECTOR_DEF(uint32_t)
__VECTOR_DEF(uint64_t)
__VECTOR_DEF(char)
__VECTOR_DEF(short)
__VECTOR_DEF(int)
__VECTOR_DEF(unsigned)
__VECTOR_DEF(long)

#undef __VECTOR_DEF_IMPL
#undef __VECTOR_DEF
#undef __LLVM_OFFLOAD_DEVICE_BUILTIN

#endif
