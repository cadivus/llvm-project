//===-- LanguageRuntime.cpp - Kernel Language runtime API implementation --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageRuntime.h"

#ifndef LANGUAGE
#error This file should be included, or used, with a LANGUAGE macro set.
#endif

#include "ExportedAPI.h"
#include "Types.h"

#include "OffloadAPI.h"

#include "DefineLanguageNames.inc"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define STR(X) #X
#define LANGUAGE_STR STR(LANGUAGE)

Error_t olKConvertResult(ol_result_t Result) { return Success; }

Error_t Malloc(void **DevPtr, size_t Size) {
  ol_device_handle_t Device = olKGetDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_DEVICE, Size, DevPtr);
  return olKConvertResult(Result);
}

Error_t Free(void *DevPtr) {
  ol_result_t Result = olMemFree(DevPtr);
  return olKConvertResult(Result);
}

Error_t Memcpy(void *Dst, const void *Src, size_t Size, MemcpyKind Kind) {
  ol_queue_handle_t Queue = olKGetDefaultQueue();

  ol_result_t Result;
  switch (Kind) {
  case MemcpyHostToHost: {
    ol_device_handle_t Host = olKGetHostDevice();
    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Host, Size,
                      nullptr);
    break;
  }
  case MemcpyHostToDevice: {
    ol_device_handle_t Device = olKGetDefaultDevice();
    ol_device_handle_t Host = olKGetHostDevice();
    Result = olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Host, Size,
                      nullptr);
    break;
  }
  case MemcpyDeviceToHost: {
    ol_device_handle_t Device = olKGetDefaultDevice();
    ol_device_handle_t Host = olKGetHostDevice();

    Result = olMemcpy(Queue, Dst, Host, const_cast<void *>(Src), Device, Size,
                      nullptr);
    break;
  }
  case MemcpyDeviceToDevice: {
    ol_device_handle_t Device = olKGetDefaultDevice();

    Result = olMemcpy(Queue, Dst, Device, const_cast<void *>(Src), Device, Size,
                      nullptr);
    break;
  }
  case MemcpyDefault:
    fprintf(stderr, LANGUAGE_STR "MemcpyDefault is not implemented yet");
    abort();
  };

  olWaitQueue(Queue);

  return olKConvertResult(Result);
}

Error_t HostAlloc(void **Ptr, size_t Size, unsigned int Flags) {
  // TODO:
  ol_device_handle_t Device = olKGetDefaultDevice();
  ol_result_t Result = olMemAlloc(Device, OL_ALLOC_TYPE_HOST, Size, Ptr);
  return olKConvertResult(Result);
}

Error_t MallocHost(void **Ptr, size_t Size) {
  return HostAlloc(Ptr, Size, /* HostAllocDefault */0);
}

Error_t HostFree(void *Ptr) {
  ol_result_t Result = olMemFree(Ptr);
  return olKConvertResult(Result);
}
