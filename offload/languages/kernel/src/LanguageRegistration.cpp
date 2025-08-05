//===---- LanguageRegistration.h - Language (CUDA/HIP) registration api ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "LanguageRegistration.h"
#include "OffloadAPI.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <inttypes.h>

typedef struct __attribute__((__packed__)) {
  uint32_t Magic;
  uint16_t Version;
  uint16_t HeaderSize;
  uint64_t FatSize;
} CudaFatbinHeader;

// Inspired by
// https://github.com/n-eiling/cuda-fatbin-decompression/blob/master/fatbin-decompress.h
typedef struct __attribute__((__packed__)) {
  uint16_t Kind;
  uint16_t Unknown1;
  uint32_t HeaderSize;
  uint64_t Size;
  uint32_t CompressedSize;
  uint32_t Unknown2;
  uint16_t Minor;
  uint16_t Major;
  uint32_t Arch;
  uint32_t ObjNameOffset;
  uint32_t ObjNameLen;
  uint64_t Flags;
  uint64_t Zero;
  uint64_t DecompressedSize;
} CudaFatbinTextHeader;

static void readTUFatbin(const char *Binary, const FatbinWrapperTy *FW) {
  ol_device_handle_t Device = olKGetDefaultDevice();

  const CudaFatbinHeader *Header =
      reinterpret_cast<const CudaFatbinHeader *>(FW->Data);
  size_t HeaderSize = static_cast<size_t>(Header->HeaderSize); // Usually 16
  size_t FatbinSize = static_cast<size_t>(Header->FatSize);

  const void *ProgramData = nullptr;
  size_t ProgramSize = 0;
  uint32_t ProgramArch = 0;

  const char *ReadPosition = FW->Data + HeaderSize;
  while (ReadPosition < (FW->Data + FatbinSize)) {
    const CudaFatbinTextHeader *TextHeader =
        reinterpret_cast<const CudaFatbinTextHeader *>(ReadPosition);
    size_t TextHeaderSize =
        static_cast<size_t>(TextHeader->HeaderSize); // Usually 64
    size_t CubinSize = static_cast<size_t>(TextHeader->Size);
    const void *CubinData =
        static_cast<const char *>(ReadPosition + TextHeaderSize);

    uint32_t Arch = TextHeader->Arch;
    bool IsCompatible = false;
    ol_result_t CompatibilityCheckResult = olElfIsCompatibleWithDevice(
        Device, CubinData, CubinSize, &IsCompatible);

    if (CompatibilityCheckResult && CompatibilityCheckResult->Code) {
      fprintf(stderr, "Failed to check for device compatibility (%i): %s\n",
              CompatibilityCheckResult->Code,
              CompatibilityCheckResult->Details);
      abort();
    }

    if (IsCompatible && Arch > ProgramArch) {
      ProgramData = CubinData;
      ProgramSize = CubinSize;
      ProgramArch = Arch;
    }

    ReadPosition += TextHeaderSize + CubinSize;
  }

  if (ProgramData == nullptr) {
    fprintf(stderr, "Failed to find compatible binary\n");
    abort();
  }

  ol_program_handle_t Program = nullptr;

  ol_result_t Result =
      olCreateProgram(Device, ProgramData, ProgramSize, &Program);

  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }

  olKRegisterProgram(Binary, Program);
}

static void readHIPFatbinEntries(const char *Binary, const char *HIPFatbinPtr) {
  ol_device_handle_t Device = olKGetDefaultDevice();
  const char *DataIt = HIPFatbinPtr;

  std::advance(DataIt, HIP_FATBIN_MAGIC_STR_LEN);
  uint64_t NumBundles = readAndAdvance<uint64_t>(DataIt);
  // printf("NB %lu\n", NumBundles);
  for (uint64_t BundleId = 0; BundleId < NumBundles; ++BundleId) {
    uint64_t BundleOffset = readAndAdvance<uint64_t>(DataIt);
    uint64_t BundleSize = readAndAdvance<uint64_t>(DataIt);

    uint64_t BundleTripleSize = readAndAdvance<uint64_t>(DataIt);
    // printf("Bundle %lu: %lu @ %lu, %s\n", BundleId, BundleSize, BundleOffset,
    //        DataIt);
    //  TODO: We should read the triple and use it to verify what devices are
    //  eligible.
    std::advance(DataIt, BundleTripleSize);

    if (!BundleSize)
      continue;

    ol_program_handle_t Program = nullptr;
    ol_result_t Result = olCreateProgram(Device, HIPFatbinPtr + BundleOffset,
                                         BundleSize, &Program);
    if (Result && Result->Code) {
      fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
              Result->Details);
      abort();
    }
    // printf("Program :: %p\n", Program);

    olKRegisterProgram(Binary, Program);
  }
}

/// Hidden, but exported, Registration API
///{
extern "C" {

void llvmRegisterFunction(const char *Binary, const char *KernelID,
                          char *KernelName, const char *KernelName1, int,
                          uint3 *, uint3 *, dim3 *, dim3 *, int *) {
  // TODO: Implement kernel launches
  fprintf(stderr, "RegisterFunction is not implemented!");
}

const char *llvmRegisterFatBinary(const char *Binary) {

  const auto *FW = reinterpret_cast<const FatbinWrapperTy *>(Binary);
  // printf("%s : %i : %s (%p:%p) :: %i\n", __PRETTY_FUNCTION__, FW->Magic,
  //        FW->Data, FW->Data, FW->DataEnd, FW->Version);

  // printf("%s : %s : %lu\n", FW->Data, HIP_FATBIN_MAGIC_STR,
  //        HIP_FATBIN_MAGIC_STR_LEN);
  if (FW->Magic == 0x466243b1) {
    readTUFatbin(Binary, FW);
  } else if (FW->Magic == 0x48495046) {
    if (!memcmp(FW->Data, HIP_FATBIN_MAGIC_STR, HIP_FATBIN_MAGIC_STR_LEN))
      readHIPFatbinEntries(Binary, FW->Data);
    else
      readTUFatbin(Binary, FW);
  } else {
    fprintf(stderr, "Unknown fatbin format");
  }

  return Binary;
}

void llvmUnregisterFatBinary(void *Handle) {}

void llvmRegisterVar(void **, char *, char *, const char *, int, int, int,
                     int) {
  fprintf(stderr, "RegisterVar is not implemented!");
}

void llvmRegisterManagedVar(void **, char *, char *, const char *, size_t,
                            unsigned) {
  fprintf(stderr, "RegisterManagedVar is not implemented!");
}

void llvmRegisterSurface(void **, const struct surfaceReference *,
                         const void **, const char *, int, int) {
  fprintf(stderr, "RegisterSurface is not implemented!");
}

void llvmRegisterTexture(void **, const struct textureReference *,
                         const void **, const char *, int, int, int) {
  fprintf(stderr, "RegisterTexture is not implemented!");
}
}
///}
