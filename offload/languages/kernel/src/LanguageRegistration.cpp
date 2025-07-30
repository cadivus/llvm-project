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
#include <inttypes.h>

typedef struct __attribute__((__packed__))
{
  uint32_t Magic;
  uint16_t Version;
  uint16_t HeaderSize;
  uint64_t FatSize;
} cuda_fatbin_header_t;

typedef struct __attribute__((__packed__))
{
    uint16_t kind;
    uint16_t unknown1;
    uint8_t header_size;
    uint64_t size;
    uint32_t compressed_size;       // Size of compressed data
    uint32_t unknown2;              // Address size for PTX?
    uint16_t minor;
    uint16_t major;
    uint32_t arch;
    uint32_t obj_name_offset;
    uint32_t obj_name_len;
    uint64_t flags;
    uint64_t zero;                  // Alignment for compression?
    uint64_t decompressed_size;     // Length of compressed data in decompressed representation.
                                    // There is an uncompressed footer so this is generally smaller
                                    // than size.
} fat_text_header;

static void readTUFatbin(const char *Binary, const FatbinWrapperTy *FW) {
  ol_device_handle_t Device = olKGetDefaultDevice();

  const cuda_fatbin_header_t* Header = reinterpret_cast<const cuda_fatbin_header_t*>(FW->Data);

  printf("Magic: 0x%08x\n", Header->Magic);
  printf("Version: %u\n", Header->Version);
  printf("HeaderSize: %u\n", Header->HeaderSize); // Usually 16
  printf("FatSize: %llu\n\n\n", Header->FatSize);


  const fat_text_header* Header2 = reinterpret_cast<const fat_text_header*>(FW->Data + 16);


  printf("Header2 header_size: %" PRIu32 "\n", Header2->header_size);

  size_t Size = static_cast<size_t>(Header->FatSize);
  size_t HeaderSize = static_cast<size_t>(Header->HeaderSize);
  printf("%p : %p :: %zu \n", FW->Data, FW->DataEnd, Size);
  ol_program_handle_t Program = nullptr;

  const void* ProgramData = static_cast<const char*>(FW->Data + (HeaderSize + 64));
  ol_result_t Result = olCreateProgram(Device, ProgramData, Size, &Program);

  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register device code (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }
  printf("Program :: %p\n", Program);

  olKRegisterProgram(Binary, Program);
}

static void readHIPFatbinEntries(const char *Binary, const char *HIPFatbinPtr) {
  ol_device_handle_t Device = olKGetDefaultDevice();
  const char *DataIt = HIPFatbinPtr;

  std::advance(DataIt, HIP_FATBIN_MAGIC_STR_LEN);
  uint64_t NumBundles = readAndAdvance<uint64_t>(DataIt);
  //printf("NB %lu\n", NumBundles);
  for (uint64_t BundleId = 0; BundleId < NumBundles; ++BundleId) {
    uint64_t BundleOffset = readAndAdvance<uint64_t>(DataIt);
    uint64_t BundleSize = readAndAdvance<uint64_t>(DataIt);

    uint64_t BundleTripleSize = readAndAdvance<uint64_t>(DataIt);
    //printf("Bundle %lu: %lu @ %lu, %s\n", BundleId, BundleSize, BundleOffset,
    //       DataIt);
    // TODO: We should read the triple and use it to verify what devices are
    // eligible.
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
    //printf("Program :: %p\n", Program);

    olKRegisterProgram(Binary, Program);
  }
}

/// Hidden, but exported, Registration API
///{
extern "C" {

void llvmRegisterFunction(const char *Binary, const char *KernelID,
                          char *KernelName, const char *KernelName1, int,
                          uint3 *, uint3 *, dim3 *, dim3 *, int *) {
  //printf("%s :: %p :: %p : %s : %s \n", __PRETTY_FUNCTION__, Binary, KernelID,
  //       KernelName, KernelName1);
  ol_kernel_handle_t Kernel;
  ol_program_handle_t Program = olKGetProgram(Binary);
  ol_result_t Result = olGetKernel(Program, KernelName, &Kernel);
  if (Result && Result->Code) {
    fprintf(stderr, "Failed to register kernel (%i): %s\n", Result->Code,
            Result->Details);
    abort();
  }

  //printf("K %p : %p\n", KernelID, Kernel);
  olKRegisterKernel(KernelID, Kernel);
}

const char *llvmRegisterFatBinary(const char *Binary) {

  const auto *FW = reinterpret_cast<const FatbinWrapperTy *>(Binary);
  //printf("%s : %i : %s (%p:%p) :: %i\n", __PRETTY_FUNCTION__, FW->Magic,
  //       FW->Data, FW->Data, FW->DataEnd, FW->Version);

  //printf("%s : %s : %lu\n", FW->Data, HIP_FATBIN_MAGIC_STR,
  //       HIP_FATBIN_MAGIC_STR_LEN);
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
