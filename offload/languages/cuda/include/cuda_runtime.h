/*===---- cuda_runtime.h - CUDA runtime api declarations -------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#pragma once

#define LANGUAGE cuda

#include "../../kernel/include/DefineLanguageNames.inc"

#include "../../kernel/include/LanguageRuntime.h"

#include "../../kernel/include/UndefineLanguageNames.inc"

#undef LANGUAGE

using cudaDeviceProp = cudaDeviceProp_t;
