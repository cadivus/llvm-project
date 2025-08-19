/*===---- ExportedAPI.h - Kernel language runtime - exported api  ----------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#pragma once

#include "OffloadAPI.h"

extern "C" {
  ol_device_handle_t olKGetDefaultDevice();

  ol_device_handle_t olKGetHostDevice();

  ol_queue_handle_t olKGetDefaultQueue();
    
  ol_symbol_handle_t olKGetKernel(const void *ID);

  void olKRegisterProgram(const void *ID, ol_program_handle_t Program);

  ol_program_handle_t olKGetProgram(const void *ID);
}
