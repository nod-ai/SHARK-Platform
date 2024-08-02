// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/globals.h"

#include "shortfin/support/logging.h"

namespace shortfin {

void Hidden() { logging::info("Hello there."); }

void GlobalInitialize() { Hidden(); }

}  // namespace shortfin
