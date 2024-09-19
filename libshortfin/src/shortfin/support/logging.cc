// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/logging.h"

#include "spdlog/cfg/env.h"

namespace shortfin::logging {

void InitializeFromEnv() {
  // TODO: Also support our own env vars.
  spdlog::cfg::load_env_levels();
}

}  // namespace shortfin::logging
