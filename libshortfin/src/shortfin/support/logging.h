// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_LOGGING_H
#define SHORTFIN_SUPPORT_LOGGING_H

#include "spdlog/spdlog.h"

namespace shortfin::logging {

// TODO: Re-export doesn't really work like this. Need to define API
// exported trampolines for cross library use.
using spdlog::debug;
using spdlog::error;
using spdlog::info;
using spdlog::warn;

}  // namespace shortfin::logging

#endif  // SHORTFIN_SUPPORT_LOGGING_H
