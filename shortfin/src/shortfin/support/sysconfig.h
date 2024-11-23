// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_SYSCONFIG_H
#define SHORTFIN_SUPPORT_SYSCONFIG_H

#include <cstdint>
#include <utility>

namespace shortfin::sysconfig {

// Attempts to ensure that the given number of file descriptors can be created.
// If the system does not support such a thing (i.e. GetOpenFileLimit() returns
// nothing), then nothing is done and true is returned. If the system does
// support it and heuristics say this should be allowed, then true will return.
// Otherwise, a warning will be logged and false returned.
// This is a best effort attempt.
bool EnsureFileLimit(unsigned needed_limit);

}  // namespace shortfin::sysconfig

#endif  // SHORTFIN_SUPPORT_SYSCONFIG_H
