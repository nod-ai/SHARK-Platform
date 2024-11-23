// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/sysconfig.h"

#include "shortfin/support/logging.h"

#ifdef __linux__
#include <sys/resource.h>
#endif

namespace shortfin::sysconfig {

// -----------------------------------------------------------------------------
// File handle limits
// -----------------------------------------------------------------------------

#ifdef __linux__

bool EnsureFileLimit(unsigned needed_limit) {
  struct rlimit limit;
  if (getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    return {};
  }

  if (limit.rlim_cur >= needed_limit) return true;
  unsigned requested_limit = needed_limit;
  if (limit.rlim_max >= needed_limit) {
    logging::debug(
        "Estimated number of open file handles ({}) < current limit ({}) but "
        "within max limit ({}): Increasing limit",
        needed_limit, limit.rlim_cur, limit.rlim_max);
  } else if (limit.rlim_max > limit.rlim_cur) {
    logging::warn(
        "Esimated number of open file handles ({}) < current ({}) and max ({}) "
        "limit: Increasing to max",
        needed_limit, limit.rlim_cur, limit.rlim_max);
    requested_limit = limit.rlim_max;
  } else {
    logging::warn("Esimated number of open file handles ({}) < max ({})",
                  needed_limit, limit.rlim_max);
    return false;
  }

  limit.rlim_cur = requested_limit;
  if (setrlimit(RLIMIT_NOFILE, &limit) != 0) {
    logging::error("Could not set open file handle limit to {}",
                   requested_limit);
    return false;
  }

  return limit.rlim_cur >= needed_limit;
}

#else
// Fallback implementation.
bool EnsureFileLimit(unsigned needed_limit) { return true; }
#endif

}  // namespace shortfin::sysconfig
