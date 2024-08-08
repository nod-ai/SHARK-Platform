// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/array.h"

#include "fmt/core.h"
#include "fmt/ranges.h"

namespace shortfin::array {

// -------------------------------------------------------------------------- //
// base_array
// -------------------------------------------------------------------------- //

// -------------------------------------------------------------------------- //
// device_array
// -------------------------------------------------------------------------- //

std::string device_array::to_s() const {
  return fmt::format("device_array([{}], dtype='{}', {})",
                     fmt::join(shape(), ", "), dtype().name(),
                     storage_.device().to_s());
}

// -------------------------------------------------------------------------- //
// host_array
// -------------------------------------------------------------------------- //

std::string host_array::to_s() const {
  return fmt::format("host_array([{}], dtype='{}', {})",
                     fmt::join(shape(), ", "), dtype().name(),
                     storage_.device().to_s());
}

}  // namespace shortfin::array
