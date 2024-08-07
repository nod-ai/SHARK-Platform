// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/array.h"

#include "fmt/core.h"

namespace shortfin::array {

namespace detail {
void ThrowIllegalDeviceAffinity(LocalDevice *first, LocalDevice *second) {
  throw std::invalid_argument(fmt::format(
      "Cannot combine unrelated devices into a DeviceAffinity: {} vs {}",
      first->name(), second->name()));
}
}  // namespace detail

}  // namespace shortfin::array
