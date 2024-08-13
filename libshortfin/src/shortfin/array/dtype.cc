// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/dtype.h"

#include "fmt/core.h"

namespace shortfin::array {

iree_device_size_t DType::compute_dense_nd_size(std::span<const size_t> dims) {
  if (!is_byte_aligned()) {
    throw std::invalid_argument(
        "Computing size of non byte aligned nd array not yet supported");
  }
  iree_device_size_t accum = dense_byte_count();
  for (size_t dim : dims) {
    accum *= dim;
  }
  return accum;
}

}  // namespace shortfin::array
