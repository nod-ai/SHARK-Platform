// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/dtype.h"

#include <unordered_map>

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

DType DType::import_element_type(iree_hal_element_type_t et) {
  static std::unordered_map<iree_hal_element_type_t, DType> static_canonical =
      ([]() {
        std::unordered_map<iree_hal_element_type_t, DType> c;
        auto add = [&](DType dt) { c.emplace(std::make_pair(dt.et_, dt)); };
#define SHORTFIN_DTYPE_HANDLE(et, ident) add(DType(et, #ident));
#include "shortfin/array/dtypes.inl"
#undef SHORTFIN_DTYPE_HANDLE
        return c;
      })();

  auto &c = static_canonical;
  auto it = c.find(et);
  if (it != c.end()) return it->second;
  return DType(et, "opaque_imported");
}

}  // namespace shortfin::array
