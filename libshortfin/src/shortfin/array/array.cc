// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/array.h"

#include <sstream>

#include "fmt/core.h"
#include "fmt/ranges.h"
#include "shortfin/array/xtensor_bridge.h"

namespace shortfin::array {

template class InlinedDims<std::size_t>;

// -------------------------------------------------------------------------- //
// device_array
// -------------------------------------------------------------------------- //

const mapping device_array::data() const { return storage_.MapRead(); }

mapping device_array::data() { return storage_.MapRead(); }

mapping device_array::data_rw() { return storage_.MapReadWrite(); }

mapping device_array::data_w() { return storage_.MapWriteDiscard(); }

std::optional<mapping> device_array::map_memory_for_xtensor() {
  if (storage_.is_mappable_for_read_write()) {
    return storage_.MapReadWrite();
  } else if (storage_.is_mappable_for_read()) {
    return storage_.MapRead();
  }
  return {};
}

std::string device_array::to_s() const {
  std::string contents;
  const char *contents_prefix = " ";
  if (!storage_.is_mappable_for_read()) {
    contents = "<unmappable for host read>";
  } else {
    auto maybe_contents = contents_to_s();
    if (maybe_contents) {
      contents = std::move(*maybe_contents);
      contents_prefix = "\n";
    } else {
      contents = "<unsupported dtype or unmappable storage>";
    }
  }

  return fmt::format("device_array([{}], dtype='{}', device={}({})) ={}{}",
                     fmt::join(shape(), ", "), dtype().name(),
                     storage_.device().to_s(), storage_.formatted_memory_type(),
                     contents_prefix, contents);
}

}  // namespace shortfin::array
