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

std::string device_array::to_s() const {
  std::string contents;
  if (!storage_.is_mappable_for_read()) {
    contents = "<unmappable for host read>";
  } else if (dtype() == DType::float32()) {
    auto flat = typed_data<float>();
    // std::vector<size_t> dims(shape().begin(), shape().end());
    auto a = xt::adapt(flat.data(), shape_container());
    std::stringstream out;
    out << "\n" << a;
    contents = out.str();
  } else {
    contents = "<unrepresented dtype>";
  }

  return fmt::format("device_array([{}], dtype='{}', device={}({})) = {}",
                     fmt::join(shape(), ", "), dtype().name(),
                     storage_.device().to_s(), storage_.formatted_memory_type(),
                     contents);
}

}  // namespace shortfin::array
