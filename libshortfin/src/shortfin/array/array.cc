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

template class InlinedDims<iree_hal_dim_t>;

// -------------------------------------------------------------------------- //
// device_array
// -------------------------------------------------------------------------- //

const mapping device_array::data() const { return storage_.map_read(); }

mapping device_array::data() { return storage_.map_read(); }

mapping device_array::data_rw() { return storage_.map_read_write(); }

mapping device_array::data_w() { return storage_.map_write_discard(); }

std::optional<mapping> device_array::map_memory_for_xtensor() {
  if (storage_.is_mappable_for_read_write()) {
    return storage_.map_read_write();
  } else if (storage_.is_mappable_for_read()) {
    return storage_.map_read();
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

  return fmt::format(
      "device_array([{}], dtype='{}', device={}(type={}, usage={}, access={})) "
      "={}{}",
      fmt::join(shape(), ", "), dtype().name(), storage_.device().to_s(),
      storage_.formatted_memory_type(), storage_.formatted_buffer_usage(),
      storage_.formatted_memory_access(), contents_prefix, contents);
}

device_array::operator iree::vm_opaque_ref() {
  auto dims_span = shape();
  iree_hal_buffer_view_t *buffer_view;
  SHORTFIN_THROW_IF_ERROR(iree_hal_buffer_view_create(
      storage_, dims_span.size(), dims_span.data(), dtype(),
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, storage_.host_allocator(),
      &buffer_view));

  iree::vm_opaque_ref ref;
  *(&ref) = iree_hal_buffer_view_move_ref(buffer_view);
  return ref;
}

}  // namespace shortfin::array
