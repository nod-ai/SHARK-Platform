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

void device_array::AddAsInvocationArgument(
    local::ProgramInvocation *inv, local::ProgramResourceBarrier barrier) {
  auto dims_span = shape();
  iree_hal_buffer_view_t *buffer_view;
  SHORTFIN_THROW_IF_ERROR(iree_hal_buffer_view_create(
      storage_, dims_span.size(), dims_span.data(), dtype(),
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, storage_.host_allocator(),
      &buffer_view));

  iree::vm_opaque_ref ref;
  *(&ref) = iree_hal_buffer_view_move_ref(buffer_view);
  inv->AddArg(std::move(ref));

  // TODO: Add barriers.
}

iree_vm_ref_type_t device_array::invocation_marshalable_type() {
  return iree_hal_buffer_view_type();
}

device_array device_array::CreateFromInvocationResultRef(
    local::ProgramInvocation *inv, iree::vm_opaque_ref ref) {
  // We don't retain the buffer view in the device array, so just deref it
  // vs stealing the ref.
  iree_hal_buffer_view_t *bv = iree_hal_buffer_view_deref(*ref.get());
  iree::hal_buffer_ptr buffer =
      iree::hal_buffer_ptr::borrow_reference(iree_hal_buffer_view_buffer(bv));

  // TODO: We've lost information at this point needed to determine which
  // device the buffer exists on. This isn't a great state of affairs and
  // we just punt and associate to device 0 to start. This should really
  // be associated with the index 0 device of the proper physical hal device.
  // Hilarity will ensure with this like it is with true multi-device.
  local::ScopedDevice device = inv->scope()->device(0);

  auto imported_storage = storage::import_buffer(device, std::move(buffer));
  std::span<const iree_hal_dim_t> shape(iree_hal_buffer_view_shape_dims(bv),
                                        iree_hal_buffer_view_shape_rank(bv));
  return device_array(
      std::move(imported_storage), shape,
      DType::import_element_type(iree_hal_buffer_view_element_type(bv)));
}

}  // namespace shortfin::array
