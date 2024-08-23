// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_STORAGE_H
#define SHORTFIN_ARRAY_STORAGE_H

#include <string_view>

#include "shortfin/local/scope.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Array storage backed by an IREE buffer of some form.
class SHORTFIN_API storage {
 public:
  local::ScopedDevice &device() { return device_; }
  local::Scope &scope() { return device_.scope(); }
  const local::ScopedDevice &device() const { return device_; }
  local::Scope &scope() const { return device_.scope(); }

  // Allocates device storage, compatible with the given device affinity.
  // By default, this will be IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE.
  static storage AllocateDevice(local::ScopedDevice &device,
                                iree_device_size_t allocation_size);

  // Allocates host storage, compatible with the given device affinity.
  // By default, if there are any affinity bits set in the device, then
  // the storage will be device visible and have permitted usage for transfers.
  // This default policy can be overriden based on device defaults or explicit
  // options.
  static storage AllocateHost(local::ScopedDevice &device,
                              iree_device_size_t allocation_size);

  // Creates a subspan view of the current storage given a byte offset and
  // length. The returned storage shares the underlying allocation and
  // scheduling control block.
  storage Subspan(iree_device_size_t byte_offset,
                  iree_device_size_t byte_length);

  // Enqueues a fill of the storage with an arbitrary pattern of the given
  // size. The pattern size must be 1, 2, or 4.
  void Fill(const void *pattern, iree_host_size_t pattern_length);

  // Performs either a d2h, h2d or d2d transfer from a source storage to this
  // storage.
  void CopyFrom(storage &source_storage);

  iree_device_size_t byte_length() const {
    return iree_hal_buffer_byte_length(buffer_.get());
  }

  std::string to_s() const;

 private:
  storage(local::ScopedDevice device, iree::hal_buffer_ptr buffer,
          local::detail::TimelineResource::Ref timeline_resource)
      : hal_device_ownership_baton_(iree::hal_device_ptr::borrow_reference(
            device.raw_device()->hal_device())),
        buffer_(std::move(buffer)),
        device_(device),
        timeline_resource_(std::move(timeline_resource)) {}
  // TODO(ownership): Since storage is a free-standing object in the system,
  // it needs an ownership baton that keeps the device/driver alive. Otherwise,
  // it can outlive the backing device and then then crashes on buffer
  // deallocation. For now, we stash an RAII hal_device_ptr, which keeps
  // everything alive. This isn't quite what we want but keeps us going for now.
  // When fixing, add a test that creates an array, destroys the System, and
  // then frees the array.
  iree::hal_device_ptr hal_device_ownership_baton_;
  iree::hal_buffer_ptr buffer_;
  local::ScopedDevice device_;
  local::detail::TimelineResource::Ref timeline_resource_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_STORAGE_H
