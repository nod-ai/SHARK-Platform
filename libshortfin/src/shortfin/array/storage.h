// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_STORAGE_H
#define SHORTFIN_ARRAY_STORAGE_H

#include <string_view>

#include "shortfin/local_scope.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Array storage backed by an IREE buffer of some form.
class SHORTFIN_API storage {
 public:
  ScopedDevice &device() { return device_; }
  LocalScope &scope() { return device_.scope(); }
  const ScopedDevice &device() const { return device_; }
  LocalScope &scope() const { return device_.scope(); }

  // Allocates device storage, compatible with the given device affinity.
  // By default, this will be IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE.
  static storage AllocateDevice(ScopedDevice &device,
                                iree_device_size_t allocation_size);

  // Allocates host storage, compatible with the given device affinity.
  // By default, if there are any affinity bits set in the device, then
  // the storage will be device visible and have permitted usage for transfers.
  // This default policy can be overriden based on device defaults or explicit
  // options.
  static storage AllocateHost(ScopedDevice &device,
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
  storage(ScopedDevice device, iree_hal_buffer_ptr buffer,
          ScopedScheduler::TimelineResource::Ref timeline_resource)
      : buffer_(std::move(buffer)),
        device_(device),
        timeline_resource_(std::move(timeline_resource)) {}
  iree_hal_buffer_ptr buffer_;
  ScopedDevice device_;
  ScopedScheduler::TimelineResource::Ref timeline_resource_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_STORAGE_H
