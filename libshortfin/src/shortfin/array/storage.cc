// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/storage.h"

#include "fmt/core.h"

namespace shortfin::array {

namespace detail {
void ThrowIllegalDeviceAffinity(LocalDevice *first, LocalDevice *second) {
  throw std::invalid_argument(fmt::format(
      "Cannot combine unrelated devices into a DeviceAffinity: {} vs {}",
      first->name(), second->name()));
}
}  // namespace detail

storage storage::AllocateDevice(ScopedDevice &device,
                                iree_device_size_t allocation_size) {
  if (!device.raw_device()) {
    throw std::invalid_argument("Cannot allocate with a null device affinity");
  }
  auto allocator = iree_hal_device_allocator(device.raw_device()->hal_device());
  iree_hal_buffer_ptr buffer;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
      .queue_affinity = device.affinity().queue_affinity(),
  };
  SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, buffer.for_output()));
  return storage(device, std::move(buffer),
                 device.scope().NewTimelineResource(device));
}

storage storage::AllocateHost(ScopedDevice &device,
                              iree_device_size_t allocation_size) {
  if (!device.raw_device()) {
    throw std::invalid_argument("Cannot allocate with a null device affinity");
  }
  auto allocator = iree_hal_device_allocator(device.raw_device()->hal_device());
  iree_hal_buffer_ptr buffer;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST,
      .queue_affinity = device.affinity().queue_affinity(),
  };
  if (device.affinity().queue_affinity() != 0) {
    params.usage |= IREE_HAL_BUFFER_USAGE_TRANSFER;
  }
  SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, buffer.for_output()));
  return storage(device, std::move(buffer),
                 device.scope().NewTimelineResource(device));
}

storage storage::Subspan(iree_device_size_t byte_offset,
                         iree_device_size_t byte_length) {
  storage new_storage(device_, {}, timeline_resource_);
  SHORTFIN_THROW_IF_ERROR(iree_hal_buffer_subspan(
      buffer_, byte_offset, byte_length, new_storage.buffer_.for_output()));
  return new_storage;
}

void storage::Fill(const void *pattern, iree_host_size_t pattern_length) {
  device_.scope().scheduler().AppendCommandBuffer(
      device_, ScopedScheduler::TransactionType::TRANSFER,
      [&](ScopedScheduler::Account &account) {
        // TODO: I need to join the submission dependencies on the account
        // with the timeline resource idle fence to ensure that
        // write-after-access is properly sequenced.
        SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_fill_buffer(
            account.active_command_buffer(),
            iree_hal_make_buffer_ref(buffer_, /*offset=*/0,
                                     /*length=*/IREE_WHOLE_BUFFER),
            pattern, pattern_length));
      });
}

void storage::CopyFrom(storage &source_storage) {
  // TODO
}

std::string storage::to_s() const {
  return fmt::format("<storage {} size {}>", static_cast<void *>(buffer_.get()),
                     byte_length());
}

}  // namespace shortfin::array
