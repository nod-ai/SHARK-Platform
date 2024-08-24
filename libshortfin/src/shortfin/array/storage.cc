// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/storage.h"

#include "fmt/core.h"
#include "shortfin/support/logging.h"

namespace shortfin::array {

using namespace local;
using namespace local::detail;

// -------------------------------------------------------------------------- //
// storage
// -------------------------------------------------------------------------- //

namespace detail {
void ThrowIllegalDeviceAffinity(Device *first, Device *second) {
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
  iree::hal_buffer_ptr buffer;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
      .queue_affinity = device.affinity().queue_affinity(),
  };
  SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, buffer.for_output()));
  return storage(device, std::move(buffer),
                 device.scope().NewTimelineResource());
}

storage storage::AllocateHost(ScopedDevice &device,
                              iree_device_size_t allocation_size) {
  if (!device.raw_device()) {
    throw std::invalid_argument("Cannot allocate with a null device affinity");
  }
  auto allocator = iree_hal_device_allocator(device.raw_device()->hal_device());
  iree::hal_buffer_ptr buffer;
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
                 device.scope().NewTimelineResource());
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
      device_, TransactionType::TRANSFER, [&](Account &account) {
        // Must depend on all of this buffer's use dependencies to avoid
        // write-after-read hazard.
        account.active_deps_extend(timeline_resource_->use_barrier());
        // And depend on any prior mutation in order to avoid a
        // write-after-write hazard.
        account.active_deps_extend(timeline_resource_->mutation_barrier());

        // TODO: I need to join the submission dependencies on the account
        // with the timeline resource idle fence to ensure that
        // write-after-access is properly sequenced.
        SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_fill_buffer(
            account.active_command_buffer(),
            iree_hal_make_buffer_ref(
                buffer_, /*offset=*/0,
                /*length=*/iree_hal_buffer_byte_length(buffer_)),
            pattern, pattern_length));

        // And move our own mutation barrier to the current pending timeline
        // value.
        timeline_resource_->set_mutation_barrier(
            account.timeline_sem(), account.timeline_idle_timepoint());
      });
}

void storage::CopyFrom(storage &source_storage) {
  throw std::logic_error("CopyFrom NYI");
}

bool storage::is_mappable_for_read() const {
  return (iree_hal_buffer_allowed_usage(buffer_) &
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         (iree_hal_buffer_allowed_access(buffer_) &
          IREE_HAL_MEMORY_ACCESS_READ);
}

void storage::MapExplicit(mapping &mapping, iree_hal_memory_access_t access) {
  assert(access != IREE_HAL_MEMORY_ACCESS_NONE);
  mapping.reset();
  SHORTFIN_THROW_IF_ERROR(iree_hal_buffer_map_range(
      buffer_, IREE_HAL_MAPPING_MODE_SCOPED, access,
      /*byte_offset=*/0, byte_length(), &mapping.mapping_));
  mapping.access_ = access;
  mapping.hal_device_ownership_baton_ =
      iree::hal_device_ptr::borrow_reference(hal_device_ownership_baton_);
}

iree_hal_memory_type_t storage::memory_type() const {
  return iree_hal_buffer_memory_type(buffer_);
}
iree_hal_memory_access_t storage::memory_access() const {
  return iree_hal_buffer_allowed_access(buffer_);
}
iree_hal_buffer_usage_t storage::buffer_usage() const {
  return iree_hal_buffer_allowed_usage(buffer_);
}

// Formatted type and access.
std::string storage::formatted_memory_type() const {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_hal_memory_type_format(memory_type(), &temp);
  return std::string(sv.data, sv.size);
}

std::string storage::formatted_memory_access() const {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_hal_memory_access_format(memory_access(), &temp);
  return std::string(sv.data, sv.size);
}

std::string storage::formatted_buffer_usage() const {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_hal_buffer_usage_format(buffer_usage(), &temp);
  return std::string(sv.data, sv.size);
}

std::string storage::to_s() const {
  return fmt::format("<storage {} size {}>", static_cast<void *>(buffer_.get()),
                     byte_length());
}

// -------------------------------------------------------------------------- //
// mapping
// -------------------------------------------------------------------------- //

mapping::mapping() { std::memset(&mapping_, 0, sizeof(mapping_)); }

mapping::~mapping() noexcept { reset(); }

void mapping::reset() noexcept {
  if (*this) {
    // Crash the process on failure to unmap. We don't have a good mitigation,
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping_));
    access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    hal_device_ownership_baton_.reset();
  }
}

}  // namespace shortfin::array
