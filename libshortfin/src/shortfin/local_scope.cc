// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local_scope.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "shortfin/support/logging.h"

namespace shortfin {

// -------------------------------------------------------------------------- //
// LocalScope
// -------------------------------------------------------------------------- //

LocalScope::LocalScope(
    iree_allocator_t host_allocator,
    std::span<const std::pair<std::string_view, LocalDevice *>> devices)
    : host_allocator_(host_allocator), scheduler_(host_allocator) {
  for (auto &it : devices) {
    AddDevice(it.first, it.second);
  }
  Initialize();
}

LocalScope::LocalScope(iree_allocator_t host_allocator,
                       std::span<LocalDevice *const> devices)
    : host_allocator_(host_allocator), scheduler_(host_allocator) {
  for (auto *device : devices) {
    AddDevice(device->address().logical_device_class, device);
  }
  Initialize();
}

LocalScope::~LocalScope() = default;

void LocalScope::Initialize() { scheduler_.Initialize(devices_); }

void LocalScope::AddDevice(std::string_view device_class, LocalDevice *device) {
  device_class = interner_.intern(device_class);
  auto &count = device_class_count_[device_class];
  std::string_view device_name =
      interner_.intern(fmt::format("{}{}", device_class, count++));
  named_devices_[device_name] = device;
  devices_.push_back(device);
}

LocalDevice *LocalScope::raw_device(std::string_view name) const {
  auto it = named_devices_.find(name);
  if (it == named_devices_.end()) [[unlikely]] {
    throw std::invalid_argument(
        fmt::format("Device '{}' not found (available: {})", name,
                    fmt::join(device_names(), ", ")));
  }
  return it->second;
}

LocalDevice *LocalScope::raw_device(int ordinal) const {
  if (ordinal < 0 || ordinal >= devices_.size()) {
    throw std::invalid_argument(
        fmt::format("Device ordinal ({}) out of bounds", ordinal));
  }
  return devices_[ordinal];
}

std::vector<std::string_view> LocalScope::device_names() const {
  std::vector<std::string_view> names;
  names.reserve(named_devices_.size());
  for (auto &it : named_devices_) {
    names.push_back(it.first);
  }
  return names;
}

// -------------------------------------------------------------------------- //
// ScopedScheduler::Account
// -------------------------------------------------------------------------- //

ScopedScheduler::Account::Account(LocalDevice *device)
    : device_(device), hal_device_(device->hal_device()) {}

void ScopedScheduler::Account::Initialize() {
  SHORTFIN_THROW_IF_ERROR(iree_hal_semaphore_create(
      hal_device(), idle_timepoint_, sem_.for_output()));
  Reset();
}

void ScopedScheduler::Account::Reset() {
  active_tx_type_ = TransactionType::NONE;
  active_command_buffer_.reset();
}

void ScopedScheduler::Account::active_deps_extend(
    iree_hal_semaphore_list_t sem_list) {
  for (iree_host_size_t i = 0; i < sem_list.count; ++i) {
    SHORTFIN_THROW_IF_ERROR(iree_hal_fence_insert(
        active_deps_, sem_list.semaphores[i], sem_list.payload_values[i]));
  }
}

// -------------------------------------------------------------------------- //
// ScopedScheduler::TimelineResource
// -------------------------------------------------------------------------- //

ScopedScheduler::TimelineResource::TimelineResource(
    iree_allocator_t host_allocator, size_t semaphore_capacity) {
  SHORTFIN_THROW_IF_ERROR(iree_hal_fence_create(
      semaphore_capacity, host_allocator, use_barrier_fence_.for_output()));
}

void ScopedScheduler::TimelineResource::use_barrier_insert(
    iree_hal_semaphore_t *sem, uint64_t timepoint) {
  SHORTFIN_THROW_IF_ERROR(
      iree_hal_fence_insert(use_barrier_fence_, sem, timepoint));
}

// -------------------------------------------------------------------------- //
// ScopedScheduler
// -------------------------------------------------------------------------- //

void ScopedScheduler::Initialize(std::span<LocalDevice *const> devices) {
  for (LocalDevice *device : devices) {
    accounts_.emplace_back(device);
  }

  for (Account &account : accounts_) {
    auto [it, inserted] = accounts_by_device_id_.emplace(
        std::make_pair(account.device()->address().device_id(), &account));
    if (!inserted) {
      throw std::logic_error("Duplicate device in ScopedScheduler");
    }
    account.Initialize();
    semaphore_count_ += account.semaphore_count();
  }
}

ScopedScheduler::Account &ScopedScheduler::GetDefaultAccount(
    ScopedDevice &device) {
  auto queue_ordinal = device.affinity().lowest_queue_ordinal();
  auto device_id =
      device.raw_device()->address().device_id_for_queue(queue_ordinal);
  auto it = accounts_by_device_id_.find(device_id);
  if (it == accounts_by_device_id_.end()) [[unlikely]] {
    throw std::logic_error(
        fmt::format("Topology check failed: could not find scheduling account "
                    "for device id {:x}",
                    device_id));
  }
  return *it->second;
}

void ScopedScheduler::AppendCommandBuffer(
    ScopedDevice &device, TransactionType tx_type,
    std::function<void(Account &)> callback) {
  Account &account = GetDefaultAccount(device);
  logging::info("AppendCommandBuffer({})", static_cast<void *>(&account));

  auto needed_affinity_bits = device.affinity().queue_affinity();

  // Initialize a fresh command buffer if needed.
  if (!account.active_command_buffer_) {
    logging::info("Create command buffer");
    // Map to a command buffer category.
    iree_hal_command_category_t category;
    switch (tx_type) {
      case TransactionType::TRANSFER:
        category = IREE_HAL_COMMAND_CATEGORY_TRANSFER;
        break;
      case TransactionType::SEQUENTIAL_DISPATCH:
      case TransactionType::PARALLEL_DISPATCH:
        category = IREE_HAL_COMMAND_CATEGORY_DISPATCH;
        break;
      default:
        throw std::logic_error(fmt::format("Unsupported transaction type {}",
                                           static_cast<int>(tx_type)));
    }

    // Set up the command buffer.
    iree_hal_command_buffer_ptr new_cb;
    iree_hal_fence_ptr new_active_deps;
    SHORTFIN_THROW_IF_ERROR(iree_hal_fence_create(
        semaphore_count_, host_allocator_, new_active_deps.for_output()));
    SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_create(
        account.hal_device(),
        /*mode=*/IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        /*command_categories=*/category,
        /*queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0,  // Never indirect so no bindings needed.
        new_cb.for_output()));
    SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_begin(new_cb));

    // Memoize what mode we are in now.
    account.active_tx_type_ = tx_type;
    account.active_queue_affinity_bits_ = needed_affinity_bits;
    account.active_deps_ = std::move(new_active_deps);
    account.active_command_buffer_ = std::move(new_cb);
    account.idle_timepoint_ += 1;
  }

  // Perform the mutation.
  callback(account);

  // Flush.
  if (tx_mode_ == TransactionMode::EAGER) {
    Flush();
  }
}

void ScopedScheduler::Flush() {
  logging::info("Flush");
  // This loop is optimized for a small number of accounts, where it is
  // fine to just linearly probe. If this ever becomes cumbersome, we can
  // maintain a dirty list which is appended to when an account transitions
  // from idle to active.
  for (Account &account : accounts_) {
    if (!account.active_command_buffer_) continue;

    iree_hal_semaphore_t *signal_sem = account.sem_;
    uint64_t signal_timepoint = account.idle_timepoint_;
    iree_hal_command_buffer_t *active_command_buffer =
        account.active_command_buffer_;
    iree_hal_buffer_binding_table_t binding_tables =
        iree_hal_buffer_binding_table_empty();
    SHORTFIN_THROW_IF_ERROR(iree_hal_device_queue_execute(
        account.hal_device(),
        /*queue_affinity=*/account.active_queue_affinity_bits_,
        /*wait_sempahore_list=*/account.active_deps_
            ? iree_hal_fence_semaphore_list(account.active_deps_)
            : iree_hal_semaphore_list_empty(),
        /*signal_semaphore_list=*/
        iree_hal_semaphore_list_t{
            .count = 1,
            .semaphores = &signal_sem,
            .payload_values = &signal_timepoint,
        },
        /*command_buffer_count=*/1,
        /*command_buffers=*/&active_command_buffer,
        /*binding_tables=*/&binding_tables));
    account.Reset();
  }
}

}  // namespace shortfin
