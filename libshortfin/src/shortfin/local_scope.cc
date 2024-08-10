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
    std::span<const std::pair<std::string_view, LocalDevice *>> devices) {
  for (auto &it : devices) {
    AddDevice(it.first, it.second);
  }
  Initialize();
}

LocalScope::LocalScope(std::span<LocalDevice *const> devices) {
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
}

// -------------------------------------------------------------------------- //
// ScopedScheduler::TimelineResource
// -------------------------------------------------------------------------- //

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
  auto needed_affinity_bits = device.affinity().queue_affinity();

  if (account.active_tx_type_ != TransactionType::NONE) {
    // Potentially auto flush.
    bool needs_tx_type_flush = (tx_mode_ == TransactionMode::AUTO_FLUSH &&
                                account.active_tx_type_ != tx_type);
    bool needs_affinity_flush =
        needed_affinity_bits != account.active_queue_affinity_bits_;
    if (needs_tx_type_flush || needs_affinity_flush) {
      logging::info("Auto flush device {:x}",
                    account.device_->address().device_id());
    }

    // TODO: Does this need to be a flush of all devices? Or should the
    // transaction type be carried at the scheduler level. We may need to
    // do this at the scheduler level because there could in theory be causality
    // cycles across devices, I think?
  }
}

}  // namespace shortfin
