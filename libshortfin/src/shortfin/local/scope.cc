// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/scope.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// Scope
// -------------------------------------------------------------------------- //

Scope::Scope(std::shared_ptr<System> system, Worker &worker,
             std::span<const std::pair<std::string_view, Device *>> devices)
    : system_(std::move(system)),
      host_allocator_(system_->host_allocator()),
      scheduler_(*system_),
      worker_(worker) {
  logging::construct("Scope", this);
  for (auto &it : devices) {
    AddDevice(it.first, it.second);
  }
  Initialize();
}

Scope::Scope(std::shared_ptr<System> system, Worker &worker,
             std::span<Device *const> devices)
    : system_(std::move(system)),
      host_allocator_(system_->host_allocator()),
      scheduler_(*system_),
      worker_(worker) {
  logging::construct("Scope", this);
  for (auto *device : devices) {
    AddDevice(device->address().logical_device_class, device);
  }
  Initialize();
}

Scope::~Scope() { logging::destruct("Scope", this); }

std::string Scope::to_s() const {
  return fmt::format("Scope(worker='{}', devices=[{}])", worker_.name(),
                     fmt::join(device_names(), ", "));
}

void Scope::Initialize() { scheduler_.Initialize(devices_); }

void Scope::AddDevice(std::string_view device_class, Device *device) {
  device_class = interner_.intern(device_class);
  auto &count = device_class_count_[device_class];
  std::string_view device_name =
      interner_.intern(fmt::format("{}{}", device_class, count++));
  named_devices_[device_name] = device;
  devices_.push_back(device);
}

Device *Scope::raw_device(std::string_view name) const {
  auto it = named_devices_.find(name);
  if (it == named_devices_.end()) [[unlikely]] {
    throw std::invalid_argument(
        fmt::format("Device '{}' not found (available: {})", name,
                    fmt::join(device_names(), ", ")));
  }
  return it->second;
}

Device *Scope::raw_device(int ordinal) const {
  if (ordinal < 0 || ordinal >= devices_.size()) {
    throw std::invalid_argument(
        fmt::format("Device ordinal ({}) out of bounds", ordinal));
  }
  return devices_[ordinal];
}

std::vector<std::string_view> Scope::device_names() const {
  std::vector<std::string_view> names;
  names.reserve(named_devices_.size());
  for (auto &it : named_devices_) {
    names.push_back(it.first);
  }
  return names;
}

// -------------------------------------------------------------------------- //
// ScopedDevice
// -------------------------------------------------------------------------- //

CompletionEvent ScopedDevice::OnSync(bool flush) {
  if (flush) {
    scope().scheduler().Flush();
  }
  auto &default_account = scope().scheduler().GetDefaultAccount(*this);
  return default_account.OnSync();
}

}  // namespace shortfin::local
