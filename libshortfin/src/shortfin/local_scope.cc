// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local_scope.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

namespace shortfin {

LocalScope::LocalScope(
    std::span<std::pair<std::string_view, LocalDevice *>> devices) {
  for (auto &it : devices) {
    AddDevice(it.first, it.second);
  }
}

LocalScope::LocalScope(std::span<LocalDevice *> devices) {
  for (auto *device : devices) {
    AddDevice(device->address().logical_device_class, device);
  }
}

LocalScope::~LocalScope() = default;

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

}  // namespace shortfin
