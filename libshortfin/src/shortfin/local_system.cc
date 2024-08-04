// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local_system.h"

#include <fmt/core.h>

#include "shortfin/support/logging.h"

namespace shortfin {

LocalSystemDevice::LocalSystemDevice(std::string device_class, int device_index,
                                     std::string driver_name,
                                     iree_hal_device_ptr hal_device,
                                     int node_affinity, bool node_locked)
    : device_class_(std::move(device_class)),
      device_index_(device_index),
      name_(fmt::format("{}:{}", device_class_, device_index_)),
      driver_name_(std::move(driver_name)),
      hal_device_(std::move(hal_device)),
      node_affinity_(node_affinity),
      node_locked_(node_locked) {}

LocalSystemDevice::~LocalSystemDevice() = default;

LocalSystem::LocalSystem(iree_allocator_t host_allocator)
    : host_allocator_(host_allocator) {}

void LocalSystem::InitializeNodes(int node_count) {
  AssertNotInitialized();
  if (!nodes_.empty()) {
    throw std::logic_error(
        "LocalSystem::InitializeNodes called more than once");
  }
  nodes_.reserve(node_count);
  for (int i = 0; i < node_count; ++i) {
    nodes_.emplace_back(i);
  }
}

void LocalSystem::InitializeHalDriver(std::string_view moniker,
                                      iree_hal_driver_ptr driver) {
  AssertNotInitialized();
  auto &slot = hal_drivers_[moniker];
  if (slot) {
    throw std::logic_error(fmt::format(
        "Cannot register multiple hal drivers with moniker '{}'", moniker));
  }
  slot.reset(driver.release());
}

void LocalSystem::InitializeHalDevice(
    std::unique_ptr<LocalSystemDevice> device) {
  AssertNotInitialized();
  auto device_name = device->name();
  auto [it, success] = named_devices_.try_emplace(device_name, device.get());
  if (!success) {
    throw std::logic_error(fmt::format(
        "Cannot register LocalSystemDevice '{}' multiple times", device_name));
  }
  devices_.push_back(device.get());
  retained_devices_.push_back(std::move(device));
}

void LocalSystem::FinishInitialization() {
  AssertNotInitialized();
  initialized_ = true;
}

}  // namespace shortfin
