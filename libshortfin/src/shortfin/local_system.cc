// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local_system.h"

#include <fmt/core.h>

#include "shortfin/support/logging.h"

namespace shortfin {

LocalSystem::LocalSystem(iree_allocator_t host_allocator)
    : host_allocator_(host_allocator) {}

void LocalSystem::InitializeHalDriver(std::string moniker,
                                      iree_hal_driver_ptr driver) {
  AssertNotInitialized();
  auto &slot = hal_drivers_[moniker];
  if (slot) {
    throw std::invalid_argument(fmt::format(
        "Cannot register multiple hal drivers with moniker '{}'", moniker));
  }
  slot.reset(driver.release());
}

void LocalSystem::InitializeHalDevices(
    std::string moniker, std::vector<iree_hal_device_ptr> devices) {
  AssertNotInitialized();
  auto &slot = hal_devices_[moniker];
  if (!slot.empty()) {
    throw std::invalid_argument(fmt::format(
        "Cannot register hal devices multiple times with the same moniker '{}'",
        moniker));
  }
  logging::info("Registered {} '{}' devices", devices.size(), moniker);
  slot.swap(devices);
}

void LocalSystem::FinishInitialization() {
  AssertNotInitialized();
  initialized_ = true;
}

}  // namespace shortfin
