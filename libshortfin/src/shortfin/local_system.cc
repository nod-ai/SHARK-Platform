// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local_system.h"

#include <fmt/core.h>

#include "shortfin/support/logging.h"

namespace shortfin {

// -------------------------------------------------------------------------- //
// LocalDeviceAddress
// -------------------------------------------------------------------------- //

LocalDeviceAddress::LocalDeviceAddress(
    std::string_view system_device_class, std::string_view logical_device_class,
    std::string_view hal_driver_prefix, iree_host_size_t instance_ordinal,
    iree_host_size_t queue_ordinal,
    std::vector<iree_host_size_t> instance_topology_address)
    : system_device_class(std::move(system_device_class)),
      logical_device_class(std::move(logical_device_class)),
      hal_driver_prefix(std::move(hal_driver_prefix)),
      instance_ordinal(instance_ordinal),
      queue_ordinal(queue_ordinal),
      instance_topology_address(instance_topology_address),
      device_name(
          fmt::format("{}:{}:{}@{}", this->system_device_class,
                      this->instance_ordinal, this->queue_ordinal,
                      fmt::join(this->instance_topology_address, ","))) {}

// -------------------------------------------------------------------------- //
// LocalDevice
// -------------------------------------------------------------------------- //

LocalDevice::LocalDevice(LocalDeviceAddress address,
                         iree_hal_device_ptr hal_device, int node_affinity,
                         bool node_locked)
    : address_(std::move(address)),
      hal_device_(std::move(hal_device)),
      node_affinity_(node_affinity),
      node_locked_(node_locked) {}

LocalDevice::~LocalDevice() = default;

// -------------------------------------------------------------------------- //
// LocalSystem
// -------------------------------------------------------------------------- //

LocalSystem::LocalSystem(iree_allocator_t host_allocator)
    : host_allocator_(host_allocator) {
  SHORTFIN_THROW_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                                  host_allocator_,
                                                  vm_instance_.for_output()));
}

LocalSystem::~LocalSystem() {
  // Worker drain and shutdown.
  for (auto &worker : workers_) {
    worker->Kill();
  }
  for (auto &worker : workers_) {
    worker->WaitForShutdown();
  }

  // Orderly destruction of heavy-weight objects.
  // Shutdown order is important so we don't leave it to field ordering.
  vm_instance_.reset();

  // Devices.
  devices_.clear();
  named_devices_.clear();
  retained_devices_.clear();

  // HAL drivers.
  hal_drivers_.clear();
}

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

void LocalSystem::InitializeHalDevice(std::unique_ptr<LocalDevice> device) {
  AssertNotInitialized();
  auto device_name = device->name();
  auto [it, success] = named_devices_.try_emplace(device_name, device.get());
  if (!success) {
    throw std::logic_error(fmt::format(
        "Cannot register LocalDevice '{}' multiple times", device_name));
  }
  devices_.push_back(device.get());
  retained_devices_.push_back(std::move(device));
}

void LocalSystem::FinishInitialization() {
  AssertNotInitialized();

  // TODO: Remove this. Just testing.
  // workers_.push_back(
  //     std::make_unique<Worker>(Worker::Options(host_allocator(),
  //     "worker:0")));
  // workers_.back()->Start();
  // workers_.back()->EnqueueCallback(
  //     []() { spdlog::info("Hi from a worker callback"); });

  initialized_ = true;
}

}  // namespace shortfin
