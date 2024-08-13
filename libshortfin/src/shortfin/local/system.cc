// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/system.h"

#include <fmt/core.h>

#include "shortfin/local/scope.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

namespace {

// A LocalScope with a back reference to the System from which it
// originated.
class ExtendingLocalScope : public LocalScope {
 public:
  using LocalScope::LocalScope;

 private:
  std::shared_ptr<System> backref_;
  friend std::shared_ptr<System> &mutable_local_scope_backref(
      ExtendingLocalScope &);
};

std::shared_ptr<System> &mutable_local_scope_backref(
    ExtendingLocalScope &scope) {
  return scope.backref_;
}

}  // namespace

// -------------------------------------------------------------------------- //
// System
// -------------------------------------------------------------------------- //

System::System(iree_allocator_t host_allocator)
    : host_allocator_(host_allocator) {
  SHORTFIN_THROW_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                                  host_allocator_,
                                                  vm_instance_.for_output()));
}

System::~System() {
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

std::unique_ptr<LocalScope> System::CreateScope() {
  auto new_scope =
      std::make_unique<ExtendingLocalScope>(host_allocator(), devices());
  mutable_local_scope_backref(*new_scope) = shared_from_this();
  return new_scope;
}

void System::InitializeNodes(int node_count) {
  AssertNotInitialized();
  if (!nodes_.empty()) {
    throw std::logic_error("System::InitializeNodes called more than once");
  }
  nodes_.reserve(node_count);
  for (int i = 0; i < node_count; ++i) {
    nodes_.emplace_back(i);
  }
}

void System::InitializeHalDriver(std::string_view moniker,
                                 iree_hal_driver_ptr driver) {
  AssertNotInitialized();
  auto &slot = hal_drivers_[moniker];
  if (slot) {
    throw std::logic_error(fmt::format(
        "Cannot register multiple hal drivers with moniker '{}'", moniker));
  }
  slot.reset(driver.release());
}

void System::InitializeHalDevice(std::unique_ptr<Device> device) {
  AssertNotInitialized();
  auto device_name = device->name();
  auto [it, success] = named_devices_.try_emplace(device_name, device.get());
  if (!success) {
    throw std::logic_error(
        fmt::format("Cannot register Device '{}' multiple times", device_name));
  }
  devices_.push_back(device.get());
  retained_devices_.push_back(std::move(device));
}

void System::FinishInitialization() {
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

}  // namespace shortfin::local
