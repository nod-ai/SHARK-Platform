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

// A Scope with a back reference to the System from which it
// originated.
class ExtendingScope : public Scope {
 public:
  using Scope::Scope;

 private:
  std::shared_ptr<System> backref_;
  friend std::shared_ptr<System> &mutable_local_scope_backref(ExtendingScope &);
};

std::shared_ptr<System> &mutable_local_scope_backref(ExtendingScope &scope) {
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
  bool needs_shutdown = false;
  {
    iree_slim_mutex_lock_guard guard(lock_);
    if (initialized_ && !shutdown_) {
      needs_shutdown = true;
    }
  }
  if (needs_shutdown) {
    logging::warn(
        "Implicit Shutdown from System destructor. Please call Shutdown() "
        "explicitly for maximum stability.");
    Shutdown();
  }
}

void System::Shutdown() {
  // Stop workers.
  std::vector<std::unique_ptr<Worker>> local_workers;
  {
    iree_slim_mutex_lock_guard guard(lock_);
    if (!initialized_ || shutdown_) return;
    shutdown_ = true;
    workers_by_name_.clear();
    local_workers.swap(workers_);
  }

  // Worker drain and shutdown.
  for (auto &worker : local_workers) {
    worker->Kill();
  }
  for (auto &worker : local_workers) {
    worker->WaitForShutdown();
  }
  local_workers.clear();

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

std::unique_ptr<Scope> System::CreateScope() {
  iree_slim_mutex_lock_guard guard(lock_);
  auto new_scope =
      std::make_unique<ExtendingScope>(host_allocator(), devices());
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

Worker &System::StartExistingWorker(std::unique_ptr<Worker> worker) {
  Worker *unowned_worker;
  {
    iree_slim_mutex_lock_guard guard(lock_);
    std::string_view name = worker->name();
    if (workers_by_name_.count(name) != 0) {
      throw std::invalid_argument(
          fmt::format("Cannot create worker with duplicate name '{}'", name));
    }
    workers_.push_back(std::move(worker));
    unowned_worker = workers_.back().get();
    workers_by_name_[name] = unowned_worker;
  }
  unowned_worker->Start();
  return *unowned_worker;
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
  iree_slim_mutex_lock_guard guard(lock_);
  AssertNotInitialized();
  initialized_ = true;
}

}  // namespace shortfin::local
