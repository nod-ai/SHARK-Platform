// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/scope.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include "iree/modules/hal/module.h"
#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// Scope
// -------------------------------------------------------------------------- //

Scope::Scope(std::shared_ptr<System> system, Worker &worker,
             std::span<const std::pair<std::string_view, Device *>> devices)
    : host_allocator_(system->host_allocator()),
      scheduler_(*system),
      system_(std::move(system)),
      worker_(worker) {
  for (auto &it : devices) {
    AddDevice(it.first, it.second);
  }
  Initialize();
}

Scope::Scope(std::shared_ptr<System> system, Worker &worker,
             std::span<Device *const> devices)
    : host_allocator_(system->host_allocator()),
      scheduler_(*system),
      system_(std::move(system)),
      worker_(worker) {
  for (auto *device : devices) {
    AddDevice(device->address().logical_device_class, device);
  }
  Initialize();
}

Scope::~Scope() = default;

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

Program Scope::LoadUnboundProgram(std::span<const ProgramModule> modules,
                                  Program::Options options) {
  std::vector<iree_vm_module_t *> all_modules;
  std::vector<iree_hal_device_t *> raw_devices;

  // By default, bind all devices in the scope in order to the program.
  for (Device *d : devices_) {
    raw_devices.push_back(d->hal_device());
  }

  // Add a HAL module.
  // TODO: at some point may want to change this to something similar to
  // what the tooling does in iree_tooling_resolve_modules - it uses
  // iree_vm_module_enumerate_dependencies to walk the dependencies and add the
  // required modules only as needed. to start you could use it just to see if
  // the hal is used, but as you add other module types for exposing sharkfin
  // functionality (or module versions; iree_vm_module_dependency_t has the
  // minimum version required so you can switch between them, and whether they
  // are optional/required).
  iree::vm_module_ptr hal_module;
  SHORTFIN_THROW_IF_ERROR(iree_hal_module_create(
      system().vm_instance(), raw_devices.size(), raw_devices.data(),
      IREE_HAL_MODULE_FLAG_NONE, system().host_allocator(),
      hal_module.for_output()));
  all_modules.push_back(hal_module);

  // Add explicit modules.
  for (auto &pm : modules) {
    all_modules.push_back(pm.vm_module());
  }

  // Create the context.
  iree::vm_context_ptr context;
  iree_vm_context_flags_t flags = IREE_VM_CONTEXT_FLAG_CONCURRENT;
  if (options.trace_execution) flags |= IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION;
  SHORTFIN_THROW_IF_ERROR(iree_vm_context_create_with_modules(
      system().vm_instance(), flags, all_modules.size(), all_modules.data(),
      system().host_allocator(), context.for_output()));

  return Program(std::move(context));
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
