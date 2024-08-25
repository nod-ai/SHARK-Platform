// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SCOPE_H
#define SHORTFIN_LOCAL_SCOPE_H

#include <functional>
#include <span>
#include <unordered_map>

#include "shortfin/local/async.h"
#include "shortfin/local/device.h"
#include "shortfin/local/program.h"
#include "shortfin/local/scheduler.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin::local {

class SHORTFIN_API Scope;
class SHORTFIN_API System;
class SHORTFIN_API Worker;

// Wraps a Scope and a DeviceAffinity together. This is used in all
// Scope based APIs as a short-hand for "device" as it contains everything
// needed to do thing with some slice of device queues.
class SHORTFIN_API ScopedDevice {
 public:
  ScopedDevice() = default;
  ScopedDevice(Scope &scope, DeviceAffinity affinity)
      : scope_(&scope), affinity_(affinity) {}
  ScopedDevice(Scope &scope, Device *device)
      : scope_(&scope), affinity_(device) {}
  ScopedDevice(const ScopedDevice &other)
      : scope_(other.scope_), affinity_(other.affinity_) {}

  Scope &scope() const {
    assert(scope_ && "scope must not be null");
    return *scope_;
  }
  DeviceAffinity affinity() const { return affinity_; }
  Device *raw_device() const { return affinity_.device(); }

  std::string to_s() const { return affinity().to_s(); }

  bool operator==(const ScopedDevice &other) const {
    return (scope_ == other.scope_) && affinity_ == other.affinity_;
  }

  // Returns a future which will be satisfied when the primary device timeline
  // of this affinity set progresses to "now". This will be true when all
  // currently queued work on the device has been completed.
  CompletionEvent OnSync(bool flush = true);

 private:
  Scope *scope_ = nullptr;
  DeviceAffinity affinity_;
};

// A logical scope of execution, consisting of participating devices,
// resources, and timelines. Most interaction with the compute resources
// is done on these instances.
//
// The scope is generally instantiated with a slice of system resources,
// and produces an arrangement that is easy to use vs maximally diverse.
//
// Devices
// -------
// The scope is initialized with a list of participating devices, which is
// a subset of all devices managed by the LocalSystem. Each device is given
// a logical name of the form `<device_class><index>`, by default using the
// DeviceAddress::logical_device_class as the `<device_class>`. In exotic
// situations, this can be customized. By default, devices are added in the
// order defined by the system and will have an `<index>` corresponding to
// their order. It is up to the constructor to produce a sensible arrangement.
class SHORTFIN_API Scope : public std::enable_shared_from_this<Scope> {
 public:
  // Initialize with devices using logical_device_class as the device class.
  Scope(std::shared_ptr<System> system, Worker &worker,
        std::span<Device *const> devices);
  // Initialize with devices with custom device class names.
  Scope(std::shared_ptr<System> system, Worker &worker,
        std::span<const std::pair<std::string_view, Device *>> devices);
  Scope(const Scope &) = delete;
  // Ensure polymorphic.
  virtual ~Scope();
  std::string to_s() const;

  // All scopes are created as shared pointers.
  std::shared_ptr<Scope> shared_ptr() { return shared_from_this(); }

  // The worker that this scope is bound to.
  Worker &worker() { return worker_; }

  // System that this scope is bound to.
  System &system() { return *system_; }

  // Device access.
  // Throws std::invalid_argument on lookup failure.
  Device *raw_device(std::string_view name) const;
  const std::unordered_map<std::string_view, Device *> named_devices() const {
    return named_devices_;
  }
  Device *raw_device(int index) const;
  Device *raw_device(Device *device) const { return device; }
  const std::vector<Device *> &raw_devices() const { return devices_; }
  std::vector<std::string_view> device_names() const;

  // Variadic helper for making a DeviceAffinity from any of:
  //  * Explicit Device*
  //  * Device name (from a Scope)
  //  * Device index (from a Scope)
  // If at any point during accumulation, the DeviceAffinity would be invalid,
  // then a std::invalid_argument exception is thrown. Any failure to resolve
  // a name or index will also throw a std::invalid_argument.
  ScopedDevice device() { return ScopedDevice(*this, DeviceAffinity()); }
  template <typename T, typename... Args>
  ScopedDevice device(T first, Args... args) {
    return ScopedDevice(
        *this, device(args...).affinity() | DeviceAffinity(raw_device(first)));
  }
  ScopedDevice device(Device *d) {
    return ScopedDevice(*this, DeviceAffinity(d));
  }
  detail::Scheduler &scheduler() { return scheduler_; }
  detail::TimelineResource::Ref NewTimelineResource() {
    return scheduler().NewTimelineResource(host_allocator_);
  }

  // Loads a program from a list of modules onto the devices managed by this
  // scope. The resulting program is not bound to this scope and can be imported
  // into compatible scopes for actual execution.
  // TODO: This is temporary during API evolution: a higher level API that
  // includes all module concepts, params, etc is needed.
  Program LoadUnboundProgram(std::span<const ProgramModule> modules,
                             Program::Options options = {});

 private:
  void AddDevice(std::string_view device_class, Device *device);
  void Initialize();  // Called after all devices are added.

  iree_allocator_t host_allocator_;
  string_interner interner_;
  // Map of `<device_class>` to the count of that class contained.
  std::unordered_map<std::string_view, int> device_class_count_;
  // Ordered devices.
  std::vector<Device *> devices_;
  // Map of `<device_class><index>` to Device.
  std::unordered_map<std::string_view, Device *> named_devices_;
  detail::Scheduler scheduler_;

  // Back reference to owning system.
  std::shared_ptr<System> system_;
  Worker &worker_;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_SCOPE_H
