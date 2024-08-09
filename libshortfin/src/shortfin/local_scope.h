// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SCOPE_H
#define SHORTFIN_LOCAL_SCOPE_H

#include <span>
#include <unordered_map>

#include "shortfin/local_device.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin {

class LocalScope;

// Wraps a LocalScope and a DeviceAffinity together. This is used in all
// Scope based APIs as a short-hand for "device" as it contains everything
// needed to do thing with some slice of device queues.
class ScopedDevice {
 public:
  ScopedDevice(LocalScope &scope, DeviceAffinity affinity)
      : scope_(scope), affinity_(affinity) {}
  ScopedDevice(LocalScope &scope, LocalDevice *device)
      : scope_(scope), affinity_(device) {}

  LocalScope &scope() const { return scope_; }
  DeviceAffinity affinity() const { return affinity_; }
  LocalDevice *raw_device() const { return affinity_.device(); }

  std::string to_s() const { return affinity().to_s(); }

  bool operator==(const ScopedDevice &other) const {
    return (&scope_ == &other.scope_) && affinity_ == other.affinity_;
  }

 private:
  LocalScope &scope_;
  DeviceAffinity affinity_;
};

// Handles scheduling state for a scope.
class SHORTFIN_API ScopedScheduler {
 public:
  class Timeline {
   public:
    Timeline(iree_hal_device_t *device);

   private:
    iree_hal_semaphore_ptr sem_;
    uint64_t now_ = 0ull;
  };

  // Accounting structure for a single iree_hal_device_t (note that multiple
  // LocalDevice instances may map to a single HAL device).
  class Account {
   public:
    Account(iree_hal_device_t *hal_device);
    iree_hal_device_t *hal_device() { return hal_device_; }
    size_t semaphore_count() const { return 2; }

   private:
    iree_hal_device_t *hal_device_ = nullptr;
    Timeline tx_timeline_;  // Transfer timeline
    Timeline ex_timeline_;  // Exec timeline
  };

 private:
  void Initialize(std::span<LocalDevice *const> devices);
  // Each distinct hal device gets an account.
  std::unordered_map<iree_hal_device_t *, Account> accounts_;
  size_t semaphore_count_ = 0;
  friend class LocalScope;
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
// LocalDeviceAddress::logical_device_class as the `<device_class>`. In exotic
// situations, this can be customized. By default, devices are added in the
// order defined by the system and will have an `<index>` corresponding to
// their order. It is up to the constructor to produce a sensible arrangement.
class SHORTFIN_API LocalScope {
 public:
  // Initialize with devices using logical_device_class as the device class.
  LocalScope(std::span<LocalDevice *const> devices);
  // Initialize with devices with custom device class names.
  LocalScope(
      std::span<const std::pair<std::string_view, LocalDevice *>> devices);
  LocalScope(const LocalScope &) = delete;
  // Ensure polymorphic.
  virtual ~LocalScope();

  // Device access.
  // Throws std::invalid_argument on lookup failure.
  LocalDevice *raw_device(std::string_view name) const;
  const std::unordered_map<std::string_view, LocalDevice *> named_devices()
      const {
    return named_devices_;
  }
  LocalDevice *raw_device(int index) const;
  LocalDevice *raw_device(LocalDevice *device) const { return device; }
  const std::vector<LocalDevice *> &raw_devices() const { return devices_; }
  std::vector<std::string_view> device_names() const;

  // Variadic helper for making a DeviceAffinity from any of:
  //  * Explicit LocalDevice*
  //  * Device name (from a LocalScope)
  //  * Device index (from a LocalScope)
  // If at any point during accumulation, the DeviceAffinity would be invalid,
  // then a std::invalid_argument exception is thrown. Any failure to resolve
  // a name or index will also throw a std::invalid_argument.
  ScopedDevice device() { return ScopedDevice(*this, DeviceAffinity()); }
  template <typename T, typename... Args>
  ScopedDevice device(T first, Args... args) {
    return ScopedDevice(
        *this, device(args...).affinity() | DeviceAffinity(raw_device(first)));
  }
  ScopedDevice device(LocalDevice *d) {
    return ScopedDevice(*this, DeviceAffinity(d));
  }

 private:
  void AddDevice(std::string_view device_class, LocalDevice *device);
  void Initialize();  // Called after all devices are added.

  string_interner interner_;

  // Map of `<device_class>` to the count of that class contained.
  std::unordered_map<std::string_view, int> device_class_count_;
  // Ordered devices.
  std::vector<LocalDevice *> devices_;
  // Map of `<device_class><index>` to LocalDevice.
  std::unordered_map<std::string_view, LocalDevice *> named_devices_;
  ScopedScheduler scheduler_;
};

}  // namespace shortfin

#endif  // SHORTFIN_LOCAL_SCOPE_H
