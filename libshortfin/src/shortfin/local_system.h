// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SYSTEM_H
#define SHORTFIN_LOCAL_SYSTEM_H

#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin {

class LocalSystem;
class LocalSystemBuilder;
class LocalSystemDevice;
class LocalSystemNode;

// NUMA node on the LocalSystem. There will always be at least one node, and
// not all NUMA nodes on the system may be included: only those applicable
// to device pinning/scheduling.
class SHORTFIN_API LocalSystemNode {
 public:
  LocalSystemNode(int node_num) : node_num_(node_num) {}

  int node_num() const { return node_num_; }

 private:
  int node_num_;
};

// Encapsulates resources attached to the local system. In most applications,
// there will be one of these, and it is used to keep long lived access to
// physical devices, connections, and other long lived resources which need
// to be available across the application lifetime.
//
// One does not generally construct a LocalSystem by hand, instead relying
// on some form of factory that constructs one to suit both the system being
// executed on and any preferences on which resources should be accessible.
//
// As the root of the hierarchy and the owner of numerous ancillary resources,
// we declare that LocalSystem is always managed via a shared_ptr, as this
// simplifies many aspects of system management.
class SHORTFIN_API LocalSystem
    : public std::enable_shared_from_this<LocalSystem> {
 public:
  LocalSystem(iree_allocator_t host_allocator);
  LocalSystem(const LocalSystem &) = delete;

  // Get a shared pointer from the instance.
  std::shared_ptr<LocalSystem> shared_ptr() { return shared_from_this(); }

  iree_allocator_t host_allocator() { return host_allocator_; }

  // Topology access.
  std::span<const LocalSystemNode> nodes() const { return {nodes_}; }
  const std::vector<LocalSystemDevice *> &devices() const { return devices_; }
  const std::unordered_map<std::string_view, LocalSystemDevice *> &
  named_devices() const {
    return named_devices_;
  }

  // Initialization APIs. Calls to these methods is only permitted between
  // construction and Initialize().
  // ------------------------------------------------------------------------ //
  void InitializeNodes(int node_count);
  void InitializeHalDriver(std::string_view moniker,
                           iree_hal_driver_ptr driver);
  void InitializeHalDevice(std::unique_ptr<LocalSystemDevice> device);
  void FinishInitialization();

 private:
  void AssertNotInitialized() {
    if (initialized_) {
      throw std::logic_error(
          "LocalSystem::Initialize* methods can only be called during "
          "initialization");
    }
  }

  const iree_allocator_t host_allocator_;

  // NUMA nodes relevant to this system.
  std::vector<LocalSystemNode> nodes_;

  // Map of retained hal drivers. These will be released as one of the
  // last steps of destruction. There are some ancillary uses for drivers
  // after initialization, but mainly this is for keeping them alive.
  std::unordered_map<std::string_view, iree_hal_driver_ptr> hal_drivers_;

  // Map of device name to a LocalSystemDevice.
  std::vector<std::unique_ptr<LocalSystemDevice>> retained_devices_;
  std::unordered_map<std::string_view, LocalSystemDevice *> named_devices_;
  std::vector<LocalSystemDevice *> devices_;

  // Whether initialization is complete. If true, various low level
  // mutations are disallowed.
  bool initialized_ = false;
};
using LocalSystemPtr = std::shared_ptr<LocalSystem>;

// Base class for configuration objects for setting up a LocalSystem.
class SHORTFIN_API LocalSystemBuilder {
 public:
  LocalSystemBuilder(iree_allocator_t host_allocator)
      : host_allocator_(host_allocator) {}
  LocalSystemBuilder() : LocalSystemBuilder(iree_allocator_system()) {}
  virtual ~LocalSystemBuilder() = default;

  iree_allocator_t host_allocator() { return host_allocator_; }

  // Construct a LocalSystem
  virtual LocalSystemPtr CreateLocalSystem() = 0;

 private:
  const iree_allocator_t host_allocator_;
};

// A device attached to the LocalSystem.
// Every device is uniquely named by a combination of its device_class
// and device_index as "{device_class}:{device_index}".
class SHORTFIN_API LocalSystemDevice {
 public:
  LocalSystemDevice(std::string device_class, int device_index,
                    std::string driver_name, iree_hal_device_ptr hal_device,
                    int node_affinity, bool node_locked);
  virtual ~LocalSystemDevice();

  std::string_view device_class() const { return device_class_; }
  int device_index() const { return device_index_; }
  std::string_view name() const { return name_; }
  std::string_view driver_name() const { return driver_name_; }
  int node_affinity() const { return node_affinity_; }
  bool node_locked() const { return node_locked_; }

 private:
  std::string device_class_;
  int device_index_;
  std::string name_;
  std::string driver_name_;
  iree_hal_device_ptr hal_device_;
  int node_affinity_;
  bool node_locked_;
};

}  // namespace shortfin

#endif  // SHORTFIN_LOCAL_SYSTEM_H
