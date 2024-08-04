// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SYSTEM_H
#define SHORTFIN_LOCAL_SYSTEM_H

#include <memory>
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

  // Get a shared pointer from the instance.
  std::shared_ptr<LocalSystem> shared_ptr() { return shared_from_this(); }

  iree_allocator_t host_allocator() { return host_allocator_; }

  // Initialization APIs. Calls to these methods is only permitted between
  // construction and Initialize().
  // ------------------------------------------------------------------------ //
  void InitializeHalDriver(std::string moniker, iree_hal_driver_ptr driver);
  void InitializeHalDevices(std::string moniker,
                            std::vector<iree_hal_device_ptr> devices);
  void FinishInitialization();

 private:
  void AssertNotInitialized() {
    if (initialized_) {
      throw std::runtime_error(
          "LocalSystem::Initialize* methods can only be called during "
          "initialization");
    }
  }

  const iree_allocator_t host_allocator_;

  // Map of retained hal drivers. These will be released as one of the
  // last steps of destruction.
  std::unordered_map<std::string, iree_hal_driver_ptr> hal_drivers_;

  // All available devices, indexed by driver name and by local ordinal.
  std::unordered_map<std::string, std::vector<iree_hal_device_ptr>>
      hal_devices_;

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

// A device attached to the LocalSystem.
class SHORTFIN_API LocalSystemDevice {
 public:
  LocalSystemDevice(iree_hal_device_ptr hal_device,
                    LocalSystemNode *node_affinity, bool node_locked)
      : hal_device_(hal_device),
        node_affinity_(node_affinity),
        node_locked_(node_locked) {}

 private:
  iree_hal_device_ptr hal_device_;
  LocalSystemNode *node_affinity_;
  bool node_locked_;
};

}  // namespace shortfin

#endif  // SHORTFIN_LOCAL_SYSTEM_H
