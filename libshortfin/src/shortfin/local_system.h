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
class LocalDevice;
class LocalNode;

// NUMA node on the LocalSystem. There will always be at least one node, and
// not all NUMA nodes on the system may be included: only those applicable
// to device pinning/scheduling.
class SHORTFIN_API LocalNode {
 public:
  LocalNode(int node_num) : node_num_(node_num) {}

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
  std::span<const LocalNode> nodes() const { return {nodes_}; }
  const std::vector<LocalDevice *> &devices() const { return devices_; }
  const std::unordered_map<std::string_view, LocalDevice *> &named_devices()
      const {
    return named_devices_;
  }

  // Initialization APIs. Calls to these methods is only permitted between
  // construction and Initialize().
  // ------------------------------------------------------------------------ //
  void InitializeNodes(int node_count);
  void InitializeHalDriver(std::string_view moniker,
                           iree_hal_driver_ptr driver);
  void InitializeHalDevice(std::unique_ptr<LocalDevice> device);
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
  std::vector<LocalNode> nodes_;

  // Map of retained hal drivers. These will be released as one of the
  // last steps of destruction. There are some ancillary uses for drivers
  // after initialization, but mainly this is for keeping them alive.
  std::unordered_map<std::string_view, iree_hal_driver_ptr> hal_drivers_;

  // Map of device name to a LocalSystemDevice.
  std::vector<std::unique_ptr<LocalDevice>> retained_devices_;
  std::unordered_map<std::string_view, LocalDevice *> named_devices_;
  std::vector<LocalDevice *> devices_;

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

// Each device exists in the local system as part of some topology that consists
// of the following levels:
//
//   Level 0: User device category / system driver prefix
//     (i.e. "hip", "cuda", "local").
//   Level 1: Device instance ordinal.
//   Level 2: Instance topology vector representing the logical organization
//     of the queues on the device instance.
//
// Concretely, this means that each leaf LocalSystemDevice instance consists
// of an iree_hal_device_t (as managed by an iree_hal_driver_t) and a
// single bit position within an iree_hal_queue_affinity_t. The total number
// of devices of a class is thus equal to the product of the device instance
// ordinal and every entry of the instance topology vector. There can be at
// most 64 queues on a device instance.
//
// How the topology is laid out is system and use case specific, with multiple
// valid arrangements which may be useful for different kinds of workloads.
// There are some general guidelines:
//
//   * All components of a device with peered memory should share the same
//     Level 1 / device instance.
//   * Whether cross-bus devices should share an instance is use case
//     specific, effectively dictated by the nature of the bus connection
//     and intended use. While an instance can be shared across a lower speed
//     link, it may be advantageous to split it and treat the corresponding
//     device leaves as independent from a memory and scheduling perspective.
//   * The instance topology should generally reflect some notion of locality
//     within the physical architecture of some hardware such that co scheduling
//     at leaf nodes of the vector may have some benefit.
//
// Examples:
//   * Large CPU system with two NUMA nodes:
//     - Split instances on NUMA node: local/2/8
//     - Unified instances for an entire chip: local/1/2,8
//     - Different exotic topologies can be represented with a longer topology
//       vector with machine specific communication cost.
//   * Machine with 8 MI300 GPUs (each with 4 memory controllers and 8
//     partitions):
//     - Split instances per host NUMA node: hip/2/4,4,2
//     - Unified instances: hip/1/8,4,2
//     - Simple 8x partition (ignore memory controller): hip/1/8,8,1
//  * Machine with 8 MI300 GPUs operating as one large GPU each: hip/1/8,1,1
//
// Generally, there is a tension that must be negotiated between how much an
// application cares about the hierarchy vs benefiting from tighter coordination
// of locality. The shape of the instance topology must match among all devices
// attached to a driver.
struct SHORTFIN_API LocalDeviceAddress {
 public:
  // Note that all string_views passed should be literals or have a lifetime
  // that exceeds the instance.
  LocalDeviceAddress(std::string_view system_device_class,
                     std::string_view logical_device_class,
                     std::string_view hal_driver_prefix,
                     iree_host_size_t instance_ordinal,
                     iree_host_size_t queue_ordinal,
                     std::vector<iree_host_size_t> instance_topology_address);

  // User driver name (i.e. 'amdgpu'). In user visible names/messages, this
  // is preferred over hal_driver_prefix, but must be 1:1 with it.
  std::string_view system_device_class;
  // User device class (i.e. 'gpu', 'cpu').
  std::string_view logical_device_class;
  // System HAL driver prefix (i.e. 'hip', 'cuda', 'local').
  std::string_view hal_driver_prefix;
  iree_host_size_t instance_ordinal;
  iree_host_size_t queue_ordinal;
  std::vector<iree_host_size_t> instance_topology_address;
  // A system-unique device name:
  //   {system_device_class}:{instance_ordinal}:{queue_ordinal}@{instance_topology_address}
  std::string device_name;
};

// A device attached to the LocalSystem.
class SHORTFIN_API LocalDevice {
 public:
  LocalDevice(LocalDeviceAddress address, iree_hal_device_ptr hal_device,
              int node_affinity, bool node_locked);
  virtual ~LocalDevice();

  const LocalDeviceAddress &address() const { return address_; }
  std::string_view name() const { return address_.device_name; }
  int node_affinity() const { return node_affinity_; }
  bool node_locked() const { return node_locked_; }

 private:
  LocalDeviceAddress address_;
  iree_hal_device_ptr hal_device_;
  int node_affinity_;
  bool node_locked_;
};

}  // namespace shortfin

#endif  // SHORTFIN_LOCAL_SYSTEM_H
