// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SYSTEMS_HOST_H
#define SHORTFIN_LOCAL_SYSTEMS_HOST_H

#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/hal/local/executable_plugin_manager.h"
#include "iree/task/api.h"
#include "shortfin/local/system.h"
#include "shortfin/support/api.h"

namespace shortfin::local::systems {

// CPU device subclass.
class SHORTFIN_API HostCPUDevice : public Device {
 public:
  using Device::Device;
};

// Configuration for building a host-based System.
class SHORTFIN_API HostSystemBuilder : public SystemBuilder {
 public:
  using SystemBuilder::SystemBuilder;
};

// Specialization of HostSystemBuilder which has CPU executors. Accelerator
// based systems which wish to also enable heterogenous CPU-based execution
// can extend this class (or provide features themselves).
class SHORTFIN_API HostCPUSystemBuilder : public HostSystemBuilder {
 public:
  HostCPUSystemBuilder(iree_allocator_t host_allocator);
  HostCPUSystemBuilder() : HostCPUSystemBuilder(iree_allocator_system()) {}
  ~HostCPUSystemBuilder() override;

  // Creates a System based purely on the CPU config. Derived classes
  // must wholly replace this method, using protected piece-wise components.
  SystemPtr CreateSystem() override;

 protected:
  // Initializes any host-cpu defaults that have not been configured yet.
  void InitializeHostCPUDefaults();
  // Initializes the host-cpu driver and registers it with a System.
  // Returns an unowned pointer to the driver that is lifetime bound to the
  // System.
  iree_hal_driver_t* InitializeHostCPUDriver(System& lsys);
  // Registers all eligible host-cpu devices with the System, given
  // a driver created from InitializeHostCPUDriver().
  void InitializeHostCPUDevices(System& lsys, iree_hal_driver_t* driver);

  struct Deps {
    Deps(iree_allocator_t host_allocator);
    ~Deps();
    iree_task_topology_t task_topology_options;
    iree_task_executor_options_t task_executor_options;
    iree_hal_task_device_params_t task_params;
    iree_hal_executable_plugin_manager_t* plugin_manager = nullptr;
    iree_hal_executable_loader_t* loaders[8] = {nullptr};
    iree_host_size_t loader_count = 0;
    iree_task_executor_t* executor = nullptr;
    iree_hal_allocator_t* device_allocator = nullptr;
  } host_cpu_deps_;
};

}  // namespace shortfin::local::systems

#endif  // SHORTFIN_LOCAL_SYSTEMS_HOST_H
