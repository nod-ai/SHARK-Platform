// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/systems/host.h"

#include "shortfin/support/iree_helpers.h"
#include "shortfin/support/logging.h"

namespace shortfin::systems {

// -------------------------------------------------------------------------- //
// HostCPUSystemConfig
// -------------------------------------------------------------------------- //

HostCPUSystemConfig::HostCPUSystemConfig(iree_allocator_t host_allocator)
    : HostSystemConfig(host_allocator) {
  iree_task_executor_options_initialize(&host_cpu_deps_.task_executor_options);
  iree_hal_task_device_params_initialize(&host_cpu_deps_.task_params);
}

HostCPUSystemConfig::~HostCPUSystemConfig() {
  iree_task_topology_deinitialize(&host_cpu_deps_.task_topology_options);
}

void HostCPUSystemConfig::InitializeHostCPUDefaults() {
  // Give it a default device allocator.
  if (!host_cpu_deps_.device_allocator) {
    logging::info("Using default heap allocator for host CPU devices");
    SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_create_heap(
        iree_make_cstring_view("local"), host_allocator(), host_allocator(),
        &host_cpu_deps_.device_allocator));
  }
}

LocalSystemPtr HostCPUSystemConfig::CreateLocalSystem() {
  InitializeHostCPUDefaults();
  auto lsys = std::make_shared<LocalSystem>(host_allocator());
  auto *driver = InitializeHostCPUDriver(*lsys);
  InitializeHostCPUDevices(*lsys, driver);
  lsys->FinishInitialization();
  return lsys;
}

iree_hal_driver_t *HostCPUSystemConfig::InitializeHostCPUDriver(
    LocalSystem &lsys) {
  // TODO: Kill these flag variants in favor of settings on the config
  // object.
  SHORTFIN_THROW_IF_ERROR(iree_task_executor_options_initialize_from_flags(
      &host_cpu_deps_.task_executor_options));
  // TODO: Do something smarter than pinning to NUMA node 0.
  SHORTFIN_THROW_IF_ERROR(iree_task_topology_initialize_from_flags(
      /*node_id=*/0, &host_cpu_deps_.task_topology_options));

  SHORTFIN_THROW_IF_ERROR(
      iree_task_executor_create(host_cpu_deps_.task_executor_options,
                                &host_cpu_deps_.task_topology_options,
                                host_allocator(), &host_cpu_deps_.executor));

  // Create the driver and save it in the LocalSystem.
  iree_hal_driver_ptr driver;
  iree_hal_driver_t *unowned_driver;
  SHORTFIN_THROW_IF_ERROR(iree_hal_task_driver_create(
      IREE_SV("local-task"), &host_cpu_deps_.task_params, /*queue_count=*/1,
      &host_cpu_deps_.executor, host_cpu_deps_.loader_count,
      host_cpu_deps_.loaders, host_cpu_deps_.device_allocator, host_allocator(),
      driver.for_output()));
  unowned_driver = driver.get();
  lsys.InitializeHalDriver("host-cpu", std::move(driver));
  return unowned_driver;
}

void HostCPUSystemConfig::InitializeHostCPUDevices(LocalSystem &lsys,
                                                   iree_hal_driver_t *driver) {
  iree_host_size_t device_info_count = 0;
  allocated_ptr<iree_hal_device_info_t> device_infos(host_allocator());
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_query_available_devices(
      driver, host_allocator(), &device_info_count, &device_infos));

  std::vector<iree_hal_device_ptr> devices;
  devices.reserve(device_info_count);
  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    iree_hal_device_ptr device;
    iree_hal_device_info_t *it = &device_infos.ptr[i];
    SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
        driver, it->device_id, 0, nullptr, host_allocator(),
        device.for_output()));
    devices.push_back(std::move(device));
  }
  lsys.InitializeHalDevices("host-cpu", std::move(devices));
}

}  // namespace shortfin::systems
