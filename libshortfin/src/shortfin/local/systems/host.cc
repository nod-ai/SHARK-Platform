// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/systems/host.h"

#include "iree/hal/local/loaders/registration/init.h"
#include "shortfin/support/iree_helpers.h"
#include "shortfin/support/logging.h"

namespace shortfin::local::systems {

namespace {
const std::string_view SYSTEM_DEVICE_CLASS = "host-cpu";
const std::string_view LOGICAL_DEVICE_CLASS = "cpu";
const std::string_view HAL_DRIVER_PREFIX = "local";
}  // namespace

// -------------------------------------------------------------------------- //
// HostCPUSystemBuilder
// -------------------------------------------------------------------------- //

HostCPUSystemBuilder::Deps::Deps(iree_allocator_t host_allocator) {
  iree_task_executor_options_initialize(&task_executor_options);
  iree_hal_task_device_params_initialize(&task_params);
  iree_task_topology_initialize(&task_topology_options);

#ifndef NDEBUG
  // TODO: In normal IREE programs, this is exposed as --task_abort_on_failure.
  // It is a critical debug feature as it forces an eager program crash at
  // the point encountered vs as a later, rolled up async status. Since it
  // guards things that are API usage bugs in how we are using the runtime,
  // from our perspective, it is assert like, and we treat it as such.
  // However, it would be best to be independently controllable.
  task_params.queue_scope_flags |= IREE_TASK_SCOPE_FLAG_ABORT_ON_FAILURE;
#endif
}

HostCPUSystemBuilder::Deps::~Deps() {
  iree_task_topology_deinitialize(&task_topology_options);
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  if (device_allocator) {
    iree_hal_allocator_release(device_allocator);
  }
  if (executor) {
    iree_task_executor_release(executor);
  }
  if (plugin_manager) {
    iree_hal_executable_plugin_manager_release(plugin_manager);
  }
}

HostCPUSystemBuilder::HostCPUSystemBuilder(iree_allocator_t host_allocator)
    : HostSystemBuilder(host_allocator), host_cpu_deps_(host_allocator) {}

HostCPUSystemBuilder::~HostCPUSystemBuilder() = default;

void HostCPUSystemBuilder::InitializeHostCPUDefaults() {
  // Give it a default device allocator.
  if (!host_cpu_deps_.device_allocator) {
    SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_create_heap(
        iree_make_cstring_view("local"), host_allocator(), host_allocator(),
        &host_cpu_deps_.device_allocator));
  }

  // And loaders.
  if (host_cpu_deps_.loader_count == 0) {
    SHORTFIN_THROW_IF_ERROR(iree_hal_create_all_available_executable_loaders(
        /*plugin_manager=*/nullptr, IREE_ARRAYSIZE(host_cpu_deps_.loaders),
        &host_cpu_deps_.loader_count, host_cpu_deps_.loaders,
        host_allocator()));
  }
}

SystemPtr HostCPUSystemBuilder::CreateSystem() {
  auto lsys = std::make_shared<System>(host_allocator());
  // TODO: Real NUMA awareness.
  lsys->InitializeNodes(1);
  InitializeHostCPUDefaults();
  auto *driver = InitializeHostCPUDriver(*lsys);
  InitializeHostCPUDevices(*lsys, driver);
  lsys->FinishInitialization();
  return lsys;
}

iree_hal_driver_t *HostCPUSystemBuilder::InitializeHostCPUDriver(System &lsys) {
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

  // Create the driver and save it in the System.
  iree::hal_driver_ptr driver;
  iree_hal_driver_t *unowned_driver;
  SHORTFIN_THROW_IF_ERROR(iree_hal_task_driver_create(
      /*identifier=*/
      {
          .data = HAL_DRIVER_PREFIX.data(),
          .size = HAL_DRIVER_PREFIX.size(),
      },
      &host_cpu_deps_.task_params, /*queue_count=*/1, &host_cpu_deps_.executor,
      host_cpu_deps_.loader_count, host_cpu_deps_.loaders,
      host_cpu_deps_.device_allocator, host_allocator(), driver.for_output()));
  unowned_driver = driver.get();
  lsys.InitializeHalDriver(SYSTEM_DEVICE_CLASS, std::move(driver));
  return unowned_driver;
}

void HostCPUSystemBuilder::InitializeHostCPUDevices(System &lsys,
                                                    iree_hal_driver_t *driver) {
  iree_host_size_t device_info_count = 0;
  iree::allocated_ptr<iree_hal_device_info_t> device_infos(host_allocator());
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_query_available_devices(
      driver, host_allocator(), &device_info_count, &device_infos));

  for (iree_host_size_t i = 0; i < device_info_count; ++i) {
    iree::hal_device_ptr device;
    iree_hal_device_info_t *it = &device_infos.get()[i];
    SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
        driver, it->device_id, 0, nullptr, host_allocator(),
        device.for_output()));
    lsys.InitializeHalDevice(std::make_unique<HostCPUDevice>(
        DeviceAddress(
            /*system_device_class=*/SYSTEM_DEVICE_CLASS,
            /*logical_device_class=*/LOGICAL_DEVICE_CLASS,
            /*hal_driver_prefix=*/HAL_DRIVER_PREFIX,
            /*instance_ordinal=*/i,
            /*queue_ordinal=*/0,
            /*instance_topology_address=*/{0}),
        /*hal_device=*/std::move(device),
        /*node_affinity=*/0,
        /*node_locked=*/false));
  }
}

}  // namespace shortfin::local::systems
