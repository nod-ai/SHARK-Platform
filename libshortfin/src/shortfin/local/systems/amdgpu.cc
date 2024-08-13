// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/systems/amdgpu.h"

#include "shortfin/support/logging.h"

namespace shortfin::systems {

namespace {
const std::string_view SYSTEM_DEVICE_CLASS = "amdgpu";
const std::string_view LOGICAL_DEVICE_CLASS = "gpu";
const std::string_view HAL_DRIVER_PREFIX = "hip";
}  // namespace

AMDGPUSystemBuilder::AMDGPUSystemBuilder(iree_allocator_t host_allocator)
    : HostCPUSystemBuilder(host_allocator) {
  InitializeDefaultSetting();
  iree_hal_hip_device_params_initialize(&default_device_params_);
}

AMDGPUSystemBuilder::~AMDGPUSystemBuilder() = default;

void AMDGPUSystemBuilder::InitializeDefaultSetting() {
  char *raw_dylib_path_env_cstr = std::getenv("IREE_HIP_DYLIB_PATH");
  if (raw_dylib_path_env_cstr) {
    std::string_view rest(raw_dylib_path_env_cstr);
    for (;;) {
      auto pos = rest.find(';');
      if (pos == std::string_view::npos) {
        hip_lib_search_paths.emplace_back(rest);
        break;
      }
      std::string_view first = rest.substr(0, pos);
      rest = rest.substr(pos + 1);
      hip_lib_search_paths.emplace_back(first);
    }
  }
}

void AMDGPUSystemBuilder::Enumerate() {
  if (hip_hal_driver_) return;

  iree_hal_hip_driver_options_t driver_options;
  iree_hal_hip_driver_options_initialize(&driver_options);

  // Search path.
  std::vector<iree_string_view_t> hip_lib_search_path_sv;
  hip_lib_search_path_sv.resize(hip_lib_search_paths.size());
  for (size_t i = 0; i < hip_lib_search_paths.size(); ++i) {
    hip_lib_search_path_sv[i].data = hip_lib_search_paths[i].data();
    hip_lib_search_path_sv[i].size = hip_lib_search_paths[i].size();
  }
  driver_options.hip_lib_search_paths = hip_lib_search_path_sv.data();
  driver_options.hip_lib_search_path_count = hip_lib_search_path_sv.size();

  SHORTFIN_THROW_IF_ERROR(iree_hal_hip_driver_create(
      IREE_SV("hip"), &driver_options, &default_device_params_,
      host_allocator(), hip_hal_driver_.for_output()));

  // Get available devices and filter into visible_devices_.
  iree_host_size_t available_devices_count = 0;
  allocated_ptr<iree_hal_device_info_t> raw_available_devices(host_allocator());
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_query_available_devices(
      hip_hal_driver_, host_allocator(), &available_devices_count,
      raw_available_devices.for_output()));
  for (iree_host_size_t i = 0; i < available_devices_count; ++i) {
    iree_hal_device_info_t *info = &raw_available_devices.get()[i];
    // TODO: Filter based on visibility list.
    visible_devices_.push_back(*info);
    logging::info("Enumerated visible AMDGPU device: {} ({})",
                  to_string_view(visible_devices_.back().path),
                  to_string_view(visible_devices_.back().name));
  }
}

LocalSystemPtr AMDGPUSystemBuilder::CreateLocalSystem() {
  auto lsys = std::make_shared<LocalSystem>(host_allocator());
  Enumerate();
  // TODO: Real NUMA awareness.
  lsys->InitializeNodes(1);
  lsys->InitializeHalDriver(SYSTEM_DEVICE_CLASS, hip_hal_driver_);

  // Initialize all visible GPU devices.
  for (size_t i = 0; i < visible_devices_.size(); ++i) {
    auto &it = visible_devices_[i];
    iree_hal_device_ptr device;
    SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
        hip_hal_driver_, it.device_id, 0, nullptr, host_allocator(),
        device.for_output()));
    lsys->InitializeHalDevice(std::make_unique<AMDGPUDevice>(
        LocalDeviceAddress(
            /*system_device_class=*/SYSTEM_DEVICE_CLASS,
            /*logical_device_class=*/LOGICAL_DEVICE_CLASS,
            /*hal_driver_prefix=*/HAL_DRIVER_PREFIX,
            /*instance_ordinal=*/i,
            /*queue_ordinal=*/0,
            /*instance_topology_address=*/{0}),
        std::move(device), /*node_affinity=*/0,
        /*node_locked=*/false));
  }

  // Initialize CPU devices if requested.
  if (cpu_devices_enabled) {
    // Delegate to the HostCPUSystemConfig to configure CPU devices.
    // This will need to become more complicated and should happen after
    // GPU configuration when mating NUMA nodes, etc.
    InitializeHostCPUDefaults();
    auto *driver = InitializeHostCPUDriver(*lsys);
    InitializeHostCPUDevices(*lsys, driver);
  }

  lsys->FinishInitialization();
  return lsys;
}

}  // namespace shortfin::systems
