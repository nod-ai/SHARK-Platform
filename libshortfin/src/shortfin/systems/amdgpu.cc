// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/systems/amdgpu.h"

#include "shortfin/support/logging.h"

namespace shortfin::systems {

AMDGPUSystemConfig::AMDGPUSystemConfig(iree_allocator_t host_allocator)
    : HostCPUSystemConfig(host_allocator) {
  InitializeDefaultSetting();
  iree_hal_hip_device_params_initialize(&default_device_params_);
}

AMDGPUSystemConfig::~AMDGPUSystemConfig() = default;

void AMDGPUSystemConfig::InitializeDefaultSetting() {
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

void AMDGPUSystemConfig::Enumerate() {
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

LocalSystemPtr AMDGPUSystemConfig::CreateLocalSystem() {
  auto lsys = std::make_shared<LocalSystem>(host_allocator());
  Enumerate();
  lsys->InitializeHalDriver("amdgpu", hip_hal_driver_);

  // Initialize all visible GPU devices.
  std::vector<iree_hal_device_ptr> devices;
  devices.reserve(visible_devices_.size());
  for (auto &it : visible_devices_) {
    iree_hal_device_ptr device;
    SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
        hip_hal_driver_, it.device_id, 0, nullptr, host_allocator(),
        device.for_output()));
    devices.push_back(std::move(device));
  }
  lsys->InitializeHalDevices("gpu", std::move(devices));

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
