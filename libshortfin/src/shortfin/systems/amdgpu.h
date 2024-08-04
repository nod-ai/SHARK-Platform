// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SYSTEMS_AMDGPU_H
#define SHORTFIN_SYSTEMS_AMDGPU_H

#include <vector>

#include "iree/hal/drivers/hip/api.h"
#include "shortfin/local_system.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"
#include "shortfin/systems/host.h"

namespace shortfin::systems {

// System configuration for some subset of AMD GPUs connected to the local
// system. Note that this inherits from HostCPUSystemConfig, allowing joint
// configuration of a heterogenous CPU/GPU system. Depending on the specific
// system, this can involve more than simple starting CPU drivers: datacenter
// GPU systems have specific NUMA configurations that need to be mated.
class SHORTFIN_API AMDGPUSystemConfig : public HostCPUSystemConfig {
 public:
  AMDGPUSystemConfig(iree_allocator_t host_allocator);
  AMDGPUSystemConfig() : AMDGPUSystemConfig(iree_allocator_system()) {}
  ~AMDGPUSystemConfig();

  // Triggers driver setup and initial device enumeration. No-op if already
  // done.
  void Enumerate();

  LocalSystemPtr CreateLocalSystem() override;

  // Settings.
  bool cpu_devices_enabled = false;

  // See iree_hal_hip_driver_options_t::hip_lib_search_paths. Each is either
  // a directory or "file:" prefixed path to a specific HIP dynamic library.
  // This is typically libamdhip64.so or amdhip64.dll.
  // If the environment variable IREE_HIP_DYLIB_PATH is present, then it is
  // split on ';' and each entry added here (for compatibility with IREE
  // tools).
  // Changing these paths after enumeration has no effect.
  std::vector<std::string> hip_lib_search_paths;

 private:
  void InitializeDefaultSetting();

  // Valid at construction time.
  iree_hal_hip_device_params_t default_device_params_;

  // Valid post enumeration.
  iree_hal_driver_ptr hip_hal_driver_;
  std::vector<iree_hal_device_info_t> visible_devices_;
};

}  // namespace shortfin::systems

#endif  // SHORTFIN_SYSTEMS_AMDGPU_H
