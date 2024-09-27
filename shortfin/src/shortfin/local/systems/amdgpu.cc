// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/systems/amdgpu.h"

#include "shortfin/support/logging.h"

namespace shortfin::local::systems {

namespace {
const std::string_view SYSTEM_DEVICE_CLASS = "amdgpu";
const std::string_view LOGICAL_DEVICE_CLASS = "gpu";
const std::string_view HAL_DRIVER_PREFIX = "hip";
}  // namespace

AMDGPUSystemBuilder::AMDGPUSystemBuilder(iree_allocator_t host_allocator,
                                         ConfigOptions options)
    : HostCPUSystemBuilder(host_allocator, std::move(options)),
      available_devices_(host_allocator) {
  InitializeDefaultSettings();
  iree_hal_hip_device_params_initialize(&default_device_params_);
}

AMDGPUSystemBuilder::~AMDGPUSystemBuilder() = default;

void AMDGPUSystemBuilder::InitializeDefaultSettings() {
  // Library search path.
  std::optional<std::string_view> search_path =
      config_options().GetOption("amdgpu_hip_dylib_path");
  if (!search_path) {
    // Fall back to the raw "IREE_HIP_DYLIB_PATH" for compatibility with IREE
    // tools.
    search_path = config_options().GetRawEnv("IREE_HIP_DYLIB_PATH");
  }
  if (search_path) {
    for (auto entry : config_options().Split(*search_path, ';')) {
      hip_lib_search_paths_.push_back(std::string(entry));
    }
  }

  // CPU devices.
  cpu_devices_enabled_ = config_options().GetBool("amdgpu_cpu_devices_enabled");

  // Visible devices.
  std::optional<std::string_view> visible_devices_option =
      config_options().GetOption("amdgpu_visible_devices");
  if (visible_devices_option) {
    auto splits = config_options().Split(*visible_devices_option, ';');
    visible_devices_.emplace();
    for (auto split_sv : splits) {
      visible_devices_->emplace_back(split_sv);
    }
  }
}

void AMDGPUSystemBuilder::Enumerate() {
  if (hip_hal_driver_) return;

  iree_hal_hip_driver_options_t driver_options;
  iree_hal_hip_driver_options_initialize(&driver_options);

  // Search path.
  std::vector<iree_string_view_t> hip_lib_search_path_sv;
  hip_lib_search_path_sv.resize(hip_lib_search_paths_.size());
  for (size_t i = 0; i < hip_lib_search_paths_.size(); ++i) {
    hip_lib_search_path_sv[i].data = hip_lib_search_paths_[i].data();
    hip_lib_search_path_sv[i].size = hip_lib_search_paths_[i].size();
  }
  driver_options.hip_lib_search_paths = hip_lib_search_path_sv.data();
  driver_options.hip_lib_search_path_count = hip_lib_search_path_sv.size();

  SHORTFIN_THROW_IF_ERROR(iree_hal_hip_driver_create(
      IREE_SV("hip"), &driver_options, &default_device_params_,
      host_allocator(), hip_hal_driver_.for_output()));

  // Get available devices and filter into visible_devices_.
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_query_available_devices(
      hip_hal_driver_, host_allocator(), &available_devices_count_,
      available_devices_.for_output()));
  for (iree_host_size_t i = 0; i < available_devices_count_; ++i) {
    iree_hal_device_info_t *info = &available_devices_.get()[i];
    logging::debug("Enumerated available AMDGPU device: {} ({})",
                   to_string_view(info->path), to_string_view(info->name));
  }
}

std::vector<std::string> AMDGPUSystemBuilder::GetAvailableDeviceIds() {
  Enumerate();
  std::vector<std::string> results;
  for (iree_host_size_t i = 0; i < available_devices_count_; ++i) {
    iree_hal_device_info_t *info = &available_devices_.get()[i];
    results.emplace_back(to_string_view(info->path));
  }
  return results;
}

SystemPtr AMDGPUSystemBuilder::CreateSystem() {
  auto lsys = std::make_shared<System>(host_allocator());
  Enumerate();

  // TODO: Real NUMA awareness.
  lsys->InitializeNodes(1);
  lsys->InitializeHalDriver(SYSTEM_DEVICE_CLASS, hip_hal_driver_);

  // Must have some device visible.
  if (available_devices_count_ == 0 &&
      (!visible_devices_ || visible_devices_->empty())) {
    throw std::invalid_argument("No AMDGPU devices found/visible");
  }

  // If a visibility list, process that.
  std::vector<iree_hal_device_id_t> used_device_ids;
  if (visible_devices_) {
    used_device_ids.reserve(visible_devices_->size());
    // In large scale partitioned cases, there could be 64+ devices, so we want
    // to avoid a linear scan. Also, in some cases with partitioned physical
    // devices, there can be multiple devices with the same id. In this case,
    // the visibility list also connotes order/repetition, so we store with
    // vectors.
    std::unordered_map<std::string_view,
                       std::vector<std::optional<iree_hal_device_id_t>>>
        visible_device_hal_ids;
    for (size_t i = 0; i < available_devices_count_; ++i) {
      iree_hal_device_info_t *info = &available_devices_.get()[i];
      visible_device_hal_ids[to_string_view(info->path)].push_back(
          info->device_id);
    }

    for (auto &visible_device_id : *visible_devices_) {
      auto found_it = visible_device_hal_ids.find(visible_device_id);
      if (found_it == visible_device_hal_ids.end()) {
        throw std::invalid_argument(fmt::format(
            "Requested visible device '{}' was not found on the system "
            "(available: '{}')",
            visible_device_id, fmt::join(GetAvailableDeviceIds(), ";")));
      }

      bool found = false;
      auto &bucket = found_it->second;
      for (auto &hal_id : bucket) {
        if (hal_id) {
          found = true;
          used_device_ids.push_back(*hal_id);
          hal_id.reset();
        }
      }

      if (!found) {
        throw std::invalid_argument(
            fmt::format("Requested visible device '{}' was requested more "
                        "times than present on the system ({})",
                        visible_device_id, bucket.size()));
      }
    }
  } else {
    for (iree_host_size_t i = 0; i < available_devices_count_; ++i) {
      iree_hal_device_info_t *info = &available_devices_.get()[i];
      used_device_ids.push_back(info->device_id);
    }
  }

  // Initialize all used GPU devices.
  for (size_t instance_ordinal = 0; instance_ordinal < used_device_ids.size();
       ++instance_ordinal) {
    iree_hal_device_id_t device_id = used_device_ids[instance_ordinal];
    iree::hal_device_ptr device;
    SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
        hip_hal_driver_, device_id, 0, nullptr, host_allocator(),
        device.for_output()));
    lsys->InitializeHalDevice(std::make_unique<AMDGPUDevice>(
        DeviceAddress(
            /*system_device_class=*/SYSTEM_DEVICE_CLASS,
            /*logical_device_class=*/LOGICAL_DEVICE_CLASS,
            /*hal_driver_prefix=*/HAL_DRIVER_PREFIX,
            /*instance_ordinal=*/instance_ordinal,
            /*queue_ordinal=*/0,
            /*instance_topology_address=*/{0}),
        /*hal_device=*/device,
        /*node_affinity=*/0,
        /*capabilities=*/static_cast<uint32_t>(Device::Capabilities::NONE)));
  }

  // Initialize CPU devices if requested.
  if (cpu_devices_enabled_) {
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

}  // namespace shortfin::local::systems