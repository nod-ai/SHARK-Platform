# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

shortfin_cc_component(
  NAME
    shortfin_systems_host
  HDRS
    host.h
  SRCS
    host.cc
  COMPONENTS
    shortfin_local
    shortfin_support
  DEPS
    iree_hal_drivers_local_task_task_driver
    iree_hal_local_local
    iree_hal_local_executable_loader
    iree_hal_local_executable_plugin
    iree_hal_local_executable_plugin_manager
    iree_task_api
    iree_task_task
)

shortfin_cc_component(
  NAME
    shortfin_systems_amdgpu
  HDRS
    amdgpu.h
  SRCS
    amdgpu.cc
  COMPONENTS
    shortfin_local
    shortfin_support
  DEPS
    iree_hal_drivers_hip_hip
)
