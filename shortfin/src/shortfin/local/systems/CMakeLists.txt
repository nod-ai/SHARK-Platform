# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_SYSTEM_COMPONENTS)

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
  DEFINES
    SHORTFIN_HAVE_HOSTCPU
  DEPS
    iree_hal_drivers_local_task_task_driver
    iree_hal_local_executable_loader
    iree_hal_local_executable_plugin
    iree_hal_local_executable_plugin_manager
    iree_hal_local_loaders_registration_registration
    iree_hal_local_local
    iree_task_api
    iree_task_task
)
list(APPEND _SYSTEM_COMPONENTS shortfin_systems_host)

if(SHORTFIN_SYSTEMS_AMDGPU)
  shortfin_cc_component(
    NAME
      shortfin_systems_amdgpu
    HDRS
      amdgpu.h
    SRCS
      amdgpu.cc
    DEFINES
      SHORTFIN_HAVE_AMDGPU
    COMPONENTS
      shortfin_local
      shortfin_support
    DEPS
      iree_hal_drivers_hip_hip
  )
  list(APPEND _SYSTEM_COMPONENTS shortfin_systems_amdgpu)
endif()

shortfin_cc_component(
  NAME
    shortfin_systems_factory
  SRCS
    factory.cc
  COMPONENTS
    ${_SYSTEM_COMPONENTS}
)

set_property(GLOBAL APPEND
  PROPERTY SHORTFIN_LIB_OPTIONAL_COMPONENTS ${_SYSTEM_COMPONENTS})
