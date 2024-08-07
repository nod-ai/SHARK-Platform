# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

add_subdirectory(array)
add_subdirectory(support)
add_subdirectory(systems)

shortfin_cc_component(
  NAME
    shortfin_local_system
  HDRS
    local_device.h
    local_scope.h
    local_system.h
    worker.h
  SRCS
    local_device.cc
    local_scope.cc
    local_system.cc
    worker.cc
  COMPONENTS
    shortfin_support
  DEPS
    iree_base_base
    iree_base_loop_sync
    iree_hal_hal
    iree_modules_hal_hal
    iree_vm_vm
)
