# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

add_subdirectory(support)
add_subdirectory(systems)

shortfin_cc_component(
  NAME
    shortfin_local_system
  HDRS
    local_system.h
  SRCS
    local_system.cc
  DEPS
    iree_base_base
    iree_hal_hal
    iree_modules_hal_hal
    iree_vm_vm
)
