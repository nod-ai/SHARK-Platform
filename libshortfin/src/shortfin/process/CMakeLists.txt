# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

shortfin_cc_component(
  NAME
    shortfin_process
  HDRS
    worker.h
  SRCS
    worker.cc
  COMPONENTS
    shortfin_support
  DEPS
    iree_base_base
    iree_base_loop_sync
)
