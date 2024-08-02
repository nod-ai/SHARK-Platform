# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

add_subdirectory(support)

shortfin_cc_component(
  NAME
    shortfin_core
  HDRS
    globals.h
  SRCS
    globals.cc
)
