# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

shortfin_cc_component(
  NAME
    shortfin_support
  HDRS
    api.h
    globals.h
    iree_helpers.h
    logging.h
  SRCS
    globals.cc
    iree_helpers.cc
    logging.cc
  DEPS
    iree_base_base
    iree_hal_hal
    iree_modules_hal_types
    spdlog::spdlog
)
