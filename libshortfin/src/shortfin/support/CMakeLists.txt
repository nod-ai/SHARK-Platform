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
    blocking_executor.h
    globals.h
    iree_helpers.h
    iree_concurrency.h
    logging.h
    stl_extras.h
  SRCS
    blocking_executor.cc
    globals.cc
    iree_helpers.cc
    logging.cc
  DEPS
    iree_base_base
    # TODO: Maybe reclassify some of these low level, shared support entities
    # as externally usable.
    iree_base_internal_threading
    iree_hal_hal
    iree_modules_hal_types
    spdlog::spdlog
)

shortfin_gtest_test(
  NAME shortfin_support_test
  SRCS
    # Order is specific: lower level tests before higher level.
    iree_helpers_test.cc
    iree_concurrency_test.cc
    blocking_executor_test.cc
    stl_extras_test.cc
)
