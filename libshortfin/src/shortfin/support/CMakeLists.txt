# Copyright 2024 Advanced Micro Devices, Inc.
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
    config.h
    globals.h
    iree_helpers.h
    iree_concurrency.h
    logging.h
    stl_extras.h
  SRCS
    blocking_executor.cc
    config.cc
    globals.cc
    iree_helpers.cc
    logging.cc
  DEPS
    iree_base_base
    # TODO: Maybe reclassify some of these low level, shared support entities
    # as externally usable.
    iree_base_internal_threading
    iree_io_file_handle
    iree_hal_hal
    iree_io_parameter_index
    iree_io_parameter_index_provider
    iree_io_parameter_provider
    iree_modules_hal_types
    iree_task_api
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
