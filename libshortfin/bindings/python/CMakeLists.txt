# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

# libshortfin publishes multiple python packages: - _shortfin: Trampoline
# __init__.py which looks at environment variables to load an appropriate native
# library. - _shortfin_default.lib: Native library as a default, uninstrumented
# build. - _shortfin_tracing.lib: Native library with tracing enabled (TODO). -
# Others.

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(shortfin_python_extension NB_STATIC LTO
  array_binding.cc
  lib_ext.cc
)

set_target_properties(shortfin_python_extension
                      PROPERTIES OUTPUT_NAME "_shortfin_default/lib")

target_link_libraries(shortfin_python_extension
  # TODO: This should be configurable as to whether we link to the static
  # or dynamic version.
  PRIVATE shortfin
)

nanobind_add_stub(
  shortfin_python_extension_stub
  MODULE _shortfin_default.lib
  OUTPUT _shortfin_default/lib.pyi
  DEPENDS shortfin_python_extension
)
