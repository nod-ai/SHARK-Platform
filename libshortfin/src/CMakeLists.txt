# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

add_subdirectory(shortfin)

# Common definitions exported from both static and dynamic libraries.
add_library(_shortfin_defs INTERFACE)
target_include_directories(
  _shortfin_defs INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>)


set(_INIT_INTERNAL_DEPS)
if(SHORTFIN_SYSTEMS_AMDGPU)
  list(APPEND _INIT_INTERNAL_DEPS shortfin_systems_amdgpu)
endif()

shortfin_public_library(
  NAME
  ${SHORTFIN_LINK_LIBRARY_NAME}
  COMPONENTS
    shortfin_array
    shortfin_local
    shortfin_support
    shortfin_systems_host
    ${_INIT_INTERNAL_DEPS}
)

set_target_properties(${SHORTFIN_LINK_LIBRARY_NAME}
                       PROPERTIES
                       VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
                       SOVERSION ${SOVERSION})
