# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

function(shortfin_public_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "COMPONENTS"
    ${ARGN}
  )

  set(_STATIC_COMPONENTS ${_RULE_COMPONENTS})
  list(TRANSFORM _STATIC_COMPONENTS APPEND ".objects")

  set(_DYLIB_COMPONENTS ${_RULE_COMPONENTS})
  list(TRANSFORM _DYLIB_COMPONENTS APPEND ".dylib.objects")

  # Static library.
  add_library("${_RULE_NAME}-static" STATIC)
  target_link_libraries(
    "${_RULE_NAME}-static" PUBLIC ${_STATIC_COMPONENTS}
  )

  # Dylib library.
  add_library("${_RULE_NAME}" SHARED)
  target_compile_definitions("${_RULE_NAME}" INTERFACE _SHORTFIN_USING_DYLIB)
  target_link_libraries(
    "${_RULE_NAME}" PUBLIC ${_DYLIB_COMPONENTS}
  )
endfunction()

function(shortfin_cc_component)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "HDRS;SRCS;DEPS"
    ${ARGN}
  )
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  set(_STATIC_OBJECTS_NAME "${_RULE_NAME}.objects")
  set(_DYLIB_OBJECTS_NAME "${_RULE_NAME}.dylib.objects")

  # Static object library.
  add_library(${_STATIC_OBJECTS_NAME} OBJECT)
  target_sources(${_STATIC_OBJECTS_NAME}
    PRIVATE
      ${_RULE_SRCS}
      ${_RULE_HDRS}
  )
  target_link_libraries(${_STATIC_OBJECTS_NAME}
    PUBLIC
      ${_RULE_DEPS}
      _shortfin_defs
    )

  # Dylib object library.
  add_library(${_DYLIB_OBJECTS_NAME} OBJECT)
  target_sources(${_DYLIB_OBJECTS_NAME}
    PRIVATE
      ${_RULE_SRCS}
      ${_RULE_HDRS}
  )
  target_link_libraries(${_DYLIB_OBJECTS_NAME}
    PUBLIC
      ${_RULE_DEPS}
      _shortfin_defs
  )
  set_target_properties(
    ${_DYLIB_OBJECTS_NAME}
    PROPERTIES CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
  target_compile_definitions(${_DYLIB_OBJECTS_NAME}
    PRIVATE _SHORTFIN_BUILDING_DYLIB)
endfunction()
