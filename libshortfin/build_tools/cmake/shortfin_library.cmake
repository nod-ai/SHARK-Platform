# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

set(SHORTFIN_DEFAULT_COPTS
  # General clang and GCC options application to C and C++.
  $<$<C_COMPILER_ID:AppleClang,Clang,GNU>:
  -Wall
  -Werror
  >

  # General MSVC options applicable to C and C++.
  $<$<C_COMPILER_ID:MSVC>:
  >
)

function(shortfin_public_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "COMPONENTS"
    ${ARGN}
  )
  if(SHORTFIN_BUILD_STATIC)
    # Static library.
    shortfin_components_to_static_libs(_STATIC_COMPONENTS ${_RULE_COMPONENTS})
    add_library("${_RULE_NAME}-static" STATIC)
    target_link_libraries(
      "${_RULE_NAME}-static" PUBLIC ${_STATIC_COMPONENTS}
    )
  endif()

  if(SHORTFIN_BUILD_DYNAMIC)
    # Dylib library.
    shortfin_components_to_dynamic_libs(_DYLIB_COMPONENTS ${_RULE_COMPONENTS})
    add_library("${_RULE_NAME}" SHARED)
    target_compile_definitions("${_RULE_NAME}" INTERFACE _SHORTFIN_USING_DYLIB)
    target_link_libraries(
      "${_RULE_NAME}" PUBLIC ${_DYLIB_COMPONENTS}
    )
    set_target_properties("${_RULE_NAME}" PROPERTIES
      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
      SOVERSION ${SOVERSION}
    )
  endif()
endfunction()

function(shortfin_cc_component)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "HDRS;SRCS;DEPS;COMPONENTS"
    ${ARGN}
  )
  if(SHORTFIN_BUILD_STATIC)
    # Static object library.
    set(_STATIC_OBJECTS_NAME "${_RULE_NAME}.objects")
    shortfin_components_to_static_libs(_STATIC_COMPONENTS ${_RULE_COMPONENTS})
    add_library(${_STATIC_OBJECTS_NAME} OBJECT)
    target_sources(${_STATIC_OBJECTS_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_HDRS}
    )
    target_compile_options(${_STATIC_OBJECTS_NAME} PRIVATE ${SHORTFIN_DEFAULT_COPTS})
    target_link_libraries(${_STATIC_OBJECTS_NAME}
      PUBLIC
        _shortfin_defs
        ${_STATIC_COMPONENTS}
        ${_RULE_DEPS}
    )
  endif()

  if(SHORTFIN_BUILD_DYNAMIC)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    set(_DYLIB_OBJECTS_NAME "${_RULE_NAME}.dylib.objects")
    shortfin_components_to_dynamic_libs(_DYLIB_COMPONENTS ${_RULE_COMPONENTS})
    # Dylib object library.
    add_library(${_DYLIB_OBJECTS_NAME} OBJECT)
    target_sources(${_DYLIB_OBJECTS_NAME}
      PRIVATE
        ${_RULE_SRCS}
        ${_RULE_HDRS}
    )
    target_compile_options(${_DYLIB_OBJECTS_NAME} PRIVATE ${SHORTFIN_DEFAULT_COPTS})
    target_link_libraries(${_DYLIB_OBJECTS_NAME}
      PUBLIC
        _shortfin_defs
        ${_DYLIB_COMPONENTS}
        ${_RULE_DEPS}
    )
    set_target_properties(
      ${_DYLIB_OBJECTS_NAME} PROPERTIES
      CXX_VISIBILITY_PRESET hidden
      C_VISIBILITY_PRESET hidden
      VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_definitions(${_DYLIB_OBJECTS_NAME}
      PRIVATE _SHORTFIN_BUILDING_DYLIB)
  endif()
endfunction()

function(shortfin_components_to_static_libs out_static_libs)
  set(_LIBS ${ARGN})
  list(TRANSFORM _LIBS APPEND ".objects")
  set(${out_static_libs} ${_LIBS} PARENT_SCOPE)
endfunction()

function(shortfin_components_to_dynamic_libs out_dynamic_libs)
  set(_LIBS ${ARGN})
  list(TRANSFORM _LIBS APPEND ".dylib.objects")
  set(${out_dynamic_libs} "${_LIBS}" PARENT_SCOPE)
endfunction()

function(shortfin_gtest_test)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;DEPS"
    ${ARGN}
  )

  if(NOT SHORTFIN_BUILD_TESTS)
    return()
  endif()

  add_executable(${_RULE_NAME} ${_RULE_SRCS})
  target_link_libraries(${_RULE_NAME} PRIVATE
    ${_RULE_DEPS}
    ${SHORTFIN_LINK_LIBRARY_NAME}
    GTest::gmock
    GTest::gtest_main
  )
  gtest_discover_tests(${_RULE_NAME})
endfunction()
