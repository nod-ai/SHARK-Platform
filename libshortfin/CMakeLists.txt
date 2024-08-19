# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
    FATAL_ERROR
      "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source."
  )
endif()

project(
  "libshortfin"
  VERSION 0.9
  LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
# https://discourse.cmake.org/t/cmake-3-28-cmake-cxx-compiler-clang-scan-deps-notfound-not-found/9244/3
set(CMAKE_CXX_SCAN_FOR_MODULES 0)

option(SHORTFIN_BUILD_PYTHON_BINDINGS "Builds Python Bindings" OFF)

# Includes.
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(shortfin_library)

# Dependencies.
find_package(spdlog)
find_package(xtensor)
find_package(IREERuntime)

add_subdirectory(src)

if(SHORTFIN_BUILD_PYTHON_BINDINGS)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  add_subdirectory(bindings/python)
  set(SHORTFIN_PYTHON_CPP_PREBUILT "TRUE") # See setup.py.
  configure_file(setup.py setup.py @ONLY)
  configure_file(pyproject.toml pyproject.toml COPYONLY)
endif()
