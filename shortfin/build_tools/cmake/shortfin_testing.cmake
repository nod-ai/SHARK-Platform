# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Downloads some test data file as part of configure.
# This does a download->rename in an attempt to be robust to partial downloads.
# It should not be used to manage large test data files or anything sensitive
# enough to require a hash check.
# The output file is added as an additional clean file on the global
# shortfin_testdata_deps target, meaning the "ninja clean" will remove it.
# It is also added to the current directories list of configure depends, which
# means that if ninja is run and it is not present, cmake will be re-invoked.
function(shortfin_download_test_data)
  cmake_parse_arguments(
    _RULE
    ""
    "URL;OUTPUT_FILE"
    ""
    ${ARGN}
  )
  if(NOT EXISTS "${_RULE_OUTPUT_FILE}")
    set(_stage_file "${_RULE_OUTPUT_FILE}.stage")
    message(STATUS "Downloading test data ${_RULE_URL} -> ${_RULE_OUTPUT_FILE}")
    file(DOWNLOAD "${_RULE_URL}" "${_stage_file}" STATUS _status)
    list(POP_FRONT _status _status_code)
    if(_status_code EQUAL "0")
      file(RENAME "${_stage_file}" "${_RULE_OUTPUT_FILE}")
    else()
      message(SEND_ERROR "Error downloading file ${_RULE_URL} -> ${_RULE_OUTPUT_FILE}")
    endif()
  endif()

  # Make clean remove it.
  set_property(
    TARGET shortfin_testdata_deps
    APPEND PROPERTY ADDITIONAL_CLEAN_FILES
      "${CMAKE_CURRENT_BINARY_DIR}/tokenizer.json"
  )

  # And make us reconfigure if it isn't there.
  set_property(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${_RULE_OUTPUT_FILE}")
endfunction()
