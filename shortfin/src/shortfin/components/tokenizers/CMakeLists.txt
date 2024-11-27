# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT SHORTFIN_ENABLE_TOKENIZERS)
  return()
endif()

shortfin_cc_component(
  NAME
    shortfin_tokenizers
  HDRS
    tokenizers.h
  SRCS
    tokenizers.cc
  DEFINES
    SHORTFIN_HAVE_TOKENIZERS
  COMPONENTS
    shortfin_support
  DEPS
    tokenizers_cpp
)
set_property(GLOBAL APPEND
  PROPERTY SHORTFIN_LIB_OPTIONAL_COMPONENTS
  shortfin_tokenizers)
target_compile_definitions(shortfin_public_defs INTERFACE SHORTFIN_HAVE_TOKENIZERS)

# Download test data.
shortfin_download_test_data(
  URL "https://huggingface.co/google-bert/bert-base-cased/resolve/cd5ef92a9fb2f889e972770a36d4ed042daf221e/tokenizer.json"
  OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/tokenizer.json"
)

# Note that tests run from the binary dir of the project.
shortfin_gtest_test(
  NAME shortfin_tokenizers_test
  SRCS
    tokenizers_test.cc
)
