// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/dims.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>

namespace shortfin::array {

TEST(array_dims, empty) {
  Dims dims;
  EXPECT_TRUE(dims.empty());
  EXPECT_EQ(dims.size(), 0);
}

TEST(array_dims, inline_init) {
  Dims dims(3, 42);
  EXPECT_EQ(dims.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dims[i], 42);
  }

  Dims copy(dims);
  EXPECT_EQ(dims.size(), copy.size());
  EXPECT_TRUE(std::equal(dims.begin(), dims.end(), copy.begin()));
  EXPECT_TRUE(std::equal(dims.cbegin(), dims.cend(), copy.begin()));

  Dims move = std::move(copy);
  EXPECT_EQ(dims.size(), move.size());
  EXPECT_TRUE(std::equal(dims.begin(), dims.end(), move.begin()));

  Dims assign;
  assign = dims;
  EXPECT_EQ(dims.size(), assign.size());
  EXPECT_TRUE(std::equal(dims.begin(), dims.end(), assign.begin()));

  EXPECT_EQ(*dims.data(), *assign.data());

  assign.clear();
  EXPECT_TRUE(assign.empty());
}

TEST(array_dims, dynamic_init) {
  Dims dims(12, 42);
  EXPECT_EQ(dims.size(), 12);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(dims[i], 42);
  }

  Dims copy(dims);
  EXPECT_EQ(dims.size(), copy.size());
  EXPECT_TRUE(std::equal(dims.begin(), dims.end(), copy.begin()));
  EXPECT_TRUE(std::equal(dims.cbegin(), dims.cend(), copy.begin()));

  Dims move = std::move(copy);
  EXPECT_EQ(dims.size(), move.size());
  EXPECT_TRUE(std::equal(dims.begin(), dims.end(), move.begin()));

  Dims assign;
  assign = dims;
  EXPECT_EQ(dims.size(), assign.size());
  EXPECT_TRUE(std::equal(dims.begin(), dims.end(), assign.begin()));

  EXPECT_EQ(*dims.data(), *assign.data());

  assign.clear();
  EXPECT_TRUE(assign.empty());
}

TEST(array_dims, resize_same_size) {
  Dims dims(3, 64);
  dims.resize(3, 32);
  EXPECT_EQ(dims.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
}

TEST(array_dims, resize_inline_to_inline) {
  Dims dims(3, 64);
  dims.resize(5, 32);
  EXPECT_EQ(dims.size(), 5);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
  for (size_t i = 3; i < 5; ++i) {
    EXPECT_EQ(dims[i], 32);
  }
}

TEST(array_dims, resize_inline_to_dynamic) {
  Dims dims(3, 64);
  dims.resize(12, 32);
  EXPECT_EQ(dims.size(), 12);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
  for (size_t i = 3; i < 12; ++i) {
    EXPECT_EQ(dims[i], 32);
  }
}

TEST(array_dims, resize_inline_truncate) {
  Dims dims(5, 64);
  dims.resize(2, 32);
  EXPECT_EQ(dims.size(), 2);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
}

TEST(array_dims, resize_dynamic_to_dynamic) {
  Dims dims(12, 64);
  dims.resize(15, 32);
  EXPECT_EQ(dims.size(), 15);
  for (size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
  for (size_t i = 12; i < 15; ++i) {
    EXPECT_EQ(dims[i], 32);
  }
}

TEST(array_dims, resize_truncate_to_inline) {
  Dims dims(12, 64);
  dims.resize(3, 32);
  EXPECT_EQ(dims.size(), 3);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
}

TEST(array_dims, resize_truncate_to_dynamic) {
  Dims dims(12, 64);
  dims.resize(10, 32);
  EXPECT_EQ(dims.size(), 10);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(dims[i], 64);
  }
}

}  // namespace shortfin::array
