// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/dtype.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>

namespace shortfin::array {

TEST(array_dtype, basics) {
  EXPECT_EQ(DType::complex64().name(), "complex64");
  EXPECT_EQ(static_cast<iree_hal_element_type_t>(DType::complex64()),
            IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64);
  EXPECT_TRUE(DType::complex64() == DType::complex64());
  EXPECT_TRUE(DType::complex64() != DType::complex128());
}

TEST(array_dtype, compure_dense_nd_size) {
  // 0d special case.
  EXPECT_EQ(DType::float32().compute_dense_nd_size({}), 4);
  // 0 extent special case.
  EXPECT_EQ(DType::float32().compute_dense_nd_size(std::array<size_t, 2>{0, 4}),
            0);
  EXPECT_EQ(DType::float32().compute_dense_nd_size(std::array<size_t, 2>{2, 4}),
            32);
}

}  // namespace shortfin::array
