// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/iree_concurrency.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace shortfin {

TEST(iree_shared_event, copy) {
  auto se1 = iree::shared_event::create(true);
  EXPECT_EQ(se1.ref_count(), 1);
  EXPECT_TRUE(se1);
  {
    iree::shared_event::ref se2 = se1;
    EXPECT_EQ(se1.ref_count(), 2);
    EXPECT_EQ(se2.ref_count(), 2);
    EXPECT_TRUE(se2);
  }
  EXPECT_EQ(se1.ref_count(), 1);
}

TEST(iree_shared_event, move) {
  auto se1 = iree::shared_event::create(true);
  EXPECT_EQ(se1.ref_count(), 1);
  EXPECT_TRUE(se1);
  {
    iree::shared_event::ref se2 = std::move(se1);
    EXPECT_FALSE(se1);
    EXPECT_EQ(se2.ref_count(), 1);
    EXPECT_TRUE(se2);
  }
}

}  // namespace shortfin
