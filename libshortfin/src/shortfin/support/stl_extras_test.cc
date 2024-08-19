// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/stl_extras.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace shortfin {

TEST(string_interner, basics) {
  string_interner si;

  std::string_view s1("One");
  std::string_view s1_intern = si.intern(s1);
  EXPECT_NE(s1.data(), s1_intern.data());
  std::string_view s2_intern = si.intern(s1);
  EXPECT_EQ(s1_intern.data(), s2_intern.data());
}

}  // namespace shortfin
