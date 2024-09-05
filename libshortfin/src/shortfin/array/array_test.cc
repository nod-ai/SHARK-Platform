// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>

#include "shortfin/array/api.h"
#include "shortfin/local/systems/host.h"

using namespace shortfin;
using namespace shortfin::local;
using namespace shortfin::array;

namespace {

class DeviceArrayTest : public testing::Test {
 protected:
  DeviceArrayTest() {}

  void SetUp() override {
    system = systems::HostCPUSystemBuilder().CreateSystem();
    scope = system->CreateScope(system->init_worker(), system->devices());
    device = scope->device(0);
  }
  void TearDown() override {
    system->Shutdown();
    system.reset();
  }

  SystemPtr system;
  std::shared_ptr<Scope> scope;
  ScopedDevice device;
};

TEST_F(DeviceArrayTest, contents_to_s_valid) {
  device_array ary1 = device_array::for_host(
      device, std::to_array<size_t>({2, 3}), DType::float32());
  {
    auto map = ary1.typed_data_w<float>();
    std::fill(map.begin(), map.end(), 42.0);
  }

  std::optional<std::string> contents = ary1.contents_to_s();
  ASSERT_TRUE(contents);
  EXPECT_EQ(*contents, "{{ 42.,  42.,  42.},\n { 42.,  42.,  42.}}");
}

TEST_F(DeviceArrayTest, contents_to_s_invalid) {
  device_array ary1 = device_array::for_host(
      device, std::to_array<size_t>({2, 3}), DType::opaque32());
  // No xtensor adaptor for opaque32.
  std::optional<std::string> contents = ary1.contents_to_s();
  ASSERT_FALSE(contents);
}

}  // namespace
